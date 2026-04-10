import math
import torch

from idpbind_cot.src.relaxation_engine.energy_functions.bonded_energy import compute_bond_angle_energy
from idpbind_cot.src.relaxation_engine.energy_functions.dihedral import compute_dihedral_energy, compute_cmap_energy
from idpbind_cot.src.relaxation_engine.energy_functions.non_bonded_interactions import compute_dense_nonbonded_energy, compute_14_energy
from idpbind_cot.src.relaxation_engine.energy_functions.generalized_born import compute_obc2_gb_energy
from idpbind_cot.src.relaxation_engine.utils.topology import build_topology_masks
from idpbind_cot.src.relaxation_engine.utils.fire_opt import FIRE
from idpbind_cot.src.relaxation_engine.utils.constants import (
    DEFAULT_CUTOFF,
    DIELECTRIC_WATER,
    ALPHA_INITIAL,
    ALPHA_DECAY,
    CA_RESTRAINT_K,
    VDW_SCALE_14,
    ELEC_SCALE_14,
)

def get_alpha(step):
    return ALPHA_INITIAL * math.exp(-ALPHA_DECAY * step)

def compute_total_energy(
    coords,
    topology,
    params,
    ca_mask,
    ref_coords,
    step,
    indices_14,
    sigma_14,
    epsilon_14,
    charges_14,
):
    cutoff = DEFAULT_CUTOFF

    global_metadata = params.get('global_metadata', {})
    vdw_scale_14 = global_metadata.get('vdw_scale_14', VDW_SCALE_14)
    elec_scale_14 = global_metadata.get('elec_scale_14', ELEC_SCALE_14)

    current_alpha = get_alpha(step)
    elec_weight = max(0.0, 1.0 - (current_alpha / ALPHA_INITIAL))
    
    e_bond, e_angle, _ = compute_bond_angle_energy(
        coords, topology['springs'], params['k_b'], params['b_0'],
        topology['angles'], params['k_theta'], params['theta_0']
    )
    e_dihedral = compute_dihedral_energy(
        coords, topology['dihedrals'], params['k_phi'], params['n'], params['gamma']
    )

    e_vdw = compute_dense_nonbonded_energy(
        coords=coords, 
        sigma=params['sigma'],
        epsilon=params['epsilon'],
        alpha=current_alpha
    )
    
    e_elec = compute_obc2_gb_energy(
        coords=coords,
        params=params,
    )
                                            
    e_vdw_14, e_elec_14 = compute_14_energy(
        coords, indices_14, sigma_14, epsilon_14, charges_14,
        alpha=current_alpha, vdw_scale=vdw_scale_14, elec_scale=elec_scale_14
    )

    e_cmap = torch.tensor(0.0, device=coords.device)
    if 'cmap' in topology and params.get('global_cmap_grids') is not None:
        e_cmap = compute_cmap_energy(
            coords, topology['cmap'], params['cmap_assignments'], params['global_cmap_grids']
        )

    diff = coords - ref_coords
    dist_sq = torch.sum(diff * diff, dim=-1)
    e_restraint = torch.sum(CA_RESTRAINT_K * dist_sq * ca_mask)

    return e_bond, e_angle, e_dihedral, e_vdw, e_elec, e_vdw_14, e_elec_14, e_cmap, e_restraint




def relax_structure(coords, topology, params, ca_mask, 
                    max_steps=100, tol=2.0, e_tol=1.0, 
                    force_max_steps=False, 
                    print_freq=100, verbose=False,
                    use_as_loss_function=False):
    """
    Executes FIRE energy minimization on a protein structure.

    Args:
        coords: Tensor of shape (N, 3). Must not be a leaf tensor initially if
                it doesn't have requires_grad=True.
        topology: Dictionary containing all structural index tensors.
        params: Dictionary containing all force field parameter tensors.
        ca_mask: Boolean/Float tensor of shape (N,) marking C-alpha atoms.
        max_steps: Maximum number of FIRE optimization steps.
        tol: Max-force tolerance for convergence criteria.
        e_tol: Energy tolerance for convergence criteria.
        force_max_steps: Whether to force the maximum number of steps.

        print_freq: frequency in steps to print the values.
        verbose: whether we print things
        use_as_loss_function: whether to use the final energy as a loss function for backpropagation
            Faster when not used as a loss function because we can skip gradient calculations and just return the final energy value.
            Leaving as an option in case I want to implement into the generative model training loop, 
            but this is also a relaxation function that should be as fast as possible when not used for training, so defaulting to False.
    Returns:
        Optimized coordinate Tensor.
    """
    # 1. Save static reference for restraints (always detached)
    ref_coords = coords.detach().clone()
    
    if use_as_loss_function:
        opt_coords = coords.clone()
        if not opt_coords.requires_grad:
            opt_coords.requires_grad_(True)
        else:
            opt_coords.retain_grad()
        
        optimizer = FIRE([opt_coords], dt_init=0.01, dt_max=0.1, max_step=0.2)
        max_force_t = torch.zeros((), device=opt_coords.device, dtype=opt_coords.dtype)
    else:
        # INFERENCE MODE: Must have requires_grad=True to calculate forces
        opt_coords = coords.detach().clone().requires_grad_(True)
        v = torch.zeros_like(opt_coords)
        dt = torch.tensor(0.01, device=opt_coords.device, dtype=opt_coords.dtype)
        dt_max = torch.tensor(0.1, device=opt_coords.device, dtype=opt_coords.dtype)
        alpha = torch.tensor(0.1, device=opt_coords.device, dtype=opt_coords.dtype)
        N_delay = torch.tensor(5, device=opt_coords.device, dtype=torch.int64)
        N_steps = torch.zeros((), device=opt_coords.device, dtype=torch.int64)
        f_inc = torch.tensor(1.1, device=opt_coords.device, dtype=opt_coords.dtype)
        f_dec = torch.tensor(0.5, device=opt_coords.device, dtype=opt_coords.dtype)
        alpha_start = torch.tensor(0.1, device=opt_coords.device, dtype=opt_coords.dtype)
        f_alpha = torch.tensor(0.99, device=opt_coords.device, dtype=opt_coords.dtype)
        max_force_t = torch.zeros((), device=opt_coords.device, dtype=opt_coords.dtype)
    
    num_atoms = coords.shape[0]
    exclusion_set, indices_14 = build_topology_masks(topology['covalent_bonds'], num_atoms)

    sigma_14 = (params['sigma'][indices_14[0]] + params['sigma'][indices_14[1]]) * 0.5
    epsilon_14 = torch.sqrt(params['epsilon'][indices_14[0]] * params['epsilon'][indices_14[1]])
    charges_14 = params['charges'][indices_14[0]] * params['charges'][indices_14[1]]

    cutoff = DEFAULT_CUTOFF
    eps_rf = DIELECTRIC_WATER
    k_rf = (eps_rf - 1.0) / ((2.0 * eps_rf + 1.0) * (cutoff ** 3))
    c_rf = (3.0 * eps_rf) / ((2.0 * eps_rf + 1.0) * cutoff)

    for step in range(max_steps):
    
        if use_as_loss_function:
            # --- STATEFUL MODE (Training) ---
            if opt_coords.grad is not None:
                opt_coords.grad.zero_()
                
            e_comps = compute_total_energy(
                opt_coords,
                topology,
                params,
                ca_mask,
                ref_coords,
                step,
                indices_14,
                sigma_14,
                epsilon_14,
                charges_14,
            )
            (e_bond, e_angle, e_dihedral, e_vdw, e_elec, e_vdw_14, e_elec_14, e_cmap, e_restraint) = e_comps
            total_energy = sum(e_comps)
            
            forces = torch.autograd.grad(total_energy, opt_coords, create_graph=True)[0]
            opt_coords.grad = forces
            max_force_t = opt_coords.grad.abs().max()
            torch.nn.utils.clip_grad_norm_([opt_coords], max_norm=100.0)
            optimizer.step() 
            
        else:
            # --- INFERENCE MODE ---
            if opt_coords.grad is not None:
                opt_coords.grad.zero_()

            e_comps = compute_total_energy(
                opt_coords,
                topology,
                params,
                ca_mask,
                dynamic_neighbors,
                ref_coords,
                step,
                indices_14,
                sigma_14,
                epsilon_14,
                charges_14,
                k_rf,
                c_rf,
            )
            (e_bond, e_angle, e_dihedral, e_vdw, e_elec, e_vdw_14, e_elec_14, e_cmap, e_restraint) = e_comps
            total_energy = sum(e_comps)
            
            # Standard backward pass. Graph is instantly destroyed afterward.
            total_energy.backward()
            
            # Detach the forces so we can use them safely
            forces = -opt_coords.grad.detach() 
            
            # We MUST use no_grad() to do in-place math on opt_coords
            with torch.no_grad():
                max_force_t = forces.abs().max()
                clip_scale = torch.clamp(100.0 / (max_force_t + 1e-12), max=1.0)
                forces.mul_(clip_scale)

                P = torch.sum(forces * v)
                p_positive = P > 0.0

                n_steps_candidate = N_steps + 1
                N_steps = torch.where(p_positive, n_steps_candidate, torch.zeros_like(N_steps))
                can_grow = N_steps > N_delay

                dt_pos = torch.where(can_grow, torch.minimum(dt * f_inc, dt_max), dt)
                alpha_pos = torch.where(can_grow, alpha * f_alpha, alpha)

                dt = torch.where(p_positive, dt_pos, dt * f_dec)
                alpha = torch.where(p_positive, alpha_pos, alpha_start)
                v = torch.where(p_positive, v, torch.zeros_like(v))

                f_norm = torch.linalg.norm(forces)
                v_norm = torch.linalg.norm(v)

                dir_f = forces / (f_norm + 1e-12)
                steer_gate = (f_norm > 1e-8).to(dtype=opt_coords.dtype)
                v = v * (1.0 - alpha) + dir_f * (alpha * v_norm * steer_gate)

                # In-place integration
                v.add_(forces * dt)
                opt_coords.add_(v * dt)

        # Calculate Energy Delta for convergence check
        t_e_val = None
        max_force_val = None
        if step % 10 == 0 or (verbose and step % print_freq == 0):
            t_e_val = total_energy.detach().item()
            max_force_val = max_force_t.detach().item()

        # Logging
        if verbose and step % print_freq == 0:
            print(
                f"Step {step:4d} | Tot: {t_e_val:>8.1f} | "
                f"Bnd: {e_bond.item():>6.1f} | Ang: {e_angle.item():>6.1f} | "
                f"Dih: {e_dihedral.item():>7.1f} | VDW: {e_vdw.item():>7.1f} | "
                f"Elec: {e_elec.item():>7.1f} | 14-V: {e_vdw_14.item():>6.1f} | "
                f"14-E: {e_elec_14.item():>7.1f} | Res: {e_restraint.item():>6.1f} | "
                f"MaxF: {max_force_val:.2f}"
            )

        # Check for convergence
        if step % 10 == 0:
            if t_e_val is None:
                t_e_val = total_energy.detach().item()
            if max_force_val is None:
                max_force_val = max_force_t.detach().item()

            delta_e = abs(t_e_val - prev_energy)
            prev_energy = t_e_val

            if not force_max_steps and step > 50:
                if max_force_val < tol:
                    if verbose:
                        print(f"Converged at step {step} due to Max Force ({max_force_val:.4f} < {tol}).")
                    break
                
                # If the energy changed by less than e_tol (1.0 kcal/mol) over the last 10 steps
                if delta_e < e_tol:
                    if verbose:
                        print(f"Converged at step {step} due to Energy Plateau (10-step Delta E: {delta_e:.4f} < {e_tol}).")
                    break
            
    if use_as_loss_function:
        return opt_coords, total_energy
    else:
        return opt_coords.detach(), total_energy.item()