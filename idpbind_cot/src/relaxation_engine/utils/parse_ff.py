"""
Note: This isn't used if we precompile the OpenMM System into PyTorch tensors, but it is still useful
if we decide we want to test something without a saved network. Requires openmm.
"""
import torch
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from idpbind_cot.src.relaxation_engine.utils.constants import KJ_TO_KCAL, NM_TO_ANGSTROM

def compile_pytorch_tensors(openmm_system, openmm_topology, device=torch.device('cpu')):
    """
    Traverses an OpenMM System and compiles flattened, vectorized PyTorch tensors 
    for the relaxation engine. Upgraded to handle Urey-Bradley/Covalent decoupling 
    and CMAP grids for AMBER ff19SB and CHARMM36 compatibility.
    
    Args:
        openmm_system: An instantiated openmm.System object.
        openmm_topology: The corresponding openmm.app.Topology object.
        device: The target PyTorch device.
        
    Returns:
        topology: Dict of integer index tensors.
        params: Dict of float32 parameter tensors.
    """
    
    topology = {}
    params = {}
    
    # --- BUILD TRUE COVALENT GRAPH ---
    true_covalent_edges = set()
    for bond in openmm_topology.bonds():
        true_covalent_edges.add((bond[0].index, bond[1].index))
        true_covalent_edges.add((bond[1].index, bond[0].index))

    # 1. Parse Harmonic Bonds (Decoupled Springs vs Covalent)
    for force in openmm_system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            springs_idx, covalent_bonds_idx, k_b_list, b_0_list = [], [], [], []
            
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                
                # All terms are springs
                springs_idx.append([p1, p2])
                b_0_list.append(length.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)
                k_val = k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                k_b_list.append(k_val * KJ_TO_KCAL / (NM_TO_ANGSTROM**2))
                
                # Only true chemical bonds go to the masking graph
                if (p1, p2) in true_covalent_edges:
                    covalent_bonds_idx.append([p1, p2])
            
            # Note: 'bonds' is gone. Replaced by 'springs' and 'covalent_bonds'.
            topology['springs'] = torch.tensor(springs_idx, dtype=torch.long, device=device).t()
            topology['covalent_bonds'] = torch.tensor(covalent_bonds_idx, dtype=torch.long, device=device).t()
            params['k_b'] = torch.tensor(k_b_list, dtype=torch.float32, device=device)
            params['b_0'] = torch.tensor(b_0_list, dtype=torch.float32, device=device)

    # 2. Parse Harmonic Angles
    for force in openmm_system.getForces():
        if isinstance(force, mm.HarmonicAngleForce):
            angles_idx, k_theta_list, theta_0_list = [], [], []
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                angles_idx.append([p1, p2, p3])
                theta_0_list.append(angle.value_in_unit(unit.radian))
                k_val = k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                k_theta_list.append(k_val * KJ_TO_KCAL)
                
            topology['angles'] = torch.tensor(angles_idx, dtype=torch.long, device=device).t()
            params['k_theta'] = torch.tensor(k_theta_list, dtype=torch.float32, device=device)
            params['theta_0'] = torch.tensor(theta_0_list, dtype=torch.float32, device=device)

    # 3. Parse Periodic Torsions (Dihedrals)
    for force in openmm_system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            dih_idx, k_phi_list, phase_list, n_list = [], [], [], []
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                dih_idx.append([p1, p2, p3, p4])
                n_list.append(periodicity)
                phase_list.append(phase.value_in_unit(unit.radian))
                k_val = k.value_in_unit(unit.kilojoule_per_mole)
                k_phi_list.append(k_val * KJ_TO_KCAL)
                
            topology['dihedrals'] = torch.tensor(dih_idx, dtype=torch.long, device=device).t()
            params['n'] = torch.tensor(n_list, dtype=torch.float32, device=device)
            params['gamma'] = torch.tensor(phase_list, dtype=torch.float32, device=device)
            params['k_phi'] = torch.tensor(k_phi_list, dtype=torch.float32, device=device)
            
    # 4. Parse CMAP (if present)
    for force in openmm_system.getForces():
        if isinstance(force, mm.CMAPTorsionForce):
            cmap_idx, cmap_assignments, global_cmap_grids = [], [], []
            
            # Extract the 2D energy maps
            for i in range(force.getNumMaps()):
                size, energy_list = force.getMapParameters(i)
                grid = torch.tensor(energy_list, dtype=torch.float32, device=device).view(size, size) * KJ_TO_KCAL
                global_cmap_grids.append(grid)
                
            if global_cmap_grids:
                params['global_cmap_grids'] = torch.stack(global_cmap_grids).unsqueeze(0)
            
            # Extract the torsions mapped to the grids
            for i in range(force.getNumTorsions()):
                map_id, a1, a2, a3, a4, a5, a6, a7, a8 = force.getTorsionParameters(i)
                cmap_idx.append([a1, a2, a3, a4, a5, a6, a7, a8])
                cmap_assignments.append(map_id)
                
            topology['cmap'] = torch.tensor(cmap_idx, dtype=torch.long, device=device).t()
            params['cmap_assignments'] = torch.tensor(cmap_assignments, dtype=torch.long, device=device)

    # 5. Parse Non-Bonded Parameters
    for force in openmm_system.getForces():
        if isinstance(force, mm.NonbondedForce):
            num_particles = force.getNumParticles()
            charges, sigmas, epsilons = [], [], []
            for i in range(num_particles):
                charge, sigma, epsilon = force.getParticleParameters(i)
                charges.append(charge.value_in_unit(unit.elementary_charge))
                sigmas.append(sigma.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)
                epsilons.append(epsilon.value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL)
                
            params['charges'] = torch.tensor(charges, dtype=torch.float32, device=device)
            params['sigma'] = torch.tensor(sigmas, dtype=torch.float32, device=device)
            params['epsilon'] = torch.tensor(epsilons, dtype=torch.float32, device=device)

    return topology, params