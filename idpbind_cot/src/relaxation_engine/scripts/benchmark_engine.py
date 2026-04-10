import time
import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import os

# Import your custom engine modules here
from bowerbird2.backend.relaxation_engine.utils.align_coordinates import ingest_and_map_structure
from bowerbird2.backend.relaxation_engine.utils.compile_polymer import PolymerCompiler
from bowerbird2.backend.relaxation_engine.energy_functions.relaxation import relax_structure
from bowerbird2.backend.relaxation_engine.relax import create_ca_restraint_mask

# Import specific energy functions for the 1-shot PyTorch evaluator
from bowerbird2.backend.relaxation_engine.energy_functions.bonded_energy import compute_bond_angle_energy
from bowerbird2.backend.relaxation_engine.energy_functions.dihedral import compute_dihedral_energy
from bowerbird2.backend.relaxation_engine.energy_functions.non_bonded_interactions import compute_nonbonded_energy, compute_14_energy
from bowerbird2.backend.relaxation_engine.utils.neighor import NeighborListManager
from bowerbird2.backend.relaxation_engine.utils.topology import build_topology_masks

def evaluate_pytorch_energy(coords_tensor, topology, params, ca_mask, ref_coords):
    """
    Evaluates the exact PyTorch energy of a coordinate tensor using the established physics graph.
    Used to grade OpenMM's homework.
    """
    cutoff = 10.0
    eps_rf = 78.5
    k_rf = (eps_rf - 1.0) / ((2.0 * eps_rf + 1.0) * (cutoff ** 3))
    c_rf = (3.0 * eps_rf) / ((2.0 * eps_rf + 1.0) * cutoff)

    num_atoms = coords_tensor.shape[0]
    _, indices_14 = build_topology_masks(topology['covalent_bonds'], num_atoms)
    sigma_14 = (params['sigma'][indices_14[0]] + params['sigma'][indices_14[1]]) * 0.5
    epsilon_14 = torch.sqrt(params['epsilon'][indices_14[0]] * params['epsilon'][indices_14[1]])
    charges_14 = params['charges'][indices_14[0]] * params['charges'][indices_14[1]]

    nl_manager = NeighborListManager(cutoff=cutoff)
    dynamic_neighbors = nl_manager.build_list(coords_tensor, topology.get('exclusion_set'))

    with torch.no_grad():
        e_bond, e_angle, _ = compute_bond_angle_energy(
            coords_tensor, topology['springs'], params['k_b'], params['b_0'],
            topology['angles'], params['k_theta'], params['theta_0']
        )
        e_dihedral = compute_dihedral_energy(
            coords_tensor, topology['dihedrals'], params['k_phi'], params['n'], params['gamma']
        )
        
        # Alpha=0.0 because we are evaluating the final state, no soft-core needed
        e_vdw, e_elec = compute_nonbonded_energy(
            coords=coords_tensor, neighbor_indices=dynamic_neighbors, 
            sigma=params['sigma'], epsilon=params['epsilon'], charges=params['charges'],
            alpha=0.0, cutoff=cutoff, k_rf=k_rf, c_rf=c_rf, elec_weight=1.0
        )
        e_vdw_14, e_elec_14 = compute_14_energy(
            coords_tensor, indices_14, sigma_14, epsilon_14, charges_14,
            alpha=0.0, vdw_scale=0.5, elec_scale=1.0/1.2 
        )
        
        diff = coords_tensor - ref_coords
        e_restraint = torch.sum(10.0 * torch.sum(diff * diff, dim=-1) * ca_mask)
        
        total = e_bond + e_angle + e_dihedral + e_vdw + e_elec + e_vdw_14 + e_elec_14 + e_restraint

    return total.item()

def benchmark_openmm(pdb_path, forcefield_xml='amber14-all.xml', tolerance=10.0):
    print(f"\n--- OpenMM Benchmark ---")
    t0_setup = time.time()
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField(forcefield_xml)
    
    system = ff.createSystem(
        pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, 
        nonbondedCutoff=1.0 * unit.nanometers, constraints=None, rigidWater=False
    )
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'})
    context.setPositions(pdb.positions)
    setup_time = time.time() - t0_setup
    
    print(f"Setup Time:    {setup_time:.4f} s")
    
    t0_min = time.time()
    force_tolerance = tolerance * unit.kilojoules_per_mole / unit.nanometer
    mm.LocalEnergyMinimizer.minimize(context, tolerance=force_tolerance)
    min_time = time.time() - t0_min
    
    # EXPORT THE OPENMM COORDINATES
    final_state = context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    final_positions_np = final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    
    print(f"Minimize Time: {min_time:.4f} s")
    print(f"Final Energy (OpenMM Math): {final_energy:>12.2f} kcal/mol")
    
    return min_time, final_energy, final_positions_np

def benchmark_pytorch(pdb_path, templates_dict_path, device='cuda:0', max_steps=100):
    print(f"\n--- PyTorch (Bowerbird2) Benchmark ---")
    t0_setup = time.time()
    templates = torch.load(os.path.join(templates_dict_path, 'amber14all.pt'), weights_only=False)
    
    starting_coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(
        pdb_path, templates, device=device
    )
    
    # Bypass Hydrogen Builder so we use the exact OpenMM atoms
    print("Bypassing PyTorch H-Builder to strictly match OpenMM topology...")
    full_coords = starting_coords    
    
    compiler = PolymerCompiler(templates, device=device)
    topology, params = compiler.compile_sequence(sequence, chain_types, chain_ids)
    ca_mask = create_ca_restraint_mask(atom_metadata, device=device)
    
    setup_time = time.time() - t0_setup
    print(f"Setup Time:    {setup_time:.4f} s")

    t0_min = time.time()
    relaxed_coords, final_energy = relax_structure(
        coords=full_coords, topology=topology, params=params, 
        ca_mask=ca_mask, max_steps=max_steps, tol=1e-3
    )
    torch.cuda.synchronize() 
    min_time = time.time() - t0_min
    
    print(f"Minimize Time: {min_time:.4f} s")
    print(f"Final Energy (PyTorch Math): {final_energy:>12.2f} kcal/mol")
    
    # Return EVERYTHING needed for the PyTorch evaluator
    return min_time, final_energy, relaxed_coords, starting_coords, topology, params, ca_mask

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    
    # WE USE THE EXACT SAME PDB FOR BOTH ENGINES
    pdb_fi2 = os.path.join(data_dir, "synuclein_STARLING_allatom_fixed.pdb")
    device = 'cuda:0'
    
    # 1. Run OpenMM
    omm_min_time, omm_native_energy, omm_coords_np = benchmark_openmm(pdb_fi2, 'amber14-all.xml')

    # 2. Run PyTorch
    pt_min_time, pt_native_energy, pt_coords, ref_coords, pt_top, pt_params, ca_mask = benchmark_pytorch(pdb_fi2, data_dir, device=device)
    
# ---------------------------------------------------------
    # 3. SPATIAL ALIGNMENT HACK (The 3-Atom Fix)
    # ---------------------------------------------------------
    print("\nAligning OpenMM output to PyTorch topology...")
    pdb_omm = app.PDBFile(pdb_fi2)
    omm_init_np = np.array(pdb_omm.positions.value_in_unit(unit.angstrom))
    pt_init_np = ref_coords.detach().cpu().numpy()
    
    mapping = []
    # Find the exact OpenMM index for every PyTorch atom based on initial starting positions
    for pt_pos in pt_init_np:
        dist_sq = np.sum((omm_init_np - pt_pos)**2, axis=1)
        mapping.append(np.argmin(dist_sq))
        
    # Slice the minimized OpenMM coordinates using this exact mapping
    omm_coords_np_aligned = omm_coords_np[mapping]
    
    # Convert the sliced numpy array into a PyTorch tensor
    omm_tensor = torch.tensor(omm_coords_np_aligned, dtype=torch.float32, device=device)
    # ---------------------------------------------------------
    #     
    # 3. Grade OpenMM using PyTorch Math
    print("\nGrading OpenMM's structure using the PyTorch energy graph...")
    
    # Convert OpenMM numpy array into a PyTorch tensor on the GPU
    #omm_tensor = torch.tensor(omm_coords_np, dtype=torch.float32, device=device)
    
    omm_graded_energy = evaluate_pytorch_energy(
        coords_tensor=omm_tensor, 
        topology=pt_top, 
        params=pt_params, 
        ca_mask=ca_mask, 
        ref_coords=ref_coords
    )
    
    print("\n--- The Final Verdict (Apples to Apples) ---")
    print(f"OpenMM Minimization Time:  {omm_min_time:.4f} s")
    print(f"PyTorch Minimization Time: {pt_min_time:.4f} s")
    print(f"--------------------------------------------------")
    print(f"OpenMM Output graded by PyTorch:  {omm_graded_energy:>10.2f} kcal/mol")
    print(f"PyTorch Output graded by PyTorch: {pt_native_energy:>10.2f} kcal/mol")
    
    energy_diff = abs(omm_graded_energy - pt_native_energy)
    print(f"Energy Difference:                {energy_diff:>10.2f} kcal/mol")