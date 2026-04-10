import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import math
from idpbind_cot.src.relaxation_engine.utils.align_coordinates import ingest_and_map_structure
from idpbind_cot.src.relaxation_engine.utils.compile_polymer import PolymerCompiler
from idpbind_cot.src.relaxation_engine.utils.export_coords import export_relaxed_coordinates
from idpbind_cot.src.relaxation_engine.energy_functions.relaxation import relax_structure
from idpbind_cot.src.relaxation_engine.hydrogen_bond_mods.place_hydrogens import VectorizedHydrogenBuilder
from idpbind_cot.src.relaxation_engine.hydrogen_bond_mods.hydrogen_constants import H_RULES
# get abspath to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# now use to get loc of folder where params are. 
params_dir = os.path.join(current_dir, 'data')

# useful function for diagnostics. Leaving in because it's saved my bacon on building hydrogens
def print_worst_angles(coords, topology, params, atom_metadata, top_k=20):
    angles_idx = topology['angles']
    # Corrected indexing for (3, N) shape
    idx_i, idx_j, idx_k = angles_idx[0], angles_idx[1], angles_idx[2]
    
    pos_i, pos_j, pos_k = coords[idx_i], coords[idx_j], coords[idx_k]
    
    v1 = pos_i - pos_j
    v2 = pos_k - pos_j
    
    # Normalize vectors safely
    v1_norm = v1 / torch.clamp(torch.norm(v1, dim=-1, keepdim=True), min=1e-6)
    v2_norm = v2 / torch.clamp(torch.norm(v2, dim=-1, keepdim=True), min=1e-6)
    
    cosine = torch.sum(v1_norm * v2_norm, dim=-1)
    angle_rad = torch.acos(torch.clamp(cosine, -0.999999, 0.999999))
    
    k_theta = params['k_theta']
    theta_0 = params['theta_0']
    
    # E = k * (theta - theta_0)^2
    energy = k_theta * (angle_rad - theta_0)**2
    
    # Safely handle systems with fewer angles than top_k
    top_k = min(top_k, energy.numel())
    top_energies, top_indices = torch.topk(energy, top_k)
    
    print("\n" + "="*80)
    print(f"{'TOP 20 WORST ANGLES':^80}")
    print("="*80)
    print(f"{'Atom 1':<14} | {'Center':<14} | {'Atom 3':<14} | {'Energy':>8} | {'Ideal':>7} | {'Actual':>7}")
    print("-" * 80)
    
    for e, idx in zip(top_energies, top_indices):
        # Extract scalar indices safely
        i = idx_i[idx].item()
        j = idx_j[idx].item()
        k = idx_k[idx].item()
        
        m_i, m_j, m_k = atom_metadata[i], atom_metadata[j], atom_metadata[k]
        
        name_i = f"{m_i['res_name']}{m_i['res_seq']}-{m_i['atom_name']}"
        name_j = f"{m_j['res_name']}{m_j['res_seq']}-{m_j['atom_name']}"
        name_k = f"{m_k['res_name']}{m_k['res_seq']}-{m_k['atom_name']}"
        
        ideal_deg = math.degrees(theta_0[idx].item())
        actual_deg = math.degrees(angle_rad[idx].item())
        
        print(f"{name_i:<14} | {name_j:<14} | {name_k:<14} | {e.item():8.2f} | {ideal_deg:6.1f}° | {actual_deg:6.1f}°")
    
    print("="*80)
    print(f"Total Angle Energy: {energy.sum().item():.2f} kcal/mol\n")

def create_ca_restraint_mask(atom_metadata, device='cuda:0'):
    """
    Creates a float tensor of shape (N,) that is 1.0 for C-alpha atoms, 0.0 otherwise.
    """
    mask = torch.tensor(
        [1.0 if meta['atom_name'] == 'CA' else 0.0 for meta in atom_metadata],
        dtype=torch.float32, device=device
    )
    return mask


def _prepare_relaxation_inputs(
    starting_coords,
    sequence,
    chain_types,
    atom_metadata,
    chain_ids,
    templates,
    device='cuda:0',
    use_restraints=True
):
    """Shared setup used by both single and batched relaxation pipelines."""
    h_builder = VectorizedHydrogenBuilder(sequence, atom_metadata, H_RULES, device=device)
    full_coords = h_builder.build_hydrogens(starting_coords)

    compiler = PolymerCompiler(templates, device=device)
    topology, params = compiler.compile_sequence(sequence, chain_types, chain_ids)

    ca_mask = create_ca_restraint_mask(atom_metadata, device=device)
    if not use_restraints:
        ca_mask = torch.zeros_like(ca_mask)
        
    return full_coords, topology, params, ca_mask

def run_relaxation(target_filepath, output_filepath, device='cuda:0', verbose=False, max_steps=100,
                             use_as_loss_function=False, use_restraints=True):
    """
    The master OpenMM-free pipeline.
    """
    print("Loading master tensor database...")
    templates = torch.load(os.path.join(params_dir, 'amber14all.pt'), weights_only=False)
    
    # 1. Native I/O Ingestion
    starting_coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(
        target_filepath, templates, device=device
    )
    # 2. Native Z-Matrix Hydrogen Placement
    print("Building ideal hydrogens...")
    print("Compiling sequence topology...")
    full_coords, topology, params, ca_mask = _prepare_relaxation_inputs(
        starting_coords,
        sequence,
        chain_types,
        atom_metadata,
        chain_ids,
        templates,
        device=device,
        use_restraints=use_restraints
    )

    # 5. Vectorized PyTorch FIRE Minimization
    print("Starting FIRE relaxation...")
    relaxed_coords, final_energy = relax_structure(
        coords=full_coords, 
        topology=topology, 
        params=params, 
        ca_mask=ca_mask,
        max_steps=max_steps, 
        tol=1e-3,
        verbose=verbose,
        use_as_loss_function=use_as_loss_function
    )
    # 6. Native I/O Export (With shape safeguard)
    # Prepare a torch tensor shaped (1, N, 3) so export_relaxed_coordinates can index [0]
    out_coords = relaxed_coords.detach().cpu().unsqueeze(0)

    try:
        export_relaxed_coordinates(out_coords, atom_metadata, output_filepath)
    except ValueError as e:
        # If the export function expects a different axis order, attempt a safe transpose
        print(f"Standard export failed with {e}. Trying axis transpose...")
        try:
            alt = out_coords.permute(0, 2, 1)
            export_relaxed_coordinates(alt, atom_metadata, output_filepath)
        except Exception:
            raise
        
    print("Pipeline finished successfully.")


