import torch
from collections import defaultdict
from idpbind_cot.src.relaxation_engine.utils.io import write_cif, write_pdb

# Assume write_cif and write_pdb from your I/O script are imported here

def export_relaxed_coordinates(relaxed_coords, atom_metadata, output_filepath):
    """
    Packs the relaxed PyTorch coordinate tensor into your custom dictionary schema 
    and exports it using your native Python CIF/PDB writers. Zero OpenMM dependency.
    
    Args:
        relaxed_coords: PyTorch Tensor of shape (N, 3) in Ångstroms.
        atom_metadata: A flat list of N dictionaries containing the identifiers for each atom.
                       (e.g., [{'chain_id': 'A', 'res_seq': '10', 'ins_code': '', 
                                'res_name': 'ALA', 'atom_name': 'CA', 'element': 'C'}, ...])
        output_filepath: String path to the target output file (.pdb or .cif).
    """
    print(f"Packing coordinates for export to {output_filepath}...")
    
    # Extract coordinates from the PyTorch autograd graph
    if relaxed_coords.dim() == 3:
        coords_np = relaxed_coords[0].detach().cpu().numpy()
    elif relaxed_coords.dim() == 2:
        coords_np = relaxed_coords.detach().cpu().numpy()
    else:
        raise ValueError(
            f"Expected relaxed_coords with 2 or 3 dimensions, got {relaxed_coords.dim()}D "
            f"tensor with shape {tuple(relaxed_coords.shape)}"
        )
    
    if len(coords_np) != len(atom_metadata):
        raise ValueError(f"Shape mismatch: Tensor has {len(coords_np)} atoms, but metadata has {len(atom_metadata)}.")
    
    # Initialize the nested dictionary structure your writers expect
    structure_dict = defaultdict(lambda: defaultdict(list))
    
    # Map the metadata and coordinates together
    for meta, coord in zip(atom_metadata, coords_np):
        
        chain_id = meta['chain_id']
        res_seq = meta['res_seq']
        ins_code = meta.get('ins_code', '?')
        if not ins_code or ins_code in [' ', '.']:
            ins_code = "?"
            
        # Construct the unique residue key exactly as your parse_cif does (e.g. "10" or "10A")
        res_key = f"{res_seq}{ins_code if ins_code != '?' else ''}"
        
        # Build the atom dictionary mapping your schema
        atom_dict = {
            "group_PDB": "ATOM",
            "label_atom_id": meta['atom_name'],
            "auth_atom_id": meta['atom_name'],
            "label_comp_id": meta['res_name'],
            "auth_comp_id": meta['res_name'],
            "label_asym_id": chain_id,
            "auth_asym_id": chain_id,
            "label_seq_id": res_seq,
            "auth_seq_id": res_seq,
            "pdbx_PDB_ins_code": ins_code,
            "type_symbol": meta['element'],
            
            # Inject PyTorch coordinates (already in Angstroms)
            "Cartn_x": f"{coord[0]:.3f}",
            "Cartn_y": f"{coord[1]:.3f}",
            "Cartn_z": f"{coord[2]:.3f}",
            
            "occupancy": "1.00",
            "B_iso_or_equiv": "0.00",
            "label_alt_id": "."
        }
        
        structure_dict[chain_id][res_key].append(atom_dict)

    final_dict = {k: dict(v) for k, v in structure_dict.items()}
    
    # Route to your custom writers
    if output_filepath.endswith('.pdb'):
        write_pdb(final_dict, output_filepath)
    elif output_filepath.endswith('.cif') or output_filepath.endswith('.mmcif'):
        write_cif(final_dict, output_filepath)
    else:
        raise ValueError("Output file must end with .pdb, .cif, or .mmcif")