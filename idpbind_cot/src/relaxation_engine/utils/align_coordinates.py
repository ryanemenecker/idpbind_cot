import torch
import re

from idpbind_cot.src.relaxation_engine.utils.io import parse_cif, parse_pdb

def ingest_and_map_structure(structure_filepath, templates_dict, device='cuda:0'):
    """
    Pure Python structure ingestion. Reads a fully-hydrogenated CIF/PDB, infers 
    the sequence, and maps the coordinates strictly to the PyTorch templates.
    Zero OpenMM dependency.
    """
    print(f"Loading structure via native parser: {structure_filepath}")
    
    if structure_filepath.endswith('.pdb'):
        parsed_data = parse_pdb(structure_filepath)
    elif structure_filepath.endswith('.cif') or structure_filepath.endswith('.mmcif'):
        parsed_data = parse_cif(structure_filepath)
    else:
        raise ValueError("Unsupported file format.")

    sequence = []
    chain_types = []
    chain_ids = []
    aligned_coords = []
    atom_metadata = []
    
    dna_names = {'DA', 'DC', 'DG', 'DT'}
    rna_names = {'A', 'C', 'G', 'U'}

    # Helper to sort residues identically to your writers
    def residue_sort_key(res_key):
        match = re.match(r"(-?\d+)(.*)", str(res_key))
        if match: 
            return (int(match.group(1)), match.group(2))
        return (0, res_key)

    # 1. Iterate through your custom parsed hierarchy
    for chain_id in sorted(parsed_data.keys()):
        residues = parsed_data[chain_id]
        sorted_res_keys = sorted(residues.keys(), key=residue_sort_key)
        
        for res_key in sorted_res_keys:
            atom_list = residues[res_key]
            if not atom_list: 
                continue
            
            # 2. Infer Residue Name and Aliasing
            raw_res_name = atom_list[0].get("label_comp_id", atom_list[0].get("auth_comp_id", "UNK"))
            res_name = raw_res_name
            
            # Dynamic AMBER-specific naming fallbacks
            if res_name == 'HIS':
                if 'HIE' in templates_dict: res_name = 'HIE'
                elif 'HID' in templates_dict: res_name = 'HID'
                elif 'HIP' in templates_dict: res_name = 'HIP'
                elif 'HIS' in templates_dict: res_name = 'HIS'
            elif res_name == 'CYS':
                # If standard CYS is missing but disulfide CYX is present
                if 'CYS' not in templates_dict and 'CYX' in templates_dict: 
                    res_name = 'CYX'
            elif res_name == 'HOH': 
                continue  # Strip explicit water
            elif res_name not in templates_dict:
                print(f"Warning: Skipping unrecognized residue {res_name} at {chain_id}:{res_key}")
                continue
                
            sequence.append(res_name)
            chain_ids.append(chain_id)
            
            # 3. Infer Chain Type
            if res_name in dna_names: 
                chain_types.append('dna')
            elif res_name in rna_names: 
                chain_types.append('rna')
            else: 
                chain_types.append('protein')
            
            # 4. Build fast atom lookup for this residue
            pdb_atoms = {}
            for atom in atom_list:
                name = atom.get("label_atom_id", atom.get("auth_atom_id"))
                pdb_atoms[name] = atom
                
            template = templates_dict[res_name]
            
            # 5. Map strictly to the template's required atoms
            for atom_name in template['atom_names']:
                target_atom = pdb_atoms.get(atom_name)
                
                # --- Task 3: Universal Hydrogen Aliasing ---
                # Handle standard PDB vs AMBER/CHARMM nomenclature aliases.
                # The goal is to find the atom from the PDB/CIF file (`pdb_atoms`) that
                # corresponds to the required atom in the template (`atom_name`).
                if not target_atom:
                    # AMBER H (backbone) <-> CHARMM HN (backbone)
                    if atom_name == 'H' and 'HN' in pdb_atoms:
                        target_atom = pdb_atoms['HN']
                    elif atom_name == 'HN' and 'H' in pdb_atoms:
                        target_atom = pdb_atoms['H']
                    
                    # AMBER N-terminal H1/H2/H3 <-> CHARMM N-terminal HT1/HT2/HT3
                    elif atom_name == 'H1' and 'HT1' in pdb_atoms:
                        target_atom = pdb_atoms['HT1']
                    elif atom_name == 'H2' and 'HT2' in pdb_atoms:
                        target_atom = pdb_atoms['HT2']
                    elif atom_name == 'H3' and 'HT3' in pdb_atoms:
                        target_atom = pdb_atoms['HT3']
                        
                    # Other existing aliases
                    elif atom_name == 'HB1' and 'HB3' in pdb_atoms:
                        target_atom = pdb_atoms['HB3']
                    elif atom_name == 'CD1' and 'CD' in pdb_atoms and res_name == 'ILE':
                        target_atom = pdb_atoms['CD']
                # ---------------------------------------------
                    
                # Check if this atom is a hydrogen
                is_hydrogen = atom_name.startswith('H') or (atom_name[0].isdigit() and atom_name[1] == 'H')

                if not target_atom:
                    if is_hydrogen:
                        # Missing Hydrogen: Inject (0,0,0) placeholder. 
                        # The Vectorized Z-Matrix Builder will overwrite this.
                        aligned_coords.append([0.0, 0.0, 0.0])
                        element = 'H'
                    else:
                        raise KeyError(f"Fatal Alignment Error: Missing required heavy atom '{atom_name}' "
                                       f"in {res_name} {chain_id}:{res_key}.")
                else:
                    # 6. Extract coordinates 
                    # (Your parsers output string values, so we cast to float)
                    x = float(target_atom["Cartn_x"])
                    y = float(target_atom["Cartn_y"])
                    z = float(target_atom["Cartn_z"])
                    aligned_coords.append([x, y, z])
                    
                    # Determine element for export metadata
                    element = target_atom.get("type_symbol", "")
                    if not element or element == "X":
                        elem_clean = ''.join([c for c in atom_name if c.isalpha()])
                        element = elem_clean[0] if elem_clean else "X"
                    
                # 7. Build the Metadata for your Exporter
                num_match = re.match(r"(-?\d+)(.*)", str(res_key))
                res_seq = num_match.group(1) if num_match else "0"
                ins_code = num_match.group(2) if num_match else ""
                
                atom_metadata.append({
                    'chain_id': chain_id,
                    'res_seq': res_seq,
                    'ins_code': ins_code,
                    'res_name': raw_res_name, # Preserve the original PDB name (e.g., HIS) for export
                    'atom_name': atom_name,   # Standardized AMBER name
                    'element': element.upper()
                })

    coords_tensor = torch.tensor(aligned_coords, dtype=torch.float32, device=device)
    print(f"Inferred sequence of {len(sequence)} residues. Mapped {coords_tensor.shape[0]} atoms.")
    
    return coords_tensor, sequence, chain_types, atom_metadata, chain_ids

