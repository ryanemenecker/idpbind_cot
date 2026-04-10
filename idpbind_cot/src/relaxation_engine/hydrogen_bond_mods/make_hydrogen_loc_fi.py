"""
Uses a file we previously placed hydrogens at to get the hyrogen positions per atom. 
Let's us bypass openmm. 

Design invariant: this is an offline, scalar rule-generation script and is
intentionally separate from place_hydrogens.py, which applies rules in a
runtime vectorized path.
"""

import torch
import math
# import your parse_cif and extract_dihedral functions here
from bowerbird2.backend.io import parse_cif
from bowerbird2.backend.ca_to_all_atom.kinematics.nerf import extract_dihedral, nerf_build

def generate_h_rules_from_ideal_cif(cif_filepath):
    """
    Reads an idealized, fully-hydrogenated CIF file and reverse-engineers the 
    NeRF geometric rules (r, theta, chi) for every hydrogen.
    """
    parsed_data = parse_cif(cif_filepath)
    h_rules = {}
    
    # Distance cutoff heuristic to find heavy-atom bonds (Å)
    BOND_CUTOFF = 1.6 

    def get_coords(atom_dict):
        return torch.tensor([
            float(atom_dict['Cartn_x']), 
            float(atom_dict['Cartn_y']), 
            float(atom_dict['Cartn_z'])
        ], dtype=torch.float32)

    for chain_id, residues in parsed_data.items():
        for res_key, atoms in residues.items():
            res_name = atoms[0].get('label_comp_id', 'UNK')
            if res_name not in h_rules:
                h_rules[res_name] = {}
                
            heavy_atoms = [a for a in atoms if a.get('type_symbol', 'X') != 'H']
            hydrogens = [a for a in atoms if a.get('type_symbol', 'X') == 'H']
            
            for h_atom in hydrogens:
                h_name = h_atom.get('label_atom_id')
                h_pos = get_coords(h_atom)
                
                # 1. Find Parent (Closest Heavy Atom)
                parent, p_pos, min_dist = None, None, 999.0
                for ha in heavy_atoms:
                    ha_pos = get_coords(ha)
                    dist = torch.linalg.norm(h_pos - ha_pos).item()
                    if dist < min_dist:
                        min_dist, parent, p_pos = dist, ha, ha_pos
                        
                if not parent or min_dist > BOND_CUTOFF:
                    continue # Skip disconnected hydrogens
                    
                # 2. Find Grandparent (Closest Heavy Atom to Parent)
                g_parent, g_pos, g_dist = None, None, 999.0
                for ha in heavy_atoms:
                    if ha['label_atom_id'] == parent['label_atom_id']: continue
                    ha_pos = get_coords(ha)
                    dist = torch.linalg.norm(p_pos - ha_pos).item()
                    if dist < g_dist:
                        g_dist, g_parent, g_pos = dist, ha, ha_pos

                # 3. Find Great-Grandparent (Closest Heavy Atom to Grandparent)
                gg_parent, gg_pos, gg_dist = None, None, 999.0
                for ha in heavy_atoms:
                    if ha['label_atom_id'] in (parent['label_atom_id'], g_parent['label_atom_id']): continue
                    ha_pos = get_coords(ha)
                    dist = torch.linalg.norm(g_pos - ha_pos).item()
                    if dist < gg_dist:
                        gg_dist, gg_parent, gg_pos = dist, ha, ha_pos
                        
                if not (parent and g_parent and gg_parent):
                    continue

                # 4. Calculate Internal Coordinates (Reverse-NeRF)
                # r = distance(Parent, H)
                r = min_dist 
                
                # theta = angle(Grandparent, Parent, H)
                v1 = g_pos - p_pos
                v2 = h_pos - p_pos
                cos_theta = torch.sum(v1 * v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))
                theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)).item()
                
                # chi = dihedral(GreatGrandparent, Grandparent, Parent, H)
                # Use your existing extract_dihedral function!
                chi = extract_dihedral(
                    gg_pos.unsqueeze(0), 
                    g_pos.unsqueeze(0), 
                    p_pos.unsqueeze(0), 
                    h_pos.unsqueeze(0)
                ).item()

                # 5. Store rule
                h_rules[res_name][h_name] = (
                    gg_parent['label_atom_id'], 
                    g_parent['label_atom_id'], 
                    parent['label_atom_id'], 
                    r, theta, chi
                )

    # 6. Print the dictionary so you can copy-paste it into your codebase
    print("H_RULES = {")
    for res, rules in h_rules.items():
        print(f"    '{res}': {{")
        for h_name, (gg, g, p, r, theta, chi) in rules.items():
            print(f"        '{h_name}': ('{gg}', '{g}', '{p}', {r:.4f}, {theta:.4f}, {chi:.4f}),")
        print("    },")
    print("}")

# Run this once offline!
generate_h_rules_from_ideal_cif('/Users/ryanemenecker/Desktop/lab_packages/bowerbird2/bowerbird2/backend/relaxation_engine/hydrogen_bond_mods/fold_rosetta_aa_model_0_added_cyx_hid_fixed.cif')
