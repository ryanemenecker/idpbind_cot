import logging
import torch
import math
from bowerbird2.backend.ca_to_all_atom.kinematics.nerf import nerf_build

# Design invariant: this module is the runtime, vectorized hydrogen placer.
# Keep it separate from make_hydrogen_loc_fi.py, which is an offline,
# scalar rule-extraction utility. They share rule format but not execution path.

class VectorizedHydrogenBuilder:
    def __init__(self, sequence, atom_metadata, h_rules, device='cuda:0'):
        self.device = device
        self.h_rules = h_rules
        
        # 1. NEW: Define which hydrogens are allowed to spin
        self.rotor_definitions = {
            'SER': ['HG'], 
            'CYS': ['HG'], 
            'THR': ['HG1'], 
            'TYR': ['HH']
        }
        
        self.res_keys = []
        self.idx_map = {}
        
        for row_idx, meta in enumerate(atom_metadata):
            res_id = (meta['chain_id'], meta['res_seq'], meta['ins_code'])
            if not self.res_keys or self.res_keys[-1] != res_id:
                self.res_keys.append(res_id)
            self.idx_map[(res_id, meta['atom_name'])] = row_idx
            
        self.ca_map = {}
        for row_idx, meta in enumerate(atom_metadata):
            res_id = (meta['chain_id'], meta['res_seq'], meta['ins_code'])
            self.ca_map[row_idx] = self.idx_map.get((res_id, 'CA'), row_idx)
            
        self.ca_index_tensor = torch.tensor(
            [self.ca_map[i] for i in range(len(atom_metadata))],
            dtype=torch.long, device=self.device
        )

        self.idx_H, self.idx_A, self.idx_B, self.idx_C = [], [], [], []
        self.lengths, self.thetas, self.chis = [], [], []
        
        # 2. NEW: Track which indices in our lists belong to free rotors
        self.local_rotor_indices = []
        
        self._compile_hydrogen_tensors(sequence)

    def _compile_hydrogen_tensors(self, sequence):
        for i, res_name in enumerate(sequence):
            current_res = self.res_keys[i]
            
            prev_res = None
            if i > 0:
                candidate_prev = self.res_keys[i-1]
                if candidate_prev[0] == current_res[0]: 
                    prev_res = candidate_prev

            # Backbone Amide
            if res_name != 'PRO':
                if prev_res is not None:
                    self._add_hydrogen_rule(current_res, prev_res, 'H', self.h_rules['GENERIC_BACKBONE']['H'])
                else:
                    fallback_rule = ('C', 'CA', 'N', 1.010, math.radians(119.8), math.radians(180.0))
                    self._add_hydrogen_rule(current_res, current_res, 'H', fallback_rule)
                
            rule_key = res_name if res_name in self.h_rules else ('HIE' if res_name == 'HIS' else None)
            if rule_key is not None:
                for h_name, rule in self.h_rules[rule_key].items():
                    success = self._add_hydrogen_rule(current_res, prev_res, h_name, rule)
                    
                    # 3. NEW: If it built successfully and is a rotor, log its local list index
                    if success and rule_key in self.rotor_definitions and h_name in self.rotor_definitions[rule_key]:
                        self.local_rotor_indices.append(len(self.idx_H) - 1)
                    
        self.idx_H = torch.tensor(self.idx_H, dtype=torch.long, device=self.device)
        self.idx_A = torch.tensor(self.idx_A, dtype=torch.long, device=self.device)
        self.idx_B = torch.tensor(self.idx_B, dtype=torch.long, device=self.device)
        self.idx_C = torch.tensor(self.idx_C, dtype=torch.long, device=self.device)
        
        self.lengths = torch.tensor(self.lengths, dtype=torch.float32, device=self.device)
        self.thetas = torch.tensor(self.thetas, dtype=torch.float32, device=self.device)
        self.chis = torch.tensor(self.chis, dtype=torch.float32, device=self.device)
        self.local_rotor_indices = torch.tensor(self.local_rotor_indices, dtype=torch.long, device=self.device)

    def _add_hydrogen_rule(self, current_res, prev_res, h_name, rule):
        atom_a, atom_b, atom_c, r, theta, chi = rule
        
        res_a = prev_res if atom_a.startswith('-') else current_res
        res_b = prev_res if atom_b.startswith('-') else current_res
        res_c = prev_res if atom_c.startswith('-') else current_res
        
        if None in (res_a, res_b, res_c):
            return False
            
        clean_a = atom_a.replace('-', '')
        clean_b = atom_b.replace('-', '')
        clean_c = atom_c.replace('-', '')
        
        try:
            row_h = self.idx_map[(current_res, h_name)]
            row_a = self.idx_map[(res_a, clean_a)]
            row_b = self.idx_map[(res_b, clean_b)]
            row_c = self.idx_map[(res_c, clean_c)]
        except KeyError:
            return False
        
        self.idx_H.append(row_h)
        self.idx_A.append(row_a)
        self.idx_B.append(row_b)
        self.idx_C.append(row_c)
        self.lengths.append(r)
        self.thetas.append(theta)
        self.chis.append(chi)
        return True

    def build_hydrogens(self, coords):
        # 1. Build the baseline structure using the rigid rules
        A_coords = coords[self.idx_A]
        B_coords = coords[self.idx_B]
        C_coords = coords[self.idx_C]
        
        h_coords = nerf_build(
            a=A_coords, b=B_coords, c=C_coords, 
            length=self.lengths, theta=self.thetas, chi=self.chis
        )
        
        coords_out = coords.clone()
        assert h_coords.shape[0] == len(self.idx_H), (
            f"Hydrogen count mismatch: NeRF produced {h_coords.shape[0]} coords "
            f"but {len(self.idx_H)} hydrogen slots were registered"
        )
        coords_out[self.idx_H] = h_coords

        # ---------------------------------------------------------
        # 4. NEW: Vectorized Steric Rotor Optimization
        # ---------------------------------------------------------
        num_rotors = len(self.local_rotor_indices)
        if num_rotors > 0:
            # Extract internal coordinates for just the rotors
            r_A = A_coords[self.local_rotor_indices]
            r_B = B_coords[self.local_rotor_indices]
            r_C = C_coords[self.local_rotor_indices]
            r_L = self.lengths[self.local_rotor_indices]
            r_T = self.thetas[self.local_rotor_indices]
            
            # Generate a 6-state rotational grid (0 to 300 degrees)
            grid_size = 36
            angles = torch.linspace(0, 2 * math.pi * (5/6), grid_size, device=self.device)
            
            # Broadcast everything to shape (num_rotors * grid_size)
            r_A_exp = r_A.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, 3)
            r_B_exp = r_B.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, 3)
            r_C_exp = r_C.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, 3)
            r_L_exp = r_L.unsqueeze(1).expand(-1, grid_size).reshape(-1)
            r_T_exp = r_T.unsqueeze(1).expand(-1, grid_size).reshape(-1)
            r_chi_exp = angles.unsqueeze(0).expand(num_rotors, -1).reshape(-1)
            
            # Build all 6 candidates for every rotor simultaneously
            candidates = nerf_build(r_A_exp, r_B_exp, r_C_exp, r_L_exp, r_T_exp, r_chi_exp)
            candidates = candidates.reshape(num_rotors, grid_size, 3)
            
            # Calculate distance to all heavy atoms (the input coords)
            # candidates: (num_rotors, 6, 1, 3) | coords: (1, 1, N, 3)
            diff = candidates.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
            dist_sq = diff.pow(2).sum(dim=-1)
            
            # Clamp distance to prevent division-by-zero from bonded heavy atoms
            dist_sq = torch.clamp(dist_sq, min=1.0)
            
            # Calculate a fast 1/r^6 repulsive steric penalty
            steric_penalty = (1.0 / dist_sq.pow(3)).sum(dim=-1) # Shape: (num_rotors, 6)
            
            # Find the rotational state with the lowest clash penalty
            best_state_idx = steric_penalty.argmin(dim=-1) # Shape: (num_rotors,)
            
            # Extract the winning coordinates
            best_coords = candidates[torch.arange(num_rotors), best_state_idx, :]
            
            # Overwrite the clashing rotors with the optimized coordinates
            global_rotor_idx = self.idx_H[self.local_rotor_indices]
            coords_out[global_rotor_idx] = best_coords
        # ---------------------------------------------------------

        # THE BULLETPROOF SAFEGUARD
        h_coords_placed = coords_out[self.idx_H]
        zeros_mask = (h_coords_placed == 0.0).all(dim=-1)
        nan_mask   = h_coords_placed.isnan().any(dim=-1)
        failure_mask = zeros_mask | nan_mask

        if failure_mask.any():
            fail_local   = torch.nonzero(failure_mask).squeeze(-1)
            fail_indices = self.idx_H[fail_local]  
            logging.warning(
                "Safeguard triggered: tethering %d missing/NaN hydrogens to CA.",
                len(fail_indices)
            )
            ca_coords = coords_out[self.ca_index_tensor[fail_indices]]
            offset = torch.zeros_like(ca_coords)
            offset[:, 0] = 1.0
            coords_out[fail_indices] = ca_coords + offset

        return coords_out