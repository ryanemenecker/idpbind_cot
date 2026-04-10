import torch

class PolymerCompiler:
    def __init__(self, templates_dict, device='cuda:0'):
        """
        Initializes the sequence compiler with the pre-calculated residue templates.
        """
        self.templates = templates_dict
        self.device = device
        
        # --- AMBER/CHARMM Inter-Residue Stitching Rules ---
        self.peptide_rules = {
            'bonds': [
                (((0, 'C'), (1, 'N')), 490.0, 1.335)  
            ],
            'angles': [
                (((0, 'CA'), (0, 'C'), (1, 'N')), 63.0, 2.035),
                (((0, 'O'), (0, 'C'), (1, 'N')), 80.0, 2.145),
                (((0, 'C'), (1, 'N'), (1, 'CA')), 63.0, 2.127),
                (((0, 'C'), (1, 'N'), (1, 'H')), 50.0, 2.094)
            ],
            'dihedrals': [
                (((0, 'CA'), (0, 'C'), (1, 'N'), (1, 'CA')), 2.5, 2.0, 3.1415), 
                (((0, 'O'), (0, 'C'), (1, 'N'), (1, 'CA')), 2.5, 2.0, 3.1415),
                (((0, 'CA'), (0, 'C'), (1, 'N'), (1, 'H')), 2.5, 2.0, 3.1415),
                (((0, 'O'), (0, 'C'), (1, 'N'), (1, 'H')), 2.5, 2.0, 3.1415),
                (((0, 'N'), (0, 'CA'), (0, 'C'), (1, 'N')), 1.7, 2.0, 3.1415),
                (((0, 'C'), (1, 'N'), (1, 'CA'), (1, 'C')), 2.0, 2.0, 3.1415)
            ],
            'cmap': [
                # Phi atoms (0:3), Psi atoms (4:7) spanning 3 residues total
                (((0, 'C'), (1, 'N'), (1, 'CA'), (1, 'C'), 
                  (1, 'N'), (1, 'CA'), (1, 'C'), (2, 'N')))
            ]
        }
        
        self.phosphodiester_rules = {
            'bonds': [
                (((0, 'O3\''), (1, 'P')), 315.0, 1.607) 
            ],
            'angles': [
                (((0, 'C3\''), (0, 'O3\''), (1, 'P')), 86.0, 2.071), 
                (((0, 'O3\''), (1, 'P'), (1, 'O5\'')), 97.0, 1.792),
                (((0, 'O3\''), (1, 'P'), (1, 'O1P')), 105.0, 1.905),
                (((0, 'O3\''), (1, 'P'), (1, 'O2P')), 105.0, 1.905)
            ],
            'dihedrals': [
                (((0, 'C4\''), (0, 'C3\''), (0, 'O3\''), (1, 'P')), 0.5, 3.0, 0.0),
                (((0, 'C3\''), (0, 'O3\''), (1, 'P'), (1, 'O5\'')), 1.2, 2.0, 0.0),
                (((0, 'O3\''), (1, 'P'), (1, 'O5\''), (1, 'C5\'')), 1.2, 2.0, 0.0)
            ]
        }

    def compile_sequence(self, sequence, chain_types, chain_ids):
        """
        Compiles a sequence of residues into flattened PyTorch tensors, seamlessly 
        handling topological index offsets and cross-boundary parameter injection.
        """
        num_residues = len(sequence)
        
        # 1. Accumulators for tensor concatenation (Note decoupled bond tracking)
        accumulated = {
            'charges': [], 'sigma': [], 'epsilon': [],
            'gb_rho': [], 'gb_screen': [],
            'springs_idx': [], 'covalent_bonds_idx': [], 'k_b': [], 'b_0': [],
            'angles_idx': [], 'k_theta': [], 'theta_0': [],
            'dihedrals_idx': [], 'k_phi': [], 'n': [], 'gamma': []
        }
        
        current_idx_offset = 0
        residue_offsets = [] 
        
        # 2. First Pass: Accumulate Intra-Residue Tensors
        for res_name in sequence:
            if res_name not in self.templates:
                available = sorted(self.templates.keys())
                raise KeyError(
                    f"No template found for residue '{res_name}'. "
                    f"Available templates: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
            t = self.templates[res_name]
            residue_offsets.append(current_idx_offset)
            
            accumulated['charges'].append(t['charges'])
            accumulated['sigma'].append(t['sigma'])
            accumulated['epsilon'].append(t['epsilon'])
            accumulated['gb_rho'].append(t.get('gb_rho', torch.zeros_like(t['charges'])))
            accumulated['gb_screen'].append(t.get('gb_screen', torch.zeros_like(t['charges'])))
            
            # Decoupled accumulation
            accumulated['springs_idx'].append(t['intra_springs'] + current_idx_offset)
            accumulated['covalent_bonds_idx'].append(t['intra_covalent_bonds'] + current_idx_offset)
            
            accumulated['angles_idx'].append(t['intra_angles'] + current_idx_offset)
            accumulated['dihedrals_idx'].append(t['intra_dihedrals'] + current_idx_offset)
            
            accumulated['k_b'].append(t['k_b'])
            accumulated['b_0'].append(t['b_0'])
            accumulated['k_theta'].append(t['k_theta'])
            accumulated['theta_0'].append(t['theta_0'])
            accumulated['k_phi'].append(t['k_phi'])
            accumulated['n'].append(t['n'])
            accumulated['gamma'].append(t['gamma'])
            
            current_idx_offset += t['num_atoms']
            
        # 3. Second Pass: Dynamic Inter-Residue Stitching
        inter_springs_idx, inter_covalent_bonds_idx, inter_kb, inter_b0 = [], [], [], []
        inter_angles_idx, inter_ktheta, inter_theta0 = [], [], []
        inter_dihedrals_idx, inter_kphi, inter_n, inter_gamma = [], [], [], []
        inter_cmap_idx, inter_cmap_assignments = [], []
        
        for i in range(num_residues - 1):
            if chain_ids[i] != chain_ids[i+1]:
                continue

            chain_i = chain_types[i]
            if chain_i != chain_types[i+1] or chain_i is None:
                continue
                
            rules = self.peptide_rules if chain_i == 'protein' else self.phosphodiester_rules
            
            # Upgraded indexer that safely evaluates arbitrary multi-residue spanning offsets
            def get_global_idx(rel_offset, atom_name):
                target_idx = i + rel_offset
                # verify the target residue hasn't crossed a chain or type boundary
                if (target_idx >= num_residues or 
                    chain_ids[target_idx] != chain_ids[i] or 
                    chain_types[target_idx] != chain_i):
                    return None # Hit a chain boundary or terminus
                    
                t_target = self.templates[sequence[target_idx]]
                if atom_name not in t_target['atom_name_to_idx']:
                    return None # Atom missing in template
                return t_target['atom_name_to_idx'][atom_name] + residue_offsets[target_idx]

            # Stitch Bonds
            for atoms, kb, b0 in rules.get('bonds', []):
                idx1 = get_global_idx(atoms[0][0], atoms[0][1])
                idx2 = get_global_idx(atoms[1][0], atoms[1][1])
                if idx1 is not None and idx2 is not None:
                    inter_springs_idx.append([idx1, idx2])
                    inter_covalent_bonds_idx.append([idx1, idx2])
                    inter_kb.append(kb)
                    inter_b0.append(b0)
                
            # Stitch Angles
            for atoms, ktheta, theta0 in rules.get('angles', []):
                idx1 = get_global_idx(atoms[0][0], atoms[0][1])
                idx2 = get_global_idx(atoms[1][0], atoms[1][1])
                idx3 = get_global_idx(atoms[2][0], atoms[2][1])
                if None not in (idx1, idx2, idx3):
                    inter_angles_idx.append([idx1, idx2, idx3])
                    inter_ktheta.append(ktheta)
                    inter_theta0.append(theta0)
                
            # Stitch Dihedrals
            for atoms, kphi, n_val, gamma_val in rules.get('dihedrals', []):
                idx1 = get_global_idx(atoms[0][0], atoms[0][1])
                idx2 = get_global_idx(atoms[1][0], atoms[1][1])
                idx3 = get_global_idx(atoms[2][0], atoms[2][1])
                idx4 = get_global_idx(atoms[3][0], atoms[3][1])
                if None not in (idx1, idx2, idx3, idx4):
                    inter_dihedrals_idx.append([idx1, idx2, idx3, idx4])
                    inter_kphi.append(kphi)
                    inter_n.append(n_val)
                    inter_gamma.append(gamma_val)
                    
            # Stitch CMAP
            for atoms in rules.get('cmap', []):
                idxs = [get_global_idx(rel, name) for rel, name in atoms]
                if None not in idxs:
                    # Central residue dictating the map assignment is i+1 based on the offsets
                    map_id = self.templates[sequence[i+1]].get('cmap_map_id', -1)
                    if map_id != -1:
                        inter_cmap_idx.append(idxs)
                        inter_cmap_assignments.append(map_id)
            
        # 4. Final Compilation: Concatenate and map to device
        params = {
            'charges': torch.cat(accumulated['charges']).to(self.device),
            'sigma': torch.cat(accumulated['sigma']).to(self.device),
            'epsilon': torch.cat(accumulated['epsilon']).to(self.device),
            'gb_rho': torch.cat(accumulated['gb_rho']).to(self.device),
            'gb_screen': torch.cat(accumulated['gb_screen']).to(self.device),
        }
        topology = {}
        
        # Merge lists and transpose index arrays to shape (N_atoms_in_term, Num_terms)
        if len(inter_springs_idx) > 0:
            accumulated['springs_idx'].append(torch.tensor(inter_springs_idx, dtype=torch.long).t())
            accumulated['covalent_bonds_idx'].append(torch.tensor(inter_covalent_bonds_idx, dtype=torch.long).t())
            accumulated['k_b'].append(torch.tensor(inter_kb, dtype=torch.float32))
            accumulated['b_0'].append(torch.tensor(inter_b0, dtype=torch.float32))
            
        if len(inter_angles_idx) > 0:
            accumulated['angles_idx'].append(torch.tensor(inter_angles_idx, dtype=torch.long).t())
            accumulated['k_theta'].append(torch.tensor(inter_ktheta, dtype=torch.float32))
            accumulated['theta_0'].append(torch.tensor(inter_theta0, dtype=torch.float32))
            
        if len(inter_dihedrals_idx) > 0:
            accumulated['dihedrals_idx'].append(torch.tensor(inter_dihedrals_idx, dtype=torch.long).t())
            accumulated['k_phi'].append(torch.tensor(inter_kphi, dtype=torch.float32))
            accumulated['n'].append(torch.tensor(inter_n, dtype=torch.float32))
            accumulated['gamma'].append(torch.tensor(inter_gamma, dtype=torch.float32))
            
        # Bind the decoupled graphs
        topology['springs'] = torch.cat(accumulated['springs_idx'], dim=1).to(self.device)
        topology['covalent_bonds'] = torch.cat(accumulated['covalent_bonds_idx'], dim=1).to(self.device)
        
        topology['angles'] = torch.cat(accumulated['angles_idx'], dim=1).to(self.device)
        topology['dihedrals'] = torch.cat(accumulated['dihedrals_idx'], dim=1).to(self.device)
        
        # Bind CMAP if present
        if len(inter_cmap_idx) > 0:
            topology['cmap'] = torch.tensor(inter_cmap_idx, dtype=torch.long).t().to(self.device)
            params['cmap_assignments'] = torch.tensor(inter_cmap_assignments, dtype=torch.long).to(self.device)
            params['global_cmap_grids'] = self.templates.get('global_cmap_grids', None)
            if params['global_cmap_grids'] is not None:
                params['global_cmap_grids'] = params['global_cmap_grids'].to(self.device)
        
        params['k_b'] = torch.cat(accumulated['k_b']).to(self.device)
        params['b_0'] = torch.cat(accumulated['b_0']).to(self.device)
        params['k_theta'] = torch.cat(accumulated['k_theta']).to(self.device)
        params['theta_0'] = torch.cat(accumulated['theta_0']).to(self.device)
        params['k_phi'] = torch.cat(accumulated['k_phi']).to(self.device)
        params['n'] = torch.cat(accumulated['n']).to(self.device)
        params['gamma'] = torch.cat(accumulated['gamma']).to(self.device)
        
        return topology, params