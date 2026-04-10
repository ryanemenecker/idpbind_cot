import os
import glob
import torch
import torch.nn.functional as F

# --- 1. KYTE-DOOLITTLE HYDROPHOBICITY SCALE ---
# Mapped to [-4.5, 4.5]. We standardize it around 0 for stable metrics.
KD_SCALE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
    'UNK': 0.0 # Catch-all
}

def get_kd_tensor(sequence, device='cpu'):
    """Returns a 1D tensor of hydrophobicity scores for the given sequence."""
    scores = [KD_SCALE.get(res, 0.0) for res in sequence]
    return torch.tensor(scores, dtype=torch.float32, device=device)

# --- 2. VECTORIZED C-BETA PROJECTION ---
def project_c_beta(n_coords, ca_coords, c_coords):
    """
    Given tensors of shape (N, 3), calculates C-beta vector via cross-products 
    of the backbone geometry without mapping to full all-atom formats.
    """
    # 1. Define vectors pointing away from CA
    v_n = n_coords - ca_coords
    v_c = c_coords - ca_coords
    
    # 2. Normalize
    v_n_norm = v_n / torch.clamp(torch.norm(v_n, dim=-1, keepdim=True), min=1e-8)
    v_c_norm = v_c / torch.clamp(torch.norm(v_c, dim=-1, keepdim=True), min=1e-8)
    
    # 3. Backbone Plane Bisector
    # The bisector points "inward" toward the N-CA-C angle.
    # In L-amino acids, CB points "down" relative to the (v_n, v_c) plane normal.
    b = v_n_norm + v_c_norm
    b_norm = b / torch.clamp(torch.norm(b, dim=-1, keepdim=True), min=1e-8)
    
    # Normal to the plane
    normal = torch.cross(v_n_norm, v_c_norm, dim=-1)
    n_norm = normal / torch.clamp(torch.norm(normal, dim=-1, keepdim=True), min=1e-8)
    
    # 4. Ideal tetrahedral combination
    # C-CA-CB and N-CA-CB angles are roughly 109.5 for sp3.
    # We construct the CB position using analytical trig scaling.
    # b_norm vector points roughly "between" N and C. CB needs to point "away" from the bisector.
    # CB = CA - sqrt(1/3)*b_norm - sqrt(2/3)*n_norm
    # Note: the exact sign config determines L/D chirality.
    
    cb_vector = -0.582734 * b_norm - 0.812674 * n_norm
    cb_coords = ca_coords + 1.526 * cb_vector
    return cb_coords

# Note: If PDBs only literally contain CA atoms, we must hallucinate the bisector directly
# from the CA trace (CA_{i-1}, CA_i, CA_{i+1}). 
def project_c_beta_from_ca_trace(ca_coords):
    """
    Fallback method: If STARLING provides strictly solitary CA atoms without N/C, 
    we trace the frame purely through CA coordinates.
    ca_coords: (N, 3)
    """
    N = ca_coords.shape[0]
    cb_coords = torch.zeros_like(ca_coords)
    
    if N < 3:
        return ca_coords # Fallback
        
    v1 = ca_coords[:-2] - ca_coords[1:-1] # CA_{i-1} -> CA_i reversed
    v2 = ca_coords[2:] - ca_coords[1:-1]  # CA_{i+1} -> CA_i reversed
    
    v1_norm = v1 / torch.clamp(torch.norm(v1, dim=-1, keepdim=True), min=1e-8)
    v2_norm = v2 / torch.clamp(torch.norm(v2, dim=-1, keepdim=True), min=1e-8)
    
    n_norm = torch.cross(v1_norm, v2_norm, dim=-1)
    n_norm = n_norm / torch.clamp(torch.norm(n_norm, dim=-1, keepdim=True), min=1e-8)
    
    b_norm = v1_norm + v2_norm
    b_norm = b_norm / torch.clamp(torch.norm(b_norm, dim=-1, keepdim=True), min=1e-8)
    
    # Reconstruct interior
    cb_interior = ca_coords[1:-1] - 0.582 * b_norm - 0.812 * n_norm
    cb_coords[1:-1] = cb_interior
    
    # Fill ends natively
    cb_coords[0] = ca_coords[0] + (cb_interior[0] - ca_coords[1])
    cb_coords[-1] = ca_coords[-1] + (cb_interior[-1] - ca_coords[-2])
    
    return cb_coords

# --- 3. FRACTIONAL CONTACT MAP ---
def compute_fractional_contact_map(ensemble_cb_coords, threshold=8.0):
    """
    ensemble_cb_coords: (M_states, N_res, 3) tensor
    Returns:
        prob_map: (N_res, N_res) containing fractional contact frequency across states.
    """
    print(f"Executing vectorized contact mapping over {ensemble_cb_coords.shape[0]} states...")
    # (M, N, 1, 3) - (M, 1, N, 3) -> (M, N, N, 3)
    delta = ensemble_cb_coords.unsqueeze(2) - ensemble_cb_coords.unsqueeze(1)
    dist_sq = torch.sum(delta**2, dim=-1)
    
    contact_masks = (dist_sq < (threshold * threshold)).float()
    
    # Average across all states (dim 0)
    prob_map = torch.mean(contact_masks, dim=0)
    return prob_map

# --- 4. MOTIF EXTRACTION ---
def extract_anchor_motif(contact_prob_map, sequence, target_residues=None, min_len=6, max_len=10):
    """
    Finds Anchor Motif window balancing Sequence Hydrophobicity against Intra-Chain Entropy.
    Anchor Motif = continuous seq with lowest integrated intra-chain contact frequency 
    and highest sequence hydrophobicity.
    """
    N = contact_prob_map.shape[0]
    kd_scores = get_kd_tensor(sequence, device=contact_prob_map.device)
    
    # Sum contacts per residue. Lower score = fewer intra-chain restrictive bonds (more available).
    # We subtract self contacts (which evaluate to 1.0 prob) natively.
    intra_contact_density = torch.sum(contact_prob_map, dim=-1) - 1.0 
    
    best_window = None
    
    if target_residues:
        print(f"Target Override Executing... Locking window to {target_residues}")
        # Build strict window evaluating around targets, or just return target bounds
        target_min, target_max = min(target_residues), max(target_residues)
        return (target_min, target_max), sequence[target_min:target_max+1]

    # Evaluate dynamic windows
    best_score = float('inf') # We now seek the minimum normalized score
    for w in range(min_len, max_len + 1):
        for i in range(N - w + 1):
            hydropathy_sum = torch.sum(kd_scores[i:i+w]).item()
            contact_sum = torch.sum(intra_contact_density[i:i+w]).item()
            
            # Heuristic Objective: Minimize internal contacts, Maximize Hydropathy
            # We subtract Hydropathy from Contacts, and normalize by window length.
            # Thus, the lowest score represents the best motif.
            normalized_score = (contact_sum - hydropathy_sum) / float(w)
            
            if normalized_score < best_score:
                best_score = normalized_score
                best_window = (i, i+w-1)
                
    start, end = best_window
    anchor_seq = sequence[start:end+1]
    
    print(f"Anchor motif discovered: Residues {start}-{end} [{'-'.join(anchor_seq)}] | Score: {best_score:.2f}")
    return best_window, anchor_seq

# --- 5. BATCH PARSER ---
def parse_starling_ensemble(directory_path, device='cpu'):
    """
    Rapid, PyTorch native tensor parsing of PDBs ignoring heavy dependencies.
    Extracts C-alpha/N/C records directly into (M, N, 3).
    """
    pdb_files = glob.glob(os.path.join(directory_path, '*.pdb'))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {directory_path}")
        
    print(f"Found {len(pdb_files)} conforming ensemble states.")
    
    ensemble_ca = []
    ensemble_cb = []
    global_seq = None
    
    for f_idx, pdb in enumerate(pdb_files):
        n_coords, ca_coords, c_coords = [], [], []
        seq = []
        with open(pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pt = [x, y, z]
                    
                    if atom_name == 'CA':
                        ca_coords.append(pt)
                        if f_idx == 0: seq.append(res_name)
                    elif atom_name == 'N':
                        n_coords.append(pt)
                    elif atom_name == 'C':
                        c_coords.append(pt)
                        
        ca_t = torch.tensor(ca_coords, dtype=torch.float32, device=device)
        
        # Check if full backbone exists
        if len(n_coords) == len(ca_coords) and len(c_coords) == len(ca_coords):
            n_t = torch.tensor(n_coords, dtype=torch.float32, device=device)
            c_t = torch.tensor(c_coords, dtype=torch.float32, device=device)
            cb_t = project_c_beta(n_t, ca_t, c_t)
        else:
            if f_idx == 0: print("Warning: N/C backbone atoms missing. Falling back to CA trace projection.")
            cb_t = project_c_beta_from_ca_trace(ca_t)
            
        ensemble_ca.append(ca_t)
        ensemble_cb.append(cb_t)
        if global_seq is None: global_seq = seq
        
    # Stack -> (M, N, 3)
    return torch.stack(ensemble_ca), torch.stack(ensemble_cb), global_seq

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract Anchor Motif from PDB Ensembles.")
    parser.add_argument('--pdb_dir', type=str, required=True, help="Directory containing coarse-grained STARLING pdbs.")
    parser.add_argument('--target_indices', type=str, default=None, help="Comma separated list of explicit target indices (e.g. 14,15,16,17,18,19). Overrides contact sliding window.")
    
    args = parser.parse_args()
    
    target_idx = None
    if args.target_indices:
        target_idx = [int(x.strip()) for x in args.target_indices.split(',')]
        
    ca_coords, cb_coords, seq = parse_starling_ensemble(args.pdb_dir)
    prob_map = compute_fractional_contact_map(cb_coords, threshold=8.0)
    
    # Execute extraction
    best_window, anchor_seq = extract_anchor_motif(prob_map, seq, target_residues=target_idx, min_len=6, max_len=10)
    
    # Slice Geometric Anchor (mean over all ensemble states to preserve single CoT target)
    start, end = best_window
    anchor_ca_mean = torch.mean(ca_coords[:, start:end+1, :], dim=0)
    anchor_cb_mean = torch.mean(cb_coords[:, start:end+1, :], dim=0)
    
    print(f"Anchor CA shape ready for Module 2: {anchor_ca_mean.shape}")
    print(f"Anchor CB shape ready for Module 2: {anchor_cb_mean.shape}")
