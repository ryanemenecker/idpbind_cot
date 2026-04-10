import torch

def build_topology_masks(covalent_bond_indices, num_atoms):
    """
    Constructs a frozen exclusion hash set and 1-4 pair indices using 
    sparse BFS graph traversal. Returns a frozen set for O(1) exclusion 
    lookups instead of an O(N²) dense matrix.
    
    Args:
        covalent_bond_indices: Tensor of shape (2, B) containing strictly true 1-2 
                               covalent bonds. (DO NOT pass Urey-Bradley springs here).
        num_atoms: Integer, the total number of atoms N.
        
    Returns:
        exclusion_set: A frozen Python set of (i, j) tuples for excluded pairs.
        indices_14: Tensor of shape (2, P_14) containing all 1-4 pairs (bidirectional).
    """
    device = covalent_bond_indices.device
    
    # 1. Build adjacency list from covalent bonds
    adj = [[] for _ in range(num_atoms)]
    src = covalent_bond_indices[0].tolist()
    dst = covalent_bond_indices[1].tolist()
    for a, b in zip(src, dst):
        adj[a].append(b)
        adj[b].append(a)
    
    # 2. BFS from each atom to depth 3 to find exclusion and 1-4 pairs
    exclusion_pairs = set()
    pairs_14_i, pairs_14_j = [], []
    
    for atom in range(num_atoms):
        # Self-exclusion
        exclusion_pairs.add((atom, atom))
        
        # Depth 1: direct bonds (1-2 pairs)
        neighbors_1 = set(adj[atom])
        for n1 in neighbors_1:
            exclusion_pairs.add((atom, n1))
        
        # Depth 2: two-bond paths (1-3 pairs)
        neighbors_2 = set()
        for n1 in neighbors_1:
            for n2 in adj[n1]:
                if n2 != atom and n2 not in neighbors_1:
                    neighbors_2.add(n2)
        for n2 in neighbors_2:
            exclusion_pairs.add((atom, n2))
        
        # Depth 3: three-bond paths (1-4 pairs)
        neighbors_3 = set()
        for n2 in neighbors_2:
            for n3 in adj[n2]:
                if n3 != atom and n3 not in neighbors_1 and n3 not in neighbors_2:
                    neighbors_3.add(n3)
        for n3 in neighbors_3:
            exclusion_pairs.add((atom, n3))
            pairs_14_i.append(atom)
            pairs_14_j.append(n3)
    
    # 3. Freeze the exclusion set for fast O(1) lookups
    exclusion_set = frozenset(exclusion_pairs)
    
    # 4. Build 1-4 pair indices (bidirectional for consistency with *0.5 in energy functions)
    if pairs_14_i:
        indices_14 = torch.tensor([pairs_14_i, pairs_14_j], dtype=torch.long, device=device)
    else:
        indices_14 = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    return exclusion_set, indices_14