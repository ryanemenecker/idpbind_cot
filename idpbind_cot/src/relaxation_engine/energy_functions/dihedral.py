import torch
import torch.nn.functional as F
import math

def compute_cmap_energy(coords, cmap_indices, map_assignments, cmap_grids):
    """
    Computes the 2D CMAP grid energy using CV bilinear/bicubic interpolation.
    
    Args:
        coords: Tensor of shape (N, 3).
        cmap_indices: Tensor of shape (8, M) for M torsions. Atoms 0-3 are phi, 4-7 are psi.
        map_assignments: Tensor of shape (M,) mapping the torsion to its specific grid.
        cmap_grids: Tensor of shape (1, num_maps, resolution, resolution).
    """
    # 1. Calculate raw dihedrals (abstracting the math from your earlier dihedral function)
    # phi = (A-B-C-D), psi = (E-F-G-H)
    def calc_dihedral(idx):
        pos = coords[idx]
        b1, b2, b3 = pos[1]-pos[0], pos[2]-pos[1], pos[3]-pos[2]
        n1, n2 = torch.linalg.cross(b1, b2, dim=-1), torch.linalg.cross(b2, b3, dim=-1)
        b2_hat = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)
        m = torch.linalg.cross(n1, b2_hat, dim=-1)
        x, y = torch.sum(n1 * n2, dim=-1), torch.sum(m * n2, dim=-1)
        return torch.atan2(y, x)

    phi = calc_dihedral(cmap_indices[0:4])
    psi = calc_dihedral(cmap_indices[4:8])
    
    # 2. Normalize angles from [-pi, pi] to [-1, 1] for grid_sample
    x_norm = phi / math.pi
    y_norm = psi / math.pi
    
    # 3. Format coordinates for grid_sample. Shape: (1, M, 1, 2)
    grid_coords = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    
    # 4. Sample all energy maps simultaneously 
    # Output shape: (1, num_maps, M, 1) -> squeeze to (num_maps, M)
    sampled_energies = F.grid_sample(
        cmap_grids, grid_coords, mode='bicubic', padding_mode='border', align_corners=True
    )  # Shape: (1, num_maps, M, 1)
    sampled_energies = sampled_energies.view(sampled_energies.shape[1], sampled_energies.shape[2])
    
    # 5. Extract the specific map energy for each of the M torsions
    M_range = torch.arange(map_assignments.shape[0], device=coords.device)
    energy = sampled_energies[map_assignments, M_range]
    
    return torch.sum(energy)

def compute_dihedral_energy(coords, dihedral_indices, k, n, gamma, eps=1e-6):
    """
    Computes the torsional energy for all dihedrals simultaneously.
    
    Args:
        coords: Tensor of shape (N, 3) containing atomic coordinates.
        dihedral_indices: Tensor of shape (4, M) containing indices for A, B, C, D atoms.
        k: Tensor of shape (M,) containing force constants.
        n: Tensor of shape (M,) containing periodicities.
        gamma: Tensor of shape (M,) containing phase offsets.
        eps: Small constant to prevent division by zero in collinearity.
        
    Returns:
        Total dihedral energy (scalar Tensor).
    """
    
    # 1. Fetch coordinates for all M dihedrals simultaneously via advanced indexing
    pos_A = coords[dihedral_indices[0]] # Shape: (M, 3)
    pos_B = coords[dihedral_indices[1]]
    pos_C = coords[dihedral_indices[2]]
    pos_D = coords[dihedral_indices[3]]

    # 2. Compute bond vectors
    b1 = pos_B - pos_A
    b2 = pos_C - pos_B
    b3 = pos_D - pos_C

    # 3. Compute normal vectors
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)

    # 4. Normalize central bond (with eps to prevent NaN during backward pass)
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
    b2_hat = b2 / (b2_norm + eps)

    # 5. Create orthogonal basis vector 'm'
    m = torch.linalg.cross(n1, b2_hat, dim=-1)

    # 6. Project n2 onto the orthogonal basis (dot products)
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m * n2, dim=-1)

    # 7. Compute the dihedral angles directly
    phi = torch.atan2(y, x)

    # 8. Calculate the standard periodic torsional energy
    energy = k * (1.0 + torch.cos(n * phi - gamma))

    return torch.sum(energy)