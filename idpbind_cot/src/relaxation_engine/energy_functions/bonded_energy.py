import torch

def compute_bond_energy(coords, bond_indices, k_b, b_0, eps=1e-6):
    """
    Computes the harmonic bond energy for all covalent bonds.
    
    Args:
        coords: Tensor of shape (N, 3) containing atomic coordinates.
        bond_indices: Tensor of shape (2, B) containing pairs of bonded atoms.
        k_b: Tensor of shape (B,) containing bond force constants.
        b_0: Tensor of shape (B,) containing equilibrium bond lengths.
        eps: Small constant to prevent NaN gradients at exactly r=0.
        
    Returns:
        Total bond energy (scalar Tensor).
    """
    pos_A = coords[bond_indices[0]] # Shape: (B, 3)
    pos_B = coords[bond_indices[1]] # Shape: (B, 3)
    
    # Compute the vector between bonded atoms
    b_vec = pos_B - pos_A
    
    # Compute distance safely using squared norm + eps
    b_sq = torch.sum(b_vec * b_vec, dim=-1)
    b = torch.sqrt(b_sq + eps)
    
    # Harmonic oscillator: OpenMM stores k for E = ½k(r-r₀)²
    delta = b - b_0
    energy = 0.5 * k_b * (delta * delta)
    
    return torch.sum(energy)


def compute_angle_energy(coords, angle_indices, k_theta, theta_0, eps=1e-6):
    """
    Computes the harmonic angle energy for all bond angles.
    
    Args:
        coords: Tensor of shape (N, 3) containing atomic coordinates.
        angle_indices: Tensor of shape (3, A) containing A, B, C atom indices (B is central).
        k_theta: Tensor of shape (A,) containing angle force constants.
        theta_0: Tensor of shape (A,) containing equilibrium angles (in radians).
        eps: Small constant to prevent division by zero and arccos singularities.
        
    Returns:
        Total angle energy (scalar Tensor).
    """
    pos_A = coords[angle_indices[0]] # Shape: (A, 3)
    pos_B = coords[angle_indices[1]] # Shape: (A, 3)
    pos_C = coords[angle_indices[2]] # Shape: (A, 3)
    
    # Vectors radiating from the central atom B
    v1 = pos_A - pos_B
    v2 = pos_C - pos_B
    
    # Compute vector norms safely
    v1_sq = torch.sum(v1 * v1, dim=-1)
    v2_sq = torch.sum(v2 * v2, dim=-1)
    v1_norm = torch.sqrt(v1_sq + eps)
    v2_norm = torch.sqrt(v2_sq + eps)
    
    # Compute the dot product
    dot_product = torch.sum(v1 * v2, dim=-1)
    
    # Calculate cosine of the angle and clamp it to avoid arccos singularity
    cos_theta = dot_product / (v1_norm * v2_norm)
    cos_theta = torch.clamp(cos_theta, min=-1.0 + eps, max=1.0 - eps)
    
    # Compute angle and harmonic energy: OpenMM stores k for E = ½k(θ-θ₀)²
    theta = torch.acos(cos_theta)
    delta = theta - theta_0
    energy = 0.5 * k_theta * (delta * delta)
    
    return torch.sum(energy)


def compute_bond_angle_energy(coords, bond_indices, k_b, b_0, angle_indices, k_theta, theta_0, eps=1e-6):
    """
    Computes the total bond and angle energy in a single fused pass.
    Returns (e_bond, e_angle, e_total)
    """
    # Bonds
    pos_A = coords[bond_indices[0]]
    pos_B = coords[bond_indices[1]]
    b_vec = pos_B - pos_A
    b_sq = torch.sum(b_vec * b_vec, dim=-1)
    b = torch.sqrt(b_sq + eps)
    delta_b = b - b_0
    e_bond = 0.5 * k_b * (delta_b * delta_b)
    # Angles
    pos_Aa = coords[angle_indices[0]]
    pos_Bb = coords[angle_indices[1]]
    pos_Cc = coords[angle_indices[2]]
    v1 = pos_Aa - pos_Bb
    v2 = pos_Cc - pos_Bb
    v1_sq = torch.sum(v1 * v1, dim=-1)
    v2_sq = torch.sum(v2 * v2, dim=-1)
    v1_norm = torch.sqrt(v1_sq + eps)
    v2_norm = torch.sqrt(v2_sq + eps)
    dot_product = torch.sum(v1 * v2, dim=-1)
    cos_theta = dot_product / (v1_norm * v2_norm)
    cos_theta = torch.clamp(cos_theta, min=-1.0 + eps, max=1.0 - eps)
    theta = torch.acos(cos_theta)
    delta_theta = theta - theta_0
    e_angle = 0.5 * k_theta * (delta_theta * delta_theta)
    # Return both and total
    total_bond = torch.sum(e_bond)
    total_angle = torch.sum(e_angle)
    return total_bond, total_angle, total_bond + total_angle