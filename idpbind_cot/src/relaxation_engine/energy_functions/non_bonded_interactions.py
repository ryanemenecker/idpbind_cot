import torch

from idpbind_cot.src.relaxation_engine.utils.constants import (
    COULOMB_CONSTANT,
    DEFAULT_CUTOFF,
    DIELECTRIC_WATER,
    ALPHA_INITIAL,
    VDW_SCALE_14,
    ELEC_SCALE_14,
)

def compute_softcore_lj_energy(coords, neighbor_indices, sigma, epsilon, alpha=ALPHA_INITIAL, cutoff=DEFAULT_CUTOFF):
    """
    Computes a clash-resolving soft-core Lennard-Jones potential.
    
    Args:
        coords: Tensor of shape (N, 3) containing atomic coordinates.
        neighbor_indices: Tensor of shape (2, P) containing pairs of interacting atoms.
        sigma: Tensor of shape (N,) containing atom-specific sigma parameters.
        epsilon: Tensor of shape (N,) containing atom-specific epsilon parameters.
        alpha: Float, dimensionless softening parameter (typically 0.1 to 0.5).
        cutoff: Float, the cutoff radius (R_c) in Angstroms.
        
    Returns:
        Total soft-core Lennard-Jones energy (scalar Tensor).
    """
    
    idx_i = neighbor_indices[0]
    idx_j = neighbor_indices[1]
    
    pos_i = coords[idx_i]
    pos_j = coords[idx_j]
    
    sigma_i, sigma_j = sigma[idx_i], sigma[idx_j]
    eps_i, eps_j = epsilon[idx_i], epsilon[idx_j]
    
    # 1. Apply Lorentz-Berthelot mixing rules
    sigma_ij = (sigma_i + sigma_j) / 2.0
    epsilon_ij = torch.sqrt(eps_i * eps_j)
    
    # 2. Compute squared distances directly (bypassing torch.linalg.norm)
    r_vec = pos_j - pos_i
    r_sq = torch.sum(r_vec * r_vec, dim=-1) # Shape: (P,)
    cutoff_sq = cutoff * cutoff
    mask = (r_sq < cutoff_sq).float()
    
    # 3. Compute 6th powers
    r_6 = r_sq * r_sq * r_sq
    sigma_6 = sigma_ij ** 6
    
    # 4. Construct the soft-core denominator
    # Even if r_sq == 0, the denominator is protected by alpha * sigma_6
    denominator = r_6 + alpha * sigma_6
    
    # 5. Compute the energy terms
    term = sigma_6 / denominator
    term_sq = term * term

    # 6. Apply the mask to the final energy
    energy = 4.0 * epsilon_ij * (term_sq - term) * mask
    
    return torch.sum(energy) * 0.5


def compute_reaction_field_electrostatics(coords, neighbor_indices, charges, 
                                          cutoff=DEFAULT_CUTOFF, eps_rf=DIELECTRIC_WATER, delta=1e-6,
                                          k_rf=None, c_rf=None):
    """
    Computes the Generalized Reaction Field electrostatic potential.
    
    Args:
        coords: Tensor of shape (N, 3) containing atomic coordinates.
        neighbor_indices: Tensor of shape (2, P) containing interacting atom pairs.
        charges: Tensor of shape (N,) containing partial atomic charges.
        cutoff: Float, the cutoff radius (R_c) in Angstroms.
        eps_rf: Float, the dielectric constant of the continuum beyond the cutoff.
        delta: Float, a small softening parameter to prevent 1/r singularities.
        k_rf: Float, pre-computed reaction field constant. If None, computed from eps_rf/cutoff.
        c_rf: Float, pre-computed reaction field shift. If None, computed from eps_rf/cutoff.
        
    Returns:
        Total electrostatic energy (scalar Tensor).
    """
    k_e = COULOMB_CONSTANT
    
    idx_i = neighbor_indices[0]
    idx_j = neighbor_indices[1]
    
    q_i = charges[idx_i]
    q_j = charges[idx_j]
    
    # 1. Compute distances with a numerical stability shift
    r_vec = coords[idx_j] - coords[idx_i]
    r_sq = torch.sum(r_vec * r_vec, dim=-1)
    
    # We add delta *before* the square root to strictly bound the derivative at r=0.
    r_ij = torch.sqrt(r_sq + delta)
    
    # 2. Use pre-computed Reaction Field constants or compute them
    if k_rf is None:
        k_rf = (eps_rf - 1.0) / ((2.0 * eps_rf + 1.0) * (cutoff ** 3))
    if c_rf is None:
        c_rf = (3.0 * eps_rf) / ((2.0 * eps_rf + 1.0) * cutoff)
    
    # 3. Compute the three terms of the Reaction Field potential
    term_direct = 1.0 / r_ij
    term_polarization = k_rf * r_sq
    term_shift = c_rf
    
    # 4. Apply a hard mask to ensure interactions strictly beyond the cutoff are zeroed.
    mask = (r_ij < cutoff).float()
    
    # 5. Calculate total energy
    energy = k_e * q_i * q_j * (term_direct + term_polarization - term_shift) * mask
    
    # Return halved sum to account for double-counting in neighbor lists
    return torch.sum(energy) * 0.5


def compute_dense_nonbonded_energy(coords, sigma, epsilon, alpha=ALPHA_INITIAL):
    """
    Computes soft-core Lennard-Jones potential natively on dense N * N matrices
    bypassing all sparse neighbor lists for fully differentiable SE(3) flows.
    
    Electrostatics evaluation is intentionally omitted in favor of standalone Generalized Born math.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype
    
    # 1. Distance matrix
    delta = coords.unsqueeze(1) - coords.unsqueeze(0)
    r_sq = torch.sum(delta**2, dim=-1)
    
    # Zero diagonal safely
    r_sq = r_sq + torch.eye(N, device=device, dtype=dtype) * 1e-8
    
    # 2. Parameter matrices
    sigma_i = sigma.unsqueeze(1)
    sigma_j = sigma.unsqueeze(0)
    eps_i = epsilon.unsqueeze(1)
    eps_j = epsilon.unsqueeze(0)
    
    sigma_ij = (sigma_i + sigma_j) * 0.5
    epsilon_ij = torch.sqrt(eps_i * eps_j)
    
    r_6 = r_sq * r_sq * r_sq
    sigma_6 = sigma_ij ** 6
    
    denominator = r_6 + alpha * sigma_6
    term = sigma_6 / denominator
    
    # Mask diagonal (self interaction)
    term = term * (1.0 - torch.eye(N, device=device, dtype=dtype))
    
    # E_vdw = 4 * eps * ( (s/r)**12 - (s/r)**6 )
    energy_matrix = 4.0 * epsilon_ij * (term * term - term)
    
    # Because we sum over all N*N we must multiply by 0.5 to prevent double counting pairs
    e_vdw = torch.sum(energy_matrix) * 0.5
    return e_vdw


def compute_14_energy(coords, indices_14, sigma_14, epsilon_14, charges_14,
                      alpha=ALPHA_INITIAL, vdw_scale=VDW_SCALE_14, elec_scale=ELEC_SCALE_14,
                      k_rf_14=None, c_rf_14=None, eps_rf=DIELECTRIC_WATER, delta=1e-6):
    """
    Computes fused 1-4 LJ + electrostatics using pre-gathered parameters.
    No parameter gather — sigma_14, epsilon_14, charges_14 are already 
    per-pair tensors of shape (P_14,).
    
    Args:
        coords: Tensor of shape (N, 3).
        indices_14: Tensor of shape (2, P_14).
        sigma_14: Pre-gathered sigma_ij for 1-4 pairs, shape (P_14,).
        epsilon_14: Pre-gathered epsilon_ij for 1-4 pairs, shape (P_14,).
        charges_14: Pre-gathered q_i * q_j for 1-4 pairs, shape (P_14,).
        alpha: Soft-core parameter.
        vdw_scale, elec_scale: AMBER 1-4 scaling factors.
        k_rf_14, c_rf_14: Pre-computed RF constants (or None for no-cutoff behavior).
        
    Returns:
        (e_vdw_14, e_elec_14) tuple of scalar Tensors.
    """
    pos_i = coords[indices_14[0]]
    pos_j = coords[indices_14[1]]
    r_vec = pos_j - pos_i
    r_sq = torch.sum(r_vec * r_vec, dim=-1)
    
    # LJ (no cutoff for 1-4)
    r_6 = r_sq * r_sq * r_sq
    sigma_6 = sigma_14 ** 6
    denominator = r_6 + alpha * sigma_6
    term = sigma_6 / denominator
    e_vdw_14 = torch.sum(4.0 * epsilon_14 * (term * term - term)) * 0.5 * vdw_scale
    
    # Electrostatics (no cutoff for 1-4)
    r_ij = torch.sqrt(r_sq + delta)
    e_elec_14 = torch.sum(COULOMB_CONSTANT * charges_14 / r_ij) * 0.5 * elec_scale
    
    return e_vdw_14, e_elec_14