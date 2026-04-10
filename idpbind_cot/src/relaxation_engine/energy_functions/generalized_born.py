import torch

def compute_obc2_gb_energy(coords, params, eps_in=1.0, eps_out=78.5):
    """
    Computes OBC2 Generalized Born implicit solvent energy.
    All logic is fully vectorized and strictly enforces numerical stability 
    to prevent NaN gradients during SE(3) backward passes.

    Args:
        coords: Tensor of shape (N, 3) specifying atomic positions.
        params: Dict of 1D tensors:
            - 'charge': Partial atomic charges
            - 'gb_rho': Intrinsic van der Waals radii offset for solvent probe
            - 'gb_screen': Atom-specific scaling factors
        eps_in: Solute dielectric constant (default 1.0)
        eps_out: Solvent dielectric constant (default 78.5)
        
    Returns:
        Scalar tensor of total GB energy in kcal/mol.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype
    
    # 1. Parameter Extraction 
    # Use standard 0.9 A dielectric offset 
    offset = 0.9
    rho_intrinsic = params['gb_rho'].to(dtype=dtype)
    rho = rho_intrinsic - offset # (N,) array
    
    # Atom-specific scaling factors
    S = params['gb_screen'].to(dtype=dtype)
    s = S * rho # (N,) scaled radii of neighbors
    
    q = params['charges'].to(dtype=dtype)
    
    # Expand for pairwise broadcast
    rho_mat = rho.unsqueeze(1) # shape (N, 1) - index i
    s_mat = s.unsqueeze(0)     # shape (1, N) - index j
    
    # 2. Distance Matrix Calculation
    delta = coords.unsqueeze(1) - coords.unsqueeze(0) # (N, N, 3)
    dist_sq = torch.sum(delta**2, dim=-1)
    
    # SAFEGUARD: Add epsilon to the diagonal to prevent infinite gradients from sqrt(0)
    eye_mask = torch.eye(N, device=device, dtype=dtype)
    dist_sq = dist_sq + eye_mask * 1e-8
    r = torch.sqrt(dist_sq) # (N, N)
    
    # 3. Piecewise OBC2 Integral evaluation
    H = torch.zeros((N, N), device=device, dtype=dtype)
    
    # Safe variants of vectors for denominator division where overlap logic dictates.
    # We must explicitly clamp denominators to avoid Inf prior to masking,
    # because autograd will trace all branches of torch.where regardless of the boolean filter.
    
    r_plus_s = r + s_mat
    r_minus_s = r - s_mat
    s_minus_r = s_mat - r
    r2_minus_s2 = r**2 - s_mat**2
    
    # Clamp denominators
    r2_minus_s2_safe = r2_minus_s2.clone()
    # Ensure it's never zero. 
    r2_minus_s2_safe[r2_minus_s2_safe.abs() < 1e-6] = 1e-6
    
    r_plus_s_safe = r_plus_s.clamp(min=1e-6)
    r_safe = r.clone()
    r_safe.diagonal().fill_(1.0) # Completely hide the distance diagonal from zero division
    
    rho_safe = rho_mat.clamp(min=1e-6)
    s_minus_r_safe = s_minus_r.clamp(min=1e-6)
    
    # Regime 1: r > rho_i + s_j
    mask1 = r > (rho_mat + s_mat)
    arg1 = (r_minus_s / r_plus_s_safe).clamp(min=1e-8)
    H1 = 0.5 * (s_mat / r2_minus_s2_safe + (1.0 / (2 * r_safe)) * torch.log(arg1))
    
    # Regime 2: r > |rho_i - s_j| AND r <= rho_i + s_j
    mask2 = (r > torch.abs(rho_mat - s_mat)) & (r <= rho_mat + s_mat)
    arg2 = (rho_mat / r_plus_s_safe).clamp(min=1e-8)
    H2 = 0.25 * (
        (1.0 / rho_safe) 
        - (1.0 / r_plus_s_safe) 
        - (r**2 - s_mat**2 + rho_mat**2) / (2 * r_safe * rho_safe**2) 
        + (1.0 / (2 * r_safe)) * torch.log(arg2)
    )
    
    # Regime 3: r <= s_j - rho_i
    mask3 = r <= s_minus_r
    arg3 = (s_minus_r_safe / r_plus_s_safe).clamp(min=1e-8)
    H3 = 0.5 * (
        s_mat / r2_minus_s2_safe 
        + 2.0 / s_minus_r_safe 
        + (1.0 / (2 * r_safe)) * torch.log(arg3)
    )
    
    # Broadcast branches onto the empty H tensor
    H = torch.where(mask1, H1, H)
    H = torch.where(mask2, H2, H)
    H = torch.where(mask3, H3, H)
    
    # Diagonal self-interactions contribute 0 to the sum I_i
    H = H * (1.0 - eye_mask)
    
    # Evaluate integral (sum over j)
    I = torch.sum(H, dim=1) 
    
    # 4. Total GB Energy Calculation
    # Standard OBC2 parameters
    alpha = 1.0
    beta = 0.8
    gamma = 4.85
    
    # Radii calculation
    # Ignoring the 1 / R_cut metric because we operate on dense N=2000 calculations globally.
    R_inv = (1.0 / rho_safe.squeeze(1)) + alpha * I - beta * I**2 + gamma * I**3 
    
    # Shape (N,)
    R = 1.0 / R_inv.clamp(min=1e-6) 
    
    # Compute the effective pairwise GB interaction term f_GB
    R_i = R.unsqueeze(1)
    R_j = R.unsqueeze(0)
    
    f_gb_sq = dist_sq + R_i * R_j * torch.exp(-dist_sq / (4.0 * R_i * R_j).clamp(min=1e-6))
    f_gb = torch.sqrt(f_gb_sq)
    
    # Calculate energy constants
    k_elec = 166.0 # kcal/mol AMBER electrostatic constant
    eps_factor = (1.0 / eps_in) - (1.0 / eps_out)
    prefactor = -k_elec * eps_factor
    
    # 5. Separation of self vs pairwise evaluations
    q_i = q.unsqueeze(1)
    q_j = q.unsqueeze(0)
    
    # Pairwise energy (sum over all i, j excluding diagonal)
    E_pair_mat = (q_i * q_j) / f_gb
    E_pair_mat = E_pair_mat * (1.0 - eye_mask) # Zero out self-pairing mathematically
    
    # Note: 166 is half of standard Coulomb 332, meaning sum(all i!=j) precisely accounts for identical double counting symmetry.
    E_pair = prefactor * torch.sum(E_pair_mat) 
    
    # Self solvation energy
    E_self = prefactor * torch.sum((q**2) / R)
    
    # Return total energy directly summing pairs and self
    E_total = E_pair + E_self
    
    return E_total
