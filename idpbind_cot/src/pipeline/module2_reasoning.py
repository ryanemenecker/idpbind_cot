import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoTReasoningEngine(nn.Module):
    """
    Differentiable Physics Engine for IDR Anchor Motif and Seed interactions.
    Executes structural refinement globally tracking spatial coordinates against
    discrete categorical amino acid limits softly interpolated.
    """
    def __init__(self, anchor_ca, anchor_cb, seed_len=3, device='cpu'):
        super().__init__()
        self.device = device
        
        # Fixed Anchor (Micro-flexibility allowed via tether)
        self.register_buffer('anchor_ca_0', anchor_ca.clone().detach())
        self.anchor_ca = nn.Parameter(anchor_ca.clone())
        self.anchor_cb = nn.Parameter(anchor_cb.clone()) # We can rigidly tie CB to CA later
        
        # Dummy Seed Spatial Extents
        # Initialized adjacent to the anchor
        anchor_center = torch.mean(anchor_ca, dim=0)
        # Shift 5 Å away in an arbitrary clear vector
        shift_vector = torch.tensor([5.0, 5.0, 5.0], device=device)
        
        # Initialize seed linearly
        self.seed_ca = nn.Parameter(
            anchor_center.unsqueeze(0) + shift_vector + torch.randn(seed_len, 3, device=device)*1.5
        )
        # CB vectors pointing dynamically toward the motif's geometric center
        ca_to_motif = anchor_center.unsqueeze(0) - self.seed_ca
        ca_to_motif_norm = ca_to_motif / torch.clamp(torch.norm(ca_to_motif, dim=-1, keepdim=True), min=1e-8)
        self.seed_cb = nn.Parameter(self.seed_ca + 1.526 * ca_to_motif_norm)
        
        # Sequence Logits N x 20
        # Initialize Uniformly
        self.seq_logits = nn.Parameter(torch.randn(seed_len, 20, device=device) * 0.1)
        
        # --- Physics Properties for the 20 categorical AA representations ---
        # Heuristic CG Sigma (radii) and Epsilon (well-depth)
        # We model Gly as tightest, Trp as largest
        self.aa_sigma = torch.linspace(2.5, 4.0, 20, device=device) 
        self.aa_epsilon = torch.linspace(0.1, 0.4, 20, device=device)
        self.tether_k = 10.0 # kcal/mol/A^2

    def soft_sequence_parameters(self, tau):
        """
        Uses Gumbel-Softmax to draw differentiable one-hot representation,
        then maps to physical parameters.
        """
        # Hard=True forces forward pass discrete, backward pass continuous.
        # But for full gradient continuity learning physics, Hard=False is smoother initially.
        soft_one_hot = F.gumbel_softmax(self.seq_logits, tau=tau, hard=False)
        
        # Interpolate params
        seed_sigmas = torch.matmul(soft_one_hot, self.aa_sigma) # (Seed_Len,)
        seed_epsilons = torch.matmul(soft_one_hot, self.aa_epsilon) # (Seed_Len,)
        
        return soft_one_hot, seed_sigmas, seed_epsilons

    def forward(self, tau=1.0):
        """
        Calculates L_total.
        """
        _, seed_sigmas, seed_epsilons = self.soft_sequence_parameters(tau)
        
        # Anchor params (Heuristic average for the fixed motif)
        # Assuming Anchor is also uniformly represented roughly as Sigma=3.0, Eps=0.2
        anchor_sigmas = torch.ones(self.anchor_ca.shape[0], device=self.device) * 3.0
        anchor_epsilons = torch.ones(self.anchor_ca.shape[0], device=self.device) * 0.2
        
        # 1. Inter-chain Lennard-Jones (Seed CB <-> Motif CB)
        delta_cb = self.seed_cb.unsqueeze(1) - self.anchor_cb.unsqueeze(0)
        dist_sq_cb = torch.sum(delta_cb**2, dim=-1) + 1e-8
        dist_cb = torch.sqrt(dist_sq_cb)
        
        s_i = seed_sigmas.unsqueeze(1)
        s_j = anchor_sigmas.unsqueeze(0)
        sigma_ij = (s_i + s_j) * 0.5
        eps_ij = torch.sqrt(seed_epsilons.unsqueeze(1) * anchor_epsilons.unsqueeze(0))
        
        # Standard LJ 12-6
        # Attractive & Repulsive
        ratio = sigma_ij / dist_cb
        term6 = ratio ** 6
        term12 = term6 ** 2
        lj_energy_cb = 4.0 * eps_ij * (term12 - term6)
        
        # 2. Inter-chain Backbone Sterics (Seed CA <-> Motif CA)
        # Prevents backbone collapse. Highly repulsive penalty (r < 3.8A)
        delta_ca = self.seed_ca.unsqueeze(1) - self.anchor_ca.unsqueeze(0)
        dist_ca = torch.sqrt(torch.sum(delta_ca**2, dim=-1) + 1e-8)
        steric_ca = torch.sum(F.relu(3.8 - dist_ca)**2) * 50.0 # Repulsive spring
        
        # Seed Intra-chain constraints
        # Ensure adjacent CA's are ~3.8A
        if self.seed_ca.shape[0] > 1:
            seed_ca_delta = self.seed_ca[:-1] - self.seed_ca[1:]
            seed_ca_dist = torch.sqrt(torch.sum(seed_ca_delta**2, dim=-1) + 1e-8)
            seed_chain_tether = torch.sum((seed_ca_dist - 3.8)**2) * 50.0
            
            # Ensure CB is attached to CA correctly
            seed_cb_delta = self.seed_cb - self.seed_ca
            seed_cb_dist = torch.sqrt(torch.sum(seed_cb_delta**2, dim=-1) + 1e-8)
            seed_cb_tether = torch.sum((seed_cb_dist - 1.53)**2) * 50.0
        else:
            seed_chain_tether = 0.0
            seed_cb_tether = 0.0

        # 3. Anchor Harmonnic Tethering U = k(x - x_0)^2
        # Micro-flexibility: allowing Anchor to move slightly
        anchor_tether_energy = torch.sum(self.tether_k * torch.sum((self.anchor_ca - self.anchor_ca_0)**2, dim=-1))
        
        # 4. Solvation (Exposed hydrophobic surface area proxy)
        # Minimizing exposed interaction surface
        solvation_penalty = torch.sum(term12) * 0.1 # Dampened simple penalty proxy
        
        L_total = torch.sum(lj_energy_cb) + steric_ca + seed_chain_tether + seed_cb_tether + anchor_tether_energy + solvation_penalty
        
        return L_total


def run_reasoning_loop(anchor_ca, anchor_cb, seed_len=3, steps=500, device='cpu'):
    """
    Executes Adam Optimization continuously decreasing tau.
    """
    model = CoTReasoningEngine(anchor_ca, anchor_cb, seed_len=seed_len, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    tau_init = 5.0
    tau_final = 0.1
    
    print("Beginning PyTorch CoT Inter-Chain Optimization...")
    for step in range(steps):
        # Anneal tau
        progress = step / float(steps)
        tau = tau_init * math.exp(math.log(tau_final / tau_init) * progress)
        
        optimizer.zero_grad()
        loss = model(tau=tau)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Tau: {tau:.3f} | L_total: {loss.item():.4f}")
            
    # Finalize Discrete Sequence
    opt_logits = model.seq_logits.detach()
    idx = torch.argmax(opt_logits, dim=-1)
    
    # Generic Mapping
    AA_MAPPING = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    seq = [AA_MAPPING[i] for i in idx]
    
    print(f"Optimized Seed Sequence: {''.join(seq)}")
    
    return {
        'seed_seq': ''.join(seq),
        'seed_ca': model.seed_ca.detach(),
        'seed_cb': model.seed_cb.detach(),
        'anchor_ca_opt': model.anchor_ca.detach(),
        'anchor_cb_opt': model.anchor_cb.detach()
    }
