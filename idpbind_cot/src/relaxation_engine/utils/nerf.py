"""
nerf.py

Differentiable internal-coordinate to Cartesian builder for 23-atom protein
residues.

Architecture
------------
**Backbone (N, CA, C, O)** is placed using the **Cα pseudo-frame** approach
(identical to ``ca_to_backbone``):

  1. For each residue *i* build a local SO(3) frame from the vectors
     Cα_{i-1}->Cα_i and Cα_i->Cα_{i+1} via Gram-Schmidt.
  2. Predict N and C as ideal-geometry local offsets, rotated by the
     predicted phi and psi angles, then projected back into global coordinates.
  3. Place O via the standard peptide-plane rule (trans to N across C).

This avoids the fatal flaw of the old NeRF approach which tried to interpret
Ramachandran dihedrals as if the reference frame were (Cα_prev, Cα, Cα_next),
producing coordinates ~89 Å from reality.

**Cb** is placed from the actual N-Cα-C frame using ideal tetrahedral
L-amino-acid chirality.

**Sidechain (CG->CD->CE->CZ)** uses the standard NeRF placement algorithm
from the backbone atoms, walking along the chi1-4 torsion angles.

The Cα pseudo-torsion angle (tau = Cα_{i-1}-Cα_i-Cα_{i+1}-Cα_{i+2}) is also
exported as a feature -- its correlation with the peptide plane orientation
provides a strong geometric prior for backbone reconstruction.

.. warning::

   **Non-standard dihedral convention.** The ``extract_dihedral()`` function
   in this module uses vectors A→B and C→D, **not** the standard IUPAC
   convention (B→A and D→C).  As a result, trans peptide bonds evaluate
   to 0 rad here vs. π rad in standard tools.  Do **not** mix dihedrals
   from this module with values from external tools (e.g. BioPython,
   MDAnalysis) without explicit sign/offset conversion.

References
----------
- Engh & Huber, Acta Cryst. A47, 392 (1991) -- ideal bond lengths/angles.
- Parsons et al. J. Comput. Chem. 26, 1063 (2005) -- NeRF algorithm.


IMPORTANT NOTE:
THE WAY THAT WE CALCULATE EXTRACT_DIHEDRAL USES
b1 = b - a  # Vector from A to B
b3 = d - c  # Vector from C to D

In a trans peptide bond ($C\alpha_{prev}$ -- $C_{prev}$ -- $N_{curr}$
-- $C\alpha_{curr}$):$A$ ($C\alpha_{prev}$) and $D$ 
($C\alpha_{curr}$) sit on opposite sides of the central peptide bond.
Because they are on opposite sides, the vector pointing from $A$ 
into the bond (b1), and the vector pointing away from the 
bond to $D$ (b3), are actually pointing in the same 
relative direction when projected onto the plane perpendicular 
to the bond.Since your v and w vectors point in the same direction, 
their dot product (x = torch.sum(v * w)) evaluates to +1.When $x$ 
is positive and $y$ is near zero, torch.atan2(0, 1) evaluates to 
$0$ radians.The Standard ConventionThe standard IUPAC definition 
for dihedrals uses the vector from $B \to A$ ($a - b$), not $A \to 
B$.

Because standard algorithms project vectors pointing 
away from the central bond on both sides, those projected 
vectors point in opposite directions for a trans bond. 
The dot product of the normal vectors is -1 for trans peptides, 
not +1. torch.atan2(0, -1) evaluates to pi radians (180 degrees).

THIS IS NOT STANDARD CONVENTION BUT IT WORKS.
See the warning block in the module docstring above.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from idpbind_cot.src.relaxation_engine.utils.common_utils import safe_norm, safe_normalize
from idpbind_cot.src.relaxation_engine.utils.constants import (
    IDEAL_ATOM_MASK,
    BOND_N_CA, BOND_CA_C, BOND_C_N, BOND_C_O, BOND_CA_CB,
    ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_CA_C_O, ANGLE_C_N_CA, ANGLE_N_CA_CB,
    _GENERIC_BL, _GENERIC_BA,
    _SC_BOND_LENGTHS, _SC_BOND_ANGLES,
    IDEAL_N_LOCAL, IDEAL_C_LOCAL,
)

# ---------------------------------------------------------------------------
# Sidechain generic bond lengths and angles (nerf-only fallbacks used when
# seq is not provided).
# ---------------------------------------------------------------------------
BOND_CB_CG: float = 1.530
BOND_CG_CD: float = 1.520
BOND_CD_CE: float = 1.510
BOND_CE_CZ: float = 1.510
ANGLE_CA_CB_CG: float = math.radians(113.8)
ANGLE_CB_CG_CD: float = math.radians(113.0)
ANGLE_CG_CD_CE: float = math.radians(113.0)
ANGLE_CD_CE_CZ: float = math.radians(113.0)

# ---------------------------------------------------------------------------
# Per-residue CG2 geometry for branched residues (VAL, ILE, THR).
#
# CG2 branches off Cβ at the same bond origin as CG1/OG1 but at a different
# dihedral angle (~120° away, tetrahedral sp3 branching).  Derived from the
# rigid_group_atom_positions CG2 entries in the chi1 frame:
#   bond_length = norm(position)
#   bond_angle  = acos(-x / bond_length)
#   dihedral_offset = atan2(z, y)    [relative to chi1]
#
# Non-branched residue entries are placeholders (never used because CG2 is
# only placed when a branched-residue mask is active).
# ---------------------------------------------------------------------------
_CG2_BOND_LENGTHS = torch.tensor([
    _GENERIC_BL,  # 0  PAD
    _GENERIC_BL,  # 1  ALA
    _GENERIC_BL,  # 2  CYS
    _GENERIC_BL,  # 3  ASP
    _GENERIC_BL,  # 4  GLU
    _GENERIC_BL,  # 5  PHE
    _GENERIC_BL,  # 6  GLY
    _GENERIC_BL,  # 7  HIS
    1.5315,        # 8  ILE  CG2 = (0.540, -0.785, -1.199)
    _GENERIC_BL,  # 9  LYS
    _GENERIC_BL,  # 10 LEU
    _GENERIC_BL,  # 11 MET
    _GENERIC_BL,  # 12 ASN
    _GENERIC_BL,  # 13 PRO
    _GENERIC_BL,  # 14 GLN
    _GENERIC_BL,  # 15 ARG
    _GENERIC_BL,  # 16 SER
    1.5251,        # 17 THR  CG2 = (0.550, -0.718, -1.228)
    1.5276,        # 18 VAL  CG2 = (0.533, -0.776,  1.203)
    _GENERIC_BL,  # 19 TRP
    _GENERIC_BL,  # 20 TYR
    _GENERIC_BL,  # 21 UNK
], dtype=torch.float32)  # shape (22,)

_CG2_BOND_ANGLES = torch.tensor([
    _GENERIC_BA,  # 0  PAD
    _GENERIC_BA,  # 1  ALA
    _GENERIC_BA,  # 2  CYS
    _GENERIC_BA,  # 3  ASP
    _GENERIC_BA,  # 4  GLU
    _GENERIC_BA,  # 5  PHE
    _GENERIC_BA,  # 6  GLY
    _GENERIC_BA,  # 7  HIS
    1.9311,        # 8  ILE  110.65°
    _GENERIC_BA,  # 9  LYS
    _GENERIC_BA,  # 10 LEU
    _GENERIC_BA,  # 11 MET
    _GENERIC_BA,  # 12 ASN
    _GENERIC_BA,  # 13 PRO
    _GENERIC_BA,  # 14 GLN
    _GENERIC_BA,  # 15 ARG
    _GENERIC_BA,  # 16 SER
    1.9397,        # 17 THR  111.14°
    1.9272,        # 18 VAL  110.42°
    _GENERIC_BA,  # 19 TRP
    _GENERIC_BA,  # 20 TYR
    _GENERIC_BA,  # 21 UNK
], dtype=torch.float32)  # shape (22,)

_CG2_DIHEDRAL_OFFSETS = torch.tensor([
    0.0,           # 0  PAD
    0.0,           # 1  ALA
    0.0,           # 2  CYS
    0.0,           # 3  ASP
    0.0,           # 4  GLU
    0.0,           # 5  PHE
    0.0,           # 6  GLY
    0.0,           # 7  HIS
    2.1505,        # 8  ILE  +123.21°
    0.0,           # 9  LYS
    0.0,           # 10 LEU
    0.0,           # 11 MET
    0.0,           # 12 ASN
    0.0,           # 13 PRO
    0.0,           # 14 GLN
    0.0,           # 15 ARG
    0.0,           # 16 SER
    2.0999,        # 17 THR  +120.31°
    -2.1437,       # 18 VAL  −122.82°
    0.0,           # 19 TRP
    0.0,           # 20 TYR
    0.0,           # 21 UNK
], dtype=torch.float32)  # shape (22,)

# ---------------------------------------------------------------------------
# Aromatic ring atom coordinates in the chi2 rigid-group frame.
#
# For aromatic residues (PHE, HIS, TRP, TYR), the ring atoms form a planar
# rigid body that rotates as a unit around the CB→CG axis (chi2).  Rather
# than walking a linear NeRF chain—which cannot close a ring—we store ideal
# atom positions from AlphaFold’s rigid_group_atom_positions (group 5) and
# transform them as a rigid body into global coordinates.
#
# Shape: (22, 10, 3) — [residue_type, ring_atom_idx, xyz]
#   ring_atom_idx 0 → slot 9  (CD1 / ND1)
#   ring_atom_idx 1 → slot 10 (CD2)
#   ring_atom_idx 2 → slot 12 (CE1 / NE1)
#   ring_atom_idx 3 → slot 13 (CZ / CZ2)
#   ring_atom_idx 4 → slot 14 (CE2)
#   ring_atom_idx 5 → slot 15 (NE2)
#   ring_atom_idx 6 → slot 16 (CE3)
#   ring_atom_idx 7 → slot 17 (CZ3)
#   ring_atom_idx 8 → slot 18 (CH2)
#   ring_atom_idx 9 → slot 19 (OH)
#
# Non-aromatic entries are zero (never accessed because of aromatic mask).
# ---------------------------------------------------------------------------
_RING_COORDS = torch.zeros(22, 10, 3)

# PHE (5): hexagonal ring — CD1, CD2, CE1, CZ, CE2
_RING_COORDS[5, 0] = torch.tensor([0.709, 1.195, 0.000])      # CD1 → slot 9
_RING_COORDS[5, 1] = torch.tensor([0.706, -1.196, 0.000])     # CD2 → slot 10
_RING_COORDS[5, 2] = torch.tensor([2.102, 1.198, 0.000])      # CE1 → slot 12
_RING_COORDS[5, 3] = torch.tensor([2.794, -0.003, -0.001])    # CZ  → slot 13
_RING_COORDS[5, 4] = torch.tensor([2.100, -1.199, -0.001])    # CE2 → slot 14

# HIS (7): imidazole ring — ND1, CD2, CE1, NE2
_RING_COORDS[7, 0] = torch.tensor([0.744, 1.160, 0.000])      # ND1 → slot 9
_RING_COORDS[7, 1] = torch.tensor([0.889, -1.021, 0.003])     # CD2 → slot 10
_RING_COORDS[7, 2] = torch.tensor([2.030, 0.851, 0.002])      # CE1 → slot 12
_RING_COORDS[7, 5] = torch.tensor([2.094, -0.534, 0.002])     # NE2 → slot 15

# TRP (19): indole (fused 6+5 ring) — CD1, CD2, NE1, CZ2, CE2, CE3, CZ3, CH2
_RING_COORDS[19, 0] = torch.tensor([0.824, 1.091, 0.000])     # CD1 → slot 9
_RING_COORDS[19, 1] = torch.tensor([0.854, -1.148, -0.005])   # CD2 → slot 10
_RING_COORDS[19, 2] = torch.tensor([2.140, 0.690, -0.004])    # NE1 → slot 12
_RING_COORDS[19, 3] = torch.tensor([3.283, -1.543, -0.011])   # CZ2 → slot 13
_RING_COORDS[19, 4] = torch.tensor([2.228, -0.659, -0.007])   # CE2 → slot 14
_RING_COORDS[19, 6] = torch.tensor([0.622, -2.529, -0.007])   # CE3 → slot 16
_RING_COORDS[19, 7] = torch.tensor([1.843, -3.402, -0.011])   # CZ3 → slot 17
_RING_COORDS[19, 8] = torch.tensor([3.057, -2.910, -0.013])   # CH2 → slot 18

# TYR (20): hydroxyphenyl ring — CD1, CD2, CE1, CZ, CE2, OH
_RING_COORDS[20, 0] = torch.tensor([0.716, 1.195, 0.000])     # CD1 → slot 9
_RING_COORDS[20, 1] = torch.tensor([0.713, -1.194, -0.001])   # CD2 → slot 10
_RING_COORDS[20, 2] = torch.tensor([2.107, 1.200, -0.002])    # CE1 → slot 12
_RING_COORDS[20, 3] = torch.tensor([2.791, -0.001, -0.003])   # CZ  → slot 13
_RING_COORDS[20, 4] = torch.tensor([2.104, -1.201, -0.003])   # CE2 → slot 14
_RING_COORDS[20, 9] = torch.tensor([4.168, -0.002, -0.005])   # OH  → slot 19

_IS_AROMATIC = torch.zeros(22, dtype=torch.bool)
_IS_AROMATIC[5] = True    # PHE
_IS_AROMATIC[7] = True    # HIS
_IS_AROMATIC[19] = True   # TRP
_IS_AROMATIC[20] = True   # TYR

# ---------------------------------------------------------------------------
# Chi-2 → slot routing and branch-atom geometry for slots 10 and 12.
#
# For non-aromatic residues, chi2 places an atom in EITHER slot 8 (CD/SD)
# or slot 9 (CD1/OD1/ND1), never both.  A secondary "branch" atom at slot 10
# (CD2/OD2/ND2) may exist in the same chi2 rigid group, offset by ~120-180°
# in dihedral.  Similarly, at the chi3 level, a branch at slot 12 (OE2/NE2)
# may exist for carboxylate/amide residues.
# ---------------------------------------------------------------------------

# True => chi2 atom goes to slot 8 (CD/SD); False => slot 9 (CD1/OD1).
_CHI2_TO_SLOT8 = torch.zeros(22, dtype=torch.bool)
_CHI2_TO_SLOT8[4]  = True   # GLU  chi2→CD (slot 8)
_CHI2_TO_SLOT8[9]  = True   # LYS  chi2→CD (slot 8)
_CHI2_TO_SLOT8[11] = True   # MET  chi2→SD (slot 8)
_CHI2_TO_SLOT8[13] = True   # PRO  chi2→CD (slot 8)
_CHI2_TO_SLOT8[14] = True   # GLN  chi2→CD (slot 8)
_CHI2_TO_SLOT8[15] = True   # ARG  chi2→CD (slot 8)

# -- Slot 10 branch: CD2 / OD2 / ND2 (chi2-frame offset from slot 9 atom) --
_HAS_SLOT10 = torch.zeros(22, dtype=torch.bool)
_HAS_SLOT10[3]  = True   # ASP  OD2
_HAS_SLOT10[10] = True   # LEU  CD2
_HAS_SLOT10[12] = True   # ASN  ND2

_SLOT10_BL = torch.full((22,), _GENERIC_BL)
_SLOT10_BA = torch.full((22,), _GENERIC_BA)
_SLOT10_DOFF = torch.zeros(22)
_SLOT10_BL[3]  = 1.2501;  _SLOT10_BA[3]  = 2.0641;  _SLOT10_DOFF[3]  =  3.1389   # ASP OD2
_SLOT10_BL[10] = 1.5249;  _SLOT10_BA[10] = 1.9293;  _SLOT10_DOFF[10] = -2.1436   # LEU CD2
_SLOT10_BL[12] = 1.3278;  _SLOT10_BA[12] = 2.0338;  _SLOT10_DOFF[12] = -3.1408   # ASN ND2

# -- Slot 12 branch: OE2 / NE2 (chi3-frame offset from slot 11 atom) -------
_HAS_SLOT12 = torch.zeros(22, dtype=torch.bool)
_HAS_SLOT12[4]  = True   # GLU  OE2
_HAS_SLOT12[14] = True   # GLN  NE2

_SLOT12_BL = torch.full((22,), _GENERIC_BL)
_SLOT12_BA = torch.full((22,), _GENERIC_BA)
_SLOT12_DOFF = torch.zeros(22)
_SLOT12_BL[4]  = 1.2513;  _SLOT12_BA[4]  = 2.0609;  _SLOT12_DOFF[4]  =  3.1407   # GLU OE2
_SLOT12_BL[14] = 1.3287;  _SLOT12_BA[14] = 2.0334;  _SLOT12_DOFF[14] =  3.1408   # GLN NE2

# C-beta placement dihedral for L-amino acid chirality (radians)
DIHEDRAL_CB: float = math.radians(122.6)

# --- Add this right below the global tensor definitions ---
_CACHED_DEVICE_TENSORS = {}

def _get_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype = None) -> torch.Tensor:
    """Fetches a tensor and caches it on the device to prevent PCIe bottlenecks."""
    # Use id(tensor) to uniquely identify the global constant
    key = (id(tensor), device, dtype)
    if key not in _CACHED_DEVICE_TENSORS:
        _CACHED_DEVICE_TENSORS[key] = tensor.to(device=device, dtype=dtype)
    return _CACHED_DEVICE_TENSORS[key]

# ---------------------------------------------------------------------------
# Cα pseudo-frame construction (parallel across all residues)
# ---------------------------------------------------------------------------
def construct_ca_pseudo_frames(
    ca_coords: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-residue SO(3) frames from a Cα trace.

    Terminal residues get extrapolated virtual neighbours so frames are
    valid everywhere.

    Parameters
    ----------
    ca_coords : Tensor (B, L, 3)

    Returns
    -------
    R : Tensor (B, L, 3, 3)   -- rotation matrices (columns = e1, e2, e3)
    t : Tensor (B, L, 3)      -- translations (= Cα positions)
    """
    B, L, _ = ca_coords.shape

    if L == 0:
        return (
            ca_coords.new_zeros((B, 0, 3, 3)),
            ca_coords,
        )
    if L == 1:
        eye = torch.eye(3, device=ca_coords.device, dtype=ca_coords.dtype)
        R = eye.view(1, 1, 3, 3).expand(B, 1, 3, 3).clone()
        return R, ca_coords

    # Extrapolate termini: generate non-collinear virtual neighbours so the
    # Gram-Schmidt process produces valid orthogonal frames *everywhere*,
    # including the first and last residues.  The old approach mirrored the
    # nearest Cα through the terminus, yielding anti-parallel v_fwd/v_bwd
    # whose cross-product is zero — and then papered over the NaN with
    # ``R[:, 0] = R[:, 1]``, destroying terminal frame accuracy.  Instead we
    # add a small perpendicular displacement so the terminal frame faithfully
    # represents the local backbone direction at each end.

    ca_prev = torch.empty_like(ca_coords)
    ca_prev[:, 1:] = ca_coords[:, :-1]

    ca_next = torch.empty_like(ca_coords)
    ca_next[:, :-1] = ca_coords[:, 1:]

    # --- N-terminus (residue 0): extrapolate backward with perpendicular kick
    fwd_0 = ca_coords[:, 1] - ca_coords[:, 0]  # (B, 3)
    fwd_0_norm = safe_normalize(fwd_0, dim=-1)
    # Pick a reference axis that is least parallel to fwd_0
    abs_fwd = fwd_0_norm.abs()
    # Minimum-component index per batch element → one-hot
    _min_idx = abs_fwd.argmin(dim=-1)  # (B,)
    ref = torch.zeros_like(fwd_0_norm)
    ref.scatter_(-1, _min_idx.unsqueeze(-1), 1.0)
    perp_0 = torch.cross(fwd_0_norm, ref, dim=-1)
    perp_0 = safe_normalize(perp_0, dim=-1)
    ca_prev[:, 0] = ca_coords[:, 0] - fwd_0 + 0.3 * safe_norm(fwd_0, dim=-1, keepdim=True) * perp_0

    # --- C-terminus (residue L-1): extrapolate forward with perpendicular kick
    bwd_L = ca_coords[:, -2] - ca_coords[:, -1]  # (B, 3)
    bwd_L_norm = safe_normalize(bwd_L, dim=-1)
    abs_bwd = bwd_L_norm.abs()
    _min_idx_L = abs_bwd.argmin(dim=-1)
    ref_L = torch.zeros_like(bwd_L_norm)
    ref_L.scatter_(-1, _min_idx_L.unsqueeze(-1), 1.0)
    perp_L = torch.cross(bwd_L_norm, ref_L, dim=-1)
    perp_L = safe_normalize(perp_L, dim=-1)
    ca_next[:, -1] = ca_coords[:, -1] - bwd_L + 0.3 * safe_norm(bwd_L, dim=-1, keepdim=True) * perp_L

    v1 = ca_next - ca_coords  # forward
    v2 = ca_prev - ca_coords  # backward

    e1 = safe_normalize(v1, dim=-1)

    dot = torch.sum(e1 * v2, dim=-1, keepdim=True)
    u2 = v2 - dot * e1
    e2 = safe_normalize(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # (B, L, 3, 3)  columns = basis

    return R, ca_coords


def local_to_global(
    R: torch.Tensor,
    t: torch.Tensor,
    local_coords: torch.Tensor,
) -> torch.Tensor:
    """Project local offsets into global coordinates.

    Parameters
    ----------
    R : (B, L, 3, 3)          -- rotation (column-major basis)
    t : (B, L, 3)             -- origin (Cα)
    local_coords : (B, L, A, 3)  -- local positions for A atoms

    Returns
    -------
    global_coords : (B, L, A, 3)
    """
    # Column-major convention: R columns = basis vectors.
    # Multiplying R @ local gives global coords (inverse of global_to_local).
    rotated_coords = torch.einsum('b l i j, b l a j -> b l a i', R, local_coords)
    global_coords = rotated_coords + t.unsqueeze(-2)
    return global_coords


# ---------------------------------------------------------------------------
# Cα pseudo-torsion (useful as a geometric feature for the network)
# ---------------------------------------------------------------------------
def compute_ca_pseudo_torsion(ca_coords: torch.Tensor) -> torch.Tensor:
    """Compute Cα pseudo-torsion tau_i = Cα_{i-1}-Cα_i-Cα_{i+1}-Cα_{i+2}.

    This angle is strongly correlated with the peptide plane orientation
    between residues i and i+1.

    Parameters
    ----------
    ca_coords : Tensor (B, L, 3)

    Returns
    -------
    tau : Tensor (B, L) in [-pi, pi].  Terminal residues = 0.
    """
    B, L, _ = ca_coords.shape
    device = ca_coords.device
    dtype = ca_coords.dtype

    tau = torch.zeros(B, L, device=device, dtype=dtype)
    if L < 4:
        return tau

    a = ca_coords[:, :-3, :]
    b = ca_coords[:, 1:-2, :]
    c = ca_coords[:, 2:-1, :]
    d = ca_coords[:, 3:, :]

    tau[:, 1:-2] = extract_dihedral(a, b, c, d)
    return tau


def compute_ca_virtual_bond_angle(ca_coords: torch.Tensor) -> torch.Tensor:
    """Compute Cα virtual bond angle θ_i = angle(Cα_{i-1}, Cα_i, Cα_{i+1}).

    This is a key local geometric feature that constrains backbone
    conformation.  Together with the pseudo-torsion τ, it provides a
    complete local description of the Cα trace geometry.

    Parameters
    ----------
    ca_coords : Tensor (B, L, 3)

    Returns
    -------
    theta : Tensor (B, L) in [0, π].  Terminal residues = π (straight).
    """
    B, L, _ = ca_coords.shape
    device = ca_coords.device
    dtype = ca_coords.dtype

    theta = torch.full((B, L), math.pi, device=device, dtype=dtype)
    if L < 3:
        return theta

    v1 = ca_coords[:, :-2, :] - ca_coords[:, 1:-1, :]  # Cα_{i-1} - Cα_i
    v2 = ca_coords[:, 2:, :] - ca_coords[:, 1:-1, :]   # Cα_{i+1} - Cα_i

    cos_angle = torch.sum(
        safe_normalize(v1, dim=-1) * safe_normalize(v2, dim=-1), dim=-1
    ).clamp(-1 + 1e-6, 1 - 1e-6)
    theta[:, 1:-1] = torch.acos(cos_angle)
    return theta


# ---------------------------------------------------------------------------
# Core NeRF placement (for sidechain extension)
# ---------------------------------------------------------------------------
def nerf_build(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length: Union[float, torch.Tensor],
    theta: Union[float, torch.Tensor],
    chi: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Place atom D given three reference atoms A, B, C and internal coords.

    The convention follows Parsons et al. (2005):
        * D is bonded to C with bond length  ``length``.
        * The angle B-C-D equals ``theta`` (interior angle, radians).
        * The dihedral A-B-C-D equals ``chi`` (radians).

    Parameters
    ----------
    a, b, c : Tensor (..., 3)
        Preceding reference atoms.
    length : float or Tensor (...)
        Bond length C → D.
    theta : float or Tensor (...)
        Bond angle B-C-D.
    chi : float or Tensor (...)
        Dihedral A-B-C-D.

    Returns
    -------
    d : Tensor (..., 3)
        Cartesian position of D.
    """
    # 1. Build an orthonormal frame anchored at C
    bc = safe_normalize(c - b, dim=-1)       # unit vector B→C
    ab = safe_normalize(b - a, dim=-1)       # unit vector A→B
    n = safe_normalize(torch.cross(ab, bc, dim=-1), dim=-1)  # plane normal
    m = torch.cross(n, bc, dim=-1)           # completes right-hand frame

    # 2. Broadcast scalar / 0-d tensor inputs
    def _expand(v: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=c.dtype, device=c.device)
        return v.unsqueeze(-1) if v.dim() < c.dim() else v

    length = _expand(length)
    theta = _expand(theta)
    chi = _expand(chi)

    # 3. Local coordinates of D (NeRF convention: x along -BC)
    dx = -length * torch.cos(theta)
    dy = length * torch.sin(theta) * torch.cos(chi)
    dz = -length * torch.sin(theta) * torch.sin(chi)

    # 4. Transform to global frame
    d = c + dx * bc + dy * m + dz * n
    return d


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def place_cbeta(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Place Cβ using ideal sp3 tetrahedral geometry with correct L-chirality.

    Parameters
    ----------
    n, ca, c : Tensor (..., 3)
        Backbone N, Cα, C atoms.

    Returns
    -------
    cb : Tensor (..., 3)
    """
    return nerf_build(
        a=c,
        b=n,
        c=ca,
        length=BOND_CA_CB,
        theta=ANGLE_N_CA_CB,
        chi=DIHEDRAL_CB,
    )


# ---------------------------------------------------------------------------
# Dihedral extraction
# ---------------------------------------------------------------------------
def extract_dihedral(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Differentiably compute the dihedral angle A-B-C-D (radians, ±π).

    Uses the Blondel & Bhatt stable atan2 formulation to avoid numerical
    issues at 0° and 180°.

    Parameters
    ----------
    a, b, c, d : Tensor (..., 3)

    Returns
    -------
    angle : Tensor (...)  in [-π, π]
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    b2_hat = safe_normalize(b2, dim=-1)

    # Remove component of b1/b3 along b2
    v = b1 - torch.sum(b1 * b2_hat, dim=-1, keepdim=True) * b2_hat
    w = b3 - torch.sum(b3 * b2_hat, dim=-1, keepdim=True) * b2_hat

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b2_hat, v, dim=-1) * w, dim=-1)

    x = x + 1e-7
    y = y + 1e-7

    return torch.atan2(y, x)


# ---------------------------------------------------------------------------
# Full backbone + sidechain builder from a Cα trace + torsion angles
# ---------------------------------------------------------------------------

def _refine_peptide_bonds(
    n_global: torch.Tensor,
    c_global: torch.Tensor,
    ca_global: torch.Tensor,
    num_iters: int = 3,
    target_length: float = BOND_C_N,
    max_shift: float = 0.25,
    target_c_n_ca_angle: float = ANGLE_C_N_CA,
    angle_max_shift: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative differentiable peptide-bond refinement.

    Each iteration performs two stages:

    1. **Distance correction** — for each pair (C_i, N_{i+1}), shifts both
       atoms along the C→N direction by half the distance error to the
       ideal C-N bond length (1.329 Å), clamped per atom.
    2. **Angle correction** — adjusts N_{i+1} to bring the
       C_i-N_{i+1}-Cα_{i+1} angle toward the ideal 121.7°, clamped per
       atom.  This jointly constrains the peptide plane geometry.

    Parameters
    ----------
    n_global : (B, L, 3)
    c_global : (B, L, 3)
    ca_global : (B, L, 3)
    num_iters : int
    target_length : float
    max_shift : float
    target_c_n_ca_angle : float
    angle_max_shift : float

    Returns
    -------
    n_refined, c_refined : (B, L, 3)
    """
    if n_global.shape[1] < 2:
        return n_global, c_global

    n_refined = n_global.clone()
    c_refined = c_global.clone()

    for _it in range(num_iters):
        # ── Stage 1: Distance correction ──────────────────────────
        c_to_n = n_refined[:, 1:] - c_refined[:, :-1]                  # (B, L-1, 3)
        dist = safe_norm(c_to_n, dim=-1, keepdim=True)
        direction = c_to_n / dist

        error = dist - target_length                                 # (B, L-1, 1)
        half_correction = 0.5 * error * direction                    # (B, L-1, 3)

        corr_mag = safe_norm(half_correction, dim=-1, keepdim=True)
        scale = torch.clamp(corr_mag, max=max_shift) / corr_mag
        half_correction = half_correction * scale

        c_refined[:, :-1] = c_refined[:, :-1] + half_correction
        n_refined[:, 1:]  = n_refined[:, 1:]  - half_correction

        # ── Stage 2: C_i-N_{i+1}-Cα_{i+1} angle correction ───────
        n_to_c = c_refined[:, :-1] - n_refined[:, 1:]                # (B, L-1, 3)
        n_to_ca = ca_global[:, 1:] - n_refined[:, 1:]                # (B, L-1, 3)

        n_to_c_norm = safe_normalize(n_to_c, dim=-1)
        n_to_ca_norm = safe_normalize(n_to_ca, dim=-1)

        cos_angle = torch.sum(n_to_c_norm * n_to_ca_norm, dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        current_angle = torch.acos(cos_angle)  # (B, L-1, 1)
        angle_error = current_angle - target_c_n_ca_angle

        bisector = safe_normalize(n_to_c_norm + n_to_ca_norm, dim=-1)
        perp = n_to_c_norm - torch.sum(n_to_c_norm * bisector, dim=-1, keepdim=True) * bisector
        perp = safe_normalize(perp, dim=-1)

        cn_len = safe_norm(n_to_c, dim=-1, keepdim=True)
        angle_shift = angle_error * cn_len * perp  # (B, L-1, 3)

        a_mag = safe_norm(angle_shift, dim=-1, keepdim=True)
        a_scale = torch.clamp(a_mag, max=angle_max_shift) / a_mag
        angle_shift = angle_shift * a_scale

        n_refined[:, 1:] = n_refined[:, 1:] + angle_shift

    return n_refined, c_refined


def build_structure_from_angles(
    coords_ca: torch.Tensor,
    angles_rad: torch.Tensor,
    seq: Optional[torch.Tensor] = None,
    ca_frames: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    backbone_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reconstruct a 23-atom representation from Cα positions and torsions.

    The atom ordering per residue (23 slots) is::

        0=N  1=CA  2=C  3=O  4=CB  5=CG  6=CG1  7=CG2
        8=CD  9=CD1  10=CD2  11=CE  12=CE1  13=CZ
        14=CE2  15=NE2  16=CE3  17=CZ3  18=CH2
        19=OH  20=NH1  21=NH2  22=OXT

    **Backbone N/C placement** uses direct regression offsets predicted by
    the network in the Cα pseudo-frame.  When ``backbone_offsets`` is
    ``None``, ideal geometry (IDEAL_N_LOCAL / IDEAL_C_LOCAL) is used.

    **Carbonyl O** is placed deterministically via the peptide-plane
    coplanarity constraint: Cα_i, C_i, O_i, and N_{i+1} are strictly
    coplanar, so O_i is the NeRF solution with chi = π using N_{i+1}
    as the dihedral reference.  The C-terminal residue (no N_{i+1})
    falls back to a ψ-based NeRF placement.

    **Cβ** is placed from the refined N, Cα, C triangle.

    **Sidechain** is extended using standard NeRF from the placed backbone.

    Parameters
    ----------
    coords_ca : Tensor (B, L, 3)
        Fixed Cα trace.
    angles_rad : Tensor (B, L, 7)
        Torsion angles: [ω, φ, ψ, χ1, χ2, χ3, χ4] in radians.
    seq : Tensor (B, L), optional
        Integer-coded amino acid sequence (Gly=6 masks Cβ+).
    ca_frames : tuple of (R, t), optional
        Pre-computed Cα pseudo-frames ``(R: (B,L,3,3), t: (B,L,3))``.
        When ``None`` (default), frames are built internally via
        ``construct_ca_pseudo_frames``.
    backbone_offsets : Tensor (B, L, 6), optional
        Direct backbone atom offsets in the Cα pseudo-frame:
        [0:3] N local offset, [3:6] C local offset.
        When ``None``, ideal geometry is used (useful for relaxation).

    Returns
    -------
    all_atom : Tensor (B, L, 23, 3)
    """
    B, L, _ = coords_ca.shape
    device = coords_ca.device
    dtype = coords_ca.dtype

    all_atom = torch.zeros(B, L, 23, 3, device=device, dtype=dtype)
    all_atom[:, :, 1, :] = coords_ca  # slot 1 = Cα

    # Unpack angles
    omega = angles_rad[:, :, 0]
    phi = angles_rad[:, :, 1]            # (B, L)
    psi = angles_rad[:, :, 2]            # (B, L)
    chi1 = angles_rad[:, :, 3]
    chi2 = angles_rad[:, :, 4]
    chi3 = angles_rad[:, :, 5]
    chi4 = angles_rad[:, :, 6]

    # -- Step 1: Cα pseudo-frames ------------------------------------------
    if ca_frames is not None:
        R, t = ca_frames
    else:
        R, t = construct_ca_pseudo_frames(coords_ca)  # R: (B,L,3,3), t: (B,L,3)

    # -- Steps 2–3: Backbone N/C placement ---------------------------------
    if backbone_offsets is not None:
        # ── Direct regression mode ──────────────────────────────────
        # The network predicts full 3D offsets for N and C in the Cα
        # pseudo-frame.
        local_n = backbone_offsets[:, :, :3]   # (B, L, 3)
        local_c = backbone_offsets[:, :, 3:6]  # (B, L, 3)
    else:
        # ── Ideal geometry fallback (used by relaxer) ───────────────
        local_n = IDEAL_N_LOCAL.to(device=device, dtype=dtype).expand(B, L, 3)  # (B, L, 3)
        local_c = IDEAL_C_LOCAL.to(device=device, dtype=dtype).expand(B, L, 3)

    # -- Step 4: Project to global ------------------------------------------
    n_global = local_to_global(R, t, local_n.unsqueeze(-2)).squeeze(-2)
    c_global = local_to_global(R, t, local_c.unsqueeze(-2)).squeeze(-2)

    # NOTE: The previous omega-based N override was removed.  It used wrong
    # NeRF parameters (BOND_C_N=1.329 Å applied as an N-Cα distance instead
    # of the correct ~1.37 Å from IDEAL_N_LOCAL, plus incorrect bond angle
    # and dihedral convention), causing ~0.1-0.3 Å systematic N-atom error.
    # The pseudo-frame placement above is consistent with the training
    # targets (pseudo-frame rotation angles) and produces correct N-Cα
    # geometry.

    # -- Step 4b: Peptide bond refinement -----------------------------------
    n_global, c_global = _refine_peptide_bonds(n_global, c_global, coords_ca)

    all_atom[:, :, 0, :] = n_global
    all_atom[:, :, 2, :] = c_global

    # -- Step 5: Carbonyl O ------------------------------------------------
    # Deterministic coplanar placement: Cα_i, C_i, O_i, and N_{i+1} are
    # strictly coplanar in the peptide plane.  O_i is trans to N_{i+1}
    # across the Cα_i–C_i bond, so chi = π gives the exact coplanar
    # position.  The last residue has no N_{i+1} and falls back to ψ-based
    # NeRF.

    B_sz, L_sz = coords_ca.shape[:2]

    if L_sz > 1:
        # Residues 0 .. L-2: dihedral ref = N(i+1), chi = π (coplanar)
        o_main = nerf_build(
            a=n_global[:, 1:, :],
            b=coords_ca[:, :-1, :],
            c=c_global[:, :-1, :],
            length=BOND_C_O,
            theta=ANGLE_CA_C_O,
            chi=math.pi,
        )
        # Last residue: fall back to ψ-based NeRF
        o_last = nerf_build(
            a=n_global[:, -1:, :],
            b=coords_ca[:, -1:, :],
            c=c_global[:, -1:, :],
            length=BOND_C_O,
            theta=ANGLE_CA_C_O,
            chi=-psi[:, -1:] - math.pi,
        )
        o_coords = torch.cat([o_main, o_last], dim=1)        # (B, L, 3)
    else:
        # Single residue: only ψ-based available
        o_coords = nerf_build(
            a=n_global, b=coords_ca, c=c_global,
            length=BOND_C_O, theta=ANGLE_CA_C_O,
            chi=-psi - math.pi,
        )

    all_atom[:, :, 3, :] = o_coords

    # -- Step 6: Cβ from refined backbone (N, Cα, C) -----------------------
    cb_coords = place_cbeta(n_global, coords_ca, c_global)
    all_atom[:, :, 4, :] = cb_coords

    # -- Step 7: Sidechain extension via NeRF from real backbone ------------
    # Look up per-residue bond lengths and angles for each chi step.
    # This replaces the old generic constants (BOND_CB_CG etc.) with
    # residue-specific values that correctly model C-S, C-O, C_sp2,
    # and aromatic bond lengths and angles.
    if seq is not None:
        seq_idx = seq.clamp(0, 21)
        bl = _get_tensor(_SC_BOND_LENGTHS, device, dtype)
        ba = _get_tensor(_SC_BOND_ANGLES, device, dtype)
        bl_chi1 = bl[seq_idx, 0]   # (B, L)
        ba_chi1 = ba[seq_idx, 0]
        bl_chi2 = bl[seq_idx, 1]
        ba_chi2 = ba[seq_idx, 1]
        bl_chi3 = bl[seq_idx, 2]
        ba_chi3 = ba[seq_idx, 2]
        bl_chi4 = bl[seq_idx, 3]
        ba_chi4 = ba[seq_idx, 3]
    else:
        bl_chi1, ba_chi1 = BOND_CB_CG, ANGLE_CA_CB_CG
        bl_chi2, ba_chi2 = BOND_CG_CD, ANGLE_CB_CG_CD
        bl_chi3, ba_chi3 = BOND_CD_CE, ANGLE_CG_CD_CE
        bl_chi4, ba_chi4 = BOND_CE_CZ, ANGLE_CD_CE_CZ

    # Chi-1 atom: CG, CG1, OG1, SG, or OG depending on residue type.
    sc1 = nerf_build(
        n_global, coords_ca, cb_coords,
        length=bl_chi1, theta=ba_chi1, chi=chi1,
    )

    if seq is not None:
        # VAL(18) and ILE(8): chi1 → slot 6 (CG1); slot 5 empty.
        # All others:          chi1 → slot 5 (CG/OG1/SG/OG); slot 6 empty.
        cg1_mask = ((seq == 8) | (seq == 18)).unsqueeze(-1)  # (B, L, 1)
        all_atom[:, :, 5, :] = torch.where(cg1_mask, torch.zeros_like(sc1), sc1)
        all_atom[:, :, 6, :] = torch.where(cg1_mask, sc1, torch.zeros_like(sc1))

        # CG2 (slot 7) for branched residues (VAL, ILE, THR).
        # CG2 branches off CB at a ~120° dihedral offset from the chi1 atom.
        cg2_bl = _get_tensor(_CG2_BOND_LENGTHS, device, dtype)[seq_idx]
        cg2_ba = _get_tensor(_CG2_BOND_ANGLES, device, dtype)[seq_idx]
        cg2_off = _get_tensor(_CG2_DIHEDRAL_OFFSETS, device, dtype)[seq_idx]

        cg2 = nerf_build(
            n_global, coords_ca, cb_coords,
            length=cg2_bl, theta=cg2_ba, chi=chi1 + cg2_off,
        )
        branched_mask = ((seq == 8) | (seq == 17) | (seq == 18)).unsqueeze(-1)
        all_atom[:, :, 7, :] = torch.where(branched_mask, cg2, torch.zeros_like(cg2))
    else:
        all_atom[:, :, 5, :] = sc1

    # -- Aromatic ring builder (PHE, HIS, TRP, TYR) -----------------------
    # Aromatic ring atoms form a planar rigid body that rotates as a unit
    # around the CB→CG axis (chi2).  We transform ideal ring coordinates
    # from the chi2 frame into global coordinates rather than walking a
    # linear NeRF chain, which cannot close a ring.
    ring_global = None
    if seq is not None:
        aro_mask = _get_tensor(_IS_AROMATIC, device)[seq_idx]
        if aro_mask.any():
            ring_local = _get_tensor(_RING_COORDS, device, dtype)[seq_idx]

            # NeRF frame at CG (sc1) with CA, CB as the reference atoms.
            cg_cb = sc1 - cb_coords
            bc = safe_normalize(cg_cb, dim=-1)
            n_hat = safe_normalize(
                torch.cross(coords_ca - cb_coords, cg_cb, dim=-1), dim=-1,
            )
            m_hat = torch.cross(n_hat, bc, dim=-1)

            # Rotate local (y, z) by chi2 around the BC axis.
            # The builder's local frame (x=CB→CG, y=m_hat, z=n_hat) has a
            # sign + π phase offset relative to the standard chi2 dihedral
            # convention (CA-CB-CG-CD1).  Applying chi2_ring = -(chi2 + π)
            # corrects both so the measured dihedral matches the input.
            rx = ring_local[..., 0]                       # (B, L, 10)
            ry = ring_local[..., 1]
            rz = ring_local[..., 2]
            chi2_ring = -chi2 - math.pi
            c2 = torch.cos(chi2_ring).unsqueeze(-1)       # (B, L, 1)
            s2 = torch.sin(chi2_ring).unsqueeze(-1)
            ry_r = ry * c2 - rz * s2
            rz_r = ry * s2 + rz * c2

            # Global positions: CG + x·bc + y’·m + z’·n
            ring_global = (
                sc1.unsqueeze(2)
                + rx.unsqueeze(-1) * bc.unsqueeze(2)
                + ry_r.unsqueeze(-1) * m_hat.unsqueeze(2)
                + rz_r.unsqueeze(-1) * n_hat.unsqueeze(2)
            )  # (B, L, 10, 3) → indices [slot9, slot10, slot12, slot13, slot14-19]
            # Zero out positions for atom slots that don't exist for a
            # given residue type (e.g. HIS has no slot-13 atom).
            ring_valid = safe_norm(ring_local, dim=-1, keepdim=True) > 0.01
            ring_global = ring_global * ring_valid.to(dtype=dtype)
    # -- Chi2+ linear NeRF chain (used by non-aromatic residues) -----------
    sc2 = nerf_build(
        coords_ca, cb_coords, sc1,
        length=bl_chi2, theta=ba_chi2, chi=chi2,
    )
    ce_coords = nerf_build(
        cb_coords, sc1, sc2,
        length=bl_chi3, theta=ba_chi3, chi=chi3,
    )
    cz_coords = nerf_build(
        sc1, sc2, ce_coords,
        length=bl_chi4, theta=ba_chi4, chi=chi4,
    )

    # -- Slot 10 branch: CD2 (LEU), OD2 (ASP), ND2 (ASN) ------------------
    # These atoms branch off CG (sc1) in the chi2 frame, offset from the
    # primary chi2 atom by ~120-180° in dihedral.
    slot10_coords = None
    if seq is not None:
        has10 = _get_tensor(_HAS_SLOT10, device)[seq_idx]
        if has10.any():
            s10_bl = _get_tensor(_SLOT10_BL, device, dtype)[seq_idx]
            s10_ba = _get_tensor(_SLOT10_BA, device, dtype)[seq_idx]
            s10_do = _get_tensor(_SLOT10_DOFF, device, dtype)[seq_idx]
            slot10_coords = nerf_build(
                coords_ca, cb_coords, sc1,
                length=s10_bl, theta=s10_ba, chi=chi2 + s10_do,
            )

    # -- Slot 12 branch: OE2 (GLU), NE2 (GLN) -----------------------------
    # These atoms branch off CD (sc2) in the chi3 frame, offset from the
    # primary chi3 atom by ~180° in dihedral.
    slot12_coords = None
    if seq is not None:
        has12 = _get_tensor(_HAS_SLOT12, device)[seq_idx]   # (B, L)
        if has12.any():
            s12_bl = _get_tensor(_SLOT12_BL, device, dtype)[seq_idx]
            s12_ba = _get_tensor(_SLOT12_BA, device, dtype)[seq_idx]
            s12_do = _get_tensor(_SLOT12_DOFF, device, dtype)[seq_idx]
            slot12_coords = nerf_build(
                cb_coords, sc1, sc2,
                length=s12_bl, theta=s12_ba, chi=chi3 + s12_do,
            )

    # -- Assign atoms to slots 8-13 ----------------------------------------
    zero = torch.zeros_like(sc2)

    if seq is not None:
        aro_3d = aro_mask.unsqueeze(-1) if ring_global is not None else None

        # Slot 8 (CD/SD): filled only for non-aromatic chi2→slot8 residues.
        s8_mask = _get_tensor(_CHI2_TO_SLOT8, device)[seq_idx].unsqueeze(-1)
        slot8_val = torch.where(s8_mask, sc2, zero)
        if aro_3d is not None:
            slot8_val = torch.where(aro_3d, zero, slot8_val)
        all_atom[:, :, 8, :] = slot8_val

        # Slot 9 (CD1/OD1/ND1): filled for non-aromatic chi2→slot9 residues,
        # or ring atom [0] for aromatics.
        slot9_val = torch.where(s8_mask, zero, sc2)
        if aro_3d is not None:
            slot9_val = torch.where(aro_3d, ring_global[:, :, 0, :], slot9_val)
        all_atom[:, :, 9, :] = slot9_val

        # Slot 10 (CD2/OD2/ND2): branching atom or ring atom [1].
        slot10_val = zero
        if slot10_coords is not None:
            slot10_val = torch.where(has10.unsqueeze(-1), slot10_coords, zero)
        if aro_3d is not None:
            slot10_val = torch.where(aro_3d, ring_global[:, :, 1, :], slot10_val)
        all_atom[:, :, 10, :] = slot10_val

        # Slot 11 (CE/OE1/NE): linear chain atom for non-aromatics.
        slot11_val = ce_coords
        if aro_3d is not None:
            slot11_val = torch.where(aro_3d, zero, slot11_val)
        all_atom[:, :, 11, :] = slot11_val

        # Slot 12 (CE1/NE1 or OE2/NE2): ring atom [2] for aromatics,
        # branching atom for GLU/GLN, zero otherwise.
        # NOTE: GLN's branching atom is NE2 which maps to slot 15 in the
        # 23-atom layout (matching the parser), not slot 12.  GLU's OE2
        # stays at slot 12.
        is_gln = (seq == 14).unsqueeze(-1)  # (B, L, 1)
        slot12_val = zero
        if slot12_coords is not None:
            # GLU OE2 → slot 12;  GLN NE2 → slot 15 (handled below)
            glu_only = has12.unsqueeze(-1) & ~is_gln
            slot12_val = torch.where(glu_only, slot12_coords, zero)
        if aro_3d is not None:
            slot12_val = torch.where(aro_3d, ring_global[:, :, 2, :], slot12_val)
        all_atom[:, :, 12, :] = slot12_val

        # Slot 13 (CZ/NZ/OH): linear chain for non-aromatics, ring atom [3]
        # for aromatics.
        slot13_val = cz_coords
        if aro_3d is not None:
            slot13_val = torch.where(aro_3d, ring_global[:, :, 3, :], slot13_val)
        all_atom[:, :, 13, :] = slot13_val

        # Slots 14-19: additional ring atoms (CE2, NE2, CE3, CZ3, CH2, OH).
        # These exist only for aromatic residues.  ring_valid (from norm > 0.01
        # check) already zeroes non-applicable entries per residue type.
        if ring_global is not None and aro_3d is not None:
            _ring_slot_map = [(4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19)]
            for ring_idx, slot in _ring_slot_map:
                all_atom[:, :, slot, :] = torch.where(
                    aro_3d, ring_global[:, :, ring_idx, :], zero,
                )

        # GLN NE2 → slot 15 (the parser maps NE2 to slot 15 for both
        # HIS and GLN; the ring builder already handles HIS above).
        if slot12_coords is not None and is_gln.any():
            gln_ne2 = torch.where(is_gln, slot12_coords, zero)
            all_atom[:, :, 15, :] = all_atom[:, :, 15, :] + gln_ne2

        # Slots 20-21: NH1/NH2 for ARG (15) guanidinium.
        # The guanidinium group is planar sp2: NE-CZ-NH1 and NE-CZ-NH2
        # are each ~120° with dihedral offsets 0 and π (trans/cis to CD).
        is_arg = (seq == 15).unsqueeze(-1)  # (B, L, 1)
        if is_arg.any():
            nh1 = nerf_build(
                sc2, ce_coords, cz_coords,
                length=1.326, theta=math.radians(120.0), chi=0.0,
            )
            nh2 = nerf_build(
                sc2, ce_coords, cz_coords,
                length=1.326, theta=math.radians(120.0), chi=math.pi,
            )
            all_atom[:, :, 20, :] = torch.where(is_arg, nh1, zero)
            all_atom[:, :, 21, :] = torch.where(is_arg, nh2, zero)
    else:
        all_atom[:, :, 8, :] = sc2
        all_atom[:, :, 9, :] = sc2
        all_atom[:, :, 11, :] = ce_coords
        all_atom[:, :, 13, :] = cz_coords

    # -- Step 8: Glycine mask (no Cb or sidechain) -------------------------
    if seq is not None:
        gly_mask = (seq == 6).unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        sc_kill = torch.ones(B, L, 23, 3, device=device, dtype=dtype)
        sc_kill[:, :, 4:, :] = 0.0
        all_atom = all_atom * torch.where(gly_mask, sc_kill, torch.ones_like(sc_kill))

    # -- Step 9: Zero out phantom atoms ------------------------------------
    if seq is not None:
        ideal_mask = _get_tensor(IDEAL_ATOM_MASK, device)[seq_idx]
        all_atom = all_atom * ideal_mask.unsqueeze(-1).to(dtype)

    return all_atom



# ---------------------------------------------------------------------------
# Ground-truth target correction for the pseudo-frame builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def recompute_backbone_sincos(
    ca_coords: torch.Tensor,       # (L, 3)
    all_atom_coords: torch.Tensor,  # (L, A, 3)
    atom_mask: torch.Tensor,        # (L, A) bool
    latent_target: torch.Tensor,    # (L, 14) float32
) -> torch.Tensor:
    """Recompute phi/psi sin/cos targets to match the Cα pseudo-frame builder.

    The data pipeline computes ground-truth phi and psi as standard
    Ramachandran dihedrals.  However, ``build_structure_from_angles``
    interprets phi and psi as rotation angles around the local e1 axis
    of the Cα pseudo-frame, which is a different quantity.

    This function projects the ground-truth N and C atoms into each
    residue's local pseudo-frame and solves for the rotation angle around
    e1 that maps the ideal offset to the actual local position.  It then
    replaces slots 2-5 (phi_sin, phi_cos, psi_sin, psi_cos) in the
    latent target while leaving omega (0-1) and chi angles (6-13)
    unchanged.

    Invariant: keep this runtime torch implementation aligned with
    ``data.parse._np_recompute_backbone_sincos``. They are intentionally
    separate because preprocessing and training/inference run in different
    execution environments.

    Call this once per chain at dataset-load time.

    .. note::

       Although the backbone head predicts direct 3D N/C offsets (not
       angles), the corrected phi/psi values are still consumed by
       ``_angles_from_sc_sincos`` during training-time structure builds
       for physics losses.  They are **not** used by the flow matcher,
       which only sees chi-angle slots 6-13.

    Parameters
    ----------
    ca_coords : Tensor (L, 3)
    all_atom_coords : Tensor (L, NUM_ATOMS, 3)
    atom_mask : Tensor (L, NUM_ATOMS) bool
    latent_target : Tensor (L, 14)

    Returns
    -------
    corrected : Tensor (L, 14)
    """
    L = ca_coords.shape[0]
    out = latent_target.clone()

    if L < 2:
        return out

    R, _t = construct_ca_pseudo_frames(ca_coords.unsqueeze(0))
    R = R.squeeze(0)  # (L, 3, 3)

    ideal_n = IDEAL_N_LOCAL.to(dtype=ca_coords.dtype, device=ca_coords.device)
    ideal_c = IDEAL_C_LOCAL.to(dtype=ca_coords.dtype, device=ca_coords.device)

    n_gt = all_atom_coords[:, 0, :]  # (L, 3)
    c_gt = all_atom_coords[:, 2, :]  # (L, 3)

    local_n = torch.einsum("lij, lj -> li", R.transpose(-1, -2), n_gt - ca_coords)
    local_c = torch.einsum("lij, lj -> li", R.transpose(-1, -2), c_gt - ca_coords)

    def _solve_angle(
        local_gt: torch.Tensor,
        ideal: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y_gt = local_gt[:, 1]
        z_gt = local_gt[:, 2]
        y0, z0 = ideal[1], ideal[2]
        sin_a = z_gt * y0 - y_gt * z0
        cos_a = y_gt * y0 + z_gt * z0
        angle = torch.atan2(sin_a, cos_a)
        return torch.where(mask, angle, torch.zeros_like(angle))

    phi = _solve_angle(local_n, ideal_n, atom_mask[:, 0])
    psi = _solve_angle(local_c, ideal_c, atom_mask[:, 2])

    out[:, 2] = torch.sin(phi)
    out[:, 3] = torch.cos(phi)
    out[:, 4] = torch.sin(psi)
    out[:, 5] = torch.cos(psi)

    return out