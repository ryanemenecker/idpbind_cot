"""
constants.py

Single source of truth for shared constants used across the
generative_approach module.

Centralises residue identity maps, atom-slot definitions, backbone and
sidechain ideal geometry, symmetry definitions, and other constants that
were previously scattered across parse.py, nerf.py, loss.py, network.py,
and train.py.

Every other module should import from here rather than re-defining values.
parse.py re-exports the constants it previously owned for backward compat.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


# ===================================================================
# 1. Residue identity maps
# ===================================================================
RESIDUE_MAP: Dict[str, int] = {
    "ALA": 1, "CYS": 2, "ASP": 3, "GLU": 4, "PHE": 5,
    "GLY": 6, "HIS": 7, "ILE": 8, "LYS": 9, "LEU": 10,
    "MET": 11, "ASN": 12, "PRO": 13, "GLN": 14, "ARG": 15,
    "SER": 16, "THR": 17, "VAL": 18, "TRP": 19, "TYR": 20,
    "UNK": 21,
}

INT_TO_RES: Dict[int, str] = {v: k for k, v in RESIDUE_MAP.items()}


# ===================================================================
# 2. 23-atom representation
# ===================================================================
# Slot :  0   1   2   3   4   5    6    7    8    9   10   11   12   13
# Name :  N  CA   C   O  CB  CG  CG1  CG2  CD  CD1  CD2  CE  CE1  CZ
#
# Slot : 14   15   16   17   18   19   20   21   22
# Name : CE2  NE2  CE3  CZ3  CH2  OH  NH1  NH2  OXT
NUM_ATOMS: int = 23

ATOM_NAMES: List[str] = [
    "N", "CA", "C", "O", "CB",
    "CG", "CG1", "CG2", "CD", "CD1",
    "CD2", "CE", "CE1", "CZ",
    "CE2", "NE2", "CE3", "CZ3",
    "CH2", "OH", "NH1", "NH2", "OXT",
]

ATOM_14_NAMES: List[str] = ATOM_NAMES[:14]

ATOM_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ATOM_NAMES)}

# Atom names that don't appear in the canonical 23-slot list but occupy
# a specific slot for certain residue types.
_EXTRA_ATOM_MAP: Dict[str, int] = {
    "OG":  5,    # Ser
    "OG1": 5,    # Thr (alias for CG slot, but correct: Thr OG1 goes to slot 5)
    "SG":  5,    # Cys
    "SD":  8,    # Met
    "OD1": 9,    # Asp
    "OD2": 10,   # Asp
    "OE1": 11,   # Glu
    "OE2": 12,   # Glu  (shares CE1 slot — kept for backward compat of old datasets;
                  #        new datasets should map to a unique branch slot if desired)
    "ND1": 9,    # His
    "ND2": 10,   # Asn
    "NE":  11,   # Arg
    "NE1": 12,   # Trp
    "NE2": 15,   # His / Gln  → slot 15
    "NZ":  13,   # Lys
    "NH1": 20,   # Arg        → slot 20
    "NH2": 21,   # Arg        → slot 21
    "OH":  19,   # Tyr        → slot 19
    "CE2": 14,   # Phe/Tyr/Trp → slot 14
    "CZ2": 13,   # Trp  (remains in CZ slot – same ring body as old mapping)
    "CZ3": 17,   # Trp        → slot 17
    "CH2": 18,   # Trp        → slot 18
    "CE3": 16,   # Trp        → slot 16
    "OXT": 22,   # C-terminal → slot 22
}


# ===================================================================
# 3. Per-residue valid atom slots and naming
# ===================================================================
# Maps: Residue Name -> {Slot Index -> Correct Atom Name}
_RESIDUE_VALID_SLOTS: Dict[str, Dict[int, str]] = {
    "ALA": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB"},
    "CYS": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "SG"},
    "ASP": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "OD1", 10: "OD2"},
    "GLU": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "CD", 11: "OE1", 12: "OE2"},
    "PHE": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "CD1", 10: "CD2", 12: "CE1", 13: "CZ", 14: "CE2"},
    "GLY": {0: "N", 1: "CA", 2: "C", 3: "O"},
    "HIS": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "ND1", 10: "CD2", 12: "CE1", 15: "NE2"},
    "ILE": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 6: "CG1", 7: "CG2", 9: "CD1"},
    "LYS": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "CD", 11: "CE", 13: "NZ"},
    "LEU": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "CD1", 10: "CD2"},
    "MET": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "SD", 11: "CE"},
    "ASN": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "OD1", 10: "ND2"},
    "PRO": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "CD"},
    "GLN": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "CD", 11: "OE1", 15: "NE2"},
    "ARG": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 8: "CD", 11: "NE", 13: "CZ", 20: "NH1", 21: "NH2"},
    "SER": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "OG"},
    "THR": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "OG1", 7: "CG2"},
    "VAL": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 6: "CG1", 7: "CG2"},
    "TRP": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "CD1", 10: "CD2", 12: "NE1", 13: "CZ2", 14: "CE2", 16: "CE3", 17: "CZ3", 18: "CH2"},
    "TYR": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB", 5: "CG", 9: "CD1", 10: "CD2", 12: "CE1", 13: "CZ", 14: "CE2", 19: "OH"},
    "UNK": {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB"},
}

RESIDUE_ATOM_NAMES: Dict[str, Dict[int, str]] = dict(_RESIDUE_VALID_SLOTS)

# Boolean mask (22 residue types × 23 atom slots) indicating which atom
# slots are occupied for each residue type.
IDEAL_ATOM_MASK = torch.zeros(22, NUM_ATOMS, dtype=torch.bool)
for _res_name, _slot_dict in _RESIDUE_VALID_SLOTS.items():
    _res_idx = RESIDUE_MAP.get(_res_name)
    if _res_idx is not None:
        for _slot_idx in _slot_dict.keys():
            IDEAL_ATOM_MASK[_res_idx, _slot_idx] = True
# PAD (0) has no atoms
IDEAL_ATOM_MASK[0, :] = False
del _res_name, _slot_dict, _res_idx, _slot_idx


# ===================================================================
# 4. Chi-angle atom definitions
# ===================================================================
# Maps three-letter residue code → list of (A, B, C, D) slot-index tuples
# defining each chi dihedral angle.
_CHI_ATOMS: Dict[str, List[Tuple[int, ...]]] = {
    "ALA": [],
    "GLY": [],
    "VAL": [(0, 1, 4, 6)],
    "LEU": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "ILE": [(0, 1, 4, 6), (1, 4, 6, 9)],
    "PRO": [(0, 1, 4, 5), (1, 4, 5, 8)],
    "PHE": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "TYR": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "TRP": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "SER": [(0, 1, 4, 5)],
    "THR": [(0, 1, 4, 5)],
    "CYS": [(0, 1, 4, 5)],
    "MET": [(0, 1, 4, 5), (1, 4, 5, 8), (4, 5, 8, 11)],
    "ASP": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "ASN": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "GLU": [(0, 1, 4, 5), (1, 4, 5, 8), (4, 5, 8, 11)],
    "GLN": [(0, 1, 4, 5), (1, 4, 5, 8), (4, 5, 8, 11)],
    "HIS": [(0, 1, 4, 5), (1, 4, 5, 9)],
    "LYS": [(0, 1, 4, 5), (1, 4, 5, 8), (4, 5, 8, 11), (5, 8, 11, 13)],
    "ARG": [(0, 1, 4, 5), (1, 4, 5, 8), (4, 5, 8, 11), (5, 8, 11, 13)],
}

# Per-residue chi-angle count (index = integer residue ID from RESIDUE_MAP).
# Index:  0  1    2    3    4    5    6    7    8    9   10
# Name: pad ALA  CYS  ASP  GLU  PHE  GLY  HIS  ILE  LYS  LEU
# Index: 11  12   13   14   15   16   17   18   19   20   21
# Name: MET  ASN  PRO  GLN  ARG  SER  THR  VAL  TRP  TYR  UNK
_NUM_CHI_PER_RES: List[int] = [
    0, 0, 1, 2, 3, 2, 0, 2, 2, 4, 2, 3, 2, 2, 3, 4, 1, 1, 1, 2, 2, 0,
]


# ===================================================================
# 5. Backbone ideal geometry (Engh-Huber)
# ===================================================================
# Individual float constants (used by nerf.py)
BOND_N_CA: float = 1.458   # N  → Cα
BOND_CA_C: float = 1.525   # Cα → C
BOND_C_N: float = 1.329    # C  → N (peptide bond)
BOND_C_O: float = 1.231    # C  = O (carbonyl)
BOND_CA_CB: float = 1.522  # Cα → Cβ

ANGLE_N_CA_C: float = math.radians(111.2)   # N–Cα–C
ANGLE_CA_C_N: float = math.radians(116.6)   # Cα–C–N(next)
ANGLE_CA_C_O: float = math.radians(120.8)   # Cα–C=O
ANGLE_C_N_CA: float = math.radians(121.7)   # C(prev)–N–Cα
ANGLE_N_CA_CB: float = math.radians(110.5)  # N–Cα–Cβ

# Dict form (used by loss.py)
IDEAL_BOND_LENGTHS: Dict[str, float] = {
    "N_CA": BOND_N_CA,
    "CA_C": BOND_CA_C,
    "C_N": BOND_C_N,
    "C_O": BOND_C_O,
    "CA_CB": BOND_CA_CB,
}

IDEAL_BOND_ANGLES_RAD: Dict[str, float] = {
    "N_CA_C": ANGLE_N_CA_C,
    "CA_C_N": ANGLE_CA_C_N,
    "CA_C_O": ANGLE_CA_C_O,
    "C_N_CA": ANGLE_C_N_CA,
    "N_CA_CB": ANGLE_N_CA_CB,
}


# ===================================================================
# 6. Ideal local-frame offsets for N and C atoms in the Cα pseudo-frame
# ===================================================================
# Derived empirically from 10,000 AF2 structures + 10,000 IDR simulations.
IDEAL_N_LOCAL = torch.tensor([-0.5939, 1.2149, -0.2696])
IDEAL_C_LOCAL = torch.tensor([1.41375, 0.01535, -0.2023])


# ===================================================================
# 7. Per-residue sidechain bond lengths (Å) and bond angles (radians)
# ===================================================================
# Generic fallback values for residues/chi steps that don't exist.
_GENERIC_BL: float = 1.522
_GENERIC_BA: float = math.radians(113.8)

# Rows = residue integer IDs (0=PAD … 21=UNK).
# Columns = [chi1_step, chi2_step, chi3_step, chi4_step].
_SC_BOND_LENGTHS = torch.tensor([
    # 0  PAD
    [_GENERIC_BL, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 1  ALA  (no sidechain beyond CB)
    [_GENERIC_BL, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 2  CYS  chi1: CB→SG  (C–S bond)
    [1.810, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 3  ASP  chi1: CB→CG  chi2: CG→OD1  (C_sp2=O)
    [1.519, 1.250, _GENERIC_BL, _GENERIC_BL],
    # 4  GLU  chi1: CB→CG  chi2: CG→CD  chi3: CD→OE1
    [1.522, 1.520, 1.252, _GENERIC_BL],
    # 5  PHE  chi1: CB→CG(sp2)  chi2: CG→CD1(aromatic)
    [1.505, 1.390, _GENERIC_BL, _GENERIC_BL],
    # 6  GLY  (no CB or sidechain)
    [_GENERIC_BL, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 7  HIS  chi1: CB→CG(sp2)  chi2: CG→ND1(imidazole)
    [1.496, 1.378, _GENERIC_BL, _GENERIC_BL],
    # 8  ILE  chi1: CB→CG1  chi2: CG1→CD1
    [1.533, 1.523, _GENERIC_BL, _GENERIC_BL],
    # 9  LYS  chi1: CB→CG  chi2: CG→CD  chi3: CD→CE  chi4: CE→NZ
    [1.522, 1.523, 1.523, 1.494],
    # 10 LEU  chi1: CB→CG  chi2: CG→CD1
    [1.530, 1.525, _GENERIC_BL, _GENERIC_BL],
    # 11 MET  chi1: CB→CG  chi2: CG→SD(C–S)  chi3: SD→CE(S–C)
    [1.520, 1.835, 1.814, _GENERIC_BL],
    # 12 ASN  chi1: CB→CG(sp2)  chi2: CG→OD1(C=O)
    [1.516, 1.234, _GENERIC_BL, _GENERIC_BL],
    # 13 PRO  chi1: CB→CG(ring)  chi2: CG→CD
    [1.495, 1.502, _GENERIC_BL, _GENERIC_BL],
    # 14 GLN  chi1: CB→CG  chi2: CG→CD  chi3: CD→OE1
    [1.523, 1.517, 1.235, _GENERIC_BL],
    # 15 ARG  chi1: CB→CG  chi2: CG→CD  chi3: CD→NE  chi4: NE→CZ(sp2)
    [1.520, 1.522, 1.460, 1.330],
    # 16 SER  chi1: CB→OG  (C–O bond)
    [1.417, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 17 THR  chi1: CB→OG1 (C–O bond)
    [1.433, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 18 VAL  chi1: CB→CG1
    [1.528, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
    # 19 TRP  chi1: CB→CG(sp2)  chi2: CG→CD1(indole)
    [1.499, 1.367, _GENERIC_BL, _GENERIC_BL],
    # 20 TYR  chi1: CB→CG(sp2)  chi2: CG→CD1(aromatic)
    [1.510, 1.393, _GENERIC_BL, _GENERIC_BL],
    # 21 UNK
    [_GENERIC_BL, _GENERIC_BL, _GENERIC_BL, _GENERIC_BL],
], dtype=torch.float32)  # shape (22, 4)

_SC_BOND_ANGLES = torch.tensor([
    # 0  PAD
    [_GENERIC_BA, _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 1  ALA
    [_GENERIC_BA, _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 2  CYS  CA-CB-SG = 114.4°
    [math.radians(114.4), _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 3  ASP  CA-CB-CG = 113.0°  CB-CG-OD1 = 119.2° (sp2)
    [math.radians(113.0), math.radians(119.2), _GENERIC_BA, _GENERIC_BA],
    # 4  GLU  CA-CB-CG = 113.8°  CB-CG-CD = 113.2°  CG-CD-OE1 = 121.5° (sp2)
    [math.radians(113.8), math.radians(113.2), math.radians(121.5), _GENERIC_BA],
    # 5  PHE  CA-CB-CG = 113.8°  CB-CG-CD1 = 120.7° (aromatic sp2)
    [math.radians(113.8), math.radians(120.7), _GENERIC_BA, _GENERIC_BA],
    # 6  GLY
    [_GENERIC_BA, _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 7  HIS  CA-CB-CG = 113.7°  CB-CG-ND1 = 122.6° (imidazole sp2)
    [math.radians(113.7), math.radians(122.6), _GENERIC_BA, _GENERIC_BA],
    # 8  ILE  CA-CB-CG1 = 110.4°  CB-CG1-CD1 = 114.0°
    [math.radians(110.4), math.radians(114.0), _GENERIC_BA, _GENERIC_BA],
    # 9  LYS  114.0°  111.6°  111.6°  CE-NZ = 111.8°
    [math.radians(114.0), math.radians(111.6), math.radians(111.6), math.radians(111.8)],
    # 10 LEU  CA-CB-CG = 116.3°  CB-CG-CD1 = 110.4°
    [math.radians(116.3), math.radians(110.4), _GENERIC_BA, _GENERIC_BA],
    # 11 MET  CA-CB-CG = 113.8°  CB-CG-SD = 107.4°  CG-SD-CE = 100.2° (C-S-C)
    [math.radians(113.8), math.radians(107.4), math.radians(100.2), _GENERIC_BA],
    # 12 ASN  CA-CB-CG = 112.6°  CB-CG-OD1 = 120.9° (sp2)
    [math.radians(112.6), math.radians(120.9), _GENERIC_BA, _GENERIC_BA],
    # 13 PRO  CA-CB-CG = 103.2° (ring)  CB-CG-CD = 103.0° (ring)
    [math.radians(103.2), math.radians(103.0), _GENERIC_BA, _GENERIC_BA],
    # 14 GLN  CA-CB-CG = 113.8°  CB-CG-CD = 112.8°  CG-CD-OE1 = 120.9° (sp2)
    [math.radians(113.8), math.radians(112.8), math.radians(120.9), _GENERIC_BA],
    # 15 ARG  113.9°  111.8°  CD-NE = 111.6°  NE-CZ = 124.7° (guanidinium sp2)
    [math.radians(113.9), math.radians(111.8), math.radians(111.6), math.radians(124.7)],
    # 16 SER  CA-CB-OG = 110.8°
    [math.radians(110.8), _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 17 THR  CA-CB-OG1 = 109.2°
    [math.radians(109.2), _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 18 VAL  CA-CB-CG1 = 110.7°
    [math.radians(110.7), _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
    # 19 TRP  CA-CB-CG = 114.0°  CB-CG-CD1 = 127.0° (indole sp2)
    [math.radians(114.0), math.radians(127.0), _GENERIC_BA, _GENERIC_BA],
    # 20 TYR  CA-CB-CG = 113.7°  CB-CG-CD1 = 120.9° (aromatic sp2)
    [math.radians(113.7), math.radians(120.9), _GENERIC_BA, _GENERIC_BA],
    # 21 UNK
    [_GENERIC_BA, _GENERIC_BA, _GENERIC_BA, _GENERIC_BA],
], dtype=torch.float32)  # shape (22, 4)


# ===================================================================
# 8. Chemical symmetry definitions
# ===================================================================
# Torsion-angle symmetry: residues whose terminal heavy atoms are
# chemically equivalent under a π rotation of a specific chi angle.
# Maps residue ID → tuple of torsion indices (0-based in the 7-angle
# [ω, φ, ψ, χ1, χ2, χ3, χ4] layout).
SYMMETRIC_CHI: Dict[int, Tuple[int, ...]] = {
    3:  (4,),   # ASP – chi2: OD1 ↔ OD2
    4:  (5,),   # GLU – chi3: OE1 ↔ OE2
    5:  (4,),   # PHE – chi2: CD1 ↔ CD2, CE1 ↔ CE2
    20: (4,),   # TYR – chi2: CD1 ↔ CD2, CE1 ↔ CE2
}

# Coordinate-level symmetry swap pairs for the 23-atom layout.
# Residue ID → list of (slot_A, slot_B) pairs whose coordinates can swap.
SYM_SWAPS: Dict[int, List[Tuple[int, int]]] = {
    3:  [(9, 10)],             # ASP: OD1 ↔ OD2
    4:  [(11, 12)],            # GLU: OE1 ↔ OE2
    5:  [(9, 10), (12, 14)],   # PHE: (CD1,CD2), (CE1,CE2)
    15: [(20, 21)],            # ARG: NH1 ↔ NH2
    20: [(9, 10), (12, 14)],   # TYR: (CD1,CD2), (CE1,CE2)
}


# ===================================================================
# 9. ESM-2 vocabulary mapping
# ===================================================================
# Maps internal RESIDUE_MAP IDs (0-21) to ESM-2 tokenizer IDs.
# 0=PAD, 1=ALA, 2=CYS, 3=ASP, 4=GLU, 5=PHE, 6=GLY, 7=HIS, 8=ILE,
# 9=LYS, 10=LEU, 11=MET, 12=ASN, 13=PRO, 14=GLN, 15=ARG, 16=SER,
# 17=THR, 18=VAL, 19=TRP, 20=TYR, 21=UNK(X)
ESM_VOCAB_MAP: List[int] = [
    1, 5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 24,
]

# misc

# Proline ring-closure: ideal N–CD bond length in the pyrrolidine ring.
# Engh & Huber value for the N(sp2)–C(sp3) bond closing the 5-membered ring.
BOND_N_CD_PRO: float = 1.473
