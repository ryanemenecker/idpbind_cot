import math

# --- AMBER Equilibrium Bond Lengths (Angstroms) ---
L_CH_ALI = 1.090  # Aliphatic C-H (sp3)
L_CH_ARO = 1.080  # Aromatic C-H (sp2)
L_NH     = 1.010  # Amide/Amine N-H
L_OH     = 0.960  # Hydroxyl O-H
L_SH     = 1.336  # Sulfhydryl S-H

# --- Idealized Bond Angles (Radians) ---
A_SP3   = math.radians(109.5)  # Tetrahedral
A_SP2   = math.radians(120.0)  # Planar
A_AMIDE = math.radians(119.8)  # Backbone N-H
A_OH    = math.radians(108.5)  # Hydroxyl
A_SH    = math.radians(96.0)   # Thiol

# --- Idealized Dihedrals (Radians) ---
D_TRANS = math.radians(180.0)
D_CIS   = math.radians(0.0)
D_STAG_P = math.radians(60.0)   # Staggered +60
D_STAG_M = math.radians(-60.0)  # Staggered -60
D_120_P  = math.radians(120.0)
D_120_M  = math.radians(-120.0)


# Format: 'H_name': ('GreatGrandparent', 'Grandparent', 'Parent', r, theta, chi)
H_RULES = {
    'GENERIC_BACKBONE': {
        'H': ('-CA', '-C', 'N', L_NH, A_AMIDE, D_CIS),
    },
    'GLY': {
        'HA2': ('N', 'C', 'CA', L_CH_ALI, A_SP3, A_SP2),
        'HA3': ('N', 'C', 'CA', L_CH_ALI, A_SP3, D_120_M),
    },
    'ALA': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB1': ('N', 'CA', 'CB', L_CH_ALI, A_SP3, D_TRANS),
        'HB2': ('N', 'CA', 'CB', L_CH_ALI, A_SP3, D_STAG_P),
        'HB3': ('N', 'CA', 'CB', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'SER': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('OG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('OG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG': ('CA', 'CB', 'OG', L_OH, A_OH, D_TRANS),
    },
    'CYS': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('SG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('SG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG': ('CA', 'CB', 'SG', L_SH, A_SH, D_TRANS),
    },
    'THR': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB': ('CA', 'OG1', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HG1': ('CA', 'CB', 'OG1', L_OH, A_OH, D_TRANS),
        'HG21': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_TRANS),
        'HG22': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_P),
        'HG23': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'MET': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('SD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('SD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HE1': ('CG', 'SD', 'CE', L_CH_ALI, A_SP3, D_TRANS),
        'HE2': ('CG', 'SD', 'CE', L_CH_ALI, A_SP3, D_STAG_P),
        'HE3': ('CG', 'SD', 'CE', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'PHE': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD1': ('CE1', 'CG', 'CD1', L_CH_ARO, A_SP2, D_TRANS),
        'HD2': ('CE2', 'CG', 'CD2', L_CH_ARO, A_SP2, D_TRANS),
        'HE1': ('CD1', 'CZ', 'CE1', L_CH_ARO, A_SP2, D_TRANS),
        'HE2': ('CD2', 'CZ', 'CE2', L_CH_ARO, A_SP2, D_TRANS),
        'HZ': ('CE1', 'CE2', 'CZ', L_CH_ARO, A_SP2, D_TRANS),
    },
    'TYR': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD1': ('CE1', 'CG', 'CD1', L_CH_ARO, A_SP2, D_TRANS),
        'HD2': ('CE2', 'CG', 'CD2', L_CH_ARO, A_SP2, D_TRANS),
        'HE1': ('CD1', 'CZ', 'CE1', L_CH_ARO, A_SP2, D_TRANS),
        'HE2': ('CD2', 'CZ', 'CE2', L_CH_ARO, A_SP2, D_TRANS),
        'HH': ('CE1', 'CZ', 'OH', L_OH, A_OH, D_TRANS),
    },
    'TRP': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD1': ('CE2', 'CG', 'CD1', L_CH_ARO, A_SP2, D_TRANS),
        'HE1': ('CD1', 'CE2', 'NE1', L_NH, A_SP2, D_TRANS),
        'HE3': ('CZ3', 'CD2', 'CE3', L_CH_ARO, A_SP2, D_TRANS),
        'HZ2': ('CH2', 'CE2', 'CZ2', L_CH_ARO, A_SP2, D_TRANS),
        'HZ3': ('CH2', 'CE3', 'CZ3', L_CH_ARO, A_SP2, D_TRANS),
        'HH2': ('CZ3', 'CZ2', 'CH2', L_CH_ARO, A_SP2, D_TRANS),
    },
    'HIE': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD2': ('NE2', 'CG', 'CD2', L_CH_ARO, A_SP2, D_TRANS),
        'HE1': ('ND1', 'NE2', 'CE1', L_CH_ARO, A_SP2, D_TRANS),
        'HE2': ('CD2', 'CE1', 'NE2', L_NH, A_SP2, D_TRANS),
    },
    'HID': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD2': ('NE2', 'CG', 'CD2', L_CH_ARO, A_SP2, D_TRANS),
        'HE1': ('ND1', 'NE2', 'CE1', L_CH_ARO, A_SP2, D_TRANS),
        'HD1': ('CE1', 'CG', 'ND1', L_NH, A_SP2, D_TRANS),
    },
    'ASP': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
    },
    'GLU': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
    },
    'LYS': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HD2': ('CE', 'CG', 'CD', L_CH_ALI, A_SP3, A_SP2),
        'HD3': ('CE', 'CG', 'CD', L_CH_ALI, A_SP3, D_120_M),
        'HE2': ('NZ', 'CD', 'CE', L_CH_ALI, A_SP3, A_SP2),
        'HE3': ('NZ', 'CD', 'CE', L_CH_ALI, A_SP3, D_120_M),
        'HZ1': ('CD', 'CE', 'NZ', L_NH, A_SP3, D_TRANS),
        'HZ2': ('CD', 'CE', 'NZ', L_NH, A_SP3, D_STAG_P),
        'HZ3': ('CD', 'CE', 'NZ', L_NH, A_SP3, D_STAG_M),
    },
    'ARG': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HD2': ('NE', 'CG', 'CD', L_CH_ALI, A_SP3, A_SP2),
        'HD3': ('NE', 'CG', 'CD', L_CH_ALI, A_SP3, D_120_M),
        'HE': ('CG', 'CD', 'NE', L_NH, A_SP2, D_TRANS),
        'HH11': ('NE', 'CZ', 'NH1', L_NH, A_SP2, D_TRANS),
        'HH12': ('NE', 'CZ', 'NH1', L_NH, A_SP2, D_CIS),
        'HH21': ('NE', 'CZ', 'NH2', L_NH, A_SP2, D_TRANS),
        'HH22': ('NE', 'CZ', 'NH2', L_NH, A_SP2, D_CIS),
    },
    'VAL': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB': ('CA', 'CG1', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG11': ('CA', 'CB', 'CG1', L_CH_ALI, A_SP3, D_TRANS),
        'HG12': ('CA', 'CB', 'CG1', L_CH_ALI, A_SP3, D_STAG_P),
        'HG13': ('CA', 'CB', 'CG1', L_CH_ALI, A_SP3, D_STAG_M),
        'HG21': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_TRANS),
        'HG22': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_P),
        'HG23': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'LEU': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG': ('CB', 'CD1', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HD11': ('CB', 'CG', 'CD1', L_CH_ALI, A_SP3, D_TRANS),
        'HD12': ('CB', 'CG', 'CD1', L_CH_ALI, A_SP3, D_STAG_P),
        'HD13': ('CB', 'CG', 'CD1', L_CH_ALI, A_SP3, D_STAG_M),
        'HD21': ('CB', 'CG', 'CD2', L_CH_ALI, A_SP3, D_TRANS),
        'HD22': ('CB', 'CG', 'CD2', L_CH_ALI, A_SP3, D_STAG_P),
        'HD23': ('CB', 'CG', 'CD2', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'ILE': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB': ('CA', 'CG1', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HG12': ('CD1', 'CB', 'CG1', L_CH_ALI, A_SP3, A_SP2),
        'HG13': ('CD1', 'CB', 'CG1', L_CH_ALI, A_SP3, D_120_M),
        'HG21': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_TRANS),
        'HG22': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_P),
        'HG23': ('CA', 'CB', 'CG2', L_CH_ALI, A_SP3, D_STAG_M),
        'HD11': ('CB', 'CG1', 'CD1', L_CH_ALI, A_SP3, D_TRANS),
        'HD12': ('CB', 'CG1', 'CD1', L_CH_ALI, A_SP3, D_STAG_P),
        'HD13': ('CB', 'CG1', 'CD1', L_CH_ALI, A_SP3, D_STAG_M),
    },
    'PRO': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HD2': ('N', 'CG', 'CD', L_CH_ALI, A_SP3, A_SP2),
        'HD3': ('N', 'CG', 'CD', L_CH_ALI, A_SP3, D_120_M),
    },
    'ASN': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HD21': ('CB', 'CG', 'ND2', L_NH, A_SP2, D_TRANS),
        'HD22': ('CB', 'CG', 'ND2', L_NH, A_SP2, D_CIS),
    },
    'GLN': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('CG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
        'HG2': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, A_SP2),
        'HG3': ('CD', 'CB', 'CG', L_CH_ALI, A_SP3, D_120_M),
        'HE21': ('CG', 'CD', 'NE2', L_NH, A_SP2, D_TRANS),
        'HE22': ('CG', 'CD', 'NE2', L_NH, A_SP2, D_CIS),
    },
    'CYX': {
        'HA': ('N', 'CB', 'CA', L_CH_ALI, A_SP3, D_120_M),
        'HB2': ('SG', 'CA', 'CB', L_CH_ALI, A_SP3, A_SP2),
        'HB3': ('SG', 'CA', 'CB', L_CH_ALI, A_SP3, D_120_M),
    },
}