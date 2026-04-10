NM_TO_ANGSTROM = 10.0
KJ_TO_KCAL = 1.0 / 4.184

# ── Physics constants ───────────────────────────────────────────────
COULOMB_CONSTANT = 332.0636  # kcal·Å/(mol·e²), AMBER convention

# ── Non-bonded interaction defaults ─────────────────────────────────
DEFAULT_CUTOFF = 10.0        # Å, non-bonded interaction cutoff radius
DIELECTRIC_WATER = 78.5      # relative dielectric constant of water
VDW_SCALE_14 = 1.0 / 2.0    # AMBER 1-4 van der Waals scaling factor
ELEC_SCALE_14 = 1.0 / 1.2   # AMBER 1-4 electrostatic scaling factor

# ── Soft-core annealing ────────────────────────────────────────────
ALPHA_INITIAL = 0.5          # initial soft-core α parameter
ALPHA_DECAY = 0.05           # exponential decay rate for α(t) = α₀·e^(−decay·t)

# ── Restraints ──────────────────────────────────────────────────────
CA_RESTRAINT_K = 10.0        # kcal/(mol·Å²), Cα positional restraint spring constant
