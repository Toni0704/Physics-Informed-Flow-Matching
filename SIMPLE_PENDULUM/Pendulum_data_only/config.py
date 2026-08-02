"""
config.py
---------
Central configuration for all hyperparameters.
All other modules import from here — nothing is hardcoded elsewhere.
"""

# ── Physical time ──────────────────────────────────────────────
N_STEPS   = 30      # number of physical time steps per trajectory
DT_PHYS   = 0.1     # physical time step Δτ  →  T_total = 3.0s

# ── State representation ───────────────────────────────────────
# Raw θ is unbounded for rotating trajectories.
# We encode state as (sin θ, cos θ, ω) — all bounded in ~[-2, 2].
# This means each trajectory has shape (N_STEPS, 3) not (N_STEPS, 2).
STATE_DIM = 3       # sin(θ), cos(θ), ω

# ── Dataset ────────────────────────────────────────────────────
N_TRAIN   = 1000    # number of training trajectories
N_TEST    = 200

# Energy range — MUST stay below separatrix at E=1.0
# Above E=1.0 the pendulum rotates (θ unbounded) — topologically
# different from oscillations. Never mix the two in one dataset.
E_MIN     = 0.1
E_MAX     = 0.9     # strictly below separatrix

# ── Network ────────────────────────────────────────────────────
HIDDEN_DIM = 512
N_LAYERS   = 4
T_EMB_DIM  = 16     # sinusoidal embedding dim for FM time t

# ── Training ───────────────────────────────────────────────────
BATCH_SIZE = 128
N_EPOCHS   = 1000
LR         = 1e-3
GRAD_CLIP  = 1.0

# ── Sampling (inference) ───────────────────────────────────────
N_FM_STEPS = 200    # Euler steps for FM integration (t: 0 → 1)

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
