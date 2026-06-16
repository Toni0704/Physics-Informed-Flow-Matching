"""
config.py  --  Bratu Experiments
---------------------------------
Central configuration. All other modules import from here.

Problem:  u_xx + C * exp(u) = 0,  x in [0,1],  u(0) = u(1) = 0
Exact 1D solution exists for C < Cc ~ 3.5138.
Two branches for C < Cc: lower (u_max < u_c) and upper (u_max > u_c).
"""

# ── Spatial discretisation ─────────────────────────────────────
N_X       = 64       # number of interior grid points (excludes x=0, x=1)
                     # total solution vector u has shape (N_X,)
                     # x_i = i/(N_X+1), i = 1, ..., N_X

# ── Physical parameters ────────────────────────────────────────
C_MIN     = 0.5      # minimum C in training/test set
C_MAX     = 3.4      # maximum C  (strictly below Cc = 3.5138)
C_CRIT    = 3.513830719   # critical value, unique solution
U_CRIT    = 1.2277   # u_max at critical C — divides upper/lower branches

# ── Dataset ────────────────────────────────────────────────────
N_TRAIN   = 4000     # training samples (split evenly upper/lower across C)
N_TEST    = 500      # test samples
N_C_GRID  = 40       # number of C values in training grid

# ── Network ────────────────────────────────────────────────────
HIDDEN_DIM  = 512
N_LAYERS    = 4
T_EMB_DIM   = 16     # sinusoidal FM-time embedding dimension
C_EMB_DIM   = 32     # C encoder output dimension
B_EMB_DIM   = 32     # branch encoder output dimension

# ── Training ───────────────────────────────────────────────────
BATCH_SIZE  = 128
N_EPOCHS    = 1000
LR          = 1e-3
GRAD_CLIP   = 1.0

# ── Sampling ───────────────────────────────────────────────────
N_FM_STEPS  = 200    # Euler steps for FM integration t: 0 -> 1

# ── PCFM projection (Case B2) ──────────────────────────────────
N_PROJ_STEPS     = 20     # Gauss-Newton iterations at final t=1
PROJ_TOL         = 1e-8   # convergence tolerance for projection
PROJ_LAMBDA      = 1e-4   # Tikhonov regularisation for ill-conditioned J

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
