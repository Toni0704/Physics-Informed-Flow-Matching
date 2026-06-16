"""
data.py  --  Bratu Dataset Generation
--------------------------------------
Generates the training and test datasets for the 1D Bratu problem.

Each sample is:
    u     : np.ndarray (N_X,)   interior grid values u(x_i)
    C     : float               parameter value
    b     : int                 branch label (0=lower, 1=upper)
    u_max : float               maximum of u (at x=0.5)

Ground truth is computed using:
    - Nonstandard finite difference (NSFD) discretisation from Buckmire/Mohsen
    - Newton iteration with sinusoidal starting guess
    - Branch selected by amplitude of starting guess:
        lower: a = 0.5  (<  u_crit = 1.2277)
        upper: a = 5.0  (>> u_crit)
    - Exact solution available via analytical formula for verification

The exact solution is:
    u(x) = 2 * ln( cosh(alpha) / cosh(alpha * (1 - 2x)) )
    where alpha solves: cosh(alpha) = (4/sqrt(2C)) * alpha
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve_banded
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (
    N_X, C_MIN, C_MAX, C_CRIT, U_CRIT,
    N_TRAIN, N_TEST, N_C_GRID, BATCH_SIZE, SEED
)


# ── Grid ──────────────────────────────────────────────────────────────────────

def make_grid():
    """Interior grid x_i = i/(N_X+1), i = 1, ..., N_X."""
    h = 1.0 / (N_X + 1)
    x = np.linspace(h, 1.0 - h, N_X)
    return x, h


# ── Exact solution ────────────────────────────────────────────────────────────

def _alpha_equation(alpha, C):
    """Transcendental equation: cosh(alpha) - (4/sqrt(2C)) * alpha = 0."""
    return np.cosh(alpha) - (4.0 / np.sqrt(2.0 * C)) * alpha


def solve_alpha(C, branch=0):
    """
    Solve cosh(alpha) = (4/sqrt(2C)) * alpha for alpha > 0.
    branch=0: lower branch (smaller alpha, steeper rhs line)
    branch=1: upper branch (larger alpha)
    Returns alpha, or None if no solution exists.
    """
    if C >= C_CRIT:
        return None

    slope = 4.0 / np.sqrt(2.0 * C)

    # Bracket: cosh(alpha) - slope*alpha = 0
    # At alpha=0: cosh(0)=1 > 0
    # Find where function crosses zero — two crossings for C < Cc
    alphas = np.linspace(0.01, 20.0, 10000)
    f = np.cosh(alphas) - slope * alphas
    sign_changes = np.where(np.diff(np.sign(f)))[0]

    if len(sign_changes) < 2:
        return None

    from scipy.optimize import brentq
    if branch == 0:
        # Lower branch: first (smaller) crossing
        a0, a1 = alphas[sign_changes[0]], alphas[sign_changes[0]+1]
    else:
        # Upper branch: second (larger) crossing
        a0, a1 = alphas[sign_changes[1]], alphas[sign_changes[1]+1]

    try:
        alpha = brentq(_alpha_equation, a0, a1, args=(C,))
        return float(alpha)
    except Exception:
        return None


def exact_solution(C, branch=0, x=None):
    """
    Compute exact 1D Bratu solution.
    Returns u on interior grid, or None if no solution.
    """
    if x is None:
        x, _ = make_grid()
    alpha = solve_alpha(C, branch)
    if alpha is None:
        return None
    u = 2.0 * np.log(np.cosh(alpha) / np.cosh(alpha * (1.0 - 2.0 * x)))
    return u


def exact_umax(C, branch=0):
    """Maximum of exact solution (at x=0.5)."""
    alpha = solve_alpha(C, branch)
    if alpha is None:
        return None
    return 2.0 * np.log(np.cosh(alpha))


# ── NSFD numerical solver ─────────────────────────────────────────────────────

def _nsfd_denominator(h):
    """
    NSFD denominator: hs = 2 * ln(cosh(h))
    Reduces to h^2 as h -> 0 (consistent with SFD).
    """
    return 2.0 * np.log(np.cosh(h))


def _bratu_residual(u, C, hs):
    """
    Residual of NSFD Bratu system F(u) = 0.
    u: interior values, shape (N_X,)
    Returns residual shape (N_X,)
    """
    N = len(u)
    F = np.zeros(N)
    u_ext = np.concatenate([[0.0], u, [0.0]])  # add BCs
    for i in range(N):
        F[i] = (u_ext[i+2] - 2*u_ext[i+1] + u_ext[i]) / hs \
               + C * np.exp(u_ext[i+1])
    return F


def _bratu_jacobian(u, C, hs):
    """
    Jacobian of NSFD Bratu system dF/du.
    Tridiagonal matrix, shape (N_X, N_X).
    """
    N = len(u)
    J = np.zeros((N, N))
    for i in range(N):
        J[i, i] = -2.0 / hs + C * np.exp(u[i])
        if i > 0:
            J[i, i-1] = 1.0 / hs
        if i < N-1:
            J[i, i+1] = 1.0 / hs
    return J


def solve_bratu_nsfd(C, branch=0, tol=1e-12, max_iter=200):
    """
    Solve 1D Bratu problem using NSFD + Newton iteration.

    Args:
        C      : parameter value
        branch : 0 = lower, 1 = upper
        tol    : Newton convergence tolerance
        max_iter: maximum iterations

    Returns:
        u      : solution on interior grid, shape (N_X,)
                 or None if failed to converge
    """
    x, h = make_grid()
    hs   = _nsfd_denominator(h)

    # Starting guess: sinusoidal with appropriate amplitude
    # Lower branch: a < u_crit, Upper branch: a > u_crit
    a = 0.5 if branch == 0 else max(8.0, 2.5 / max(C, 0.1))
    u = a * np.sin(np.pi * x)

    for it in range(max_iter):
        F = _bratu_residual(u, C, hs)
        if not np.isfinite(F).all():
            return None
        if np.linalg.norm(F) < tol:
            return u
        J = _bratu_jacobian(u, C, hs)
        if not np.isfinite(J).all():
            return None
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        # Clip step to prevent overflow
        delta = np.clip(delta, -2.0, 2.0)
        u = u + delta

    # Check final residual
    F = _bratu_residual(u, C, hs)
    if np.linalg.norm(F) < 1e-6:
        return u
    return None


def classify_branch(u):
    """Classify solution as lower (0) or upper (1) based on u_max."""
    u_max = np.max(u)
    return 1 if u_max > U_CRIT else 0


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_samples, c_min=C_MIN, c_max=C_MAX,
                     seed=SEED, include_critical=False):
    """
    Generate Bratu solutions for both branches.

    Returns:
        solutions : np.ndarray (n, N_X)
        C_vals    : np.ndarray (n,)
        b_vals    : np.ndarray (n,)  int {0,1}
        umax_vals : np.ndarray (n,)
    """
    rng = np.random.default_rng(seed)

    solutions = []
    C_vals    = []
    b_vals    = []
    umax_vals = []

    n_per_config = max(1, n_samples // (N_C_GRID * 2))  # per (C, branch)
    C_grid = np.linspace(c_min, c_max, N_C_GRID)

    for C in C_grid:
        for branch in [0, 1]:
            for _ in range(n_per_config):
                u = solve_bratu_nsfd(C, branch)
                if u is None:
                    continue
                b_check = classify_branch(u)
                if b_check != branch:
                    continue  # solver landed on wrong branch
                solutions.append(u)
                C_vals.append(C)
                b_vals.append(branch)
                umax_vals.append(np.max(u))

    if include_critical:
        u = solve_bratu_nsfd(C_CRIT, branch=0)
        if u is not None:
            solutions.append(u)
            C_vals.append(C_CRIT)
            b_vals.append(classify_branch(u))
            umax_vals.append(np.max(u))

    solutions = np.array(solutions, dtype=np.float32)
    C_vals    = np.array(C_vals,    dtype=np.float32)
    b_vals    = np.array(b_vals,    dtype=np.int64)
    umax_vals = np.array(umax_vals, dtype=np.float32)

    print(f"Generated {len(solutions)} solutions "
          f"({(b_vals==0).sum()} lower, {(b_vals==1).sum()} upper)")
    return solutions, C_vals, b_vals, umax_vals


def get_dataloaders(seed=SEED):
    """
    Build train/test DataLoaders.
    Returns train_loader, test_loader, test_data dict.
    """
    print("Generating training data...")
    u_train, C_train, b_train, umax_train = generate_dataset(
        N_TRAIN, seed=seed
    )
    print("Generating test data...")
    u_test, C_test, b_test, umax_test = generate_dataset(
        N_TEST, seed=seed + 1
    )

    # Normalise u to roughly zero mean / unit variance using training stats
    u_mean = u_train.mean()
    u_std  = u_train.std() + 1e-8
    u_train_n = (u_train - u_mean) / u_std
    u_test_n  = (u_test  - u_mean) / u_std

    train_ds = TensorDataset(
        torch.tensor(u_train_n),
        torch.tensor(C_train),
        torch.tensor(b_train),
        torch.tensor(umax_train),
    )
    test_ds = TensorDataset(
        torch.tensor(u_test_n),
        torch.tensor(C_test),
        torch.tensor(b_test),
        torch.tensor(umax_test),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)

    # Raw test data for evaluation (un-normalised)
    test_raw = {
        "u":    u_test,
        "C":    C_test,
        "b":    b_test,
        "umax": umax_test,
        "u_mean": u_mean,
        "u_std":  u_std,
    }

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    return train_loader, test_loader, test_raw


# ── PDE residual ──────────────────────────────────────────────────────────────

def pde_residual(u_batch, C_batch):
    """
    Compute NSFD PDE residual for a batch of solutions.

    Args:
        u_batch : np.ndarray (B, N_X)  interior values
        C_batch : np.ndarray (B,)

    Returns:
        residuals : np.ndarray (B,)  mean squared residual per sample
    """
    _, h = make_grid()
    hs   = _nsfd_denominator(h)
    B    = u_batch.shape[0]
    res  = np.zeros(B)

    for i in range(B):
        u   = u_batch[i]
        C   = C_batch[i]
        F   = _bratu_residual(u, C, hs)
        res[i] = np.mean(F**2)
    return res


def bc_error(u_batch):
    """
    Boundary condition error. u_batch is interior only (BCs are exactly 0
    by construction of the solver, but the generative model may not honour this).
    For interior-only representation, BCs are implicitly zero — so BC error
    is measured as deviation of endpoint values from expected decay.
    Returns zeros for solver-generated data; non-zero for FM-generated data.
    """
    # We track the values at the first and last interior nodes as a proxy.
    # True BC error requires evaluating u at x=0 and x=1 which are not in grid.
    # Instead, we check that u is close to 0 at both ends of the interior grid.
    x, h = make_grid()
    # Linear extrapolation to boundaries
    u_left  = u_batch[:, 0]   # u at x=h, should be small
    u_right = u_batch[:, -1]  # u at x=1-h, should be small
    # Expected value if u(0)=0: u(h) ≈ u'(0)*h  (should be small for reasonable u)
    # We just report the magnitude at the first/last interior node
    return (u_left**2 + u_right**2) / 2.0


def branch_accuracy(u_batch, b_true):
    """
    Fraction of generated samples on the correct branch.
    Args:
        u_batch : np.ndarray (B, N_X)
        b_true  : np.ndarray (B,) int
    """
    u_max   = u_batch.max(axis=1)
    b_pred  = (u_max > U_CRIT).astype(int)
    return (b_pred == b_true).mean()
