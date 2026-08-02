"""
data.py
-------
Data generation for the simple pendulum using the Störmer-Verlet
(leapfrog) symplectic integrator.

Why symplectic?
    Standard integrators (e.g. RK4) accumulate energy drift over time.
    Störmer-Verlet exactly conserves a *shadow Hamiltonian* close to H,
    so energy oscillates with bounded error over exponentially long times.
    This means every training trajectory lies on (or extremely close to)
    a true energy contour — the physics is inherent to the data.

Pendulum equations:
    dθ/dτ = ω
    dω/dτ = -sin(θ)

    H(θ, ω) = 0.5 * ω² - cos(θ)      (conserved Hamiltonian)

One sample = one trajectory = shape (N_STEPS, 2), columns [θ, ω].
Physical time τ is implicit in the row index.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from Simple_pendulum.Pendulum_2.config import (
    N_STEPS, DT_PHYS,
    N_TRAIN, N_TEST,
    E_MIN, E_MAX,
    STATE_DIM,
    BATCH_SIZE, SEED,
)


# ─────────────────────────────────────────────────────────────
# Hamiltonian
# ─────────────────────────────────────────────────────────────

def hamiltonian(theta, omega):
    """
    H(θ, ω) = 0.5 * ω² - cos(θ)

    Works on scalars or numpy arrays of any shape.
    """
    return 0.5 * omega**2 - np.cos(theta)


# ─────────────────────────────────────────────────────────────
# State encoding / decoding
# ─────────────────────────────────────────────────────────────

def encode_trajectory(traj_raw):
    """
    Convert raw trajectory (θ, ω) → encoded (sin θ, cos θ, ω).

    Why encode?
        Raw θ is unbounded for rotating trajectories (θ → ±∞).
        Even for oscillating trajectories it can reach ±π.
        sin θ and cos θ are always in [-1, 1] — well-scaled for
        the network and robust to angle wrapping.

    Args:
        traj_raw : np.ndarray, shape (..., n_steps, 2)  columns [θ, ω]

    Returns:
        traj_enc : np.ndarray, shape (..., n_steps, 3)  columns [sin θ, cos θ, ω]
    """
    sin_th = np.sin(traj_raw[..., 0:1])   # (..., n_steps, 1)
    cos_th = np.cos(traj_raw[..., 0:1])   # (..., n_steps, 1)
    omega  = traj_raw[..., 1:2]           # (..., n_steps, 1)
    return np.concatenate([sin_th, cos_th, omega], axis=-1)


def decode_trajectory(traj_enc):
    """
    Convert encoded (sin θ, cos θ, ω) → raw (θ, ω).

    Uses arctan2(sin θ, cos θ) to recover θ ∈ (-π, π].

    Args:
        traj_enc : np.ndarray, shape (..., n_steps, 3)

    Returns:
        traj_raw : np.ndarray, shape (..., n_steps, 2)  columns [θ, ω]
    """
    theta = np.arctan2(traj_enc[..., 0], traj_enc[..., 1])  # (..., n_steps)
    omega = traj_enc[..., 2]                                 # (..., n_steps)
    return np.stack([theta, omega], axis=-1)


# ─────────────────────────────────────────────────────────────
# Störmer-Verlet integrator
# ─────────────────────────────────────────────────────────────

def stormer_verlet_step(theta, omega, dt):
    """
    One Störmer-Verlet step for  dω/dτ = -sin(θ).

    Decomposition:
        half kick  →  full drift  →  half kick

    This is a symplectic (area-preserving) map on phase space,
    which is what guarantees long-term energy conservation.

    Args:
        theta, omega : current state (scalars)
        dt           : physical time step Δτ

    Returns:
        theta_new, omega_new : updated state
    """
    omega_half = omega - 0.5 * dt * np.sin(theta)           # half kick
    theta_new  = theta + dt * omega_half                     # full drift
    omega_new  = omega_half - 0.5 * dt * np.sin(theta_new)  # half kick
    return theta_new, omega_new


def generate_trajectory(theta0, omega0, n_steps=N_STEPS, dt=DT_PHYS):
    """
    Integrate one pendulum trajectory using Störmer-Verlet.

    Args:
        theta0, omega0 : initial condition
        n_steps        : number of steps to record
        dt             : physical time step

    Returns:
        traj : np.ndarray, shape (n_steps, 2)
               columns are [θ(τ), ω(τ)]
               τ is implicit in the row index: τᵢ = i * dt
    """
    traj = np.zeros((n_steps, 2))
    theta, omega = theta0, omega0
    for i in range(n_steps):
        theta, omega = stormer_verlet_step(theta, omega, dt)
        traj[i, 0] = theta
        traj[i, 1] = omega
    return traj


# ─────────────────────────────────────────────────────────────
# Initial condition sampling
# ─────────────────────────────────────────────────────────────

def sample_initial_conditions(n, e_min=E_MIN, e_max=E_MAX, rng=None):
    """
    Sample initial conditions (θ₀, ω₀) distributed across energy levels.

    Strategy:
        1. Sample θ₀ uniformly in (-0.9π, 0.9π)
        2. Sample energy E uniformly in [e_min, e_max]
        3. Solve H(θ₀, ω₀) = E for ω₀:
               ω₀ = ± √(2(E + cos(θ₀)))
        4. Reject if unphysical (argument of sqrt < 0)

    The energy range [E_MIN, E_MAX] must stay strictly below E=1.0
    (the separatrix). Above E=1.0 the pendulum rotates continuously
    and θ becomes unbounded — topologically different from oscillations.
    Never mix the two in one dataset.

    Args:
        n     : number of initial conditions to generate
        e_min : minimum energy level
        e_max : maximum energy level
        rng   : np.random.Generator (for reproducibility)

    Returns:
        thetas, omegas : np.ndarray of shape (n,) each
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    thetas, omegas = [], []
    while len(thetas) < n:
        theta = rng.uniform(-np.pi * 0.9, np.pi * 0.9)
        E     = rng.uniform(e_min, e_max)
        val   = 2.0 * (E + np.cos(theta))
        if val < 0:
            continue  # unphysical combination — skip
        omega = rng.choice([-1.0, 1.0]) * np.sqrt(val)
        thetas.append(theta)
        omegas.append(omega)

    return np.array(thetas), np.array(omegas)


# ─────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────

def build_trajectories(n_traj, n_steps=N_STEPS, dt=DT_PHYS, seed=SEED):
    """
    Generate a dataset of pendulum trajectories.

    Returns encoded trajectories (sin θ, cos θ, ω) — not raw (θ, ω).
    Raw trajectories are also returned for evaluation purposes.

    Args:
        n_traj  : number of trajectories
        n_steps : physical time steps per trajectory
        dt      : physical time step
        seed    : random seed

    Returns:
        trajs_enc : np.ndarray, shape (n_traj, n_steps, 3)  ← network input
        trajs_raw : np.ndarray, shape (n_traj, n_steps, 2)  ← for evaluation
        theta0s   : np.ndarray, shape (n_traj,)
        omega0s   : np.ndarray, shape (n_traj,)
    """
    rng = np.random.default_rng(seed)
    theta0s, omega0s = sample_initial_conditions(n_traj, rng=rng)

    # Hard assertion — no rotating trajectories allowed
    H0 = hamiltonian(theta0s, omega0s)
    assert (H0 < 1.0).all(), (
        f"Rotating trajectories detected (E >= 1.0)! "
        f"Max energy: {H0.max():.3f}. Set E_MAX < 1.0."
    )

    trajs_raw = np.stack([
        generate_trajectory(th, om, n_steps, dt)
        for th, om in zip(theta0s, omega0s)
    ])  # (n_traj, n_steps, 2)

    trajs_enc = encode_trajectory(trajs_raw)  # (n_traj, n_steps, 3)

    return trajs_enc, trajs_raw, theta0s, omega0s


def get_dataloaders(n_train=N_TRAIN, n_test=N_TEST, batch_size=BATCH_SIZE):
    """
    Build train/test trajectories and return DataLoaders.

    Returns:
        train_loader  : DataLoader  — batches of shape (B, N_STEPS, 3)
        train_raw     : np.ndarray  — shape (n_train, N_STEPS, 2)  raw (θ, ω)
        test_raw      : np.ndarray  — shape (n_test,  N_STEPS, 2)  raw (θ, ω)
    """
    print("Generating training data (Störmer-Verlet, oscillating only)...")
    train_enc, train_raw, _, _ = build_trajectories(n_train, seed=SEED)
    test_enc,  test_raw,  _, _ = build_trajectories(n_test,  seed=SEED + 1)

    # Sanity check: energy conservation in training data
    H0 = hamiltonian(train_raw[:, 0, 0],  train_raw[:, 0, 1])
    HT = hamiltonian(train_raw[:, -1, 0], train_raw[:, -1, 1])
    print(f"  Train : {train_enc.shape}  (encoded: sin θ, cos θ, ω)")
    print(f"  Test  : {test_enc.shape}")
    print(f"  Energy range  : [{H0.min():.3f}, {H0.max():.3f}]  (all < 1.0 ✓)")
    print(f"  mean |ΔH|     : {np.mean(np.abs(HT - H0)):.2e}  (Störmer-Verlet drift)")

    # DataLoader uses encoded trajectories
    train_tensor = torch.tensor(train_enc, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, train_raw, test_raw


if __name__ == "__main__":
    loader, train_raw, test_raw = get_dataloaders()
    batch = next(iter(loader))[0]
    print(f"\nBatch shape    : {batch.shape}  (B, N_STEPS, STATE_DIM)")
    print(f"Batch mean     : {batch.mean():.4f}")
    print(f"Batch std      : {batch.std():.4f}")
    print(f"Batch range    : [{batch.min():.3f}, {batch.max():.3f}]")
    print("data.py OK")
