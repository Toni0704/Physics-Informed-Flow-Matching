"""
train.py  —  Case 1: Physics Enforced in Training  (CORRECTED)
--------------------------------------------------------------

=======================================================================
WHAT WAS WRONG IN THE PREVIOUS VERSION
=======================================================================

The previous implementation applied the physics loss to:

    u1_pred = u_t + (1-t) * v_pred

This is the model's single-step prediction of the final trajectory,
starting from a random interpolated point u_t at a random FM time t.

This has the SAME fundamental problem as the wrong Case 2 projection:
the physics penalty is applied to an object that does NOT correspond
to what the model actually generates at inference.

At inference the generated trajectory is produced by chaining
N_FM_STEPS Euler steps:

    u0 -> u_dt -> u_2dt -> ... -> u1

Each step uses the model's velocity field. But u1_pred from a single
FM step is a one-shot prediction from a random t — a completely
different object with different structure.

The training signal therefore never matched the inference procedure.
This is why energy conservation did not improve regardless of lambda.

=======================================================================
THE CORRECT APPROACH: UNROLL THE FM ODE DURING TRAINING
=======================================================================

The physics loss must be applied to the ACTUAL generated trajectory —
the one produced by running the full Euler integration chain.

Correct training step:

    Step 1: FM loss (standard, unchanged)
        u0, u1, t -> u_t -> v_pred -> L_FM = ||v_pred - (u1-u0)||^2

    Step 2: Physics loss on actual rollout
        u = u0_fresh ~ N(0, I)
        for step in 0..N_FM_STEPS_TRAIN:
            u = u + dt * v_theta(u, t)     <- WITH gradient tracking
        L_physics = Var_tau[H(theta(tau), omega(tau))]  on u

    Step 3: Combined
        L = L_FM + lambda * L_physics
        loss.backward()   <- gradients flow through both terms

This ensures the physics signal matches exactly what the model produces
at inference. The rollout uses N_FM_STEPS_TRAIN=20 steps (fewer than
inference=200) for computational tractability.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from Simple_pendulum.Pendulum_training_constraint.config import (
    N_STEPS, STATE_DIM, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED
)
from Simple_pendulum.Pendulum_training_constraint.model import VelocityFieldMLP

# Number of Euler steps to unroll for physics loss during training.
# Much fewer than inference (200) for speed while still capturing drift.
N_FM_STEPS_TRAIN = 20


# ── Differentiable Hamiltonian ────────────────────────────────────────────────

def hamiltonian_torch(theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return 0.5 * omega**2 - torch.cos(theta)


# ── Physics loss on actual rollout ────────────────────────────────────────────

def physics_loss_rollout(
    model:      VelocityFieldMLP,
    batch_size: int,
    device:     str,
    n_steps:    int = N_FM_STEPS_TRAIN,
) -> torch.Tensor:
    """
    Compute energy conservation penalty on the ACTUAL generated trajectory.

    Runs the FM Euler chain WITH gradient tracking, then penalises
    energy variance across physical time steps tau.

    This is the corrected approach — physics penalty on the same object
    the model produces at inference, not on an intermediate FM state.

    Args:
        model      : VelocityFieldMLP in train mode
        batch_size : rollout batch size
        device     : torch device
        n_steps    : Euler steps to unroll (20 is sufficient)

    Returns:
        l_phys : scalar — mean Var_tau[H] across rollout batch
    """
    dt_fm = 1.0 / n_steps

    # Start from Gaussian noise — identical to inference start
    # No torch.no_grad() — we need gradients through this rollout
    u = torch.randn(batch_size, N_STEPS, STATE_DIM, device=device)

    # Euler chain — gradients tracked through every step
    for step in range(n_steps):
        t_val = torch.full((batch_size,), step * dt_fm, device=device)
        v     = model(u, t_val)
        u     = u + dt_fm * v         # += would break autograd

    # u is now the generated trajectory — (B, N_STEPS, 3)
    # Decode (sin theta, cos theta, omega) -> (theta, omega)
    sin_th = u[:, :, 0]
    cos_th = u[:, :, 1]
    omega  = u[:, :, 2]
    theta  = torch.atan2(sin_th, cos_th)   # differentiable

    H = hamiltonian_torch(theta, omega)    # (B, N_STEPS)
    return H.var(dim=1).mean()             # Var_tau[H], mean over batch


# ── Standard FM loss ──────────────────────────────────────────────────────────

def fm_loss_step(
    model:    VelocityFieldMLP,
    u1_batch: torch.Tensor,
    device:   str,
) -> torch.Tensor:
    B     = u1_batch.shape[0]
    u0    = torch.randn_like(u1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    u_t   = (1.0 - t_exp) * u0 + t_exp * u1_batch
    v_target = u1_batch - u0
    v_pred   = model(u_t, t)
    return nn.functional.mse_loss(v_pred, v_target)


# ── Combined loss ─────────────────────────────────────────────────────────────

def combined_loss(
    model:    VelocityFieldMLP,
    u1_batch: torch.Tensor,
    device:   str,
    lam:      float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_FM  +  lambda * L_physics

    L_FM     : standard FM regression (single-step interpolation)
    L_physics: energy variance of ACTUAL rollout (corrected)

    When lambda=0 the rollout is skipped entirely for speed.
    """
    l_fm = fm_loss_step(model, u1_batch, device)

    if lam == 0.0:
        l_phys = torch.tensor(0.0, device=device)
    else:
        l_phys = physics_loss_rollout(model, u1_batch.shape[0], device)

    return l_fm + lam * l_phys, l_fm, l_phys


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model:        VelocityFieldMLP,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    lam:          float = 1.0,
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str   = "checkpoints",
) -> dict:
    torch.manual_seed(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    history   = {"total": [], "fm": [], "physics": []}
    best_loss = float("inf")

    note = f"rollout {N_FM_STEPS_TRAIN} steps" if lam > 0 else "no rollout"
    print(f"\nTraining lambda={lam}  [{note}]  ({n_epochs} epochs on {device})")
    print(f"{'Epoch':>6}  {'Total':>10}  {'FM':>10}  {'Physics':>10}  {'LR':>8}")
    print("─" * 52)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_total = epoch_fm = epoch_phys = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)
            optimizer.zero_grad()

            loss, l_fm, l_phys = combined_loss(model, u1_batch, device, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_total += loss.item()
            epoch_fm    += l_fm.item()
            epoch_phys  += l_phys.item()

        scheduler.step()
        n = len(train_loader)
        history["total"].append(epoch_total / n)
        history["fm"].append(epoch_fm / n)
        history["physics"].append(epoch_phys / n)

        if epoch_total / n < best_loss:
            best_loss = epoch_total / n
            torch.save(
                model.state_dict(),
                Path(save_dir) / f"best_model_lam{lam:.1f}.pt"
            )

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  "
                  f"{epoch_total/n:>10.5f}  "
                  f"{epoch_fm/n:>10.5f}  "
                  f"{epoch_phys/n:>10.5f}  "
                  f"{scheduler.get_last_lr()[0]:>8.1e}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    return history


# ── Sampling — unchanged from Case 3 ─────────────────────────────────────────

@torch.no_grad()
def sample(
    model:      VelocityFieldMLP,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    from Simple_pendulum.Pendulum_training_constraint.data import decode_trajectory
    torch.manual_seed(seed)
    model.eval()
    dt_fm = 1.0 / n_fm_steps

    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val)
        u     = u + dt_fm * v

    return decode_trajectory(u.cpu().numpy())