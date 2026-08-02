"""
train.py
--------
Flow Matching training loop and inference (sampling) for Case 3.

Training:
    Standard FM loss — no physics anywhere.
    The only physics is in the training data (symplectic integrator).

    Loss = E_{t, u0, u1} [ || vθ(u_t, t) - (u1 - u0) ||² ]

    where:
        u0     ~ N(0, I)          noise trajectory
        u1     ~ p_data           real pendulum trajectory
        t      ~ Uniform(0, 1)    FM time
        u_t    = (1-t)*u0 + t*u1  straight-line interpolation
        u1-u0                     target velocity (constant along path)

Sampling:
    Euler integration of the learned ODE from t=0 to t=1:
        u_{t+Δt} = u_t + Δt * vθ(u_t, t)
    Starting from Gaussian noise, ending at a generated trajectory.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from Simple_pendulum.Pendulum_inference_constraint.config import (
    N_STEPS, BATCH_SIZE, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED
)
from Simple_pendulum.Pendulum_inference_constraint.model import VelocityFieldMLP


# ─────────────────────────────────────────────────────────────
# FM Loss
# ─────────────────────────────────────────────────────────────

def fm_loss(
    model:    VelocityFieldMLP,
    u1_batch: torch.Tensor,
    device:   str,
) -> torch.Tensor:
    """
    Compute the Flow Matching loss for one batch of real trajectories.

    Args:
        model    : velocity field network
        u1_batch : real trajectories, shape (B, N_STEPS, 2)
        device   : torch device string

    Returns:
        loss : scalar tensor
    """
    B = u1_batch.shape[0]

    # ── Step 1: sample noise trajectory ────────────────────────
    u0 = torch.randn_like(u1_batch)                  # (B, N, 2)

    # ── Step 2: sample FM time ─────────────────────────────────
    t  = torch.rand(B, device=device)                # (B,)

    # ── Step 3: interpolate ────────────────────────────────────
    # u_t = (1-t)*u0 + t*u1  (straight-line path in traj space)
    t_exp = t.view(B, 1, 1)                          # broadcast over (N, 2)
    u_t   = (1.0 - t_exp) * u0 + t_exp * u1_batch   # (B, N, 2)

    # ── Step 4: target velocity ────────────────────────────────
    # Derivative of the interpolation path w.r.t. t is constant:
    #   d/dt [(1-t)*u0 + t*u1] = u1 - u0
    v_target = u1_batch - u0                         # (B, N, 2)

    # ── Step 5: predicted velocity ─────────────────────────────
    v_pred = model(u_t, t)                           # (B, N, 2)

    return nn.functional.mse_loss(v_pred, v_target)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(
    model:        VelocityFieldMLP,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    n_epochs:     int = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str = "checkpoints",
) -> list[float]:
    """
    Train the velocity field model using the FM loss.

    Args:
        model        : VelocityFieldMLP
        train_loader : DataLoader yielding batches of shape (B, N_STEPS, 2)
        device       : torch device string
        n_epochs     : number of training epochs
        lr           : learning rate
        save_dir     : directory to save best checkpoint

    Returns:
        losses : list of per-epoch average losses
    """
    torch.manual_seed(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    losses    = []
    best_loss = float("inf")

    print(f"\nTraining for {n_epochs} epochs on {device}...")
    print(f"{'Epoch':>6}  {'FM Loss':>12}  {'LR':>10}")
    print("─" * 35)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)

            optimizer.zero_grad()
            loss = fm_loss(model, u1_batch, device)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path / "best_model.pt")

        if epoch % 50 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6}  {avg_loss:>12.6f}  {current_lr:>10.2e}")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    return losses


# ─────────────────────────────────────────────────────────────
# Sampling (inference)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model:      VelocityFieldMLP,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    """
    Generate pendulum trajectories by integrating the learned ODE
    from FM time t=0 (noise) to t=1 (data).

    The model operates in encoded space (sin θ, cos θ, ω).
    Output is decoded back to raw (θ, ω) for evaluation.

    Returns:
        generated_raw : np.ndarray, shape (n_samples, N_STEPS, 2)
                        columns [θ, ω] recovered via arctan2
    """
    from Simple_pendulum.Pendulum_inference_constraint.data import decode_trajectory
    from Simple_pendulum.Pendulum_inference_constraint.config import N_STEPS, STATE_DIM

    torch.manual_seed(seed)
    model.eval()

    dt_fm = 1.0 / n_fm_steps

    # Start from Gaussian noise in encoded space
    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)

    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val)
        u     = u + dt_fm * v

    # Decode: (sin θ, cos θ, ω) → (θ, ω)
    u_np = u.cpu().numpy()
    return decode_trajectory(u_np)   # (n_samples, N_STEPS, 2)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone training run.
    Usage: python train.py
    Saves checkpoint to checkpoints/best_model.pt
    """
    import sys
    sys.path.insert(0, "..")

    from Simple_pendulum.Pendulum_inference_constraint.data  import get_dataloaders
    from Simple_pendulum.Pendulum_inference_constraint.model import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, train_trajs, test_trajs = get_dataloaders()

    # Model
    model = build_model(device)

    # Train
    losses = train(model, train_loader, device)

    # Quick sample check
    generated = sample(model, n_samples=10, device=device)
    print(f"\nSample shape: {generated.shape}")
    print("train.py OK — run eval.py for full evaluation.")
