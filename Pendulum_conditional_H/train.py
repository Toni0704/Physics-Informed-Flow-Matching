"""
train.py  —  Case 4: Conditional Generation
--------------------------------------------
Standard FM loss, but the model is conditioned on energy E:

    L = E_{t, u0, u1, E} [ || vθ(u_t, t, E) - (u1 - u0) ||² ]

where E = H(θ₀, ω₀) is the energy of the real trajectory u1.

At training time:
    For each real trajectory u1, compute its energy E from the
    initial state, and pass E as conditioning to the model.
    The model learns: "given I want energy E, how should I
    move this noisy trajectory toward a real one?"

At inference time:
    Specify a desired energy E* and sample:
        u₀ ~ N(0, I)
        Integrate: u_{t+Δt} = u_t + Δt * vθ(u_t, t, E*)
    The model should generate a trajectory at energy level E*.

No physics penalty. No projection.
The constraint is enforced purely through conditioning.

Key question:
    Does conditioning on E cause the model to implicitly learn
    the energy manifold geometry? Or does it just learn to
    generate trajectories "near" the right energy level?

    We test by conditioning on E values from the training
    distribution and measuring actual energy conservation
    in the generated trajectories.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import N_STEPS, STATE_DIM, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED
from data   import hamiltonian, encode_trajectory, decode_trajectory
from model  import ConditionalVelocityField


# ─────────────────────────────────────────────────────────────
# Conditional FM loss
# ─────────────────────────────────────────────────────────────

def conditional_fm_loss(
    model:    ConditionalVelocityField,
    u1_batch: torch.Tensor,
    device:   str,
) -> torch.Tensor:
    """
    FM loss conditioned on energy.

    Energy E is computed from the real trajectory u1_batch
    (using the initial state θ₀, ω₀) and passed to the model.

    Args:
        model    : ConditionalVelocityField
        u1_batch : real encoded trajectories, shape (B, N_STEPS, 3)
        device   : torch device

    Returns:
        loss : scalar
    """
    B = u1_batch.shape[0]

    # ── Compute energy of each real trajectory ─────────────────
    # Decode first time step to get (θ₀, ω₀)
    # u1_batch[:, 0, :] = [sin θ₀, cos θ₀, ω₀]
    sin_th0 = u1_batch[:, 0, 0]
    cos_th0 = u1_batch[:, 0, 1]
    omega0  = u1_batch[:, 0, 2]
    theta0  = torch.atan2(sin_th0, cos_th0)    # (B,)
    E       = 0.5 * omega0**2 - torch.cos(theta0)  # (B,)  Hamiltonian

    # ── Standard FM interpolation ──────────────────────────────
    u0       = torch.randn_like(u1_batch)
    t        = torch.rand(B, device=device)
    t_exp    = t.view(B, 1, 1)
    u_t      = (1.0 - t_exp) * u0 + t_exp * u1_batch
    v_target = u1_batch - u0

    # ── Conditional forward pass ───────────────────────────────
    v_pred = model(u_t, t, E)

    return nn.functional.mse_loss(v_pred, v_target)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(
    model:        ConditionalVelocityField,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    n_epochs:     int = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str = "checkpoints",
) -> list[float]:
    torch.manual_seed(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    losses    = []
    best_loss = float("inf")

    print(f"\nTraining conditional FM ({n_epochs} epochs on {device})...")
    print(f"{'Epoch':>6}  {'FM Loss':>12}  {'LR':>10}")
    print("─" * 35)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)
            optimizer.zero_grad()
            loss = conditional_fm_loss(model, u1_batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),
                       f"{save_dir}/best_model_case4.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg:>12.6f}  "
                  f"{scheduler.get_last_lr()[0]:>10.2e}")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    return losses


# ─────────────────────────────────────────────────────────────
# Conditional sampling
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_conditional(
    model:      ConditionalVelocityField,
    E_targets:  np.ndarray,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    """
    Generate trajectories conditioned on specific energy levels.

    Args:
        model      : trained ConditionalVelocityField
        E_targets  : np.ndarray (n,) — desired energy level per sample
        device     : torch device
        n_fm_steps : Euler integration steps
        seed       : random seed

    Returns:
        generated_raw : np.ndarray (n, N_STEPS, 2)  [θ, ω]
    """
    torch.manual_seed(seed)
    model.eval()
    n     = len(E_targets)
    dt_fm = 1.0 / n_fm_steps

    E_tensor = torch.tensor(E_targets, dtype=torch.float32, device=device)
    u = torch.randn(n, N_STEPS, STATE_DIM, device=device)

    for step in range(n_fm_steps):
        t_val = torch.full((n,), step * dt_fm, device=device)
        v     = model(u, t_val, E_tensor)
        u     = u + dt_fm * v

    return decode_trajectory(u.cpu().numpy())   # (n, N_STEPS, 2)


@torch.no_grad()
def sample_unconditional(
    model:      ConditionalVelocityField,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    """
    Sample without conditioning — use mean energy from training range.
    Useful as a sanity check: the model should still generate reasonable
    trajectories when given a typical energy value.
    """
    from config import E_MIN, E_MAX
    E_mean = np.full(n_samples, (E_MIN + E_MAX) / 2.0)
    return sample_conditional(model, E_mean, device, n_fm_steps, seed)
