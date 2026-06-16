"""
train.py  --  Case 5: Extended Experiments
-------------------------------------------
Training loops for all three new variants. All use the corrected
rollout-based physics loss (applied to the actual generated trajectory).

Variant A: Adaptive Lambda + physics loss
    - Standard MLP (no FiLM)
    - Lambda network outputs per-sample lambda(ic)
    - Physics loss: (1/K) sum (H_i - H_0)^2 weighted by lambda(ic)
    - Lambda network is trained jointly with the velocity field

Variant B: IC conditioning + fixed lambda physics loss
    - FiLM conditioning on (sin theta0, cos theta0, omega0)
    - Same rollout physics loss as corrected Case 1
    - Fixed lambda from config

Variant C: IC conditioning + Hamiltonian enforcement
    - Same as B but stronger physics loss:
      L_phys = (1/K) sum (H_i - H_0)^2 + alpha * (mean(H_i) - H_0)^2
      The second term penalises mean energy level deviation, not just drift.
      This is more aggressive: it also pushes the mean energy to H_0.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import N_STEPS, STATE_DIM, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED, N_FM_STEPS_TRAIN, LAMBDA_BASE, ALPHA_C
from data   import decode_trajectory
from model  import (
    VelocityFieldMLP_AdaptiveLambda,
    VelocityFieldFiLM_IC_Physics,
)



# ── Differentiable Hamiltonian ────────────────────────────────────────────────

def hamiltonian_torch(theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return 0.5 * omega**2 - torch.cos(theta)


# ── IC extraction helper ──────────────────────────────────────────────────────

def extract_ic(u1: torch.Tensor) -> torch.Tensor:
    """Extract encoded IC from first time step of real trajectory batch."""
    return u1[:, 0, :]   # (B, 3) = (sin theta0, cos theta0, omega0)


# ── Physics loss on rollout ───────────────────────────────────────────────────

def rollout(model, u0: torch.Tensor, ic=None, device: str = "cpu",
            n_steps: int = N_FM_STEPS_TRAIN):
    """
    Run FM Euler integration for n_steps WITH gradient tracking.
    Returns the generated trajectory u_K shape (B, N_STEPS, STATE_DIM).

    ic is required for FiLM-conditioned models (Variants B, C).
    ic is None for standard MLP (Variant A).
    """
    dt = 1.0 / n_steps
    u  = u0
    for k in range(n_steps):
        t_val = torch.full((u.shape[0],), k * dt, device=device)
        if ic is not None:
            v = model(u, t_val, ic)
        else:
            v = model(u, t_val)
        u = u + dt * v
    return u   # (B, N_STEPS, STATE_DIM)


def physics_loss_from_traj(u_K: torch.Tensor, variant: str = "B") -> torch.Tensor:
    """
    Compute physics loss on the actual rollout output u_K.

    Variant B: L = (1/K) sum (H_i - H_0)^2
    Variant C: L = (1/K) sum (H_i - H_0)^2 + alpha * (mean_H - H_0)^2
    """
    sin_th = u_K[:, :, 0]
    cos_th = u_K[:, :, 1]
    omega  = u_K[:, :, 2]
    theta  = torch.atan2(sin_th, cos_th)

    H  = hamiltonian_torch(theta, omega)           # (B, N_STEPS)
    H0 = H[:, 0:1]                                 # (B, 1) initial energy

    drift_loss = ((H - H0) ** 2).mean(dim=1).mean()   # mean over steps and batch

    if variant == "C":
        mean_H     = H.mean(dim=1, keepdim=True)   # (B, 1)
        level_loss = ((mean_H - H0) ** 2).mean()
        return drift_loss + ALPHA_C * level_loss
    else:
        return drift_loss


# ── Standard FM loss ──────────────────────────────────────────────────────────

def fm_loss_step(model, u1: torch.Tensor, device: str, ic=None) -> torch.Tensor:
    B     = u1.shape[0]
    u0    = torch.randn_like(u1)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    u_t   = (1.0 - t_exp) * u0 + t_exp * u1
    v_target = u1 - u0
    if ic is not None:
        v_pred = model(u_t, t, ic)
    else:
        v_pred = model(u_t, t)
    return nn.functional.mse_loss(v_pred, v_target)


# ══════════════════════════════════════════════════════════════════════════════
# Variant A training: Adaptive Lambda
# ══════════════════════════════════════════════════════════════════════════════

def train_A(
    model:        VelocityFieldMLP_AdaptiveLambda,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str   = "checkpoints",
) -> dict:
    """
    Adaptive lambda training:
    - lambda(ic) = learned positive scalar, different per sample
    - L = L_FM + lambda(ic) * L_phys(rollout)
    - Lambda network trained jointly
    """
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    history   = {"total": [], "fm": [], "physics": [], "lambda_mean": []}
    best_loss = float("inf")

    print(f"\nVariant A — Adaptive Lambda (rollout K={N_FM_STEPS_TRAIN}, {n_epochs} epochs)")
    print(f"{'Epoch':>6}  {'Total':>10}  {'FM':>10}  {'Physics':>10}  {'lam_mean':>10}")
    print("─" * 56)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_total = ep_fm = ep_phys = ep_lam = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)
            B = u1_batch.shape[0]
            ic = extract_ic(u1_batch)              # (B, 3)

            optimizer.zero_grad()

            # FM loss
            l_fm = fm_loss_step(model, u1_batch, device, ic=None)

            # Adaptive lambda
            lam = model.get_lambda(ic)             # (B,) -- per sample, gradient flows

            # Rollout physics loss
            u0_fresh = torch.randn_like(u1_batch)
            u_K      = rollout(model, u0_fresh, ic=None, device=device)
            l_phys   = physics_loss_from_traj(u_K, variant="B")

            # Weighted combination: mean over batch of lambda_i * L_phys_i
            # For simplicity use mean(lambda) as scalar weight
            lam_scalar = lam.mean()
            loss = l_fm + lam_scalar * l_phys

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            ep_total += loss.item()
            ep_fm    += l_fm.item()
            ep_phys  += l_phys.item()
            ep_lam   += lam_scalar.item()

        scheduler.step()
        n = len(train_loader)
        history["total"].append(ep_total / n)
        history["fm"].append(ep_fm / n)
        history["physics"].append(ep_phys / n)
        history["lambda_mean"].append(ep_lam / n)

        if ep_total / n < best_loss:
            best_loss = ep_total / n
            torch.save(model.state_dict(), Path(save_dir) / "best_A.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {ep_total/n:>10.5f}  {ep_fm/n:>10.5f}  "
                  f"{ep_phys/n:>10.5f}  {ep_lam/n:>10.4f}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Variants B and C training: FiLM IC + physics loss
# ══════════════════════════════════════════════════════════════════════════════

def train_BC(
    model:        VelocityFieldFiLM_IC_Physics,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    lam:          float = LAMBDA_BASE,
    variant:      str   = "B",     # "B" or "C"
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str   = "checkpoints",
) -> dict:
    """
    FiLM IC conditioning + rollout physics loss.
    Variant B: L = L_FM(ic) + lambda * (1/K) sum (H_i - H_0)^2
    Variant C: L = L_FM(ic) + lambda * [(1/K) sum (H_i-H_0)^2
                                        + alpha * (mean_H - H_0)^2]
    """
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    history   = {"total": [], "fm": [], "physics": []}
    best_loss = float("inf")

    vname = f"Variant {variant}"
    extra = f"(+ alpha={ALPHA_C} mean-energy penalty)" if variant == "C" else ""
    print(f"\n{vname} — FiLM IC + physics loss {extra}")
    print(f"  lambda={lam}, rollout K={N_FM_STEPS_TRAIN}, {n_epochs} epochs on {device}")
    print(f"{'Epoch':>6}  {'Total':>10}  {'FM':>10}  {'Physics':>10}  {'LR':>8}")
    print("─" * 52)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_total = ep_fm = ep_phys = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)
            ic = extract_ic(u1_batch)              # (B, 3)

            optimizer.zero_grad()

            # FM loss with IC conditioning
            l_fm = fm_loss_step(model, u1_batch, device, ic=ic)

            # Rollout physics loss (conditioned on IC)
            u0_fresh = torch.randn_like(u1_batch)
            u_K      = rollout(model, u0_fresh, ic=ic, device=device)
            l_phys   = physics_loss_from_traj(u_K, variant=variant)

            loss = l_fm + lam * l_phys

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            ep_total += loss.item()
            ep_fm    += l_fm.item()
            ep_phys  += l_phys.item()

        scheduler.step()
        n = len(train_loader)
        history["total"].append(ep_total / n)
        history["fm"].append(ep_fm / n)
        history["physics"].append(ep_phys / n)

        if ep_total / n < best_loss:
            best_loss = ep_total / n
            torch.save(model.state_dict(),
                       Path(save_dir) / f"best_{variant}.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {ep_total/n:>10.5f}  {ep_fm/n:>10.5f}  "
                  f"{ep_phys/n:>10.5f}  {scheduler.get_last_lr()[0]:>8.1e}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    return history


# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_A(model, n_samples: int, device: str,
             n_fm_steps: int = N_FM_STEPS, seed: int = SEED + 99) -> np.ndarray:
    """Plain Euler FM for Variant A (no IC at inference)."""
    from data import decode_trajectory
    torch.manual_seed(seed)
    model.eval()
    dt = 1.0 / n_fm_steps
    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for k in range(n_fm_steps):
        t_val = torch.full((n_samples,), k * dt, device=device)
        v     = model(u, t_val)
        u     = u + dt * v
    return decode_trajectory(u.cpu().numpy())


@torch.no_grad()
def sample_BC(model, n_samples: int, device: str,
              n_fm_steps: int = N_FM_STEPS, seed: int = SEED + 99,
              ic_raw: np.ndarray = None) -> tuple:
    """
    FM with IC conditioning for Variants B and C.

    If ic_raw is None: sample random ICs from training distribution (generative mode).
    If ic_raw is provided: use exact ICs (surrogate mode).

    Returns:
        generated_raw : np.ndarray (n, N_STEPS, 2)
        ic_used       : np.ndarray (n, 2)   [theta0, omega0] used
    """
    from config import E_MIN, E_MAX
    from data   import decode_trajectory, sample_initial_conditions, hamiltonian

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval()
    dt = 1.0 / n_fm_steps

    if ic_raw is None:
        # Random ICs from training distribution
        th0s, om0s = sample_initial_conditions(n_samples, E_MIN, E_MAX, rng)
        ic_used = np.stack([th0s, om0s], axis=-1)
    else:
        th0s = ic_raw[:, 0]
        om0s = ic_raw[:, 1]
        ic_used = ic_raw

    # Encode IC
    ic_enc = np.stack([np.sin(th0s), np.cos(th0s), om0s], axis=-1)
    ic_t   = torch.tensor(ic_enc, dtype=torch.float32, device=device)

    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for k in range(n_fm_steps):
        t_val = torch.full((n_samples,), k * dt, device=device)
        v     = model(u, t_val, ic_t)
        u     = u + dt * v

    return decode_trajectory(u.cpu().numpy()), ic_used


# ══════════════════════════════════════════════════════════════════════════════
# Variant C (CORRECTED): FiLM on IC + H0, standard FM loss only
# ══════════════════════════════════════════════════════════════════════════════

def train_C(
    model,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str   = "checkpoints",
) -> dict:
    """
    Variant C (corrected): FiLM conditioning on IC + H0.
    No physics loss. Standard FM loss only.

    The model receives BOTH:
      - ic = (sin theta0, cos theta0, omega0)  -- phase information
      - H0 = H(theta0, omega0)                 -- energy level

    IC already implies H0, but providing both explicitly gives the model
    redundant signals that reinforce each other. This tests whether
    over-specifying the conditioning (complete IC + explicit energy)
    helps beyond IC alone (Case 4b).
    """
    from data import hamiltonian
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    history   = {"total": [], "fm": []}
    best_loss = float("inf")

    print(f"\nVariant C (corrected) -- FiLM on IC + H0, no physics loss")
    print(f"  {n_epochs} epochs on {device}")
    print(f"{'Epoch':>6}  {'FM Loss':>12}  {'LR':>8}")
    print("─" * 32)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_fm = 0.0

        for (u1_batch,) in train_loader:
            u1_batch = u1_batch.to(device)
            B = u1_batch.shape[0]

            # Extract IC and H0 from the real trajectory
            ic  = u1_batch[:, 0, :]                 # (B, 3): sin t0, cos t0, om0
            th0 = torch.atan2(ic[:, 0], ic[:, 1])   # (B,)
            om0 = ic[:, 2]                           # (B,)
            H0  = 0.5 * om0**2 - torch.cos(th0)     # (B,)

            optimizer.zero_grad()

            # Standard FM loss with IC + H0 conditioning
            u0    = torch.randn_like(u1_batch)
            t     = torch.rand(B, device=device)
            t_exp = t.view(B, 1, 1)
            u_t   = (1.0 - t_exp) * u0 + t_exp * u1_batch
            v_target = u1_batch - u0
            v_pred   = model(u_t, t, ic, H0)
            loss     = torch.nn.functional.mse_loss(v_pred, v_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_fm += loss.item()

        scheduler.step()
        n = len(train_loader)
        history["total"].append(ep_fm / n)
        history["fm"].append(ep_fm / n)

        if ep_fm / n < best_loss:
            best_loss = ep_fm / n
            torch.save(model.state_dict(), Path(save_dir) / "best_C_corrected.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {ep_fm/n:>12.6f}  "
                  f"{scheduler.get_last_lr()[0]:>8.1e}")

    print(f"\nDone. Best FM loss: {best_loss:.6f}")
    return history


@torch.no_grad()
def sample_C(model, n_samples: int, device: str,
             n_fm_steps: int = N_FM_STEPS, seed: int = SEED + 99,
             ic_raw: np.ndarray = None) -> tuple:
    """
    Sample from Variant C: condition on random IC + corresponding H0.

    If ic_raw is None: sample random ICs (generative mode).
    If ic_raw is provided: use exact test ICs (surrogate mode).
    """
    from config import E_MIN, E_MAX
    from data   import decode_trajectory, sample_initial_conditions

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval()
    dt = 1.0 / n_fm_steps

    if ic_raw is None:
        th0s, om0s = sample_initial_conditions(n_samples, E_MIN, E_MAX, rng)
        ic_used = np.stack([th0s, om0s], axis=-1)
    else:
        th0s = ic_raw[:, 0]; om0s = ic_raw[:, 1]
        ic_used = ic_raw

    H0_vals = 0.5 * om0s**2 - np.cos(th0s)

    ic_enc = np.stack([np.sin(th0s), np.cos(th0s), om0s], axis=-1)
    ic_t   = torch.tensor(ic_enc,   dtype=torch.float32, device=device)
    H0_t   = torch.tensor(H0_vals,  dtype=torch.float32, device=device)

    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for k in range(n_fm_steps):
        t_val = torch.full((n_samples,), k * dt, device=device)
        v     = model(u, t_val, ic_t, H0_t)
        u     = u + dt * v

    return decode_trajectory(u.cpu().numpy()), ic_used
