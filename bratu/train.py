"""
train.py  --  Bratu Training and Sampling
------------------------------------------
Case B0: vanilla FM, no conditioning
Case B1: FiLM on C only -- must learn bimodal p(u|C) from data
Case B2: B1 model + PCFM projection at t=1 (no retraining)
Case B3: B1 model evaluated at C = Cc (critical, unique solution)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.linalg import solve

from config import (
    N_X, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED,
    C_CRIT, U_CRIT, N_PROJ_STEPS, PROJ_TOL, PROJ_LAMBDA
)
from data import (
    make_grid, exact_solution,
    _bratu_residual, _bratu_jacobian, _nsfd_denominator
)


# ── FM loss ───────────────────────────────────────────────────────────────────

def fm_loss_step(model, u1, C, device, model_type="B0"):
    B     = u1.shape[0]
    u0    = torch.randn_like(u1)
    t     = torch.rand(B, device=device)
    u_t   = (1.0 - t.view(B,1)) * u0 + t.view(B,1) * u1
    v_tgt = u1 - u0
    if model_type == "B0":
        v_pred = model(u_t, t)
    else:
        v_pred = model(u_t, t, C)
    return nn.functional.mse_loss(v_pred, v_tgt)


# ── Training loop (shared B0 / B1) ────────────────────────────────────────────

def train(model, train_loader, device, model_type="B0",
          n_epochs=N_EPOCHS, lr=LR,
          save_dir="checkpoints", save_name="best.pt"):

    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    history   = []
    best_loss = float("inf")

    labels = {"B0": "Vanilla FM (B0)", "B1": "FiLM on C (B1)"}
    print(f"\n{'='*55}\nTraining {labels.get(model_type, model_type)}")
    print(f"  {n_epochs} epochs | lr={lr} | device={device}\n{'='*55}")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'LR':>8}")
    print("─" * 32)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            u1_n, C_b, b_b, _ = batch     # b_b loaded but not used
            u1_n = u1_n.to(device)
            C_b  = C_b.to(device)
            optimizer.zero_grad()
            loss = fm_loss_step(model, u1_n, C_b, device, model_type)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        avg = ep_loss / len(train_loader)
        history.append(avg)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), Path(save_dir) / save_name)
        if epoch % 100 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg:>12.6f}  "
                  f"{scheduler.get_last_lr()[0]:>8.2e}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    return history


# ── Sampling: B0 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_B0(model, n_samples, device, u_mean, u_std,
              n_fm_steps=N_FM_STEPS, seed=SEED+99):
    """Sample from vanilla FM. No conditioning."""
    torch.manual_seed(seed)
    model.eval()
    dt = 1.0 / n_fm_steps
    u  = torch.randn(n_samples, N_X, device=device)
    for k in range(n_fm_steps):
        t_val = torch.full((n_samples,), k * dt, device=device)
        u = u + dt * model(u, t_val)
    return u.cpu().numpy() * u_std + u_mean


# ── Sampling: B1 / B3 ────────────────────────────────────────────────────────

@torch.no_grad()
def sample_B1(model, C_vals, device, u_mean, u_std,
              n_fm_steps=N_FM_STEPS, seed=SEED+99):
    """
    Sample from FiLM model conditioned on C only.
    Different noise realisations at the same C should map to different
    branches if the model has learned the bimodal structure.

    Args:
        C_vals : np.ndarray (n,)
    Returns:
        u_gen  : np.ndarray (n, N_X) un-normalised
    """
    torch.manual_seed(seed)
    model.eval()
    dt  = 1.0 / n_fm_steps
    n   = len(C_vals)
    C_t = torch.tensor(C_vals, dtype=torch.float32, device=device)
    u   = torch.randn(n, N_X, device=device)
    for k in range(n_fm_steps):
        t_val = torch.full((n,), k * dt, device=device)
        u = u + dt * model(u, t_val, C_t)
    return u.cpu().numpy() * u_std + u_mean


# ── PCFM Projection: B2 ───────────────────────────────────────────────────────

def _pde_res_and_jac(u, C):
    _, h = make_grid()
    hs   = _nsfd_denominator(h)
    return _bratu_residual(u, C, hs), _bratu_jacobian(u, C, hs)


def pcfm_project(u, C, n_steps=N_PROJ_STEPS, tol=PROJ_TOL, lam=PROJ_LAMBDA):
    """
    Gauss-Newton projection onto h(u) = u_xx + C*exp(u) = 0.
    u <- u - J^T (J J^T + lam I)^{-1} h(u)
    """
    u = u.copy()
    h_norms = []
    for _ in range(n_steps):
        F, J = _pde_res_and_jac(u, C)
        h_norms.append(np.linalg.norm(F))
        if h_norms[-1] < tol:
            break
        M = J @ J.T + lam * np.eye(len(F))
        try:
            delta_h = solve(M, F, assume_a='pos')
        except Exception:
            break
        u = u - J.T @ delta_h
    F_final, _ = _pde_res_and_jac(u, C)
    return u, {
        "h_norm_initial": h_norms[0] if h_norms else np.nan,
        "h_norm_final":   np.linalg.norm(F_final),
        "n_iter":         len(h_norms),
        "converged":      np.linalg.norm(F_final) < tol * 100,
    }


def sample_B2(model, C_vals, device, u_mean, u_std,
              n_fm_steps=N_FM_STEPS, seed=SEED+99):
    """
    B1 sampling followed by PCFM projection. No retraining.
    Returns un-normalised (u_unc, u_proj, infos).
    """
    u_unc = sample_B1(model, C_vals, device, u_mean, u_std,
                      n_fm_steps, seed)
    u_proj = np.zeros_like(u_unc)
    infos  = []
    print(f"\nApplying PCFM projection to {len(C_vals)} samples...")
    n_conv = 0
    for i in range(len(C_vals)):
        u_p, info = pcfm_project(u_unc[i], float(C_vals[i]))
        u_proj[i] = u_p
        infos.append(info)
        if info["converged"]: n_conv += 1
    print(f"  Converged: {n_conv}/{len(C_vals)} "
          f"({100*n_conv/len(C_vals):.1f}%)")
    print(f"  Mean ||h|| before: "
          f"{np.mean([d['h_norm_initial'] for d in infos]):.4e}")
    print(f"  Mean ||h|| after:  "
          f"{np.mean([d['h_norm_final']   for d in infos]):.4e}")
    return u_unc, u_proj, infos


# ── Critical case: B3 ────────────────────────────────────────────────────────

def sample_B3_critical(model, device, u_mean, u_std,
                       n_samples=200, n_fm_steps=N_FM_STEPS, seed=SEED+99):
    """
    Evaluate B1 model at C = Cc with many noise realisations.
    Since only one solution exists at Cc, variance across samples should
    be near zero if the model is well-calibrated.
    """
    C_vals  = np.full(n_samples, C_CRIT, dtype=np.float32)
    u_gen   = sample_B1(model, C_vals, device, u_mean, u_std,
                        n_fm_steps, seed)
    u_exact = exact_solution(C_CRIT, branch=0)
    return u_gen, u_exact, C_vals