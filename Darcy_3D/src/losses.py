"""
Loss functions for the Darcy_3D study -- same structure as Burgers_1D/NS_2D's
src/losses.py, adapted to a steady-state 3D field (B, nx, ny, nz) with
(permeability, BC) conditioning. There is no physical-time axis here, so
"unrolling" in pbfm_loss only ever means Euler-integrating the flow-matching
ODE in interpolation time (t -> 1), never stepping a PDE forward in physical
time -- the model outputs the whole steady-state field in one shot.

  * flow_matching_loss : pure data-driven conditional flow matching (no physics).
  * pbfm_loss          : Physics-Based Flow Matching -- FM loss plus a
                         t-weighted finite-volume flux-balance residual on the
                         differentiably-unrolled x1, returned SEPARATELY for a
                         conflict-free (ConFIG) update.
  * logit_normal_pdf   : logit-normal importance weight on the FM loss.
"""

import math

import torch
import torch.nn.functional as F


def logit_normal_pdf(t, m=0.0, s=1.0):
    eps = 1e-6
    t = torch.clamp(t, eps, 1 - eps)
    z = (torch.log(t / (1 - t)) - m) / s
    return torch.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi) * t * (1 - t))


def flow_matching_loss(model, x1, cond_k, cond_bc):
    """Conditional FM loss; prior is standard Gaussian noise, so sampling must
    start from torch.randn. x1: (B, nx, ny, nz) normalized pressure fields."""
    B = x1.size(0)
    t = torch.rand((B,), device=x1.device)
    noise = torch.randn_like(x1)
    t_exp = t.view(B, 1, 1, 1)

    xt = (1 - t_exp) * noise + t_exp * x1
    vf_target = x1 - noise
    vf_pred = model(xt, t, cond_k, cond_bc)
    return F.mse_loss(vf_pred, vf_target)


def pbfm_loss(model, x1_data, cond_k, cond_bc, cond_bc_phys, k_phys,
              residual_fn, n_steps, use_dignorm, U_MIN, U_MAX, device):
    """PBFM loss. Returns (data_loss, phys_loss) separately.

    x1_data      : (B, nx, ny, nz) normalized pressure fields
    cond_k       : (B, 1, nx, ny, nz) log-normalized permeability (conditioning)
    cond_bc      : (B, 2) normalized (p_left, p_right) (conditioning)
    cond_bc_phys : (B, 2) physical (p_left, p_right), used in the PDE residual
    k_phys       : (B, nx, ny, nz) physical permeability, used in the residual
    """
    B = x1_data.shape[0]
    eps = 1e-5
    t = torch.rand(B, device=device) * (1 - 2 * eps) + eps
    t_view = t.view(-1, 1, 1, 1)

    # 1. Base FM interpolation (sigma_min = 0)
    x0_noise = torch.randn_like(x1_data)
    xt = (1 - t_view) * x0_noise + t_view * x1_data
    vt_target = x1_data - x0_noise

    # Single forward -- serves both the FM loss and the first unroll step.
    vt_pred = model(xt, t, cond_k, cond_bc)

    if use_dignorm:
        w = logit_normal_pdf(t).view(-1, 1, 1, 1)
    else:
        w = torch.ones_like(t).view(-1, 1, 1, 1)
    data_loss = torch.mean(w * (vt_pred - vt_target) ** 2)

    # 2. Differentiable Euler unroll (interpolation time only) to x1
    t_curr = t.clone()
    dt = (1.0 - t) / n_steps
    dt_view = dt.view(-1, 1, 1, 1)

    x_pred = xt + dt_view * vt_pred
    for _ in range(1, n_steps):
        t_curr = t_curr + dt
        vt_step = model(x_pred, t_curr, cond_k, cond_bc)
        x_pred = x_pred + dt_view * vt_step

    # 3. Finite-volume flux-balance residual in physical units
    p_phys = (x_pred + 1.0) / 2.0 * (U_MAX - U_MIN) + U_MIN
    p_phys_fp64 = p_phys.to(torch.float64)
    k_phys_fp64 = k_phys.to(torch.float64)
    bc_phys_fp64 = cond_bc_phys.to(torch.float64)
    R_fp32 = residual_fn(p_phys_fp64, k_phys_fp64, bc_phys_fp64).to(torch.float32)

    phys_error = (R_fp32 ** 2).mean(dim=1)       # [B]
    phys_loss = (t * phys_error).mean()          # t-weighted, p=1
    return data_loss, phys_loss
