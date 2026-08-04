"""
Loss functions for the NS_2D study -- same structure as Burgers_1D/src/losses.py,
adapted to 4D trajectories (B, T, H, W) and (IC, forcing) conditioning.

  * flow_matching_loss : pure data-driven conditional flow matching (no physics).
  * pbfm_loss          : Physics-Based Flow Matching -- FM loss plus a t-weighted
                         spectral PDE residual on the differentiably-unrolled x1,
                         returned SEPARATELY for a conflict-free (ConFIG) update.
  * logit_normal_pdf   : logit-normal importance weight on the FM loss.
"""

import math

import torch
import torch.nn.functional as F

from .physics import ns_residual


def logit_normal_pdf(t, m=0.0, s=1.0):
    eps = 1e-6
    t = torch.clamp(t, eps, 1 - eps)
    z = (torch.log(t / (1 - t)) - m) / s
    return torch.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi) * t * (1 - t))


def flow_matching_loss(model, x1, cond_a, cond_f):
    """Conditional FM loss; prior is standard Gaussian noise, so sampling must
    start from torch.randn. x1: (B, T, H, W) normalized trajectories."""
    B = x1.size(0)
    t = torch.rand((B,), device=x1.device)
    noise = torch.randn_like(x1)
    t_exp = t.view(B, 1, 1, 1)

    xt = (1 - t_exp) * noise + t_exp * x1
    vf_target = x1 - noise
    vf_pred = model(xt, t, cond_a, cond_f)
    return F.mse_loss(vf_pred, vf_target)


def pbfm_loss(model, x1_data, cond_a, cond_f, n_steps, w_scale,
              visc=1e-3, frame_dt=1.0, use_dignorm=True):
    """PBFM loss. Returns (data_loss, phys_loss) separately.

    x1_data : (B, T, H, W) normalized trajectories
    cond_a  : (B, 1, H, W) normalized IC field (model conditioning)
    cond_f  : (B, 1, H, W) forcing, PHYSICAL units (used both as model
              conditioning and in the PDE residual)
    w_scale : normalization scale, to map the unrolled x1 back to physical
              units before the residual
    """
    B = x1_data.shape[0]
    device = x1_data.device
    eps = 1e-5
    t = torch.rand(B, device=device) * (1 - 2 * eps) + eps
    t_view = t.view(-1, 1, 1, 1)

    # 1. Base FM interpolation (sigma_min = 0)
    x0_noise = torch.randn_like(x1_data)
    xt = (1 - t_view) * x0_noise + t_view * x1_data
    vt_target = x1_data - x0_noise

    vt_pred = model(xt, t, cond_a, cond_f)

    if use_dignorm:
        w = logit_normal_pdf(t).view(-1, 1, 1, 1)
    else:
        w = torch.ones_like(t).view(-1, 1, 1, 1)
    data_loss = torch.mean(w * (vt_pred - vt_target) ** 2)

    # 2. Differentiable Euler unroll to x1
    t_curr = t.clone()
    dt = (1.0 - t) / n_steps
    dt_view = dt.view(-1, 1, 1, 1)

    x_pred = xt + dt_view * vt_pred
    for _ in range(1, n_steps):
        t_curr = t_curr + dt
        vt_step = model(x_pred, t_curr, cond_a, cond_f)
        x_pred = x_pred + dt_view * vt_step

    # 3. Spectral PDE residual in physical units
    w_phys = x_pred * w_scale                       # (B, T, H, W)
    f_phys = cond_f.squeeze(1)                      # (B, H, W)
    R = ns_residual(w_phys, f_phys, visc, frame_dt)  # (B, T-2, H, W)

    phys_error = (R ** 2).mean(dim=(1, 2, 3))       # [B]
    phys_loss = (t * phys_error).mean()             # t-weighted, p=1
    return data_loss, phys_loss
