# Key algorithms for Physics-Constrained Flow Matching (PCFM)

import math
import os
import torch
from torch.func import vmap, jacrev
import gc
from typing import Callable, Sequence

_DEBUG_GUARDS = bool(os.environ.get("PCFM_DEBUG_GUARDS"))

def compute_jacobian(fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    def fn_flat(x: torch.Tensor) -> torch.Tensor:
        return fn(x).flatten()
    J = jacrev(fn_flat)(inputs)
    m = J.shape[0]
    n = inputs.numel()
    return J.reshape(m, n)


def fast_project_batched(xi_batch: torch.Tensor, h_func: Callable[[torch.Tensor], torch.Tensor], max_iter: int = 1) -> torch.Tensor:
    """
    Final projection step in PCFM
    """
    B, n = xi_batch.shape

    def newton_step(u, xi):
        h_val = h_func(u)
        if h_val.ndim == 1:
            h_val = h_val.unsqueeze(-1)
        J = jacrev(h_func,chunk_size=len(h_val)//4)(u)
        if J.ndim == 1:
            J = J.unsqueeze(0)
        delta = (xi - u).unsqueeze(-1)
        JJt = J @ J.transpose(-2, -1)
        rhs = J @ delta + h_val
        lambda_ = torch.linalg.solve(JJt, rhs)
        du = delta - J.transpose(-2, -1) @ lambda_
        return u + du.squeeze(-1)

    def loop(xi):
        u = xi.clone()
        gc.collect()
        torch.cuda.empty_cache()
        for _ in range(max_iter):
            u = newton_step(u, xi)
        return u

    return vmap(loop)(xi_batch)

def fast_project_batched_chunk(xi_batch, h_func, max_iter=1, chunk_size=16):
    B, n = xi_batch.shape
    results = []
    for start in range(0, B, chunk_size):
        xi_chunk = xi_batch[start:start + chunk_size]

        def newton_step(u, xi):
            h_val = h_func(u)
            if h_val.ndim == 1:
                h_val = h_val.unsqueeze(-1)
            J = jacrev(h_func, chunk_size=max(1, len(h_val)//4))(u)
            if J.ndim == 1:
                J = J.unsqueeze(0)
            delta = (xi - u).unsqueeze(-1)
            JJt = J @ J.transpose(-2, -1)
            rhs = J @ delta + h_val
            lambda_ = torch.linalg.solve(JJt, rhs)
            du = delta - J.transpose(-2, -1) @ lambda_
            return u + du.squeeze(-1)

        def loop(xi):
            u = xi.clone()
            for _ in range(max_iter):
                u = newton_step(u, xi)
            return u

        results.append(vmap(loop)(xi_chunk))
        del xi_chunk
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)


def make_grid(dims: tuple[int], device='cpu', start: float | tuple[float] = 0., end: float | tuple[float] = 1.):
    ndim = len(dims)
    if not isinstance(start, (tuple, list)):
        start = [start] * ndim
    if not isinstance(end, (tuple, list)):
        end = [end] * ndim
    if ndim == 1:
        return torch.linspace(start[0], end[0], dims[0], dtype=torch.float, device=device).unsqueeze(-1)
    xs = torch.meshgrid([
        torch.linspace(start[i], end[i], dims[i], dtype=torch.float, device=device)
        for i in range(ndim)
    ], indexing='ij')
    grid = torch.stack(xs, dim=-1).view(-1, ndim)
    return grid


def relaxed_penalty_constraint_interp_linear_detached(
    u0, u1_proj, v_flat, t, dt, hfunc, lam=1e-2, step_size=1e-2, num_steps=10, safe_clamp=1e-3, lambda_schedule=None
):
    """
    Relaxed constraint correction step in PCFM algorithm
    Solves:
        min_u ||u - hat_u(t')||^2 + lam * ||h(u + gamma * v_flat)||^2
    Args:
        u0: Tensor (n,)
        u1_proj: Tensor (n,)
        v_flat: Tensor (n,), vector field at current state
        t: scalar float (flow matching time t)
        dt: scalar float 
        hfunc: constraint residual
        lam: penalty coefficient
        step_size: gradient descent step size
        num_steps: gradient descent iterations
        safe_clamp: minimum value for gamma
    Returns:
        u_corr: Tensor (n,) 
    """
    t_prime = t + dt
    gamma = max(1 - t_prime, safe_clamp)
    hat_u = (1 - t_prime) * u0 + t_prime * u1_proj
    
    if lambda_schedule is not None:
        lam = lambda_schedule(t_prime)  # Evaluate schedule at t_prime
        print(f"[t={t_prime:.3f}] Using scheduled lambda: {lam:.4e}")  # Debug print
    
    # Sanity check: if lam=0, skip optimization (vanilla FFM behavior)
    if lam == 0.0:
        return hat_u.detach()
    
    u = hat_u.detach().clone().requires_grad_(True)
    best_u, best_loss = u.detach().clone(), float('inf')

    for _ in range(num_steps):
        u_ext = u + gamma * v_flat
        penalty = hfunc(u_ext).pow(2).sum()
        loss = (u - hat_u).pow(2).sum() + lam * penalty
        loss_val = loss.item()

        # This is plain fixed-step gradient descent with no line search; for
        # ill-conditioned constraints (e.g. NS's mass residual, whose Jacobian
        # rows are highly correlated across ~thousands of spatial cells) it can
        # diverge geometrically rather than converge. Track the best-seen
        # iterate and bail out on divergence instead of returning garbage.
        if not math.isfinite(loss_val) or loss_val > best_loss * 10:
            if _DEBUG_GUARDS:
                print(f"[relaxed_penalty_interp] guard tripped: "
                      f"loss {best_loss:.3e} -> {loss_val:.3e}; bailing to best-seen iterate")
            break
        if loss_val < best_loss:
            best_loss = loss_val
            best_u = u.detach().clone()

        grad = torch.autograd.grad(loss, u)[0]
        u = (u - step_size * grad).detach().clone().requires_grad_(True)

    return best_u


# 
def pcfm_sample(
    u_flat, v_flat, t, u0_flat, dt, hfunc,
    mode='root', newtonsteps=1, eps=1e-6,
    guided_interpolation=False, interpolation_params={},
    pc_only_last_step=False     # NEW FLAG
):
    

    t_next = t + dt

    # --------------------------------------------------
    # If last-step-only and not final step → skip projection
    # --------------------------------------------------
    if pc_only_last_step and t_next < 1.0:
        ut_interp = (1.0 - t_next) * u0_flat + t_next * (u_flat + (1.0 - t) * v_flat)
        proj_vf = ((ut_interp - u_flat) / dt).detach()
        return proj_vf

    # --------------------------------------------------
    # NORMAL PCFM PROJECTION
    # --------------------------------------------------

    ut1 = u_flat + (1.0 - t) * v_flat
    u_corr = ut1.clone()

    if _DEBUG_GUARDS:
        print(f"[pcfm_sample TRACE] t={float(t):.3f}  "
              f"u_flat.norm={u_flat.norm().item():.3e}  "
              f"v_flat.norm={v_flat.norm().item():.3e}  "
              f"ut1.norm={ut1.norm().item():.3e}  "
              f"u0_flat.norm={u0_flat.norm().item():.3e}")

    # Stable reference scale for sanity-checking corrections: u0_flat is the
    # ORIGINAL noise seed for the whole trajectory, passed through unchanged
    # by the outer ODE loop (never reassigned) -- unlike u_flat/ut1, which is
    # this timestep's running state and can itself have already drifted from
    # earlier steps. Deliberately NOT max()'d with ut1.norm(): if a prior step
    # let contamination through, ut1 is already inflated, and folding it into
    # the reference would raise the threshold along with the contamination
    # instead of catching it -- anchoring on u0_flat ALONE is what makes this
    # guard resistant to gradual multi-step drift, not just single-step blowup.
    ref_scale = max(u0_flat.norm().item(), 1e-8)

    for _ in range(newtonsteps):
        res = hfunc(u_corr)
        res_norm = res.norm().item()
        J = compute_jacobian(hfunc, u_corr)
        JJt = J @ J.T
        rhs = res

        if mode == 'least_squares':
            delta = (ut1 - u_corr).unsqueeze(-1)
            rhs = J @ delta + res.unsqueeze(-1)
            rhs = rhs.squeeze(-1)

        lam = torch.linalg.solve(
            JJt + eps * torch.eye(JJt.shape[0], device=u_flat.device),
            rhs
        )
        u_new = u_corr - J.T @ lam

        # torch.linalg.solve on a near-singular JJt (eps=1e-6 is a thin
        # Tikhonov regularizer) can produce a huge lam and overshoot badly in
        # a single step -- the result stays finite (so isfinite alone won't
        # catch it) but can be many orders of magnitude off. The constraint
        # is underdetermined (far fewer residual dims than state dims), so a
        # near-singular JJt can also land on a numerically "valid"
        # least-squares solution whose norm is enormous even though hfunc's
        # reported residual looks fine -- the residual check below isn't
        # sufficient by itself, hence the absolute-magnitude check first.
        # Require the step to actually reduce the constraint residual AND
        # stay within a generous multiple of the trajectory's own scale; if
        # either fails, keep the pre-step state and stop iterating rather
        # than injecting a diverged value into the ODE trajectory, where it would
        # corrupt every subsequent timestep's Jacobian too.
        new_norm = u_new.norm().item()
        if not torch.isfinite(u_new).all() or new_norm > 50 * ref_scale:
            if _DEBUG_GUARDS:
                print(f"[pcfm_sample] MAGNITUDE guard tripped @ t={float(t):.3f}: "
                      f"new_norm={new_norm:.3e} vs 50*ref_scale={50*ref_scale:.3e} "
                      f"(ref_scale={ref_scale:.3e}); reverting this Newton step")
            break
        new_res_norm = hfunc(u_new).norm().item()
        if not math.isfinite(new_res_norm) or new_res_norm > res_norm * 10 + 1e-6:
            if _DEBUG_GUARDS:
                print(f"[pcfm_sample] RESIDUAL guard tripped @ t={float(t):.3f}: "
                      f"res_norm {res_norm:.3e} -> {new_res_norm:.3e}; "
                      f"reverting this Newton step")
            break
        u_corr = u_new

    # --------------------------------------------------
    # interpolation
    # --------------------------------------------------

    if guided_interpolation:
        if interpolation_params:
            custom_lam = interpolation_params['custom_lam']
            step_size = interpolation_params['step_size']
            num_steps = interpolation_params['num_steps']
            lambda_schedule = interpolation_params.get('lambda_schedule', None)
        else:
            custom_lam = 1e0
            step_size = 1e-2
            num_steps = 20
            lambda_schedule = None

        ut_interp = relaxed_penalty_constraint_interp_linear_detached(
            u0=u0_flat,
            u1_proj=u_corr,
            v_flat=v_flat,
            t=t.item(),
            dt=dt,
            hfunc=hfunc,
            lam=custom_lam,
            step_size=step_size,
            num_steps=num_steps,
            lambda_schedule=lambda_schedule
        )
    else:
        ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr

    proj_vf = ((ut_interp - u_flat) / dt).detach()
    return proj_vf


def pcfm_batched(ut, vf, t, u0, dt, hfunc, use_vmap=False, mode='root', newtonsteps=1, guided_interpolation=False, interpolation_params={}, eps=1e-6):
    """
    Batched PCFM projection for 1D problems (nx, nt)
    """
    B, nx, nt = ut.shape
    n = nx * nt

    def wrapped_project(u_flat, v_flat, u0_flat):
        return pcfm_sample(
            u_flat, v_flat, t, u0_flat, dt,
            hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
            guided_interpolation=guided_interpolation,
            interpolation_params=interpolation_params,
            eps=eps
        )

    u_flat = ut.view(B, n).detach().clone().requires_grad_(True)
    v_flat = vf.view(B, n)
    u0_flat = u0.view(B, n)

    if use_vmap:
        v_proj_flat = vmap(wrapped_project)(u_flat, v_flat, u0_flat)
    else:
        v_proj_list = []
        for i in range(B):
            v_proj = wrapped_project(u_flat[i], v_flat[i], u0_flat[i])
            v_proj_list.append(v_proj)
        v_proj_flat = torch.stack(v_proj_list, dim=0)

    return v_proj_flat.view(B, nx, nt)


def pcfm_2d_batched(ut, vf, t, u0, dt, hfunc, mode='root', newtonsteps=1, guided_interpolation = True, interpolation_params={}, eps=1e-6):
    """
    Batched PCFM projection for 2D problems (nx, ny, nt)
    """
    B, nx, ny, nt = ut.shape
    n = nx * ny * nt

    gc.collect()
    torch.cuda.empty_cache()

    def wrapped_project(u_flat, v_flat, u0_flat):
        return pcfm_sample(
            u_flat, v_flat, t, u0_flat, dt,
            hfunc=hfunc, mode=mode, newtonsteps=newtonsteps, 
            guided_interpolation=guided_interpolation, 
            interpolation_params=interpolation_params, 
            eps=eps
        )

    u_flat = ut.view(B, n).detach().clone().requires_grad_(True)
    v_flat = vf.view(B, n)
    u0_flat = u0.view(B, n)

    # v_proj_flat = vmap(wrapped_project)(u_flat, v_flat, u0_flat) 
    # prevent OOM: 
    v_proj_list = []
    for i in range(u_flat.shape[0]):
        v_proj = wrapped_project(u_flat[i], v_flat[i], u0_flat[i])
        v_proj_list.append(v_proj)
    v_proj_flat = torch.stack(v_proj_list, dim=0)
    return v_proj_flat.view(B, nx, ny, nt)


def pcfm_3d_batched(ut, vf, t, u0, dt, hfunc, mode='root', newtonsteps=1, guided_interpolation=True, interpolation_params={}, eps=1e-6):
    """
    Batched PCFM projection for pure-3D-space problems (nx, ny, nz), e.g.
    steady-state Darcy flow -- no time axis, unlike pcfm_batched/pcfm_2d_batched.
    """
    B, nx, ny, nz = ut.shape
    n = nx * ny * nz

    gc.collect()
    torch.cuda.empty_cache()

    def wrapped_project(u_flat, v_flat, u0_flat):
        return pcfm_sample(
            u_flat, v_flat, t, u0_flat, dt,
            hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
            guided_interpolation=guided_interpolation,
            interpolation_params=interpolation_params,
            eps=eps
        )

    u_flat = ut.view(B, n).detach().clone().requires_grad_(True)
    v_flat = vf.view(B, n)
    u0_flat = u0.view(B, n)

    v_proj_list = []
    for i in range(u_flat.shape[0]):
        v_proj = wrapped_project(u_flat[i], v_flat[i], u0_flat[i])
        v_proj_list.append(v_proj)
    v_proj_flat = torch.stack(v_proj_list, dim=0)
    return v_proj_flat.view(B, nx, ny, nz)

