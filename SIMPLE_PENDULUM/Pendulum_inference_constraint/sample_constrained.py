"""
sample_constrained.py
---------------------
Case 2: Physics Enforced in Sampling  — CORRECTED IMPLEMENTATION

=======================================================================
WHAT WAS WRONG IN THE PREVIOUS VERSION
=======================================================================

The previous implementation projected at every FM step inside the
integration loop:

    for t in [0, dt, 2dt, ..., 1]:
        u = u + dt * vtheta(u, t)    <- Euler step
        u = project(u, E0)           <- WRONG: projection inside loop

This is fundamentally wrong for two reasons:

  1. At t ~ 0, u is near-Gaussian noise. The theta values are random
     numbers with no physical meaning. Projecting omega at random theta
     positions places each state on the correct energy contour but at a
     completely meaningless position — this caused the star-shaped
     polygon phase portraits we observed.

  2. The FM integration is a continuous ODE in trajectory space.
     Interrupting it at every step with an algebraic correction
     corrupts the learned dynamics throughout the entire integration.

=======================================================================
WHAT THE PCFM PAPER ACTUALLY DOES
=======================================================================

From Utkarsh et al. (NeurIPS 2025), the correct procedure is:

  Step 1 — Plain FM integration (no interference):
      u = u0 ~ N(0, I)
      for t = 0 to 1:
          u = u + dt * vtheta(u, t)   <- standard Euler, untouched
      # u is now the fully generated trajectory u1

  Step 2 — Project ONCE onto the constraint manifold:
      u1_corrected = argmin_u ||u - u1||^2  s.t.  h(u) = 0
      (via Gauss-Newton in the paper; exact for the pendulum)

The key distinction: projection is a POST-HOC correction applied to
the FINAL output u1, not an interleaved disruption of the generative
process. The FM integration runs exactly as trained. Only the
endpoint is corrected.

This matches what the paper did for Burgers' and Navier-Stokes:
generate fully, then project the final field onto the constraint
manifold (IC satisfaction, mass conservation, etc.).

=======================================================================
WHAT THIS CORRECTLY MEASURES
=======================================================================

"If we take the FM model's best guess and then enforce the physical
constraint as a post-processing step, what is the trade-off between
constraint satisfaction and PDE fidelity?"

This is the pendulum analogue of the Burgers'/NS result in the
mid-semester report. We expect:
  - constraint satisfaction improves (std(H) goes down)
  - PDE residual increases moderately

Not the catastrophic 800x explosion from the wrong version, which
was caused by corrupting the FM integration at every step.
"""

import numpy as np
import torch
from Simple_pendulum.Pendulum_inference_constraint.config import N_STEPS, STATE_DIM, N_FM_STEPS, SEED
from Simple_pendulum.Pendulum_inference_constraint.data   import hamiltonian, encode_trajectory, decode_trajectory


# ─────────────────────────────────────────────────────────────
# Energy projection  (exact, no iteration needed for pendulum)
# ─────────────────────────────────────────────────────────────

def project_to_energy(theta, omega, E0):
    """
    Project each state (theta_i, omega_i) onto H(theta, omega) = E0
    by correcting omega while keeping theta fixed:

        omega_i = sign(omega_hat_i) * sqrt(2 * (E0 + cos(theta_i)))

    This is the exact analytical solution to:
        argmin_{omega} (omega - omega_hat)^2  s.t.  0.5*omega^2 - cos(theta) = E0

    For the pendulum this is closed-form. For Burgers'/NS the
    paper uses iterative Gauss-Newton, but the concept is identical.

    Args:
        theta : np.ndarray, shape (..., N_STEPS)
        omega : np.ndarray, shape (..., N_STEPS)
        E0    : float or np.ndarray broadcastable to theta

    Returns:
        omega_proj : np.ndarray, corrected omega, same shape as omega
        valid      : bool array, False where E0 + cos(theta) < 0 (unphysical)
    """
    val   = 2.0 * (E0 + np.cos(theta))
    valid = val >= 0.0
    omega_proj = np.where(
        valid,
        np.sign(omega) * np.sqrt(np.maximum(val, 0.0)),
        omega,   # fallback: leave unchanged if unphysical
    )
    return omega_proj, valid


# ─────────────────────────────────────────────────────────────
# Unconstrained sampler  (plain FM, identical to Case 3)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_unconstrained(
    model,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    """
    Standard Euler FM integration from t=0 to t=1.
    No projection at any point. Identical to Case 3 sampling.

    Returns:
        generated_raw : np.ndarray (n_samples, N_STEPS, 2)  [theta, omega]
    """
    torch.manual_seed(seed)
    model.eval()
    dt_fm = 1.0 / n_fm_steps

    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val)
        u     = u + dt_fm * v

    return decode_trajectory(u.cpu().numpy())   # (n, N_STEPS, 2)


# ─────────────────────────────────────────────────────────────
# Constrained sampler  (correct PCFM approach)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_constrained(
    model,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> tuple:
    """
    PCFM: generate fully with plain FM, then project ONCE at the end.

    This matches the PCFM paper procedure:
        1. Full Euler integration t: 0 -> 1  (FM runs exactly as trained)
        2. Decode final state to (theta, omega)
        3. Project each (theta_i, omega_i) onto H(theta, omega) = E0
           as a post-hoc correction

    The target energy E0 is drawn from the training distribution
    U(E_MIN, E_MAX) for each sample.

    Args:
        model      : trained VelocityFieldMLP (same weights as Case 3)
        n_samples  : number of trajectories to generate
        device     : torch device string
        n_fm_steps : Euler steps for FM integration
        seed       : random seed

    Returns:
        gen_unc  : np.ndarray (n, N_STEPS, 2) — FM output before projection
        gen_con  : np.ndarray (n, N_STEPS, 2) — after post-hoc projection
        E0s      : np.ndarray (n,)             — target energies
        diag     : dict with pre/post projection diagnostics
    """
    from Simple_pendulum.Pendulum_inference_constraint.config import E_MIN, E_MAX

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval()

    dt_fm = 1.0 / n_fm_steps

    # Target energy for each sample
    E0s = rng.uniform(E_MIN, E_MAX, size=n_samples)    # (n,)

    # ── Step 1: Plain FM integration — NO projection inside loop ──
    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)

    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val)
        u     = u + dt_fm * v

    # Raw FM output — completely untouched by any physics
    gen_unc = decode_trajectory(u.cpu().numpy())       # (n, N_STEPS, 2)

    theta = gen_unc[:, :, 0]                           # (n, N_STEPS)
    omega = gen_unc[:, :, 1]                           # (n, N_STEPS)
    E0_bc = E0s[:, np.newaxis]                         # (n, 1) for broadcasting

    # ── Step 2: Measure violation in unconstrained output ─────────
    H_before    = hamiltonian(theta, omega)             # (n, N_STEPS)
    viol_before = np.abs(H_before - E0_bc).mean()

    # ── Step 3: Project ONCE onto target energy contour ───────────
    omega_proj, valid = project_to_energy(theta, omega, E0_bc)

    H_after    = hamiltonian(theta, omega_proj)
    viol_after = np.abs(H_after - E0_bc)[valid].mean()

    gen_con = np.stack([theta, omega_proj], axis=-1)   # (n, N_STEPS, 2)

    invalid_frac = (~valid).mean()

    diag = {
        "E0s":          E0s,
        "viol_before":  viol_before,    # mean |H - E0| before projection
        "viol_after":   viol_after,     # mean |H - E0| after  projection
        "invalid_frac": invalid_frac,   # fraction of states where theta was unphysical
    }

    print(f"  Projection diagnostics:")
    print(f"    Mean |H - E0| before : {viol_before:.4e}")
    print(f"    Mean |H - E0| after  : {viol_after:.4e}")
    print(f"    Invalid projections  : {invalid_frac*100:.1f}%")

    return gen_unc, gen_con, E0s, diag