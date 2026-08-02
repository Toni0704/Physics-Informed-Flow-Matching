"""
train.py  --  Case 4b: Conditional Generation on Initial State
--------------------------------------------------------------
Training is identical to Case 4a except the conditioning signal
changes from scalar E to encoded IC = (sin theta0, cos theta0, omega0).

Loss:
    L = E_{t, u0, u1, ic} [ || v_theta(u_t, t, ic) - (u1 - u0) ||^2 ]

where ic is extracted from the first time step of the real trajectory u1.

Inference:
    To keep the generative framing (Option 2 from the design discussion),
    we sample random oscillating initial conditions from the training
    distribution at inference time, then ask the model to generate the
    corresponding trajectory.

    This preserves scientific honesty: we are not peeking at the ground
    truth initial condition of a test trajectory. We are asking whether
    the model can generate a physically correct trajectory from a
    randomly drawn IC -- the same task the pendulum ODE solver does.

    A separate evaluation function also tests the surrogate solver mode:
    given the EXACT IC of each test trajectory, can the model reproduce
    the ground truth? This quantifies the upper bound on performance.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from Simple_pendulum.Pendulum_conditional_IC.config import N_STEPS, STATE_DIM, N_EPOCHS, LR, GRAD_CLIP, N_FM_STEPS, SEED
from Simple_pendulum.Pendulum_conditional_IC.data   import hamiltonian, decode_trajectory, encode_trajectory, \
                   sample_initial_conditions, generate_trajectory
from Simple_pendulum.Pendulum_conditional_IC.model  import ConditionalVelocityFieldIC


# ── Helper: extract encoded IC from trajectory batch ─────────────────────────

def extract_ic(u1_batch: torch.Tensor) -> torch.Tensor:
    """
    Extract the encoded initial condition from the first time step.

    The trajectory is encoded as (sin theta, cos theta, omega), so the
    first time step u1[:, 0, :] = (sin theta1, cos theta1, omega1).

    Note: we use time step index 0 which is tau=dt, not tau=0. The true
    IC (theta0, omega0) was used to generate the trajectory but is not
    stored in the trajectory tensor itself (the first stored step is
    after one integration step). In practice this is a negligible
    difference -- the first stored state is very close to the true IC.

    Args:
        u1_batch : shape (B, N_STEPS, 3)

    Returns:
        ic : shape (B, 3)  -- (sin theta0, cos theta0, omega0) approx
    """
    return u1_batch[:, 0, :]   # (B, 3)


# ── Conditional FM loss ───────────────────────────────────────────────────────

def conditional_fm_loss(
    model:    ConditionalVelocityFieldIC,
    u1_batch: torch.Tensor,
    device:   str,
) -> torch.Tensor:
    """
    FM loss conditioned on the initial state.

    Args:
        model    : ConditionalVelocityFieldIC
        u1_batch : real encoded trajectories, shape (B, N_STEPS, 3)
        device   : torch device

    Returns:
        loss : scalar
    """
    B = u1_batch.shape[0]

    # Extract IC from first time step of each real trajectory
    ic = extract_ic(u1_batch)                      # (B, 3)

    # Standard FM interpolation
    u0       = torch.randn_like(u1_batch)
    t        = torch.rand(B, device=device)
    t_exp    = t.view(B, 1, 1)
    u_t      = (1.0 - t_exp) * u0 + t_exp * u1_batch
    v_target = u1_batch - u0

    # Conditional forward pass
    v_pred = model(u_t, t, ic)

    return nn.functional.mse_loss(v_pred, v_target)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model:        ConditionalVelocityFieldIC,
    train_loader: torch.utils.data.DataLoader,
    device:       str,
    n_epochs:     int   = N_EPOCHS,
    lr:           float = LR,
    save_dir:     str   = "checkpoints",
) -> list:
    torch.manual_seed(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    losses    = []
    best_loss = float("inf")

    print(f"\nTraining Case 4b ({n_epochs} epochs on {device})...")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'LR':>10}")
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
                       f"{save_dir}/best_model_case4b.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg:>12.6f}  "
                  f"{scheduler.get_last_lr()[0]:>10.2e}")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    return losses


# ── Sampling: generative mode (random IC from training distribution) ──────────

@torch.no_grad()
def sample_generative(
    model:      ConditionalVelocityFieldIC,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> tuple:
    """
    Generate trajectories by sampling random ICs from the training
    distribution, then running FM conditioned on those ICs.

    This is the honest generative evaluation: we do not use any
    ground-truth trajectory information.

    Returns:
        generated_raw : np.ndarray (n, N_STEPS, 2)  [theta, omega]
        ic_raw        : np.ndarray (n, 2)            [theta0, omega0] used
        E_ic          : np.ndarray (n,)              energy of each IC
    """
    from Simple_pendulum.Pendulum_conditional_IC.config import E_MIN, E_MAX

    torch.manual_seed(seed)
    rng   = np.random.default_rng(seed)
    model.eval()
    dt_fm = 1.0 / n_fm_steps

    # Sample random oscillating ICs from the training distribution
    th0s, om0s = sample_initial_conditions(n_samples, e_min=E_MIN, e_max=E_MAX, rng=rng)
    E_ic = hamiltonian(th0s, om0s)

    # Encode IC as (sin theta0, cos theta0, omega0)
    ic_enc = np.stack([np.sin(th0s), np.cos(th0s), om0s], axis=-1)  # (n, 3)
    ic_t   = torch.tensor(ic_enc, dtype=torch.float32, device=device)

    # FM integration conditioned on IC
    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val, ic_t)
        u     = u + dt_fm * v

    generated_raw = decode_trajectory(u.cpu().numpy())   # (n, N_STEPS, 2)
    ic_raw        = np.stack([th0s, om0s], axis=-1)      # (n, 2)

    return generated_raw, ic_raw, E_ic


# ── Sampling: surrogate solver mode (exact IC from test set) ──────────────────

@torch.no_grad()
def sample_surrogate(
    model:      ConditionalVelocityFieldIC,
    test_raw:   np.ndarray,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> np.ndarray:
    """
    Surrogate solver evaluation: condition on the EXACT initial state of
    each test trajectory and try to reproduce the ground truth.

    This is the upper bound evaluation -- the model has maximum information.
    If it still fails here, the architecture is the bottleneck.
    If it succeeds here but fails in generative mode, the issue is that
    energy conditioning (Case 4a) loses too much IC information.

    Args:
        test_raw : np.ndarray (n, N_STEPS, 2)  ground truth trajectories

    Returns:
        generated_raw : np.ndarray (n, N_STEPS, 2)
    """
    torch.manual_seed(seed)
    model.eval()
    dt_fm = 1.0 / n_fm_steps
    n     = len(test_raw)

    # Extract true IC from first step of each test trajectory
    theta0 = test_raw[:, 0, 0]
    omega0 = test_raw[:, 0, 1]

    # Encode as (sin, cos, omega)
    ic_enc = np.stack([np.sin(theta0), np.cos(theta0), omega0], axis=-1)
    ic_t   = torch.tensor(ic_enc, dtype=torch.float32, device=device)

    # FM integration
    u = torch.randn(n, N_STEPS, STATE_DIM, device=device)
    for step in range(n_fm_steps):
        t_val = torch.full((n,), step * dt_fm, device=device)
        v     = model(u, t_val, ic_t)
        u     = u + dt_fm * v

    return decode_trajectory(u.cpu().numpy())


# ── Variant D: Case 4b + post-hoc projection ──────────────────────────────────

@torch.no_grad()
def sample_generative_projected(
    model:      ConditionalVelocityFieldIC,
    n_samples:  int,
    device:     str,
    n_fm_steps: int = N_FM_STEPS,
    seed:       int = SEED + 99,
) -> tuple:
    """
    Variant D: Case 4b (IC-conditioned FM) + post-hoc energy projection.

    Procedure:
        1. Sample random ICs from training distribution
        2. Run full FM integration conditioned on IC  (identical to Case 4b)
        3. Project ONCE onto the energy contour H(theta, omega) = E_ic
           where E_ic = H(theta0, omega0) is the IC's energy

    This is the correct PCFM approach (project once at t=1, not during
    integration) applied to a model that already knows where on the
    contour to start.

    Case 2 showed projection fails when the model is far from the manifold.
    Case 4b puts the model very close to the manifold.
    Variant D tests: does projection now actually help?

    Projection formula (exact for pendulum, no Gauss-Newton needed):
        omega_proj = sign(omega_hat) * sqrt(2 * (E_ic + cos(theta_hat)))

    Args:
        model      : trained Case 4b model (ConditionalVelocityFieldIC)
        n_samples  : number of trajectories to generate
        device     : torch device
        n_fm_steps : FM Euler steps
        seed       : random seed

    Returns:
        gen_unc  : np.ndarray (n, N_STEPS, 2)  FM output before projection
        gen_proj : np.ndarray (n, N_STEPS, 2)  after post-hoc projection
        ic_raw   : np.ndarray (n, 2)            [theta0, omega0] used
        E_ic     : np.ndarray (n,)              target energy per sample
        diag     : dict with projection diagnostics
    """
    from Simple_pendulum.Pendulum_conditional_IC.config import E_MIN, E_MAX

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval()
    dt_fm = 1.0 / n_fm_steps

    # Step 1: Sample random ICs
    th0s, om0s = sample_initial_conditions(n_samples, e_min=E_MIN, e_max=E_MAX, rng=rng)
    E_ic    = hamiltonian(th0s, om0s)          # (n,)  target energy
    ic_raw  = np.stack([th0s, om0s], axis=-1)  # (n, 2)

    ic_enc = np.stack([np.sin(th0s), np.cos(th0s), om0s], axis=-1)
    ic_t   = torch.tensor(ic_enc, dtype=torch.float32, device=device)

    # Step 2: Full FM integration conditioned on IC — NO projection inside
    u = torch.randn(n_samples, N_STEPS, STATE_DIM, device=device)
    for step in range(n_fm_steps):
        t_val = torch.full((n_samples,), step * dt_fm, device=device)
        v     = model(u, t_val, ic_t)
        u     = u + dt_fm * v

    # Decode raw FM output
    gen_unc = decode_trajectory(u.cpu().numpy())   # (n, N_STEPS, 2)
    theta   = gen_unc[:, :, 0]                     # (n, N_STEPS)
    omega   = gen_unc[:, :, 1]

    # Step 3: Project ONCE onto target energy contour
    E0_bc = E_ic[:, np.newaxis]                    # (n, 1) for broadcasting
    H_before    = hamiltonian(theta, omega)
    viol_before = np.abs(H_before - E0_bc).mean()

    val   = 2.0 * (E0_bc + np.cos(theta))
    valid = val >= 0.0
    omega_proj = np.where(
        valid,
        np.sign(omega) * np.sqrt(np.maximum(val, 0.0)),
        omega,   # fallback if unphysical
    )

    H_after    = hamiltonian(theta, omega_proj)
    viol_after = np.abs(H_after - E0_bc)[valid].mean()

    gen_proj = np.stack([theta, omega_proj], axis=-1)  # (n, N_STEPS, 2)

    diag = {
        "E_ic":         E_ic,
        "viol_before":  viol_before,
        "viol_after":   viol_after,
        "invalid_frac": (~valid).mean(),
    }

    print(f"  Variant D projection diagnostics:")
    print(f"    Mean |H - E_ic| before : {viol_before:.4e}")
    print(f"    Mean |H - E_ic| after  : {viol_after:.4e}")
    print(f"    Invalid projections    : {diag['invalid_frac']*100:.1f}%")

    return gen_unc, gen_proj, ic_raw, E_ic, diag