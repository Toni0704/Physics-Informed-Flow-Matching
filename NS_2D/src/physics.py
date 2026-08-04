"""
Differentiable spectral physics residual for 2D Navier-Stokes (vorticity form).

The data (datasets/generate_ns_2d.py) solves

    w_t + u . grad(w) = nu * lap(w) + f        on the periodic torus [0,1]^2

pseudo-spectrally (Crank-Nicolson, internal dt = 1e-3), recording 50 snapshots
1.0 time units apart. This module evaluates the SAME spatial discretization
(spectral derivatives, 2/3-rule dealiasing on the nonlinear term, identical
wavenumber conventions) on a trajectory of snapshots, with finite differences
in time across frames.

Because the recorded frame spacing (dt = 1.0) is 1000x coarser than the
solver's internal step, the residual on ground truth is NOT machine zero --
the time-derivative term carries an O(dt^2) discretization error. It is still
a valid physics loss/metric: it is minimized by trajectories that follow the
PDE, and ground truth gives the reference floor that generated samples are
compared against (run experiments/check_physics.py to see the floor).

Everything here is plain torch on real/complex tensors -- fully differentiable,
so `ns_residual` can sit inside the PBFM training loss.

Convention: trajectories are channels-first, w: (B, T, H, W), matching the
model input/output layout; forcing f: (B, H, W); time is the LAST frame axis
of the solver's (H, W, T) layout transposed to the front.
"""

import math

import torch

_CACHE = {}


def _spectral_operators(n, device, dtype=torch.float32):
    """Wavenumber grids / Laplacian / dealias mask for an n x n periodic grid
    on [0,1]^2, matching generate_ns_2d.py's conventions exactly. `dtype` must
    match the (real) dtype of the fields being operated on, or mixed-precision
    promotion quietly degrades float64 evaluation to float32 accuracy."""
    key = (n, str(device), dtype)
    if key not in _CACHE:
        freqs = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype)  # 0..n/2-1, -n/2..-1
        # First-derivative wavenumbers with the Nyquist mode zeroed: on an even
        # grid k = -n/2 has no conjugate partner, so i*k*X_h breaks Hermitian
        # symmetry and the .real projection silently corrupts that mode.
        dfreqs = freqs.clone()
        if n % 2 == 0:
            dfreqs[n // 2] = 0.0
        k_x = dfreqs.view(n, 1).expand(n, n)  # varies along rows (dim 0)
        k_y = dfreqs.view(1, n).expand(n, n)  # varies along cols (dim 1)

        # The Laplacian is symmetric (k^2), so it keeps the full wavenumbers.
        f_x = freqs.view(n, 1).expand(n, n)
        f_y = freqs.view(1, n).expand(n, n)
        lap = 4 * (math.pi ** 2) * (f_x ** 2 + f_y ** 2)
        lap_inv = lap.clone()
        lap_inv[0, 0] = 1.0  # solver's hack: mean mode of psi is arbitrary

        k_max = n // 2
        dealias = (
            (f_x.abs() <= (2.0 / 3.0) * k_max)
            & (f_y.abs() <= (2.0 / 3.0) * k_max)
        )

        _CACHE[key] = (k_x, k_y, lap, lap_inv, dealias)
    return _CACHE[key]


def velocity_from_vorticity(w):
    """Velocities (u, v) from vorticity via the stream function, spectrally.

    w : (..., H, W) real. Returns (u, v), each (..., H, W) real.
    """
    n = w.shape[-1]
    k_x, k_y, _, lap_inv, _ = _spectral_operators(n, w.device, w.dtype)

    w_h = torch.fft.fft2(w)
    psi_h = w_h / lap_inv
    u = torch.fft.ifft2(1j * 2 * math.pi * k_y * psi_h).real   # u =  psi_y
    v = torch.fft.ifft2(-1j * 2 * math.pi * k_x * psi_h).real  # v = -psi_x
    return u, v


def ns_rhs(w, f, visc):
    """Spatial RHS of the vorticity equation: -u.grad(w) + nu*lap(w) + f.

    w : (..., H, W) real vorticity field(s)
    f : broadcastable to w's shape (forcing)
    """
    n = w.shape[-1]
    k_x, k_y, lap, _, dealias = _spectral_operators(n, w.device, w.dtype)

    w_h = torch.fft.fft2(w)
    u, v = velocity_from_vorticity(w)
    w_x = torch.fft.ifft2(1j * 2 * math.pi * k_x * w_h).real
    w_y = torch.fft.ifft2(1j * 2 * math.pi * k_y * w_h).real

    # Nonlinear term, dealiased in Fourier space exactly like the solver.
    adv_h = torch.fft.fft2(u * w_x + v * w_y) * dealias
    adv = torch.fft.ifft2(adv_h).real

    diff = torch.fft.ifft2(-lap * w_h).real * visc
    return -adv + diff + f


def ns_residual(w_traj, f, visc, dt=1.0):
    """PDE residual across a trajectory of snapshots.

    w_traj : (B, T, H, W) vorticity trajectory, physical units
    f      : (B, H, W) forcing per sample
    visc   : viscosity (scalar)
    dt     : time between recorded frames

    Returns (B, T-2, H, W): residual w_t - rhs at interior frames, using
    central differences for w_t (frames 1..T-2).
    """
    w_t = (w_traj[:, 2:] - w_traj[:, :-2]) / (2.0 * dt)   # (B, T-2, H, W)
    rhs = ns_rhs(w_traj[:, 1:-1], f[:, None, :, :], visc)  # (B, T-2, H, W)
    return w_t - rhs


def mass_drift(w_traj):
    """Mean-vorticity drift per frame relative to frame 0 (should be conserved
    up to forcing's mean, which is ~0 for the sinusoidal forcing used).

    w_traj : (B, T, H, W). Returns (B, T-1)."""
    mean_t = w_traj.mean(dim=(-2, -1))          # (B, T)
    return mean_t[:, 1:] - mean_t[:, :1]
