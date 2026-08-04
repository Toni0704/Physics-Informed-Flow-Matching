#!/usr/bin/env python
"""
Sanity-check the spectral NS residual (src/physics.py) against ground truth.

Go/no-go gate for the whole NS_2D study (README plan, step 2): before any
training, verify that

  1. the vorticity -> velocity spectral inversion round-trips (curl(u,v) == w
     to machine precision, mean mode excluded);
  2. the PDE residual on ground-truth trajectories is SMALL relative to the
     magnitudes of the individual PDE terms (it won't be machine zero -- the
     frame spacing dt=1.0 is 1000x the solver's internal step, so the
     time-derivative term carries O(dt^2) error);
  3. the residual sharply separates ground truth from physics-free imposters
     with identical marginal statistics (time-shuffled GT, Gaussian noise).

    python experiments/check_physics.py [--data <path.h5>] [--n 8]
"""

import argparse
import sys
from pathlib import Path

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import add_repo_to_path
from src.physics import ns_residual, ns_rhs, velocity_from_vorticity

add_repo_to_path()

VISC = 1e-3
DT = 1.0


def rms(x):
    return x.pow(2).mean().sqrt().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None,
                   help="NS .h5 file (default: datasets/data/ns_nw10_* in the repo)")
    p.add_argument("--n", type=int, default=8, help="number of (ic, forcing) samples")
    args = p.parse_args()

    repo = Path(add_repo_to_path())
    data_path = Path(args.data) if args.data else \
        repo / "datasets" / "data" / "ns_nw10_nf100_s64_t50_mu0.001.h5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    with h5py.File(data_path, "r") as fh:
        nw, nf = fh["u"].shape[:2]
        # spread picks across both axes
        pairs = [(i % nw, (i * 7) % nf) for i in range(args.n)]
        w = torch.stack([torch.from_numpy(fh["u"][i, j]) for i, j in pairs])
        f = torch.stack([torch.from_numpy(fh["f"][j][()]) for _, j in pairs])

    # (B, H, W, T) -> (B, T, H, W)
    w = w.permute(0, 3, 1, 2).to(device).double()
    f = f.to(device).double()
    print(f"data: {data_path.name}  w {tuple(w.shape)}  f {tuple(f.shape)}  [{device}]")

    # -- 1. velocity inversion round-trip --------------------------------------
    # Exactness only holds for band-limited fields: the Nyquist mode (k=-n/2)
    # is not differentiable spectrally (physics.py zeroes it in derivative
    # operators), so band-limit the test field with the solver's own 2/3-rule
    # mask -- above which GT carries no meaningful physical energy anyway.
    import math
    from src.physics import _spectral_operators
    n = w.shape[-1]
    kx, ky, _, _, dealias = _spectral_operators(n, device, w.dtype)
    w0 = w[:, 0] - w[:, 0].mean(dim=(-2, -1), keepdim=True)
    w0 = torch.fft.ifft2(torch.fft.fft2(w0) * dealias).real
    u, v = velocity_from_vorticity(w0)
    curl = (torch.fft.ifft2(1j * 2 * math.pi * kx * torch.fft.fft2(v)).real
            - torch.fft.ifft2(1j * 2 * math.pi * ky * torch.fft.fft2(u)).real)
    rt_err = rms(curl - w0) / rms(w0)
    print(f"\n[1] curl(u,v) round-trip rel error (band-limited): {rt_err:.3e}   "
          f"({'OK' if rt_err < 1e-10 else 'FAIL — spectral ops are wrong'})")

    # -- 2. residual floor on ground truth -------------------------------------
    w_t = (w[:, 2:] - w[:, :-2]) / (2 * DT)
    rhs = ns_rhs(w[:, 1:-1], f[:, None], VISC)
    res_gt = ns_residual(w, f, VISC, DT)
    scale = max(rms(w_t), rms(rhs))
    rel_gt = rms(res_gt) / scale
    print(f"\n[2] ground-truth residual\n"
          f"    rms(w_t)={rms(w_t):.3e}  rms(rhs)={rms(rhs):.3e}  "
          f"rms(residual)={rms(res_gt):.3e}\n"
          f"    relative residual = {rel_gt:.3f}")

    # -- 3. contrast against physics-free imposters -----------------------------
    perm = torch.randperm(w.shape[1])
    res_shuf = ns_residual(w[:, perm], f, VISC, DT)
    noise = torch.randn_like(w) * w.std()
    res_noise = ns_residual(noise, f, VISC, DT)
    print(f"\n[3] imposter contrast (same rms scale as GT)\n"
          f"    rms residual  GT / time-shuffled / noise : "
          f"{rms(res_gt):.3e} / {rms(res_shuf):.3e} / {rms(res_noise):.3e}\n"
          f"    separation    shuffled/GT = {rms(res_shuf)/rms(res_gt):.1f}x   "
          f"noise/GT = {rms(res_noise)/rms(res_gt):.1f}x")

    ok = rt_err < 1e-10 and rms(res_shuf) / rms(res_gt) > 2.0
    print(f"\n==> {'GATE PASSED' if ok else 'GATE FAILED'}: "
          f"{'residual is usable as a physics loss/metric' if ok else 'do not proceed to PBFM'}")


if __name__ == "__main__":
    main()
