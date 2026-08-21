#!/usr/bin/env python
"""
Evaluate the four 2D Navier-Stokes flow-matching techniques and write
per-technique metrics + figures into results/.

  technique 1  PBFM (physics in training) + FiLM(IC,forcing) -> conditioned_pbfm.{txt,png}
  technique 2  FiLM(IC,forcing) + PCFM sampling               -> conditioned_pcfm.{txt,png}
  technique 3  Pure PCFM on the unconditioned model (IC+mass
               as hard constraints at sampling)               -> pure_pcfm.{txt,png}
  technique 4  FiLM(IC,forcing) + vanilla sampling             -> conditioned_vanilla.{txt,png}

Metrics reported per technique: Data MSE/NRMSE vs GT, the spectral PDE
residual (src/physics.ns_residual -- same discretization validated by
check_physics.py), IC residual, and mass-conservation drift. This is richer
than the IC+mass-only pair used as the PCFM *projection target* (Residuals2D
in the main pcfm library): the spectral residual is an independent physics
check, not just "did the constraint solver converge."

Runnable mid-training: --ckpt-uncond/--ckpt-cond/--ckpt-pbfm override the
default weights/best_*.pt paths, so you can evaluate a --logdir checkpoint
(e.g. .../ns_logs/ns_uncond/latest.pt) from a second terminal while training
continues -- any technique whose checkpoint isn't found is skipped with a
clear message rather than failing the whole run.

Run:
    python experiments/evaluate.py --technique pure_pcfm --ckpt-uncond /kaggle/working/ns_logs/ns_uncond/latest.pt \
        --data-test /kaggle/input/datasets/rishabhj74/10-100/ns_nw10_nf100_s64_t50_mu0.001.h5
    python experiments/evaluate.py   # all techniques, default weights/ paths
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import REPO_ROOT, add_repo_to_path
from src.models import NSVelocityNet_FiLM_ICF
from src.physics import ns_residual, mass_drift

add_repo_to_path()
from models import get_flow_model                       # noqa: E402
from scripts.training.utils import load_config          # noqa: E402
from pcfm import Residuals2D, FFM_NS_sampler             # noqa: E402
from pcfm.pcfm_sampling import pcfm_2d_batched, make_grid  # noqa: E402

NS_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = NS_DIR / "results" / "figures"
MET_DIR = NS_DIR / "results" / "metrics"
WEIGHTS = NS_DIR / "weights"
VISC = 1e-3
FRAME_DT = 1.0


def _lambda_schedule_increasing(t):
    """Ramp 1e-2 -> ~1e0 over flow time t in [0,1] (same schedule PCFM's own
    scripts/training/run_pcfm_hpc.py uses for NS). Guided interpolation with a
    FLAT lam=1e0 from t=0 (pure noise, no coherent structure yet) over-
    constrains early in the trajectory and can distort the rest of the
    sample; ramping it in avoids that."""
    return 1e-2 * torch.exp(torch.tensor(5.0 * t)).item()


# Presets for the guided-interpolation step used inside pcfm_sample's Newton
# correction. 'increasing' matches run_pcfm_hpc.py's tuned NS setup and
# should be the default choice; 'fixed' is the naive flat-lambda behavior
# (what pure_pcfm used before this fix, whenever no interpolation_params
# were supplied at all); 'none' bypasses interpolation entirely -- the IC+
# mass residual is LINEAR in u, so the Newton/least-squares projection alone
# is already exact, and this isolates whether the interpolation smoothing
# itself is what hurts quality.
INTERP_PRESETS = {
    "increasing": dict(guided_interpolation=True, interpolation_params={
        "custom_lam": 1e-2, "step_size": 1e-2, "num_steps": 20,
        "lambda_schedule": _lambda_schedule_increasing,
    }),
    "fixed": dict(guided_interpolation=True, interpolation_params={
        "custom_lam": 1e0, "step_size": 1e-2, "num_steps": 20,
    }),
    "none": dict(guided_interpolation=False, interpolation_params={}),
}


# --------------------------------------------------------------------------- #
# Metrics / plotting
# --------------------------------------------------------------------------- #
def rich_metrics(w_pred, w_gt, f_phys, device):
    """Per-sample Data/Phys/IC/Mass metrics on PHYSICAL trajectories.

    w_pred, w_gt : (B, T, H, W) numpy, physical units
    f_phys       : (B, H, W) numpy, forcing
    """
    rows = []
    for i in range(w_pred.shape[0]):
        data_mse = float(np.mean((w_gt[i] - w_pred[i]) ** 2))
        gtr = float(w_gt[i].max() - w_gt[i].min())
        data_nrmse = 100.0 * np.sqrt(data_mse) / gtr if gtr > 1e-12 else 0.0

        wp = torch.from_numpy(w_pred[i:i + 1]).to(device, dtype=torch.float64)
        fp = torch.from_numpy(f_phys[i:i + 1]).to(device, dtype=torch.float64)
        res = ns_residual(wp, fp, VISC, FRAME_DT)
        phys_mse = float((res ** 2).mean().item())

        ic_mse = float(np.mean((w_gt[i, 0] - w_pred[i, 0]) ** 2))

        drift_gt = mass_drift(torch.from_numpy(w_gt[i:i + 1]).to(device, dtype=torch.float64))
        drift_pred = mass_drift(wp)
        mass_mse = float(((drift_pred - drift_gt) ** 2).mean().item())

        rows.append({
            "Sample": i + 1, "Data MSE": data_mse, "Data NRMSE (%)": round(data_nrmse, 4),
            "Phys MSE (spectral)": phys_mse, "IC MSE": ic_mse, "Mass drift MSE": mass_mse,
        })
    return pd.DataFrame(rows)


def write_rich(df, title, txt_path):
    with open(txt_path, "w") as fh:
        fh.write(f"=== {title} ===\n")
        fh.write(df.to_string(index=False) + "\n\n")
        fh.write("=== Mean Metrics Across Samples ===\n")
        for col, val in df.drop(columns=["Sample"]).mean().items():
            fh.write(f"  {col:20s}: {val:.6e}\n")
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    print(f"[written] {txt_path}")


def save_triptych(w_gt, w_pred, png_path, title, frame=25):
    n = w_pred.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(f"{title} (frame {frame})", fontsize=14, fontweight="bold", y=1.01)
    for i in range(n):
        vmin, vmax = w_gt[i, frame].min(), w_gt[i, frame].max()
        axes[i, 0].imshow(w_gt[i, frame], cmap="bwr", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title("GT")
        axes[i, 1].imshow(w_pred[i, frame], cmap="bwr", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Pred")
        im = axes[i, 2].imshow(np.abs(w_gt[i, frame] - w_pred[i, frame]), cmap="Reds")
        axes[i, 2].set_title("Absolute error")
        fig.colorbar(im, ax=axes[i, 2])
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[written] {png_path}")


# --------------------------------------------------------------------------- #
# Conditioned model helpers (techniques 1, 2, 4)
# --------------------------------------------------------------------------- #
def load_conditioned_checkpoint(path, n_t, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NSVelocityNet_FiLM_ICF(n_t=n_t).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["w_scale"])


@torch.no_grad()
def vanilla_sample(model, cond_a, cond_f, n_t, H, W, w_scale, device, timesteps=200):
    B = cond_a.shape[0]
    xt = torch.randn(B, n_t, H, W, device=device)
    dt = 1.0 / timesteps
    for i in range(timesteps):
        t = torch.full((B,), i / timesteps, device=device)
        vf = model(xt, t, cond_a, cond_f)
        xt = xt + vf * dt
    return xt * w_scale


def pcfm_sample_with_physics(model, cond_a, cond_f, n_t, H, W, w_scale, residuals_list,
                             device, timesteps=200, correction_steps=1):
    """FiLM(IC,forcing) model + PCFM hard projection (IC+mass, Residuals2D).

    residuals_list: one Residuals2D per batch item (each built against its own
    GT IC, since the projection target is per-sample). pcfm_2d_batched takes
    unflattened (B, nx, ny, nt) tensors and a SINGLE shared hfunc for its whole
    batch loop, so -- to give each sample its own hfunc -- it's called once per
    sample (B=1) rather than once for the whole batch.
    """
    B = cond_a.shape[0]
    xt = torch.randn(B, n_t, H, W, device=device)
    dt = 1.0 / timesteps

    for i in range(timesteps):
        t_val = i / timesteps
        t = torch.full((B,), t_val, device=device)
        with torch.no_grad():
            vf = model(xt, t, cond_a, cond_f)

        # (B, n_t, H, W) -> pcfm_2d_batched / Residuals2D convention (B, nx, ny, nt).
        # pcfm_2d_batched does a plain .view() internally, which requires
        # contiguous memory -- permute() alone leaves a non-contiguous view.
        ut_hwt = xt.permute(0, 2, 3, 1).contiguous()
        vf_hwt = vf.permute(0, 2, 3, 1).contiguous()

        proj_list = []
        for b in range(B):
            def hfunc_single(u_flat_in, _res=residuals_list[b], _ws=w_scale):
                u_phys = (u_flat_in * _ws).to(torch.float64)
                return _res.full_residual_ns(u_phys).to(torch.float32)

            with torch.enable_grad():
                proj_v = pcfm_2d_batched(
                    ut=ut_hwt[b:b + 1], vf=vf_hwt[b:b + 1],
                    t=torch.tensor(t_val, device=device), u0=ut_hwt[b:b + 1], dt=dt,
                    hfunc=hfunc_single, mode="least_squares",
                    newtonsteps=correction_steps, guided_interpolation=False,
                )
            proj_list.append(proj_v[0])

        vf_corrected = torch.stack(proj_list, dim=0).permute(0, 3, 1, 2)  # -> (B, n_t, H, W)
        xt = xt + vf_corrected * dt

    return xt * w_scale


# --------------------------------------------------------------------------- #
# Technique runners
# --------------------------------------------------------------------------- #
def load_ns_test_batch(data_path, num_samples, device):
    """`num_samples` (IC, forcing) pairs spanning BOTH axes of the test .h5,
    physical units. Same pick pattern as check_physics.py -- pinning forcing
    to index 0 would never exercise the forcing-FiLM branch of the
    conditioned models and would cap num_samples at nw."""
    with h5py.File(data_path, "r") as fh:
        nw, nf = fh["u"].shape[:2]
        n_t = fh["u"].shape[-1]
        H, W = fh["u"].shape[2], fh["u"].shape[3]
        pairs = [(i % nw, (i * 7) % nf) for i in range(num_samples)]
        w_gt = torch.stack([torch.from_numpy(fh["u"][i, j]) for i, j in pairs]
                           ).permute(0, 3, 1, 2).to(device)          # (B,T,H,W)
        a = torch.stack([torch.from_numpy(fh["a"][i]) for i, j in pairs]).to(device)  # (B,H,W)
        f = torch.stack([torch.from_numpy(fh["f"][j]) for i, j in pairs]).to(device)  # (B,H,W)
    return w_gt, a, f, n_t, H, W


def run_ground_truth(args, device):
    """Sanity check, not a technique: evaluates the metrics on the true trajectory
    against itself. Data MSE/IC MSE are trivially 0 (w_pred == w_gt); Phys MSE
    (spectral residual) and Mass drift MSE are the meaningful numbers here -- they
    check whether the *dataset itself* satisfies the PDE/mass-conservation to a
    reasonable tolerance, independent of any trained model.
    """
    w_gt, a_gt, f_gt, n_t, H, W = load_ns_test_batch(args.data_test, args.num_samples, device)
    w_gt_np = w_gt.cpu().numpy()
    f_np = f_gt.cpu().numpy()
    df = rich_metrics(w_gt_np, w_gt_np, f_np, device)
    write_rich(df, "Ground truth (sanity check)", MET_DIR / "ground_truth.txt")


def run_pure_pcfm(args, device):
    ckpt_path = Path(args.ckpt_uncond) if args.ckpt_uncond else WEIGHTS / "best_fm_uncond.pt"
    if not ckpt_path.exists():
        print(f"[skip] pure_pcfm: checkpoint not found at {ckpt_path}")
        return

    config = load_config(str(REPO_ROOT / "configs" / "ns.yml"))
    model = get_flow_model(config.model, config.encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"[pure_pcfm] loaded {ckpt_path}")

    w_gt, a_gt, f_gt, n_t, H, W = load_ns_test_batch(args.data_test, args.num_samples, device)
    x_grid = torch.linspace(0, 1.0, H, device=device)
    y_grid = torch.linspace(0, 1.0, W, device=device)
    t_grid = torch.linspace(0, 1.0, n_t, device=device)
    grid = make_grid((H, W, n_t), device)
    sampler = FFM_NS_sampler(model)

    w_van_list, w_pcfm_list = [], []
    for i in range(args.num_samples):
        u_true_hwT = w_gt[i].permute(1, 2, 0)  # (H,W,T), Residuals2D convention
        res = Residuals2D(data=u_true_hwT.unsqueeze(0), x=x_grid, y=y_grid, t_grid=t_grid,
                          nx=H, ny=W, nt=n_t)
        u0 = model.gp.sample(grid, (H, W, n_t), n_samples=1).to(device)
        u_van = sampler.vanilla_sample(u0, n_step=args.n_step)
        u_pcfm = sampler.pcfm_sample(u0=u0, n_step=args.n_step, hfunc=res.full_residual_ns,
                                     mode="least_squares", newtonsteps=1,
                                     **INTERP_PRESETS[args.interp])
        w_van_list.append(u_van[0].permute(2, 0, 1))   # back to (T,H,W)
        w_pcfm_list.append(u_pcfm[0].permute(2, 0, 1))

    w_gt_np = w_gt.cpu().numpy()
    f_np = f_gt.cpu().numpy()
    w_van = torch.stack(w_van_list).cpu().numpy()
    w_pcfm = torch.stack(w_pcfm_list).cpu().numpy()

    for w_pred, title, stem in [
        (w_van, "Vanilla (unconstrained)", "pure_pcfm_vanilla"),
        (w_pcfm, f"PCFM (IC+mass projection, interp={args.interp})", f"pure_pcfm_{args.interp}"),
    ]:
        df = rich_metrics(w_pred, w_gt_np, f_np, device)
        write_rich(df, title, MET_DIR / f"{stem}.txt")
        save_triptych(w_gt_np, w_pred, FIG_DIR / f"{stem}.png", title)


def run_conditioned(args, device, which):
    ckpt_arg = {"cond_pcfm": args.ckpt_cond, "cond_vanilla": args.ckpt_cond,
                "cond_pbfm": args.ckpt_pbfm}[which]
    default = WEIGHTS / ("best_pbfm.pt" if which == "cond_pbfm" else "best_fm_conditioned.pt")
    ckpt_path = Path(ckpt_arg) if ckpt_arg else default
    if not ckpt_path.exists():
        print(f"[skip] {which}: checkpoint not found at {ckpt_path}")
        return

    w_gt, a_gt, f_gt, n_t, H, W = load_ns_test_batch(args.data_test, args.num_samples, device)
    model, w_scale = load_conditioned_checkpoint(ckpt_path, n_t, device)
    print(f"[{which}] loaded {ckpt_path} (w_scale={w_scale:.4f})")
    cond_a = (a_gt / w_scale).unsqueeze(1)   # (B,1,H,W), same normalization as training
    cond_f = f_gt.unsqueeze(1)               # (B,1,H,W), physical (matches training's cond_f)

    if which == "cond_vanilla":
        w_pred = vanilla_sample(model, cond_a, cond_f, n_t, H, W, w_scale, device,
                                timesteps=args.n_step)
        title, stem = "Vanilla (unconstrained) sampling", "conditioned_vanilla"
    elif which == "cond_pbfm":
        w_pred = vanilla_sample(model, cond_a, cond_f, n_t, H, W, w_scale, device,
                                timesteps=args.n_step)
        title, stem = "PBFM (deterministic sampler)", "conditioned_pbfm"
    elif which == "cond_pcfm":
        x_grid = torch.linspace(0, 1.0, H, device=device)
        y_grid = torch.linspace(0, 1.0, W, device=device)
        t_grid = torch.linspace(0, 1.0, n_t, device=device)
        residuals_list = [
            Residuals2D(data=w_gt[i].permute(1, 2, 0).unsqueeze(0), x=x_grid, y=y_grid,
                       t_grid=t_grid, nx=H, ny=W, nt=n_t)
            for i in range(args.num_samples)
        ]
        w_pred = pcfm_sample_with_physics(model, cond_a, cond_f, n_t, H, W, w_scale,
                                          residuals_list, device, timesteps=args.n_step)
        title, stem = "FiLM(IC,forcing) + PCFM sampling", "conditioned_pcfm"
    else:
        raise ValueError(which)

    w_pred_np = w_pred.detach().cpu().numpy()
    w_gt_np = w_gt.cpu().numpy()
    f_np = f_gt.cpu().numpy()
    df = rich_metrics(w_pred_np, w_gt_np, f_np, device)
    write_rich(df, title, MET_DIR / f"{stem}.txt")
    save_triptych(w_gt_np, w_pred_np, FIG_DIR / f"{stem}.png", title)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--technique", default="all",
                   choices=["all", "pure_pcfm", "cond_pcfm", "cond_vanilla", "cond_pbfm",
                            "ground_truth"])
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--n-step", type=int, default=200,
                   help="Euler/PCFM sampling steps. Paper's Appendix H uses 200 for "
                        "Navier-Stokes (100 for the simpler heat problem) -- fewer steps "
                        "makes the per-step Newton correction ill-conditioned and can "
                        "diverge to NaN/astronomical values.")
    p.add_argument("--interp", default="none", choices=list(INTERP_PRESETS),
                   help="pure_pcfm's guided-interpolation lambda schedule. Default 'none' "
                        "(recommended): the IC+mass residual is LINEAR in u, so the "
                        "Newton/least-squares projection alone is already exact -- no need "
                        "for the interpolation smoothing. 'increasing'/'fixed' route through "
                        "relaxed_penalty_constraint_interp_linear_detached's fixed-step "
                        "gradient descent, which is numerically fragile at NS's problem size "
                        "(observed diverging to 1e9-1e23 on a real trained checkpoint) -- a "
                        "divergence guard was added, but 'none' avoids the mechanism entirely")
    p.add_argument("--data-test",
                   default=str(REPO_ROOT / "datasets" / "data" / "ns_nw10_nf100_s64_t50_mu0.001.h5"))
    p.add_argument("--ckpt-uncond", default=None, help="override weights/best_fm_uncond.pt")
    p.add_argument("--ckpt-cond", default=None, help="override weights/best_fm_conditioned.pt")
    p.add_argument("--ckpt-pbfm", default=None, help="override weights/best_pbfm.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="'cpu' forces off-GPU evaluation -- use this to run evaluate.py "
                        "in a second terminal alongside training without competing for "
                        "GPU memory (the PCFM Jacobian alone is a few GB at this problem "
                        "size, on top of whatever the training process already holds)")
    p.add_argument("--outdir", default=None,
                   help="Where to write results/{metrics,figures}/ -- defaults to the repo "
                        "checkout (NS_2D/results/), which does NOT persist across a fresh "
                        "Kaggle session (only /kaggle/working does). Pass e.g. "
                        "/kaggle/working/ns/eval to make results survive a session restart.")
    args = p.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global FIG_DIR, MET_DIR
    if args.outdir:
        FIG_DIR = Path(args.outdir) / "figures"
        MET_DIR = Path(args.outdir) / "metrics"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)

    todo = (["ground_truth", "pure_pcfm", "cond_pbfm", "cond_pcfm", "cond_vanilla"]
            if args.technique == "all" else [args.technique])
    for t in todo:
        print(f"\n{'#'*70}\n# technique: {t}\n{'#'*70}")
        if t == "ground_truth":
            run_ground_truth(args, device)
        elif t == "pure_pcfm":
            run_pure_pcfm(args, device)
        else:
            run_conditioned(args, device, t)


if __name__ == "__main__":
    main()
