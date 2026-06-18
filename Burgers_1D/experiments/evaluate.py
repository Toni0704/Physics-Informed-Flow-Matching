#!/usr/bin/env python
"""
Evaluate the four 1D Burgers' flow-matching techniques and write per-technique
metrics + figures into results/.

  technique 1  PBFM (physics in training) + FiLM   -> conditioned_pbfm.{txt,png}
  technique 2  FiLM + PCFM sampling (physics @ sampling) -> conditioned_pcfm.{txt,png}
  technique 3  Pure PCFM on the unconditioned model (physics+IC+BC as hard
               constraints at sampling)            -> pure_pcfm.{txt,png}
  technique 4  FiLM + vanilla sampling (no physics) -> conditioned_vanilla.{txt,png}

Key correctness points (vs. a naive port):
  * The conditioned FiLM model lives in normalised [-1, 1] space and flows from
    Gaussian noise -> samples start from torch.randn and are DENORMALISED to
    physical units before any metric/plot.
  * The unconditioned FFM model lives in physical space and flows from its GP
    prior -> u0 = model.gp.sample(...), not torch.randn.
  * The conditioned PCFM projection uses the low-level pcfm_sample with a hfunc
    that denormalises internally (k=5), guided_interpolation=False (matching the
    notebook), not FFM_sampler.pcfm_sample on a normalised field.

Run (all techniques, or pick one with --technique):
    python experiments/evaluate.py
    python experiments/evaluate.py --technique cond_pcfm
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import add_pcfm_to_path
PCFM_DIR = add_pcfm_to_path()

from src.dataset import BurgersConditionedDataset
from src.models import PatchedFNO, PCFM_FiLMConditionedFNO

# Top-level imports from the vendored PCFM repo (now on sys.path)
from models import get_flow_model                # noqa: E402
from scripts.training.utils import load_config   # noqa: E402
from pcfm import Residuals, FFM_sampler          # noqa: E402
from pcfm.pcfm_sampling import pcfm_sample, make_grid  # noqa: E402

NX = NT = 101
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results" / "figures"
MET_DIR = REPO_ROOT / "results" / "metrics"
WEIGHTS = REPO_ROOT / "weights"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_conditioned_model(device):
    base = PatchedFNO(n_modes=[32, 32], in_channels=3, emb_channels=32,
                      hidden_channels=64, proj_channels=256, n_layers=4).to(device)
    return PCFM_FiLMConditionedFNO(base, feature_channels=32).to(device)


def load_film_checkpoint(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "_metadata" in sd:
        del sd["_metadata"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_val_batch(data_dir, device, num_samples):
    """First `num_samples` test samples (deterministic; shuffle off)."""
    train_ds = BurgersConditionedDataset(data_dir / "burgers_train_nIC80_nBC80.h5")
    U_MIN, U_MAX = float(train_ds.u_min), float(train_ds.u_max)
    test_ds = BurgersConditionedDataset(data_dir / "burgers_test_nIC30_nBC30.h5",
                                        u_min=train_ds.u_min, u_max=train_ds.u_max)
    x0 = test_ds.data[:num_samples].to(device)
    ic = test_ds.cond_ic[:num_samples].to(device)
    bc = test_ds.cond_bc[:num_samples].to(device)
    bcp = test_ds.cond_bc_phys[:num_samples].to(device)
    return x0, ic, bc, bcp, U_MIN, U_MAX


def denorm(x_norm, U_MIN, U_MAX):
    return (x_norm + 1.0) / 2.0 * (U_MAX - U_MIN) + U_MIN


def rich_metrics(u_pred, u_gt, bc_phys, device):
    """Per-sample Data/Phys/IC/BC metrics on PHYSICAL fields. Returns a DataFrame."""
    x_grid = torch.linspace(0, 1.0, NX, dtype=torch.float64, device=device)
    t_grid = torch.linspace(0, 1.0, NT, dtype=torch.float64, device=device)
    rows = []
    for i in range(u_pred.shape[0]):
        data_mse = float(np.mean((u_gt[i] - u_pred[i]) ** 2))
        gtr = float(u_gt[i].max() - u_gt[i].min())
        data_nrmse = 100.0 * np.sqrt(data_mse) / gtr if gtr > 1e-12 else 0.0

        up = torch.from_numpy(u_pred[i]).to(device, dtype=torch.float64)
        lb = bc_phys[i].to(torch.float64).view(1)
        res = Residuals(data=up.unsqueeze(0), x=x_grid, t_grid=t_grid,
                        nx=NX, nt=NT, left_bc=lb)
        pr = res.burgers_local_multistep_residual(up.flatten(), k=5)
        phys_mse = float(torch.mean(pr ** 2).item())
        phys_l2 = float(torch.sqrt(torch.sum(pr ** 2)).item())

        icg, icp = u_gt[i, :, 0], u_pred[i, :, 0]
        ic_mse = float(np.mean((icg - icp) ** 2))
        icr = float(icg.max() - icg.min())
        ic_nrmse = 100.0 * np.sqrt(ic_mse) / icr if icr > 1e-12 else 0.0

        bcg, bcp_ = u_gt[i, 0, :], u_pred[i, 0, :]
        bc_mse = float(np.mean((bcg - bcp_) ** 2))
        bcr = float(bcg.max() - bcg.min())
        bc_nrmse = 100.0 * np.sqrt(bc_mse) / bcr if bcr > 1e-12 else 0.0

        rows.append({
            "Sample": i + 1, "Data MSE": data_mse, "Data NRMSE (%)": round(data_nrmse, 4),
            "Phys MSE": phys_mse, "Phys L2": phys_l2,
            "IC MSE": ic_mse, "IC NRMSE (%)": round(ic_nrmse, 4),
            "BC MSE": bc_mse, "BC NRMSE (%)": round(bc_nrmse, 4),
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


def save_triptych(u_gt, u_pred, png_path, title):
    n = u_pred.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    for i in range(n):
        axes[i, 0].imshow(u_gt[i].T, origin="lower", aspect="auto", cmap="bwr")
        axes[i, 0].set_title("GT")
        axes[i, 1].imshow(u_pred[i].T, origin="lower", aspect="auto", cmap="bwr")
        axes[i, 1].set_title("Pred")
        im = axes[i, 2].imshow(np.abs(u_gt[i] - u_pred[i]).T, origin="lower",
                               aspect="auto", cmap="Reds")
        axes[i, 2].set_title("Absolute error")
        fig.colorbar(im, ax=axes[i, 2])
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[written] {png_path}")


# --------------------------------------------------------------------------- #
# Samplers (conditioned model: state-first, normalised space)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vanilla_sample(model, cond_ic, cond_bc, template, U_MIN, U_MAX, device, timesteps=50):
    model.eval()
    B = cond_ic.shape[0]
    xt = torch.randn_like(template[:B])
    dt = 1.0 / timesteps
    for i in range(timesteps):
        t_tensor = torch.full((B,), i / timesteps, device=device)
        vf = model(xt, t_tensor, cond_ic, cond_bc)
        xt = xt + vf * dt
    return denorm(xt, U_MIN, U_MAX)


@torch.no_grad()
def sample_pbfm(model, cond_ic, cond_bc, template, U_MIN, U_MAX, device,
                num_steps=100, t_star=0.0):
    """Unified PBFM sampler. t_star=0 -> deterministic Euler; t_star>0 -> stochastic."""
    model.eval()
    B = cond_ic.shape[0]
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    xt = torch.randn_like(template[:B])
    for i in range(num_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        t_tensor = torch.full((B,), t_cur.item(), device=device)
        vf = model(xt, t_tensor, cond_ic, cond_bc)
        if t_cur < t_star:
            xt_one = xt + (1.0 - t_cur) * vf
            new_noise = torch.randn_like(xt_one)
            xt = (1.0 - t_next) * new_noise + t_next * xt_one
        else:
            xt = xt + (t_next - t_cur) * vf
    return denorm(xt, U_MIN, U_MAX)


def pcfm_sample_with_physics(model, cond_ic, cond_bc, cond_bc_phys, template,
                             U_MIN, U_MAX, device, timesteps=50, correction_steps=3):
    """FiLM model + PCFM hard projection (per-sample Jacobian/Newton).

    The hfunc denormalises the normalised state to physical units before the
    Burgers residual (k=5); guided_interpolation=False (matching the notebook).
    """
    model.eval()
    B = cond_ic.shape[0]
    xt = torch.randn_like(template[:B])
    dt = 1.0 / timesteps
    x_grid = torch.linspace(0, 1.0, NX, dtype=torch.float64, device=device)
    t_grid = torch.linspace(0, 1.0, NT, dtype=torch.float64, device=device)

    for i in range(timesteps):
        t_val = i / timesteps
        t_tensor = torch.full((B,), t_val, device=device)
        with torch.no_grad():
            vf = model(xt, t_tensor, cond_ic, cond_bc)
        vf_corrected = torch.zeros_like(vf)

        for b in range(B):
            u_norm_flat = xt[b].flatten()
            v_norm_flat = vf[b].flatten()
            u0_flat = torch.zeros_like(u_norm_flat)
            left_bc_b = cond_bc_phys[b].to(torch.float64).view(1)

            def hfunc_single(u_flat_in, _lb=left_bc_b):
                u_real = ((u_flat_in + 1.0) / 2.0 * (U_MAX - U_MIN)) + U_MIN
                u_fp64 = u_real.to(torch.float64)
                res_obj = Residuals(
                    data=u_fp64.view(NX, NT).unsqueeze(0),
                    x=x_grid, t_grid=t_grid, nx=NX, nt=NT, left_bc=_lb,
                )
                return res_obj.burgers_local_multistep_residual(u_fp64, k=5).to(torch.float32)

            with torch.enable_grad():
                proj_vf_flat = pcfm_sample(
                    u_flat=u_norm_flat, v_flat=v_norm_flat,
                    t=torch.tensor(t_val, device=device), u0_flat=u0_flat, dt=dt,
                    hfunc=hfunc_single, mode="least_squares",
                    newtonsteps=correction_steps, guided_interpolation=False,
                )
            vf_corrected[b] = proj_vf_flat.view(xt[b].shape)

        xt = xt + vf_corrected * dt

    return denorm(xt, U_MIN, U_MAX)


# --------------------------------------------------------------------------- #
# Technique runners
# --------------------------------------------------------------------------- #
def run_conditioned(data_dir, device, num_samples, which):
    """which in {'cond_pcfm','cond_vanilla','cond_pbfm'}."""
    x0, ic, bc, bcp, U_MIN, U_MAX = load_val_batch(data_dir, device, num_samples)
    u_gt = denorm(x0, U_MIN, U_MAX).cpu().numpy()

    if which == "cond_pcfm":
        model = load_film_checkpoint(build_conditioned_model(device),
                                     WEIGHTS / "best_fm_conditioned.pt", device)
        u_pred = pcfm_sample_with_physics(model, ic, bc, bcp, x0, U_MIN, U_MAX, device,
                                          timesteps=50, correction_steps=3).cpu().numpy()
        title, stem = "PCFM Sampling Metrics", "conditioned_pcfm"
    elif which == "cond_vanilla":
        model = load_film_checkpoint(build_conditioned_model(device),
                                     WEIGHTS / "best_fm_conditioned.pt", device)
        u_pred = vanilla_sample(model, ic, bc, x0, U_MIN, U_MAX, device,
                                timesteps=50).cpu().numpy()
        title, stem = "Vanilla (Unconstrained) Sampling Metrics", "conditioned_vanilla"
    elif which == "cond_pbfm":
        model = load_film_checkpoint(build_conditioned_model(device),
                                     WEIGHTS / "best_pbfm.pt", device)
        u_pred = sample_pbfm(model, ic, bc, x0, U_MIN, U_MAX, device,
                             num_steps=100, t_star=0.0).cpu().numpy()
        title, stem = "PBFM (Deterministic sampler) metrics", "conditioned_pbfm"
    else:
        raise ValueError(which)

    df = rich_metrics(u_pred, u_gt, bcp, device)
    write_rich(df, title, MET_DIR / f"{stem}.txt")
    save_triptych(u_gt, u_pred, FIG_DIR / f"{stem}.png", title)


def run_pure_pcfm(data_dir, device, num_samples):
    """Technique 3: unconditioned FFM model, GP prior, hard physics+IC+BC at sampling."""
    config = load_config(str(Path(PCFM_DIR) / "configs" / "burgers1d.yml"))
    model = get_flow_model(config.model, config.encoder).to(device)
    ckpt = torch.load(WEIGHTS / "best_fm_uncond.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    test_path = data_dir / "burgers_test_nIC30_nBC30.h5"
    with h5py.File(test_path, "r") as f:
        bc_val = torch.from_numpy(f["bc"][:]).to(device)
        u_all = f["u"][:num_samples, 0]   # physical, BC index 0, varying IC index

    x_grid = torch.linspace(0, 1.0, NX).to(device)
    t_grid = torch.linspace(0, 1.0, NT).to(device)
    grid = make_grid((NX, NT), device)
    sampler = FFM_sampler(model, model.gp)

    error_types = ["Full Constraint MSE", "Original Burgers Eq MSE",
                   "Initial Condition MSE", "Mass Conservation MSE"]

    def evaluate_constraints(u_sample, res):
        uf = u_sample.flatten()
        return {
            "Full Constraint MSE": torch.mean(res.full_residual_burgers(uf) ** 2).item(),
            "Original Burgers Eq MSE": torch.mean(res.burgers_local_multistep_residual(uf, k=100) ** 2).item(),
            "Initial Condition MSE": torch.mean(res.ic_residual(uf) ** 2).item(),
            "Mass Conservation MSE": torch.mean(res.mass_residual_burgers(uf) ** 2).item(),
        }

    hist = {s: {k: [] for k in error_types} for s in ("gt", "vanilla", "pcfm")}
    last = None
    for i in range(num_samples):
        u_true = torch.from_numpy(u_all[i]).to(device)
        residuals = Residuals(data=u_true.unsqueeze(0), x=x_grid, t_grid=t_grid,
                              nx=NX, nt=NT, left_bc=torch.tensor([bc_val[0]]).to(device))
        u0 = model.gp.sample(grid, (NX, NT), n_samples=1).to(device)
        u_pcfm = sampler.pcfm_sample(u0=u0, n_step=100,
                                     hfunc=residuals.full_residual_burgers,
                                     mode="least_squares", newtonsteps=1)
        u_van = sampler.vanilla_sample(u0, n_step=100)
        gt_m = evaluate_constraints(u_true, residuals)
        van_m = evaluate_constraints(u_van[0], residuals)
        pcfm_m = evaluate_constraints(u_pcfm[0], residuals)
        for k in error_types:
            hist["gt"][k].append(gt_m[k]); hist["vanilla"][k].append(van_m[k]); hist["pcfm"][k].append(pcfm_m[k])
        last = (u_true.cpu().numpy(), u_van[0].cpu().numpy(), u_pcfm[0].cpu().numpy())

    # One constraint table per sample (faithful to the notebook, which prints a
    # table per sample rather than averaging). The committed pure_pcfm.txt is one
    # such representative sample's table.
    txt = MET_DIR / "pure_pcfm.txt"
    with open(txt, "w") as fh:
        for i in range(num_samples):
            fh.write(f"Sample {i + 1} of {num_samples}\n")
            fh.write(f"{'Constraint Abidement Comparison (MSE)':^80}\n")
            fh.write("-" * 80 + "\n")
            fh.write(f"{'Metric':<25} | {'Ground Truth':<15} | {'Vanilla FFM':<15} | {'PCFM (Ours)':<15}\n")
            fh.write("-" * 80 + "\n")
            for k in error_types:
                g, v, p = hist["gt"][k][i], hist["vanilla"][k][i], hist["pcfm"][k][i]
                fh.write(f"{k:<25} | {g:<15.2e} | {v:<15.2e} | {p:<15.2e}\n")
            fh.write("-" * 80 + "\n\n")
    print(f"[written] {txt}")

    # Figure: GT / Vanilla / PCFM for the last sample
    gt_im, van_im, pcfm_im = last
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, img, ttl in zip(axes, (gt_im, van_im, pcfm_im),
                            ("Ground Truth", "Vanilla Generated", "PCFM Generated")):
        im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="bwr", vmin=0, vmax=1)
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("t")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.savefig(FIG_DIR / "pure_pcfm.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[written] {FIG_DIR / 'pure_pcfm.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--technique", default="all",
                   choices=["all", "cond_pcfm", "cond_vanilla", "cond_pbfm", "pure_pcfm"])
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--data-dir", default=str(Path(PCFM_DIR) / "datasets" / "data"))
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    todo = (["cond_pbfm", "cond_pcfm", "pure_pcfm", "cond_vanilla"]
            if args.technique == "all" else [args.technique])
    for t in todo:
        print(f"\n{'#'*70}\n# technique: {t}\n{'#'*70}")
        if t == "pure_pcfm":
            run_pure_pcfm(data_dir, device, args.num_samples)
        else:
            run_conditioned(data_dir, device, args.num_samples, t)


if __name__ == "__main__":
    main()
