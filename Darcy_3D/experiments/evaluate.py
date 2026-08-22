#!/usr/bin/env python
"""
Evaluate the two FiLM-conditioned Darcy3D techniques:

  cond_vanilla  FiLM(k, BC)-conditioned model + unconstrained Euler sampling
  cond_pcfm     FiLM(k, BC)-conditioned model + PCFM hard flux-balance-residual
                projection at sampling time

Counterpart to scripts/training/run_pcfm_darcy3d.py, which covers the
UNconditioned-backbone techniques (vanilla / PCFM-fixed / PCFM-increasing).
Same metric names ("PDE Residual MSE", "MSE vs Ground Truth") and same
--num_samples/--sample_indices batch-mode + aggregate-table convention as
that script, so results are directly comparable across all four Darcy3D
sampling techniques.

Run:
    python experiments/evaluate.py --technique cond_vanilla --num_samples 20 \
        --ckpt /kaggle/working/darcy/weights/best_fm_conditioned.pt \
        --data /kaggle/input/.../darcy3d_test_n500.h5 \
        --outdir /kaggle/working/darcy/eval/eval_conditioned
    python experiments/evaluate.py --technique all ...   # both techniques in one run
"""

import argparse
import gc
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import h5py
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import REPO_ROOT, add_repo_to_path
from src.models import PatchedFNO3D, PCFM_FiLMConditionedFNO3D

add_repo_to_path()
from pcfm import Residuals3D                      # noqa: E402
from pcfm.pcfm_sampling import pcfm_3d_batched     # noqa: E402


# --------------------------------------------------
# UTILITIES (same shape/semantics as run_pcfm_darcy3d.py, kept separate
# since this script's model/conditioning interface is different)
# --------------------------------------------------

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def plot_slice(u, title, path):
    mid = u.shape[-1] // 2
    plt.figure(figsize=(6, 5))
    plt.imshow(u[:, :, mid].cpu(), cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_darcy3d_constraints(p_sample, residuals_obj, p_true):
    p_flat = p_sample.flatten()
    full_res = residuals_obj.full_residual_darcy3d(p_flat)
    full_loss = torch.mean(full_res ** 2).item()
    mse_vs_gt = torch.mean((p_sample - p_true) ** 2).item()
    return {"PDE Residual MSE": full_loss, "MSE vs Ground Truth": mse_vs_gt}


def aggregate(per_sample_metrics_list):
    keys = per_sample_metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in per_sample_metrics_list]
        out[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def save_aggregate_table(agg_by_technique, path):
    """agg_by_technique: {technique_name: aggregate_dict}"""
    metric_keys = next(iter(agg_by_technique.values())).keys()
    rows = []
    for k in metric_keys:
        row = {"metric": k}
        n = None
        for tech, agg in agg_by_technique.items():
            row[f"{tech}_mean"] = agg[k]["mean"]
            row[f"{tech}_std"] = agg[k]["std"]
            n = agg[k]["n"]
        row["n"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_conditioned_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_fno = PatchedFNO3D(n_modes=[8, 8, 8], in_channels=4, emb_channels=32,
                            hidden_channels=32, proj_channels=128, n_layers=4).to(device)
    model = PCFM_FiLMConditionedFNO3D(base_fno, feature_channels=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    norm = {"u_min": ckpt["u_min"], "u_max": ckpt["u_max"],
            "logk_min": ckpt["logk_min"], "logk_max": ckpt["logk_max"]}
    return model, norm


def load_and_normalize_sample(data_path, sample_idx, norm, device):
    with h5py.File(data_path, "r") as f:
        p_true = torch.from_numpy(f["u"][sample_idx]).to(device)
        k_phys = torch.from_numpy(f["k"][sample_idx]).to(device)
        p_left, p_right = [float(v) for v in f["bc"][sample_idx]]

    u_min, u_max = norm["u_min"], norm["u_max"]
    logk_min, logk_max = norm["logk_min"], norm["logk_max"]

    logk = torch.log(k_phys)
    cond_k = (2.0 * (logk - logk_min) / (logk_max - logk_min + 1e-8) - 1.0).unsqueeze(0).unsqueeze(0)
    bc_phys = torch.tensor([p_left, p_right], device=device, dtype=torch.float32)
    cond_bc = (2.0 * (bc_phys - u_min) / (u_max - u_min + 1e-8) - 1.0).unsqueeze(0)

    return p_true, k_phys, bc_phys, cond_k, cond_bc


def denormalize(x_norm, norm):
    return (x_norm + 1.0) / 2.0 * (norm["u_max"] - norm["u_min"]) + norm["u_min"]


@torch.no_grad()
def vanilla_sample(model, cond_k, cond_bc, shape, device, timesteps=200):
    nx, ny, nz = shape
    xt = torch.randn(1, nx, ny, nz, device=device)
    dt = 1.0 / timesteps
    for i in range(timesteps):
        t = torch.full((1,), i / timesteps, device=device)
        vf = model(xt, t, cond_k, cond_bc)
        xt = xt + vf * dt
    return xt[0]


def pcfm_sample_with_physics(model, cond_k, cond_bc, residuals, shape, device,
                              timesteps=200, correction_steps=1, mode="least_squares",
                              eps=1e-6):
    nx, ny, nz = shape
    xt = torch.randn(1, nx, ny, nz, device=device)
    dt = 1.0 / timesteps

    def hfunc(u_flat_in):
        return residuals.full_residual_darcy3d(u_flat_in.to(torch.float64)).to(torch.float32)

    for i in range(timesteps):
        t_val = i / timesteps
        t = torch.full((1,), t_val, device=device)
        with torch.no_grad():
            vf = model(xt, t, cond_k, cond_bc)
        with torch.enable_grad():
            vf_corrected = pcfm_3d_batched(
                ut=xt, vf=vf, t=torch.tensor(t_val, device=device), u0=xt, dt=dt,
                hfunc=hfunc, mode=mode, newtonsteps=correction_steps,
                guided_interpolation=False, eps=eps,
            )
        xt = (xt + vf_corrected * dt).detach()
    return xt[0]


# --------------------------------------------------
# TECHNIQUE RUNNERS
# --------------------------------------------------

def run_technique(which, model, norm, data_path, indices, device, args, outdir):
    per_sample_metrics = []
    for idx in indices:
        p_true, k_phys, bc_phys, cond_k, cond_bc = load_and_normalize_sample(
            data_path, idx, norm, device)
        nx, ny, nz = p_true.shape
        residuals = Residuals3D(k=k_phys, p_left=bc_phys[0].item(), p_right=bc_phys[1].item(),
                                nx=nx, ny=ny, nz=nz)

        if which in ("cond_vanilla", "cond_pbfm"):
            # PBFM enforces physics at TRAINING time (via the physics loss), not at
            # sampling time -- so it uses the same plain deterministic Euler sampler
            # as cond_vanilla, just with the PBFM-trained checkpoint loaded instead
            # of the plain-FM one. Mirrors NS2D's evaluate.py convention.
            print(f"[{which}] idx {idx}: running vanilla Euler sampling...")
            p_pred_norm = vanilla_sample(model, cond_k, cond_bc, (nx, ny, nz), device,
                                         timesteps=args.n_step)
        elif which == "cond_pcfm":
            print(f"[{which}] idx {idx}: running PCFM-constrained sampling...")
            p_pred_norm = pcfm_sample_with_physics(
                model, cond_k, cond_bc, residuals, (nx, ny, nz), device,
                timesteps=args.n_step, correction_steps=args.newtonsteps,
                mode=args.mode, eps=args.eps)
        else:
            raise ValueError(which)

        p_pred = denormalize(p_pred_norm, norm)
        metrics = evaluate_darcy3d_constraints(p_pred, residuals, p_true)
        per_sample_metrics.append(metrics)

        sample_outdir = os.path.join(outdir, which, f"idx_{idx}")
        os.makedirs(sample_outdir, exist_ok=True)
        save_metrics(metrics, os.path.join(sample_outdir, "metrics.json"))
        torch.save(p_pred.cpu(), os.path.join(sample_outdir, "p_pred.pt"))
        torch.save(p_true.cpu(), os.path.join(sample_outdir, "p_true.pt"))
        plot_slice(p_pred, f"{which} (idx {idx})", os.path.join(sample_outdir, "pred.png"))

        if device == "cuda":
            torch.cuda.empty_cache()

    return per_sample_metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--technique", default="all",
                   choices=["all", "cond_vanilla", "cond_pcfm", "cond_pbfm"])
    p.add_argument("--ckpt", required=True, help="best_fm_conditioned.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--sample_indices", type=str, default=None,
                   help="comma-separated explicit indices, overrides --num_samples")
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--n_step", type=int, default=200,
                   help="Euler/PCFM sampling steps. NS2D's paper-matched default is 200; "
                        "Darcy3D's own unconditioned techniques (run_pcfm_darcy3d.py) default "
                        "to 20 and were empirically checked as sufficient there (linear PDE, "
                        "residual collapses to ~1e-8 with no sign of divergence). Not yet "
                        "cross-checked for this conditioned-model path specifically -- 200 is a "
                        "conservative starting point pending an actual check.")
    p.add_argument("--mode", default="least_squares")
    p.add_argument("--newtonsteps", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps":
        mac_ver = platform.mac_ver()[0]
        major_ver = int(mac_ver.split(".")[0]) if mac_ver else 0
        if major_ver < 14:
            print("MPS FFT requires macOS 14+. Falling back to CPU.")
            device = "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.outdir, exist_ok=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    model, norm = load_conditioned_model(args.ckpt, device)

    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",")]
    else:
        indices = list(range(args.num_samples))

    todo = ["cond_vanilla", "cond_pcfm"] if args.technique == "all" else [args.technique]
    agg_by_technique = {}
    for t in todo:
        print(f"\n{'#'*70}\n# technique: {t}\n{'#'*70}")
        per_sample = run_technique(t, model, norm, args.data, indices, device, args, args.outdir)
        agg = aggregate(per_sample)
        agg_by_technique[t] = agg
        save_metrics(agg, os.path.join(args.outdir, f"aggregate_metrics_{t}.json"))
        print(f"[{t}] N={len(indices)}: "
              f"PDE Residual MSE {agg['PDE Residual MSE']['mean']:.4e} ± "
              f"{agg['PDE Residual MSE']['std']:.4e}, "
              f"MSE vs GT {agg['MSE vs Ground Truth']['mean']:.4e} ± "
              f"{agg['MSE vs Ground Truth']['std']:.4e}")

    save_aggregate_table(agg_by_technique, os.path.join(args.outdir, "aggregate_table.csv"))
    print("\nRun complete. Aggregate table:", os.path.join(args.outdir, "aggregate_table.csv"))


if __name__ == "__main__":
    main()
