import sys
import os
import gc
import json
import argparse
import platform
import statistics
import torch
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import easydict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from models import get_flow_model
from scripts.training.utils import load_config
from pcfm import Residuals3D, FFM_Darcy3D_sampler
from pcfm.pcfm_sampling import make_grid


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_table(gt, vanilla, pcfm, pcfm_last, path):
    rows = []
    for k in gt.keys():
        rows.append({
            "metric": k,
            "ground_truth": gt[k],
            "vanilla": vanilla[k],
            "pcfm": pcfm[k],
            "pcfm_last": pcfm_last[k]
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def plot_slice(u, title, path):
    """Plot middle z-slice of the 3D pressure field"""
    mid = u.shape[-1] // 2
    plt.figure(figsize=(6, 5))
    plt.imshow(u[:, :, mid].cpu(), cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_darcy3d_constraints(p_sample, residuals_obj, p_true):
    """Evaluate constraint satisfaction and accuracy for a single sample"""
    p_flat = p_sample.flatten()

    full_res = residuals_obj.full_residual_darcy3d(p_flat)
    full_loss = torch.mean(full_res ** 2).item()

    mse_vs_gt = torch.mean((p_sample - p_true) ** 2).item()

    return {
        "PDE Residual MSE": full_loss,
        "MSE vs Ground Truth": mse_vs_gt,
    }


def load_darcy3d_model(config_path, ckpt_path, device):
    config = load_config(config_path)
    with torch.no_grad():
        model = get_flow_model(config.model, config.encoder).to(device)

        torch.serialization.add_safe_globals([easydict.EasyDict])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "state_dict" in ckpt:
            sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
        else:
            sd = ckpt["model"]

        model.load_state_dict(sd, strict=True)
        model.eval()

    del ckpt, sd
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model


def run_one_sample(model, sampler, data_path, sample_idx, device, args, outdir,
                    save_figures=True):
    """Runs vanilla / PCFM(fixed) / PCFM(increasing) on a single test sample and
    writes its per-sample outputs (metrics json/csv, prediction tensors, figures)
    into outdir. Returns the four metrics dicts (gt, vanilla, pcfm_fixed,
    pcfm_increasing) for the caller to aggregate across samples.
    """
    os.makedirs(outdir, exist_ok=True)

    f = h5py.File(data_path, "r")
    p_true = torch.from_numpy(f["u"][sample_idx]).to(device)
    k_field = torch.from_numpy(f["k"][sample_idx]).to(device)
    p_left, p_right = [float(v) for v in f["bc"][sample_idx]]
    nx, ny, nz = p_true.shape
    f.close()

    residuals = Residuals3D(k=k_field, p_left=p_left, p_right=p_right, nx=nx, ny=ny, nz=nz)

    grid = make_grid((nx, ny, nz), device)
    u0 = model.gp.sample(grid, (nx, ny, nz), n_samples=1).to(device)
    del grid
    if device == "cuda":
        torch.cuda.empty_cache()

    def lambda_schedule(t):
        return 1e-2 * torch.exp(torch.tensor(5.0 * t)).item()

    interpolation_params_fixed = {'custom_lam': 1e0, 'step_size': 1e-2, 'num_steps': 20}
    interpolation_params_increasing = {
        'custom_lam': 1e-2, 'step_size': 1e-2, 'num_steps': 20,
        'lambda_schedule': lambda_schedule,
    }

    print(f"[idx {sample_idx}] Running Vanilla FFM...")
    with torch.no_grad():
        p_vanilla = sampler.vanilla_sample(u0, n_step=args.n_step)
        vanilla_metrics = evaluate_darcy3d_constraints(p_vanilla[0], residuals, p_true)

    print(f"[idx {sample_idx}] Running PCFM (fixed lambda=1e0)...")
    p_pcfm_fixed = sampler.pcfm_sample(
        u0=u0, n_step=args.n_step, hfunc=residuals.full_residual_darcy3d,
        mode=args.mode, newtonsteps=args.newtonsteps, guided_interpolation=True,
        interpolation_params=interpolation_params_fixed, eps=args.eps,
    )
    pcfm_fixed_metrics = evaluate_darcy3d_constraints(p_pcfm_fixed[0], residuals, p_true)

    print(f"[idx {sample_idx}] Running PCFM (increasing lambda: 0->1e0)...")
    p_pcfm_increasing = sampler.pcfm_sample(
        u0=u0, n_step=args.n_step, hfunc=residuals.full_residual_darcy3d,
        mode=args.mode, newtonsteps=args.newtonsteps, guided_interpolation=True,
        interpolation_params=interpolation_params_increasing, eps=args.eps,
    )
    pcfm_increasing_metrics = evaluate_darcy3d_constraints(p_pcfm_increasing[0], residuals, p_true)

    gt_metrics = evaluate_darcy3d_constraints(p_true, residuals, p_true)

    save_metrics(gt_metrics, os.path.join(outdir, "metrics_gt.json"))
    save_metrics(vanilla_metrics, os.path.join(outdir, "metrics_vanilla.json"))
    save_metrics(pcfm_fixed_metrics, os.path.join(outdir, "metrics_pcfm_fixed.json"))
    save_metrics(pcfm_increasing_metrics, os.path.join(outdir, "metrics_pcfm_increasing.json"))
    save_table(gt_metrics, vanilla_metrics, pcfm_fixed_metrics, pcfm_increasing_metrics,
               os.path.join(outdir, "metrics_table.csv"))

    torch.save(p_true.cpu(), os.path.join(outdir, "p_true.pt"))
    torch.save(p_vanilla.cpu(), os.path.join(outdir, "p_vanilla.pt"))
    torch.save(p_pcfm_fixed.cpu(), os.path.join(outdir, "p_pcfm_fixed.pt"))
    torch.save(p_pcfm_increasing.cpu(), os.path.join(outdir, "p_pcfm_increasing.pt"))

    if save_figures:
        plot_slice(p_true, "Ground Truth", os.path.join(outdir, "gt.png"))
        plot_slice(p_vanilla[0], "Vanilla FFM", os.path.join(outdir, "vanilla.png"))
        plot_slice(p_pcfm_fixed[0], "PCFM (λ=1e0)", os.path.join(outdir, "pcfm_fixed.png"))
        plot_slice(p_pcfm_increasing[0], "PCFM (λ: 0→1e0)",
                   os.path.join(outdir, "pcfm_increasing.png"))

    if device == "cuda":
        torch.cuda.empty_cache()

    return gt_metrics, vanilla_metrics, pcfm_fixed_metrics, pcfm_increasing_metrics


def aggregate(per_sample_metrics_list):
    """per_sample_metrics_list: list of dicts (one per sample), all with the same
    keys. Returns {key: {"mean": ..., "std": ...}} across the list.
    """
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


def save_aggregate_table(gt_agg, vanilla_agg, pcfm_fixed_agg, pcfm_increasing_agg, path):
    rows = []
    for k in gt_agg.keys():
        rows.append({
            "metric": k,
            "ground_truth_mean": gt_agg[k]["mean"], "ground_truth_std": gt_agg[k]["std"],
            "vanilla_mean": vanilla_agg[k]["mean"], "vanilla_std": vanilla_agg[k]["std"],
            "pcfm_fixed_mean": pcfm_fixed_agg[k]["mean"], "pcfm_fixed_std": pcfm_fixed_agg[k]["std"],
            "pcfm_increasing_mean": pcfm_increasing_agg[k]["mean"],
            "pcfm_increasing_std": pcfm_increasing_agg[k]["std"],
            "n": gt_agg[k]["n"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--sample_idx", type=int, default=0,
                     help="Single-sample mode (default, backward-compatible): evaluate exactly "
                          "this one test sample and write outputs directly into --outdir.")
parser.add_argument("--num_samples", type=int, default=None,
                     help="Batch mode: evaluate this many samples (indices 0..num_samples-1 "
                          "unless --sample_indices is given), write each sample's outputs into "
                          "--outdir/idx_<i>/, and write an aggregate mean/std table/json into "
                          "--outdir/aggregate_table.csv and aggregate_metrics.json. Overrides "
                          "--sample_idx when set.")
parser.add_argument("--sample_indices", type=str, default=None,
                     help="Batch mode only: comma-separated explicit sample indices (e.g. "
                          "'0,50,100,200,350,499') instead of the default range(num_samples).")
parser.add_argument("--outdir", required=True)

parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cpu", "mps", "cuda"],
    help="Device to use: auto|cpu|mps|cuda"
)

parser.add_argument("--n_step", type=int, default=20,
                     help="Euler/PCFM sampling steps. NOTE: unlike NS2D (paper Appendix H "
                          "specifies 200), this default has NOT been validated for Darcy3D -- "
                          "Darcy3D is steady-state so it may not be equally sensitive to "
                          "under-stepping, but that hasn't actually been checked. If results look "
                          "off (NaN, huge residuals), try increasing this before assuming a model bug.")
parser.add_argument("--mode", default="least_squares")
parser.add_argument("--newtonsteps", type=int, default=1)
parser.add_argument("--eps", type=float, default=1e-6)

args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)

# Device selection with MPS safeguard for FFT (requires macOS 14+)
if args.device == "cpu":
    device = "cpu"
elif args.device == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device == "mps":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

if device == "mps":
    mac_ver = platform.mac_ver()[0]
    try:
        major_ver = int(mac_ver.split(".")[0]) if mac_ver else 0
    except ValueError:
        major_ver = 0
    if major_ver < 14:
        print("MPS FFT requires macOS 14+. Falling back to CPU.")
        device = "cpu"

print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

model = load_darcy3d_model(args.config, args.ckpt, device)
sampler = FFM_Darcy3D_sampler(model)

if args.num_samples is not None:
    # --------------------------------------------------
    # BATCH MODE: loop over samples, aggregate
    # --------------------------------------------------
    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",")]
    else:
        indices = list(range(args.num_samples))

    gt_list, vanilla_list, pcfm_fixed_list, pcfm_increasing_list = [], [], [], []
    for idx in indices:
        sample_outdir = os.path.join(args.outdir, f"idx_{idx}")
        gt_m, van_m, pf_m, pi_m = run_one_sample(
            model, sampler, args.data, idx, device, args, sample_outdir)
        gt_list.append(gt_m)
        vanilla_list.append(van_m)
        pcfm_fixed_list.append(pf_m)
        pcfm_increasing_list.append(pi_m)

    gt_agg = aggregate(gt_list)
    vanilla_agg = aggregate(vanilla_list)
    pcfm_fixed_agg = aggregate(pcfm_fixed_list)
    pcfm_increasing_agg = aggregate(pcfm_increasing_list)

    save_metrics(gt_agg, os.path.join(args.outdir, "aggregate_metrics_gt.json"))
    save_metrics(vanilla_agg, os.path.join(args.outdir, "aggregate_metrics_vanilla.json"))
    save_metrics(pcfm_fixed_agg, os.path.join(args.outdir, "aggregate_metrics_pcfm_fixed.json"))
    save_metrics(pcfm_increasing_agg,
                 os.path.join(args.outdir, "aggregate_metrics_pcfm_increasing.json"))
    save_aggregate_table(gt_agg, vanilla_agg, pcfm_fixed_agg, pcfm_increasing_agg,
                          os.path.join(args.outdir, "aggregate_table.csv"))

    print(f"Run complete. N={len(indices)} samples: {indices}")
    print("Per-sample outputs:", os.path.join(args.outdir, "idx_<i>/"))
    print("Aggregate table:", os.path.join(args.outdir, "aggregate_table.csv"))

else:
    # --------------------------------------------------
    # SINGLE-SAMPLE MODE (original behavior, unchanged outputs)
    # --------------------------------------------------
    run_one_sample(model, sampler, args.data, args.sample_idx, device, args, args.outdir)
    print("Run complete.")
    print("Outputs saved to:", args.outdir)
