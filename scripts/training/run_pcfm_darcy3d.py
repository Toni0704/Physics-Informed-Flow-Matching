import sys
import os
import gc
import json
import argparse
import platform
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


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--sample_idx", type=int, default=0)
parser.add_argument("--outdir", required=True)

parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cpu", "mps", "cuda"],
    help="Device to use: auto|cpu|mps|cuda"
)

parser.add_argument("--n_step", type=int, default=20)
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

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

config = load_config(args.config)

with torch.no_grad():
    model = get_flow_model(config.model, config.encoder).to(device)

    torch.serialization.add_safe_globals([easydict.EasyDict])
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

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


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

f = h5py.File(args.data, "r")
p_true = torch.from_numpy(f["u"][args.sample_idx]).to(device)
k_field = torch.from_numpy(f["k"][args.sample_idx]).to(device)
p_left, p_right = [float(v) for v in f["bc"][args.sample_idx]]
nx, ny, nz = p_true.shape

residuals = Residuals3D(
    k=k_field,
    p_left=p_left,
    p_right=p_right,
    nx=nx, ny=ny, nz=nz,
)


# --------------------------------------------------
# PRIOR
# --------------------------------------------------

grid = make_grid((nx, ny, nz), device)
u0 = model.gp.sample(grid, (nx, ny, nz), n_samples=1).to(device)
del grid
if device == "cuda":
    torch.cuda.empty_cache()


sampler = FFM_Darcy3D_sampler(model)


def lambda_schedule(t):
    """
    Schedule lambda to increase with flow time.
    t: flow time in [0, 1]
    """
    return 1e-2 * torch.exp(torch.tensor(5.0 * t)).item()

# --------------------------------------------------
# SAMPLING
# --------------------------------------------------

interpolation_params_fixed = {
    'custom_lam': 1e0,
    'step_size': 1e-2,
    'num_steps': 20
}

interpolation_params_increasing = {
    'custom_lam': 1e-2,
    'step_size': 1e-2,
    'num_steps': 20,
    'lambda_schedule': lambda_schedule
}

# 1. Vanilla FFM (no constraints)
print("Running Vanilla FFM...")
with torch.no_grad():
    p_vanilla = sampler.vanilla_sample(u0, n_step=args.n_step)
    vanilla_metrics = evaluate_darcy3d_constraints(p_vanilla[0], residuals, p_true)

# 2. PCFM with fixed lambda (standard)
print("Running PCFM (fixed lambda=1e0)...")
p_pcfm_fixed = sampler.pcfm_sample(
    u0=u0,
    n_step=args.n_step,
    hfunc=residuals.full_residual_darcy3d,
    mode=args.mode,
    newtonsteps=args.newtonsteps,
    guided_interpolation=True,
    interpolation_params=interpolation_params_fixed,
    eps=args.eps
)
pcfm_fixed_metrics = evaluate_darcy3d_constraints(p_pcfm_fixed[0], residuals, p_true)

# 3. PCFM with increasing lambda
print("Running PCFM (increasing lambda: 0->1e0)...")
p_pcfm_increasing = sampler.pcfm_sample(
    u0=u0,
    n_step=args.n_step,
    hfunc=residuals.full_residual_darcy3d,
    mode=args.mode,
    newtonsteps=args.newtonsteps,
    guided_interpolation=True,
    interpolation_params=interpolation_params_increasing,
    eps=args.eps
)
pcfm_increasing_metrics = evaluate_darcy3d_constraints(p_pcfm_increasing[0], residuals, p_true)


gt_metrics = evaluate_darcy3d_constraints(p_true, residuals, p_true)

# --------------------------------------------------
# SAVE EVERYTHING
# --------------------------------------------------

save_metrics(gt_metrics, os.path.join(args.outdir, "metrics_gt.json"))
save_metrics(vanilla_metrics, os.path.join(args.outdir, "metrics_vanilla.json"))
save_metrics(pcfm_fixed_metrics, os.path.join(args.outdir, "metrics_pcfm_fixed.json"))
save_metrics(pcfm_increasing_metrics, os.path.join(args.outdir, "metrics_pcfm_increasing.json"))

save_table(
    gt_metrics,
    vanilla_metrics,
    pcfm_fixed_metrics,
    pcfm_increasing_metrics,
    os.path.join(args.outdir, "metrics_table.csv")
)

torch.save(p_true.cpu(), os.path.join(args.outdir, "p_true.pt"))
torch.save(p_vanilla.cpu(), os.path.join(args.outdir, "p_vanilla.pt"))
torch.save(p_pcfm_fixed.cpu(), os.path.join(args.outdir, "p_pcfm_fixed.pt"))
torch.save(p_pcfm_increasing.cpu(), os.path.join(args.outdir, "p_pcfm_increasing.pt"))

plot_slice(p_true, "Ground Truth", os.path.join(args.outdir, "gt.png"))
plot_slice(p_vanilla[0], "Vanilla FFM", os.path.join(args.outdir, "vanilla.png"))
plot_slice(p_pcfm_fixed[0], "PCFM (λ=1e0)", os.path.join(args.outdir, "pcfm_fixed.png"))
plot_slice(p_pcfm_increasing[0], "PCFM (λ: 0→1e0)", os.path.join(args.outdir, "pcfm_increasing.png"))

f.close()

print("Run complete.")
print("Outputs saved to:", args.outdir)
