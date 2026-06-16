import math
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
from pcfm import Residuals2D, FFM_NS_sampler
from pcfm.pcfm_sampling import make_grid


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def plot_slice(u, title, path):
    """Plot middle time slice"""
    mid = u.shape[-1] // 2
    plt.figure(figsize=(6, 5))
    plt.imshow(u[:, :, mid].cpu(), cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def ns_equation_residual(u_flat, nx, ny, nt, device, visc=1e-3):
    # Reshape to (nx, ny, nt)
    w = u_flat.view(nx, ny, nt)

    # Time step
    T = 49.0  # Final time from solver
    dt = T / (nt - 1)

    # Spatial grid
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # Wavenumbers for Fourier differentiation (matching solver)
    k_max = nx // 2

    k_y = torch.cat((
        torch.arange(0, k_max, device=device),
        torch.arange(-k_max, 0, device=device)
    ), 0).repeat(nx, 1).float()

    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0  # Avoid division by zero

    residuals = []

    # Compute residual at interior time points
    for n in range(1, nt - 1):
        # Get vorticity at current time
        w_curr = w[:, :, n]

        # Transform to Fourier space
        w_h = torch.fft.fftn(w_curr, dim=[0, 1], norm='backward')

        # 1) Solve for stream function: ∆ψ = w
        psi_h = w_h / lap

        # 2) Compute velocity from stream function
        # u = ∂ψ/∂y
        u_vel_h = psi_h.clone()
        u_vel_h_real_temp = u_vel_h.real.clone()
        u_vel_h.real = -2 * math.pi * k_y * u_vel_h.imag
        u_vel_h.imag = 2 * math.pi * k_y * u_vel_h_real_temp
        u_vel = torch.fft.ifftn(u_vel_h, dim=[0, 1], norm='backward').real

        # v = -∂ψ/∂x
        v_vel_h = psi_h.clone()
        v_vel_h_real_temp = v_vel_h.real.clone()
        v_vel_h.real = 2 * math.pi * k_x * v_vel_h.imag
        v_vel_h.imag = -2 * math.pi * k_x * v_vel_h_real_temp
        v_vel = torch.fft.ifftn(v_vel_h, dim=[0, 1], norm='backward').real

        # 3) Compute vorticity gradients
        # ∂w/∂x
        w_x_h = w_h.clone()
        w_x_h_real_temp = w_x_h.real.clone()
        w_x_h.real = -2 * math.pi * k_x * w_x_h.imag
        w_x_h.imag = 2 * math.pi * k_x * w_x_h_real_temp
        w_x = torch.fft.ifftn(w_x_h, dim=[0, 1], norm='backward').real

        # ∂w/∂y
        w_y_h = w_h.clone()
        w_y_h_real_temp = w_y_h.real.clone()
        w_y_h.real = -2 * math.pi * k_y * w_y_h.imag
        w_y_h.imag = 2 * math.pi * k_y * w_y_h_real_temp
        w_y = torch.fft.ifftn(w_y_h, dim=[0, 1], norm='backward').real

        # 4) Advection term: u·∇w
        advection = u_vel * w_x + v_vel * w_y

        # 5) Diffusion term: ν∆w
        laplacian_h = -lap * w_h
        laplacian = torch.fft.ifftn(laplacian_h, dim=[0, 1], norm='backward').real
        diffusion = visc * laplacian

        # 6) Forcing term (approx 0)
        forcing = 0.0

        # 7) Time derivative: ∂w/∂t (central difference)
        w_prev = w[:, :, n - 1]
        w_next = w[:, :, n + 1]
        dwdt = (w_next - w_prev) / (2 * dt)

        # 8) NS equation residual: ∂w/∂t + u·∇w - ν∆w - f
        residual = dwdt + advection - diffusion - forcing
        residuals.append(residual.flatten())

    return torch.cat(residuals, dim=0)


def evaluate_ns_constraints(u_sample, residuals_obj):
    """Evaluate all constraints for a single sample"""
    u_flat = u_sample.flatten()

    full_res = residuals_obj.full_residual_ns(u_flat)
    full_loss = torch.mean(full_res ** 2).item()

    ic_res = residuals_obj.ic_residual_ns(u_flat)
    ic_loss = torch.mean(ic_res ** 2).item()

    mass_res = residuals_obj.mass_residual_ns(u_flat)
    mass_loss = torch.mean(mass_res ** 2).item()

    ns_eqn_res = ns_equation_residual(
        u_flat,
        nx=residuals_obj.nx,
        ny=residuals_obj.ny,
        nt=residuals_obj.nt,
        device=u_sample.device
    )
    ns_eqn_loss = torch.mean(ns_eqn_res ** 2).item()

    return {
        "Full Constraint MSE": full_loss,
        "Initial Condition MSE": ic_loss,
        "Mass Conservation MSE": mass_loss,
        "NS Equation MSE": ns_eqn_loss
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

parser.add_argument("--n_step", type=int, default=100)
parser.add_argument("--mode", default="least_squares")
parser.add_argument("--newtonsteps", type=int, default=2)
parser.add_argument("--guided", action="store_true")
parser.add_argument("--eps", type=float, default=1e-6)
parser.add_argument("--n_samples", type=int, default=10)

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

torch.cuda.empty_cache()

gc.collect()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

f = h5py.File(args.data, "r")
u_true = torch.from_numpy(f["u"][args.sample_idx, 0]).to(device)
nx, ny, nt = u_true.shape

x_grid = torch.linspace(0, 1, nx).to(device)
y_grid = torch.linspace(0, 1, ny).to(device)
t_grid = torch.linspace(0, 1, nt).to(device)

residuals = Residuals2D(
    data=u_true.unsqueeze(0),
    x=x_grid,
    y=y_grid,
    t_grid=t_grid,
    nx=nx, ny=ny, nt=nt
)

# --------------------------------------------------
# PRIOR + SAMPLER
# --------------------------------------------------

sampler = FFM_NS_sampler(model)


# --------------------------------------------------
# LAMBDA SCHEDULES
# --------------------------------------------------

def lambda_schedule_linear(t):
    """Linear schedule: lambda increases linearly with flow time."""
    return t * 1e0


def lambda_schedule_exponential(t):
    """Exponential schedule: lambda increases exponentially with flow time."""
    return 1e-2 * torch.exp(torch.tensor(5.0 * t)).item()


# --------------------------------------------------
# SAMPLING (MULTI-SAMPLE)
# --------------------------------------------------

interpolation_params_fixed = {
    'custom_lam': 1e0,
    'step_size': 1e-2,
    'num_steps': 20
}

interpolation_params_linear = {
    'custom_lam': 1e-2,
    'step_size': 1e-2,
    'num_steps': 20,
    'lambda_schedule': lambda_schedule_linear
}

interpolation_params_exponential = {
    'custom_lam': 1e-2,
    'step_size': 1e-2,
    'num_steps': 20,
    'lambda_schedule': lambda_schedule_exponential
}

n_samples = args.n_samples
all_vanilla_metrics = []
all_pcfm_fixed_metrics = []
all_pcfm_linear_metrics = []
all_pcfm_exponential_metrics = []

print(f"\nRunning sampling for {n_samples} samples...")

for sample_num in range(n_samples):
    print(f"\n--- Sample {sample_num + 1}/{n_samples} ---")

    grid = make_grid((nx, ny, nt), device)
    u0_sample = model.gp.sample(grid, (nx, ny, nt), n_samples=1).to(device)
    del grid
    torch.cuda.empty_cache()

    # 1. Vanilla FFM
    print("  Vanilla FFM...")
    with torch.no_grad():
        u_vanilla = sampler.vanilla_sample(u0_sample, n_step=args.n_step)
        vanilla_metrics = evaluate_ns_constraints(u_vanilla[0], residuals)
    all_vanilla_metrics.append(vanilla_metrics)

    # 2. PCFM fixed lambda
    print("  PCFM (fixed λ=1e0)...")
    u_pcfm_fixed = sampler.pcfm_sample(
        u0=u0_sample,
        n_step=args.n_step,
        hfunc=residuals.full_residual_ns,
        mode=args.mode,
        newtonsteps=1,
        guided_interpolation=True,
        interpolation_params=interpolation_params_fixed,
        eps=args.eps
    )
    pcfm_fixed_metrics = evaluate_ns_constraints(u_pcfm_fixed[0], residuals)
    all_pcfm_fixed_metrics.append(pcfm_fixed_metrics)

    # 3. PCFM linear lambda
    print("  PCFM (linear λ: 0→1e0)...")
    u_pcfm_linear = sampler.pcfm_sample(
        u0=u0_sample,
        n_step=args.n_step,
        hfunc=residuals.full_residual_ns,
        mode=args.mode,
        newtonsteps=1,
        guided_interpolation=True,
        interpolation_params=interpolation_params_linear,
        eps=args.eps
    )
    pcfm_linear_metrics = evaluate_ns_constraints(u_pcfm_linear[0], residuals)
    all_pcfm_linear_metrics.append(pcfm_linear_metrics)

    # 4. PCFM exponential lambda
    print("  PCFM (exponential λ)...")
    u_pcfm_exponential = sampler.pcfm_sample(
        u0=u0_sample,
        n_step=args.n_step,
        hfunc=residuals.full_residual_ns,
        mode=args.mode,
        newtonsteps=1,
        guided_interpolation=True,
        interpolation_params=interpolation_params_exponential,
        eps=args.eps
    )
    pcfm_exponential_metrics = evaluate_ns_constraints(u_pcfm_exponential[0], residuals)
    all_pcfm_exponential_metrics.append(pcfm_exponential_metrics)

    # Save individual sample plots
    plot_slice(u_vanilla[0], f"Vanilla (Sample {sample_num + 1})",
               os.path.join(args.outdir, f"sample_{sample_num:02d}_vanilla.png"))
    plot_slice(u_pcfm_fixed[0], f"PCFM Fixed (Sample {sample_num + 1})",
               os.path.join(args.outdir, f"sample_{sample_num:02d}_pcfm_fixed.png"))
    plot_slice(u_pcfm_linear[0], f"PCFM Linear (Sample {sample_num + 1})",
               os.path.join(args.outdir, f"sample_{sample_num:02d}_pcfm_linear.png"))
    plot_slice(u_pcfm_exponential[0], f"PCFM Exponential (Sample {sample_num + 1})",
               os.path.join(args.outdir, f"sample_{sample_num:02d}_pcfm_exponential.png"))

# --------------------------------------------------
# AVERAGE METRICS
# --------------------------------------------------

def average_metrics(metrics_list):
    avg = {}
    for key in metrics_list[0].keys():
        avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    return avg

avg_vanilla = average_metrics(all_vanilla_metrics)
avg_pcfm_fixed = average_metrics(all_pcfm_fixed_metrics)
avg_pcfm_linear = average_metrics(all_pcfm_linear_metrics)
avg_pcfm_exponential = average_metrics(all_pcfm_exponential_metrics)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

gt_metrics = evaluate_ns_constraints(u_true, residuals)

save_metrics(gt_metrics, os.path.join(args.outdir, "metrics_gt.json"))
save_metrics(avg_vanilla, os.path.join(args.outdir, "metrics_avg_vanilla.json"))
save_metrics(avg_pcfm_fixed, os.path.join(args.outdir, "metrics_avg_pcfm_fixed.json"))
save_metrics(avg_pcfm_linear, os.path.join(args.outdir, "metrics_avg_pcfm_linear.json"))
save_metrics(avg_pcfm_exponential, os.path.join(args.outdir, "metrics_avg_pcfm_exponential.json"))

rows = []
for k in gt_metrics.keys():
    rows.append({
        "metric": k,
        "ground_truth": gt_metrics[k],
        "vanilla": avg_vanilla[k],
        "pcfm_fixed": avg_pcfm_fixed[k],
        "pcfm_linear": avg_pcfm_linear[k],
        "pcfm_exponential": avg_pcfm_exponential[k]
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.outdir, "metrics_table_averaged.csv"), index=False)

plot_slice(u_true, "Ground Truth", os.path.join(args.outdir, "gt.png"))
torch.save(u_true.cpu(), os.path.join(args.outdir, "u_true.pt"))

print("\n" + "=" * 60)
print("AVERAGED RESULTS")
print("=" * 60)
print(df.to_string(index=False))
print("=" * 60)
print(f"Run complete. Outputs saved to: {args.outdir}")

f.close()
