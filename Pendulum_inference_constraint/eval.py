"""
eval.py  —  Case 2: Physics Enforced in Sampling
-------------------------------------------------
Evaluates three configurations side by side:
    A. Ground Truth          (Störmer-Verlet)
    B. Unconstrained FM      (Case 3 — same model, no projection)
    C. Constrained FM        (Case 2 — same model, projection at every step)

Key metrics:
    1. Energy conservation   std(H) across τ within each trajectory
    2. Energy drift          |H(end) - H(start)|
    3. Trajectory MSE        vs ground truth
    4. PDE residual          how well generated traj satisfies pendulum ODE

The PDE residual is the Case 2 analogue of Burgers MSE in your report —
it measures whether the trajectory actually satisfies dω/dτ = -sin(θ),
even if constraints are satisfied instantaneously.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from config  import N_STEPS, DT_PHYS, N_FM_STEPS
from data    import hamiltonian, get_dataloaders
from model   import build_model
from train   import train
from sample_constrained import sample_constrained, sample_unconstrained


# ─────────────────────────────────────────────────────────────
# PDE residual
# ─────────────────────────────────────────────────────────────

def pde_residual(trajs_raw, dt=DT_PHYS):
    """
    Measure how well a trajectory satisfies the pendulum ODE:
        dθ/dτ = ω
        dω/dτ = -sin(θ)

    Using finite differences to approximate derivatives:
        (θ_{i+1} - θ_{i-1}) / (2Δτ)  ≈  ω_i
        (ω_{i+1} - ω_{i-1}) / (2Δτ)  ≈  -sin(θ_i)

    This is the pendulum equivalent of Burgers MSE in your report.
    A trajectory can satisfy energy conservation (H = const) but
    still violate the ODE if the timing is wrong — this metric
    catches that.

    Args:
        trajs_raw : np.ndarray (n, N_STEPS, 2)  [θ, ω]
        dt        : physical time step

    Returns:
        residual_theta : mean squared residual for dθ/dτ = ω
        residual_omega : mean squared residual for dω/dτ = -sin(θ)
    """
    theta = trajs_raw[:, :, 0]   # (n, N)
    omega = trajs_raw[:, :, 1]   # (n, N)

    # Central differences — interior points only
    dtheta_dt = (theta[:, 2:] - theta[:, :-2]) / (2 * dt)  # (n, N-2)
    domega_dt = (omega[:, 2:] - omega[:, :-2]) / (2 * dt)  # (n, N-2)

    # RHS of ODE at interior points
    rhs_theta =  omega[:, 1:-1]           # ω
    rhs_omega = -np.sin(theta[:, 1:-1])   # -sin(θ)

    res_theta = np.mean((dtheta_dt - rhs_theta)**2)
    res_omega = np.mean((domega_dt - rhs_omega)**2)

    return res_theta, res_omega


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def energy_stats(trajs, label):
    H = hamiltonian(trajs[:, :, 0], trajs[:, :, 1])  # (n, N_STEPS)
    std   = H.std(axis=1)
    drift = np.abs(H[:, -1] - H[:, 0])
    print(f"\n[{label}]")
    print(f"  mean std(H)    : {std.mean():.4e}")
    print(f"  mean |ΔH|      : {drift.mean():.4e}")
    return {"H": H, "std": std, "drift": drift}


def full_metrics(trajs, label, dt=DT_PHYS):
    stats = energy_stats(trajs, label)
    r_th, r_om = pde_residual(trajs, dt)
    print(f"  PDE res (dθ)   : {r_th:.4e}")
    print(f"  PDE res (dω)   : {r_om:.4e}")
    stats["pde_theta"] = r_th
    stats["pde_omega"] = r_om
    return stats


def print_comparison_table(gt, unconstrained, constrained):
    print("\n" + "═"*65)
    print(f"{'Metric':<28} {'GT':>10}  {'Unconstr':>10}  {'Constr':>10}")
    print("─"*65)
    rows = [
        ("mean std(H)",      "std",       "{:.2e}"),
        ("mean |ΔH|",        "drift",     "{:.2e}"),
        ("PDE res dθ/dτ",    "pde_theta", "{:.2e}"),
        ("PDE res dω/dτ",    "pde_omega", "{:.2e}"),
    ]
    for name, key, fmt in rows:
        g = fmt.format(gt[key].mean()           if hasattr(gt[key],     'mean') else gt[key])
        u = fmt.format(unconstrained[key].mean() if hasattr(unconstrained[key], 'mean') else unconstrained[key])
        c = fmt.format(constrained[key].mean()   if hasattr(constrained[key],   'mean') else constrained[key])
        print(f"  {name:<26} {g:>10}  {u:>10}  {c:>10}")
    print("═"*65)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_all(test_raw, gen_unc, gen_con, stats_gt, stats_unc, stats_con,
             losses, diag, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS

    # ── Figure 1: Main 3×3 comparison grid ─────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Case 2 — Physics Enforced in Sampling\n"
        "Same model as Case 3  |  Projection onto H=E₀ at every Euler step",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    configs = [
        (test_raw, stats_gt,  "#2E86AB", "Ground Truth"),
        (gen_unc,  stats_unc, "#E84855", "Unconstrained FM (Case 3)"),
        (gen_con,  stats_con, "#3BB273", "Constrained FM   (Case 2)"),
    ]

    # Row 0: Phase portraits
    for col, (trajs, _, color, label) in enumerate(configs):
        ax = fig.add_subplot(gs[0, col])
        for i in range(min(40, len(trajs))):
            ax.plot(trajs[i,:,0], trajs[i,:,1],
                    alpha=0.3, linewidth=0.7, color=color)
        ax.set_xlabel("θ  (rad)"); ax.set_ylabel("ω  (rad/s)")
        ax.set_title(f"Phase Portrait\n{label}", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 1: Energy vs physical time
    for col, (trajs, stats, color, label) in enumerate(configs):
        ax = fig.add_subplot(gs[1, col])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, stats["H"][i],
                    alpha=0.35, linewidth=0.7, color=color)
        ax.set_xlabel("Physical time τ"); ax.set_ylabel("H(θ,ω)")
        ax.set_title(f"Energy vs τ\n{label}", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 2 left: std(H) distribution
    ax = fig.add_subplot(gs[2, 0])
    for stats, color, label in [(stats_gt,  "#2E86AB", "GT"),
                                 (stats_unc, "#E84855", "Unconstrained"),
                                 (stats_con, "#3BB273", "Constrained")]:
        ax.hist(stats["std"], bins=25, alpha=0.55, color=color,
                label=label, density=True)
    ax.set_xlabel("std(H) across τ"); ax.set_ylabel("Density")
    ax.set_title("Energy Conservation Quality"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2 middle: energy drift comparison
    ax = fig.add_subplot(gs[2, 1])
    labels_d = ["GT", "Unconstrained", "Constrained"]
    means_d  = [stats_gt["drift"].mean(),
                stats_unc["drift"].mean(),
                stats_con["drift"].mean()]
    colors_d = ["#2E86AB", "#E84855", "#3BB273"]
    bars = ax.bar(labels_d, means_d, color=colors_d, alpha=0.8)
    ax.set_ylabel("|H_end − H_start|"); ax.set_yscale("log")
    ax.set_title("Mean Energy Drift"); ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, means_d):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.3,
                f"{val:.1e}", ha="center", va="bottom", fontsize=8)

    # Row 2 right: PDE residual comparison — the key trade-off plot
    ax = fig.add_subplot(gs[2, 2])
    labels_p = ["GT", "Unconstrained", "Constrained"]
    pde_vals = [
        (stats_gt["pde_theta"]  + stats_gt["pde_omega"])  / 2,
        (stats_unc["pde_theta"] + stats_unc["pde_omega"]) / 2,
        (stats_con["pde_theta"] + stats_con["pde_omega"]) / 2,
    ]
    bars2 = ax.bar(labels_p, pde_vals, color=colors_d, alpha=0.8)
    ax.set_ylabel("Mean PDE Residual"); ax.set_yscale("log")
    ax.set_title("PDE Residual\n(does traj satisfy ODE?)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, pde_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.3,
                f"{val:.1e}", ha="center", va="bottom", fontsize=8)

    plt.savefig(f"{out_dir}/case2_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/case2_main.png")

    # ── Figure 2: 5 individual trajectory comparisons ──────────
    fig2, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig2.suptitle(
        "Case 2 — Phase portraits: GT (blue) | Unconstrained (red) | Constrained (green)",
        fontsize=11
    )
    for i, ax in enumerate(axes):
        if i < len(test_raw):
            ax.plot(test_raw[i,:,0], test_raw[i,:,1],
                    color="#2E86AB", linewidth=1.8, label="GT" if i==0 else "")
        if i < len(gen_unc):
            ax.plot(gen_unc[i,:,0], gen_unc[i,:,1],
                    color="#E84855", linewidth=1.2, linestyle="--",
                    label="Unc" if i==0 else "")
        if i < len(gen_con):
            ax.plot(gen_con[i,:,0], gen_con[i,:,1],
                    color="#3BB273", linewidth=1.2, linestyle=":",
                    label="Con" if i==0 else "")
        ax.set_title(f"Sample {i+1}"); ax.set_xlabel("θ"); ax.set_ylabel("ω")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/case2_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir}/case2_individual.png")

    # ── Figure 3: Pre vs post projection energy violation ──────
    fig3, ax = plt.subplots(figsize=(7, 4))
    categories = ['Before\nProjection', 'After\nProjection']
    values     = [diag['viol_before'], diag['viol_after']]
    bars = ax.bar(categories, values, color=["#E84855", "#3BB273"], alpha=0.85, width=0.4)
    ax.set_ylabel("Mean |H - E0|")
    ax.set_yscale("log")
    ax.set_title("Energy Violation: Before vs After Post-hoc Projection\n"
                 "(projection applied once to fully generated trajectory)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                f"{val:.2e}", ha="center", va="bottom", fontsize=10)
    ax.text(0.98, 0.95,
            f"Invalid projections: {diag['invalid_frac']*100:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/case2_violation_trace.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_dir}/case2_violation_trace.png")

    # ── Figure 4: Training loss ─────────────────────────────────
    fig4, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("FM Loss"); ax.set_yscale("log")
    ax.set_title("Training Loss (identical to Case 3)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/case2_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {out_dir}/case2_training_loss.png")


# ─────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────

def run_eval(model, losses, test_raw, device, out_dir="outputs"):
    print("\n" + "═"*65)
    print("EVALUATION — Case 2: Physics Enforced in Sampling")
    print("═"*65)
    n = len(test_raw)

    # Generate both variants in one call
    # sample_constrained now runs plain FM first (gen_unc),
    # then applies one post-hoc projection (gen_con) — matching PCFM paper
    print(f"\nSampling {n} trajectories (FM + post-hoc projection)...")
    gen_unc, gen_con, E0s, diag = sample_constrained(model, n, device)

    print(f"\n  Violation before projection: {diag['viol_before']:.4e}")
    print(f"  Violation after  projection: {diag['viol_after']:.4e}")

    # Metrics
    stats_gt  = full_metrics(test_raw, "Ground Truth")
    stats_unc = full_metrics(gen_unc,  "Unconstrained FM (Case 3 baseline)")
    stats_con = full_metrics(gen_con,  "Constrained FM   (Case 2)")

    print_comparison_table(stats_gt, stats_unc, stats_con)

    # Plots
    print("\nSaving plots...")
    plot_all(test_raw, gen_unc, gen_con,
             stats_gt, stats_unc, stats_con,
             losses, diag, out_dir)

    return {
        "gen_unc":    gen_unc,
        "gen_con":    gen_con,
        "stats_gt":   stats_gt,
        "stats_unc":  stats_unc,
        "stats_con":  stats_con,
        "E0s":        E0s,
        "diag":       diag,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, _, test_raw = get_dataloaders()
    model  = build_model(device)
    losses = train(model, train_loader, device, save_dir="checkpoints")
    results = run_eval(model, losses, test_raw, device, out_dir="outputs")

    print("\nAll done.")