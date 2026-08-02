"""
eval.py
-------
Evaluation and visualisation for Case 3: Physics Inherent to Data.

Metrics
-------
1. Energy conservation quality
       std(H) across physical time steps within each trajectory
       A perfect pendulum has std(H) = 0 — energy is constant.

2. Energy drift
       |H(τ=T) - H(τ=0)| — absolute drift from start to end

3. Trajectory MSE
       Mean squared error vs ground-truth test trajectories
       (matched by index — both start from same energy range)

4. Phase portrait visual inspection
       Generated trajectories should form closed ellipses in (θ, ω)
       space. Any spiral inward/outward = energy not conserved.

Plots saved to: outputs/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from Simple_pendulum.Pendulum_data_only.config  import N_STEPS, DT_PHYS, N_FM_STEPS
from Simple_pendulum.Pendulum_data_only.data    import hamiltonian, get_dataloaders
from Simple_pendulum.Pendulum_data_only.model   import build_model
from Simple_pendulum.Pendulum_data_only.train   import train, sample


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def energy_stats(trajs: np.ndarray, label: str) -> dict:
    """
    Compute energy conservation statistics for a set of trajectories.

    Args:
        trajs : shape (n, N_STEPS, 2)
        label : string label for printing

    Returns:
        dict with keys: H_vals, std_per_traj, drift_per_traj
    """
    # H at every physical time step: shape (n, N_STEPS)
    H_vals = hamiltonian(trajs[:, :, 0], trajs[:, :, 1])

    std_per_traj   = H_vals.std(axis=1)                    # (n,)
    drift_per_traj = np.abs(H_vals[:, -1] - H_vals[:, 0]) # (n,)

    print(f"\n[{label}]")
    print(f"  mean std(H) across τ   : {std_per_traj.mean():.4e}   ← lower is better")
    print(f"  max  std(H) across τ   : {std_per_traj.max():.4e}")
    print(f"  mean |H_end - H_start| : {drift_per_traj.mean():.4e}")

    return {
        "H_vals":        H_vals,
        "std_per_traj":  std_per_traj,
        "drift_per_traj":drift_per_traj,
    }


def trajectory_mse(generated: np.ndarray, reference: np.ndarray) -> float:
    """
    MSE between generated and reference trajectories (matched by index).

    Note: this measures distributional mismatch only approximately —
    FM generates from random noise so there's no one-to-one pairing.
    A better metric is energy statistics, but MSE is a useful baseline.
    """
    n = min(len(generated), len(reference))
    mse = float(np.mean((generated[:n] - reference[:n])**2))
    print(f"\n  Trajectory MSE (generated vs reference): {mse:.4e}")
    return mse


def summarise(stats_gt: dict, stats_gen: dict, mse: float) -> None:
    """Print a compact comparison table."""
    print("\n" + "═" * 55)
    print(f"{'Metric':<35} {'GT':>8}  {'Gen':>8}")
    print("─" * 55)
    print(f"{'mean std(H)':<35} "
          f"{stats_gt['std_per_traj'].mean():>8.2e}  "
          f"{stats_gen['std_per_traj'].mean():>8.2e}")
    print(f"{'mean |H drift|':<35} "
          f"{stats_gt['drift_per_traj'].mean():>8.2e}  "
          f"{stats_gen['drift_per_traj'].mean():>8.2e}")
    print(f"{'trajectory MSE':<35} {'—':>8}  {mse:>8.2e}")
    print("═" * 55)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_all(
    losses:    list[float],
    test_trajs: np.ndarray,
    generated:  np.ndarray,
    stats_gt:   dict,
    stats_gen:  dict,
    out_dir:    str = "outputs",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS   # physical time axis

    # ── Figure 1: Main results (2×3 grid) ──────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Case 3 — Physics Inherent to Data\n"
        "Vanilla FM  |  Störmer-Verlet training data  |  No physics in model",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (0,0) Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(losses, color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FM Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (0,1) Phase portrait — ground truth
    ax = fig.add_subplot(gs[0, 1])
    for i in range(min(40, len(test_trajs))):
        ax.plot(test_trajs[i, :, 0], test_trajs[i, :, 1],
                alpha=0.35, linewidth=0.7, color="#2E86AB")
    ax.set_xlabel("θ  (rad)")
    ax.set_ylabel("ω  (rad/s)")
    ax.set_title("Phase Portrait — Ground Truth")
    ax.grid(True, alpha=0.3)

    # (0,2) Phase portrait — generated
    ax = fig.add_subplot(gs[0, 2])
    for i in range(min(40, len(generated))):
        ax.plot(generated[i, :, 0], generated[i, :, 1],
                alpha=0.35, linewidth=0.7, color="#E84855")
    ax.set_xlabel("θ  (rad)")
    ax.set_ylabel("ω  (rad/s)")
    ax.set_title("Phase Portrait — Generated")
    ax.grid(True, alpha=0.3)

    # (1,0) Energy vs physical time — ground truth
    ax = fig.add_subplot(gs[1, 0])
    for i in range(min(20, len(test_trajs))):
        ax.plot(tau, stats_gt["H_vals"][i],
                alpha=0.35, linewidth=0.7, color="#2E86AB")
    ax.set_xlabel("Physical time τ")
    ax.set_ylabel("H(θ, ω)")
    ax.set_title("Energy vs τ — Ground Truth")
    ax.grid(True, alpha=0.3)

    # (1,1) Energy vs physical time — generated
    ax = fig.add_subplot(gs[1, 1])
    for i in range(min(20, len(generated))):
        ax.plot(tau, stats_gen["H_vals"][i],
                alpha=0.35, linewidth=0.7, color="#E84855")
    ax.set_xlabel("Physical time τ")
    ax.set_ylabel("H(θ, ω)")
    ax.set_title("Energy vs τ — Generated")
    ax.grid(True, alpha=0.3)

    # (1,2) std(H) distribution comparison
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(stats_gt["std_per_traj"],  bins=30, alpha=0.6,
            label="Ground Truth", color="#2E86AB", density=True)
    ax.hist(stats_gen["std_per_traj"], bins=30, alpha=0.6,
            label="Generated",    color="#E84855", density=True)
    ax.set_xlabel("std(H) across τ  (energy variation)")
    ax.set_ylabel("Density")
    ax.set_title("Energy Conservation Quality")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path1 = f"{out_dir}/case3_main.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Figure 2: 5 individual trajectory comparisons ──────────
    fig2, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig2.suptitle(
        "Case 3 — Phase portraits: Ground Truth (blue) vs Generated (red)",
        fontsize=11
    )
    for i, ax in enumerate(axes):
        if i < len(test_trajs):
            ax.plot(test_trajs[i, :, 0], test_trajs[i, :, 1],
                    color="#2E86AB", linewidth=1.5,
                    label="GT" if i == 0 else "")
        if i < len(generated):
            ax.plot(generated[i, :, 0], generated[i, :, 1],
                    color="#E84855", linewidth=1.5, linestyle="--",
                    label="Gen" if i == 0 else "")
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("θ")
        ax.set_ylabel("ω")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path2 = f"{out_dir}/case3_individual.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: Energy drift scatter ─────────────────────────
    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        stats_gt["H_vals"][:, 0],
        stats_gt["drift_per_traj"],
        alpha=0.5, s=15, label="Ground Truth", color="#2E86AB",
    )
    ax.scatter(
        stats_gen["H_vals"][:, 0],
        stats_gen["drift_per_traj"],
        alpha=0.5, s=15, label="Generated", color="#E84855",
        marker="^",
    )
    ax.set_xlabel("Initial energy H(τ=0)")
    ax.set_ylabel("|H(τ=T) − H(τ=0)|")
    ax.set_title("Energy Drift vs Initial Energy")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = f"{out_dir}/case3_energy_drift.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ─────────────────────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────────────────────

def run_eval(
    model,
    train_losses: list[float],
    test_trajs:   np.ndarray,
    device:       str,
    out_dir:      str = "outputs",
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        model        : trained VelocityFieldMLP
        train_losses : list of per-epoch losses from train()
        test_trajs   : ground truth test trajectories, shape (n, N_STEPS, 2)
        device       : torch device string
        out_dir      : directory to save plots

    Returns:
        results dict with all computed metrics
    """
    print("\n" + "═" * 55)
    print("EVALUATION — Case 3: Physics Inherent to Data")
    print("═" * 55)

    # Generate trajectories
    print(f"\nSampling {len(test_trajs)} trajectories...")
    generated = sample(model, n_samples=len(test_trajs), device=device)

    # Metrics
    stats_gt  = energy_stats(test_trajs, "Ground Truth")
    stats_gen = energy_stats(generated,  "Generated")
    mse       = trajectory_mse(generated, test_trajs)
    summarise(stats_gt, stats_gen, mse)

    # Plots
    print("\nSaving plots...")
    plot_all(train_losses, test_trajs, generated, stats_gt, stats_gen, out_dir)

    return {
        "generated":   generated,
        "stats_gt":    stats_gt,
        "stats_gen":   stats_gen,
        "mse":         mse,
        "train_losses":train_losses,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone evaluation.
    Runs full training + evaluation pipeline.
    Usage: python eval.py
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    train_loader, train_trajs, test_trajs = get_dataloaders()

    # Model
    model = build_model(device)

    # Train
    losses = train(model, train_loader, device, save_dir="checkpoints")

    # Evaluate
    results = run_eval(model, losses, test_trajs, device, out_dir="outputs")

    print("\nAll done.")
