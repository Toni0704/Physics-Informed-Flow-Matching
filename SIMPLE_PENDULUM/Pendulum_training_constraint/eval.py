"""
eval.py  —  Case 1: Physics Enforced in Training
-------------------------------------------------
Trains one model per λ value and compares them all against:
    - Ground Truth
    - Case 3 (λ=0, vanilla FM — the baseline)

λ sweep: [0.0, 0.1, 1.0, 10.0]

For each λ we measure:
    1. FM loss        — did the model learn the data distribution?
    2. Physics loss   — is energy conserved in generated trajectories?
    3. std(H)         — energy conservation quality
    4. PDE residual   — does trajectory satisfy the pendulum ODE?

The key question: is there a λ that improves energy conservation
WITHOUT destroying PDE residual? Or is the trade-off unavoidable
at training time too?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from Simple_pendulum.Pendulum_training_constraint.config import N_STEPS, DT_PHYS, SEED
from Simple_pendulum.Pendulum_training_constraint.data   import hamiltonian, get_dataloaders
from Simple_pendulum.Pendulum_training_constraint.model  import build_model
from Simple_pendulum.Pendulum_training_constraint.train  import train, sample

# λ values to sweep
LAMBDA_VALUES = [0.0, 0.1, 1.0, 10.0]
COLORS        = ["#E84855", "#F4A261", "#3BB273", "#2E86AB"]


# ─────────────────────────────────────────────────────────────
# Metrics (same as Case 2 for direct comparison)
# ─────────────────────────────────────────────────────────────

def pde_residual(trajs_raw, dt=DT_PHYS):
    theta = trajs_raw[:, :, 0]
    omega = trajs_raw[:, :, 1]
    dtheta_dt = (theta[:, 2:] - theta[:, :-2]) / (2 * dt)
    domega_dt = (omega[:, 2:] - omega[:, :-2]) / (2 * dt)
    rhs_theta =  omega[:, 1:-1]
    rhs_omega = -np.sin(theta[:, 1:-1])
    return (np.mean((dtheta_dt - rhs_theta)**2),
            np.mean((domega_dt - rhs_omega)**2))


def evaluate(trajs, label):
    H     = hamiltonian(trajs[:, :, 0], trajs[:, :, 1])
    std   = H.std(axis=1).mean()
    drift = np.abs(H[:, -1] - H[:, 0]).mean()
    r_th, r_om = pde_residual(trajs)
    print(f"  [{label:<20}]  std(H)={std:.3e}  |ΔH|={drift:.3e}  "
          f"PDE_θ={r_th:.3e}  PDE_ω={r_om:.3e}")
    return {"H": H, "std": std, "drift": drift,
            "pde_theta": r_th, "pde_omega": r_om}


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_all(test_raw, all_generated, all_stats, all_histories,
             stats_gt, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS

    # ── Figure 1: Phase portraits for each λ ───────────────────
    n_lam = len(LAMBDA_VALUES)
    fig, axes = plt.subplots(1, n_lam + 1, figsize=(4*(n_lam+1), 4))
    fig.suptitle("Case 1 — Phase Portraits by λ value", fontsize=12, fontweight="bold")

    # Ground truth
    ax = axes[0]
    for i in range(min(40, len(test_raw))):
        ax.plot(test_raw[i,:,0], test_raw[i,:,1],
                alpha=0.3, lw=0.7, color="#2E86AB")
    ax.set_title("Ground Truth", fontsize=9)
    ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)

    for j, (lam, gen, color) in enumerate(
            zip(LAMBDA_VALUES, all_generated, COLORS)):
        ax = axes[j+1]
        for i in range(min(40, len(gen))):
            ax.plot(gen[i,:,0], gen[i,:,1],
                    alpha=0.3, lw=0.7, color=color)
        ax.set_title(f"λ = {lam}", fontsize=9)
        ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case1_phase_portraits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/case1_phase_portraits.png")

    # ── Figure 2: Training curves (FM loss + physics loss) ─────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Case 1 — Training Curves by λ", fontsize=11)

    for hist, lam, color in zip(all_histories, LAMBDA_VALUES, COLORS):
        label = f"λ={lam}"
        axes2[0].plot(hist["total"],   color=color, lw=1.2, label=label)
        axes2[1].plot(hist["fm"],      color=color, lw=1.2, label=label)
        axes2[2].plot(hist["physics"], color=color, lw=1.2, label=label)

    for ax, title in zip(axes2, ["Total Loss", "FM Loss", "Physics Loss (Var H)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.set_yscale("log")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir}/case1_training_curves.png")

    # ── Figure 3: Metric vs λ — the trade-off plot ─────────────
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
    fig3.suptitle(
        "Case 1 — Trade-off: Physics Loss Weight λ vs Metrics\n"
        "Does training-time enforcement improve energy conservation without hurting PDE residual?",
        fontsize=10, fontweight="bold"
    )

    metrics_keys = [
        ("std",       "mean std(H)",        "Energy Conservation\n(lower = better)"),
        ("drift",     "mean |ΔH|",          "Energy Drift\n(lower = better)"),
        ("pde_theta", "PDE res dθ/dτ",      "PDE Residual θ\n(lower = better)"),
        ("pde_omega", "PDE res dω/dτ",      "PDE Residual ω\n(lower = better)"),
    ]

    lam_labels = [str(l) for l in LAMBDA_VALUES]

    for ax, (key, ylabel, title) in zip(axes3, metrics_keys):
        vals = [s[key] for s in all_stats]
        gt_val = stats_gt[key]

        bars = ax.bar(lam_labels, vals, color=COLORS, alpha=0.8)
        ax.axhline(gt_val, color="#2E86AB", linestyle="--",
                   linewidth=1.5, label=f"GT ({gt_val:.1e})")
        ax.set_xlabel("λ")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val * 1.3, f"{val:.1e}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case1_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_dir}/case1_tradeoff.png")

    # ── Figure 4: Energy vs τ for best λ vs λ=0 ───────────────
    best_idx = int(np.argmin([s["std"] for s in all_stats]))
    best_lam = LAMBDA_VALUES[best_idx]
    best_gen = all_generated[best_idx]
    base_gen = all_generated[0]   # λ=0

    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))
    fig4.suptitle(
        f"Case 1 — Energy vs Physical Time τ\n"
        f"GT vs λ=0 (baseline) vs λ={best_lam} (best energy conservation)",
        fontsize=10
    )
    for ax, trajs, color, label in [
        (axes4[0], test_raw,  "#2E86AB", "Ground Truth"),
        (axes4[1], base_gen,  "#E84855", "λ = 0 (Case 3 baseline)"),
        (axes4[2], best_gen,  COLORS[best_idx], f"λ = {best_lam}"),
    ]:
        H = hamiltonian(trajs[:, :, 0], trajs[:, :, 1])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, H[i], alpha=0.35, lw=0.7, color=color)
        ax.set_xlabel("Physical time τ"); ax.set_ylabel("H(θ, ω)")
        ax.set_title(label, fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case1_energy_vs_tau.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {out_dir}/case1_energy_vs_tau.png")

    # ── Figure 5: Individual phase portraits for each λ ────────
    # 5 sample trajectories shown per λ, GT overlaid in blue.
    # This is the same style as Cases 2, 4a, 4b so all cases
    # are directly visually comparable.
    n_samples_to_show = 5
    n_lam = len(LAMBDA_VALUES)

    fig5, axes5 = plt.subplots(
        n_lam, n_samples_to_show,
        figsize=(4 * n_samples_to_show, 3.5 * n_lam),
    )
    fig5.suptitle(
        "Case 1 — Individual Phase Portraits: GT (blue) vs Generated, per λ\n"
        "Each row is one λ value. Each column is one sample.",
        fontsize=11, fontweight="bold"
    )

    for row, (lam, gen, color) in enumerate(
            zip(LAMBDA_VALUES, all_generated, COLORS)):
        for col in range(n_samples_to_show):
            ax = axes5[row, col]

            # GT trajectory
            if col < len(test_raw):
                ax.plot(test_raw[col, :, 0], test_raw[col, :, 1],
                        color="#2E86AB", lw=1.8,
                        label="GT" if (row == 0 and col == 0) else "")

            # Generated trajectory
            if col < len(gen):
                ax.plot(gen[col, :, 0], gen[col, :, 1],
                        color=color, lw=1.2, linestyle="--",
                        label=f"λ={lam}" if col == 0 else "")

            # True energy contour for reference
            E_true = hamiltonian(test_raw[col, 0, 0], test_raw[col, 0, 1])
            th_c   = np.linspace(-np.pi * 0.9, np.pi * 0.9, 300)
            val    = 2 * (E_true + np.cos(th_c))
            mask   = val >= 0
            ax.plot(th_c,  np.where(mask, np.sqrt(np.maximum(val, 0)), np.nan),
                    "k:", lw=0.7, alpha=0.3)
            ax.plot(th_c, -np.where(mask, np.sqrt(np.maximum(val, 0)), np.nan),
                    "k:", lw=0.7, alpha=0.3)

            ax.grid(True, alpha=0.3)
            ax.set_xlabel("θ", fontsize=8)
            ax.set_ylabel("ω", fontsize=8)

            if col == 0:
                ax.set_ylabel(f"λ = {lam}\nω", fontsize=9)
            if row == 0:
                ax.set_title(f"Sample {col + 1}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case1_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved: {out_dir}/case1_individual.png")


# ─────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────

def run_eval(test_raw, device, out_dir="outputs"):
    print("\n" + "═"*65)
    print("EVALUATION — Case 1: Physics Enforced in Training")
    print("═"*65)

    train_loader, _, _ = get_dataloaders()

    all_generated = []
    all_stats     = []
    all_histories = []

    # Ground truth stats
    print("\nGround Truth:")
    stats_gt = evaluate(test_raw, "Ground Truth")

    # Train and evaluate one model per λ
    for lam in LAMBDA_VALUES:
        print(f"\n{'─'*65}")
        print(f"λ = {lam}")
        print(f"{'─'*65}")

        model   = build_model(device)
        history = train(model, train_loader, device,
                        lam=lam, save_dir=f"checkpoints/lam_{lam}")
        gen     = sample(model, len(test_raw), device)
        stats   = evaluate(gen, f"λ={lam}")

        all_generated.append(gen)
        all_stats.append(stats)
        all_histories.append(history)

    # Summary table
    print("\n" + "═"*75)
    print(f"{'λ':<8} {'std(H)':>10}  {'|ΔH|':>10}  {'PDE_θ':>10}  {'PDE_ω':>10}")
    print("─"*75)
    print(f"{'GT':<8} {stats_gt['std']:>10.3e}  "
          f"{stats_gt['drift']:>10.3e}  "
          f"{stats_gt['pde_theta']:>10.3e}  "
          f"{stats_gt['pde_omega']:>10.3e}")
    for lam, stats in zip(LAMBDA_VALUES, all_stats):
        print(f"{lam:<8} {stats['std']:>10.3e}  "
              f"{stats['drift']:>10.3e}  "
              f"{stats['pde_theta']:>10.3e}  "
              f"{stats['pde_omega']:>10.3e}")
    print("═"*75)

    # Plots
    print("\nSaving plots...")
    plot_all(test_raw, all_generated, all_stats,
             all_histories, stats_gt, out_dir)

    return {
        "all_generated": all_generated,
        "all_stats":     all_stats,
        "all_histories": all_histories,
        "stats_gt":      stats_gt,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    _, _, test_raw = get_dataloaders()
    results = run_eval(test_raw, device, out_dir="outputs")
    print("\nAll done.")