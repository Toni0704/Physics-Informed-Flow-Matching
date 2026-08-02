"""
eval.py  —  Case 4: Conditional Generation
-------------------------------------------
Evaluates how well conditioning on E controls the generated energy.

Key tests beyond the standard metrics:

1. Energy targeting accuracy
   Generate trajectories conditioned on E* ∈ [0.1, 0.9].
   Measure actual mean H of generated trajectories.
   A perfect model: mean(H_generated) = E*  for all E*.

2. Energy sweep visualisation
   Generate phase portraits at fixed E* values across the range.
   GT contours should match the generated trajectory shapes.

3. Standard comparison table
   Same metrics as Cases 2 and 3 for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from Simple_pendulum.Pendulum_conditional_H.config import N_STEPS, DT_PHYS, E_MIN, E_MAX, SEED
from Simple_pendulum.Pendulum_conditional_H.data   import hamiltonian, get_dataloaders
from Simple_pendulum.Pendulum_conditional_H.model  import build_model
from Simple_pendulum.Pendulum_conditional_H.train  import train, sample_conditional, sample_unconditional


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def pde_residual(trajs, dt=DT_PHYS):
    theta     = trajs[:, :, 0]
    omega     = trajs[:, :, 1]
    dth_dt    = (theta[:, 2:] - theta[:, :-2]) / (2*dt)
    dom_dt    = (omega[:, 2:] - omega[:, :-2]) / (2*dt)
    r_th = np.mean((dth_dt -  omega[:, 1:-1])**2)
    r_om = np.mean((dom_dt - -np.sin(theta[:, 1:-1]))**2)
    return r_th, r_om


def evaluate(trajs, label):
    H     = hamiltonian(trajs[:,:,0], trajs[:,:,1])
    std   = H.std(axis=1).mean()
    drift = np.abs(H[:,-1] - H[:,0]).mean()
    r_th, r_om = pde_residual(trajs)
    print(f"  [{label:<25}]  std(H)={std:.3e}  "
          f"|ΔH|={drift:.3e}  PDE_θ={r_th:.3e}  PDE_ω={r_om:.3e}")
    return {"H": H, "std": std, "drift": drift,
            "pde_theta": r_th, "pde_omega": r_om}


# ─────────────────────────────────────────────────────────────
# Energy targeting test
# ─────────────────────────────────────────────────────────────

def energy_targeting_test(model, device, n_per_level=50):
    """
    For a grid of target energies E*, generate trajectories and
    measure whether the actual mean energy matches E*.

    Returns:
        E_targets : np.ndarray — the requested energy levels
        E_actual  : np.ndarray — mean actual energy of generated trajs
        E_std     : np.ndarray — std of actual energy per level
    """
    E_targets = np.linspace(E_MIN + 0.05, E_MAX - 0.05, 10)
    E_actual  = []
    E_std_arr = []

    for E_star in E_targets:
        E_arr = np.full(n_per_level, E_star)
        gen   = sample_conditional(model, E_arr, device)
        H_gen = hamiltonian(gen[:,0,0], gen[:,0,1])   # energy at first step
        E_actual.append(H_gen.mean())
        E_std_arr.append(H_gen.std())

    return E_targets, np.array(E_actual), np.array(E_std_arr)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_all(test_raw, gen_unc, stats_gt, stats_gen,
             E_targets, E_actual, E_std,
             losses, model, device, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS

    # ── Figure 1: Main comparison ──────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Case 4 — Conditional Generation (FiLM conditioning on energy E)\n"
        "No physics penalty, no projection — constraint via conditioning",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Phase portrait GT
    ax = fig.add_subplot(gs[0, 0])
    for i in range(min(40, len(test_raw))):
        ax.plot(test_raw[i,:,0], test_raw[i,:,1],
                alpha=0.3, lw=0.7, color="#2E86AB")
    ax.set_title("Phase Portrait — Ground Truth", fontsize=9)
    ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)

    # Phase portrait generated (unconditional — mean E)
    ax = fig.add_subplot(gs[0, 1])
    for i in range(min(40, len(gen_unc))):
        ax.plot(gen_unc[i,:,0], gen_unc[i,:,1],
                alpha=0.3, lw=0.7, color="#9B5DE5")
    ax.set_title("Phase Portrait — Generated\n(conditioned on mean E)", fontsize=9)
    ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)

    # Energy vs tau GT
    ax = fig.add_subplot(gs[0, 2])
    for i in range(min(20, len(test_raw))):
        ax.plot(tau, stats_gt["H"][i], alpha=0.35, lw=0.7, color="#2E86AB")
    ax.set_title("Energy vs τ — Ground Truth", fontsize=9)
    ax.set_xlabel("Physical time τ"); ax.set_ylabel("H")
    ax.grid(True, alpha=0.3)

    # Energy vs tau generated
    ax = fig.add_subplot(gs[1, 0])
    for i in range(min(20, len(gen_unc))):
        ax.plot(tau, stats_gen["H"][i], alpha=0.35, lw=0.7, color="#9B5DE5")
    ax.set_title("Energy vs τ — Generated", fontsize=9)
    ax.set_xlabel("Physical time τ"); ax.set_ylabel("H")
    ax.grid(True, alpha=0.3)

    # Energy targeting: E* vs actual H
    ax = fig.add_subplot(gs[1, 1])
    ax.plot([E_MIN, E_MAX], [E_MIN, E_MAX],
            "k--", lw=1.5, label="Perfect targeting (E*=H)")
    ax.errorbar(E_targets, E_actual, yerr=E_std,
                fmt="o", color="#9B5DE5", capsize=4, lw=1.5,
                label="Generated mean H ± std")
    ax.set_xlabel("Target energy E*")
    ax.set_ylabel("Actual mean H of generated traj")
    ax.set_title("Energy Targeting Accuracy\n(dots on diagonal = perfect)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Training loss
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(losses, color="#9B5DE5", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("FM Loss")
    ax.set_title("Training Loss"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{out_dir}/case4_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/case4_main.png")

    # ── Figure 2: Energy sweep — phase portraits at different E* ──
    E_sweep = np.linspace(E_MIN + 0.05, E_MAX - 0.05, 5)
    cmap    = plt.cm.plasma
    colors  = [cmap(i / (len(E_sweep)-1)) for i in range(len(E_sweep))]

    fig2, axes2 = plt.subplots(1, len(E_sweep), figsize=(4*len(E_sweep), 4))
    fig2.suptitle(
        "Case 4 — Phase Portraits at Different Conditioning Energies E*\n"
        "Each column: 20 trajectories conditioned on same E*",
        fontsize=11
    )
    for ax, E_star, color in zip(axes2, E_sweep, colors):
        E_arr = np.full(20, E_star)
        gen   = sample_conditional(model, E_arr, device)
        H_gen = hamiltonian(gen[:,0,0], gen[:,0,1])

        for i in range(len(gen)):
            ax.plot(gen[i,:,0], gen[i,:,1],
                    alpha=0.4, lw=0.8, color=color)
        ax.set_title(f"E* = {E_star:.2f}\n"
                     f"actual H = {H_gen.mean():.2f} ± {H_gen.std():.2f}",
                     fontsize=8)
        ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)

        # Draw theoretical energy contour for reference
        theta_cont = np.linspace(-np.pi*0.9, np.pi*0.9, 300)
        val        = 2*(E_star + np.cos(theta_cont))
        mask       = val >= 0
        omega_pos  =  np.where(mask, np.sqrt(np.maximum(val,0)), np.nan)
        omega_neg  = -omega_pos
        ax.plot(theta_cont, omega_pos, "k--", lw=0.8, alpha=0.5, label="True contour")
        ax.plot(theta_cont, omega_neg, "k--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case4_energy_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir}/case4_energy_sweep.png")

    # ── Figure 3: 5 individual trajectory comparisons ──────────
    fig3, axes3 = plt.subplots(1, 5, figsize=(18, 4))
    fig3.suptitle("Case 4 — Individual Trajectories: GT (blue) vs Conditional (purple)", fontsize=11)
    for i, ax in enumerate(axes3):
        E_star = hamiltonian(test_raw[i,0,0], test_raw[i,0,1])
        gen_i  = sample_conditional(model, np.array([E_star]), device)
        ax.plot(test_raw[i,:,0], test_raw[i,:,1],
                color="#2E86AB", lw=1.8, label=f"GT (E={E_star:.2f})" if i==0 else "")
        ax.plot(gen_i[0,:,0], gen_i[0,:,1],
                color="#9B5DE5", lw=1.2, linestyle="--",
                label=f"Gen (E*={E_star:.2f})" if i==0 else "")
        # True energy contour
        theta_cont = np.linspace(-np.pi*0.9, np.pi*0.9, 300)
        val  = 2*(E_star + np.cos(theta_cont))
        mask = val >= 0
        ax.plot(theta_cont,  np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4)
        ax.plot(theta_cont, -np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4)
        ax.set_title(f"Sample {i+1}\nE*={E_star:.2f}")
        ax.set_xlabel("θ"); ax.set_ylabel("ω"); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case4_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_dir}/case4_individual.png")


# ─────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────

def run_eval(model, losses, test_raw, device, out_dir="outputs"):
    print("\n" + "═"*65)
    print("EVALUATION — Case 4: Conditional Generation")
    print("═"*65)

    # Generate with mean energy (unconditional baseline)
    print(f"\nSampling {len(test_raw)} trajectories (mean energy conditioning)...")
    gen_unc = sample_unconditional(model, len(test_raw), device)

    # Metrics
    print()
    stats_gt  = evaluate(test_raw, "Ground Truth")
    stats_gen = evaluate(gen_unc,  "Case 4 (cond. on mean E)")

    # Energy targeting test
    print("\nRunning energy targeting test...")
    E_targets, E_actual, E_std = energy_targeting_test(model, device)

    print("\nEnergy Targeting Results:")
    print(f"  {'E*':>8}  {'H_mean':>8}  {'H_std':>8}  {'error':>8}")
    print("  " + "─"*36)
    for E_s, H_m, H_s in zip(E_targets, E_actual, E_std):
        print(f"  {E_s:>8.3f}  {H_m:>8.3f}  {H_s:>8.3f}  {abs(H_m-E_s):>8.3f}")

    # Comparison table
    print("\n" + "═"*65)
    print(f"{'Metric':<28} {'GT':>10}  {'Case 4':>10}")
    print("─"*65)
    for name, key in [("mean std(H)",  "std"),
                      ("mean |ΔH|",    "drift"),
                      ("PDE res dθ/dτ","pde_theta"),
                      ("PDE res dω/dτ","pde_omega")]:
        g = stats_gt[key]
        c = stats_gen[key]
        print(f"  {name:<26} {g:>10.3e}  {c:>10.3e}")
    print("═"*65)

    # Plots
    print("\nSaving plots...")
    plot_all(test_raw, gen_unc, stats_gt, stats_gen,
             E_targets, E_actual, E_std, losses, model, device, out_dir)

    return {
        "gen":        gen_unc,
        "stats_gt":   stats_gt,
        "stats_gen":  stats_gen,
        "E_targets":  E_targets,
        "E_actual":   E_actual,
        "E_std":      E_std,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_loader, _, test_raw = get_dataloaders()
    model  = build_model(device)
    losses = train(model, train_loader, device, save_dir="checkpoints")
    results = run_eval(model, losses, test_raw, device, out_dir="outputs")
    print("\nAll done.")
