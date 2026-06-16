"""
eval.py  --  Case 5: Extended Experiments
-------------------------------------------
Evaluation for all three variants. Computes the same four metrics as
all other cases so results are directly comparable in the summary table.

Also produces:
  - Phase portrait comparisons vs GT
  - Energy vs tau plots
  - Lambda distribution plot (Variant A)
  - Side-by-side comparison across all three variants
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from config import N_STEPS, DT_PHYS, SEED
from data   import hamiltonian, get_dataloaders


# ── Metrics ───────────────────────────────────────────────────────────────────

def pde_residual(trajs, dt=DT_PHYS):
    th = trajs[:,:,0]; om = trajs[:,:,1]
    dth = (th[:,2:] - th[:,:-2]) / (2*dt)
    dom = (om[:,2:] - om[:,:-2]) / (2*dt)
    return np.mean((dth - om[:,1:-1])**2), np.mean((dom + np.sin(th[:,1:-1]))**2)

def evaluate(trajs, label=""):
    H    = hamiltonian(trajs[:,:,0], trajs[:,:,1])
    std  = H.std(axis=1).mean()
    dh   = np.abs(H[:,-1] - H[:,0]).mean()
    rth, rom = pde_residual(trajs)
    print(f"  [{label:<35}]  std(H)={std:.3e}  |DH|={dh:.3e}  "
          f"PDE_th={rth:.3e}  PDE_om={rom:.3e}")
    return {"std": std, "dh": dh, "pde_theta": rth, "pde_omega": rom, "H": H}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_variant(test_raw, gen, label, color, stats_gt, stats_gen,
                 losses, out_dir, tag, extra_plot_fn=None):
    """Standard 2x3 diagnostic panel for any variant."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Case 5{tag}: {label}", fontsize=12, fontweight="bold")

    # Phase portraits
    for ax, trajs, col, lbl in [
        (axes[0,0], test_raw, "#2E86AB", "Ground Truth"),
        (axes[0,1], gen,      color,     label),
    ]:
        for i in range(min(40, len(trajs))):
            ax.plot(trajs[i,:,0], trajs[i,:,1], alpha=0.3, lw=0.7, color=col)
        ax.set_title(f"Phase Portrait — {lbl}"); ax.set_xlabel("theta")
        ax.set_ylabel("omega"); ax.grid(True, alpha=0.3)

    # Individual samples
    ax = axes[0,2]
    for i in range(min(8, len(test_raw))):
        ax.plot(test_raw[i,:,0], test_raw[i,:,1], color="#2E86AB", lw=1.5,
                alpha=0.8, label="GT" if i==0 else "")
        ax.plot(gen[i,:,0], gen[i,:,1], color=color, lw=1.2,
                linestyle="--", alpha=0.7, label=label if i==0 else "")
    ax.set_title("Individual: GT vs Generated"); ax.legend(fontsize=8)
    ax.set_xlabel("theta"); ax.grid(True, alpha=0.3)

    # Energy vs tau
    for ax, trajs, col, lbl in [
        (axes[1,0], test_raw, "#2E86AB", "GT"),
        (axes[1,1], gen,      color,     label),
    ]:
        H = hamiltonian(trajs[:,:,0], trajs[:,:,1])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, H[i], alpha=0.35, lw=0.7, color=col)
        ax.set_title(f"Energy vs tau — {lbl}")
        ax.set_xlabel("tau"); ax.set_ylabel("H"); ax.grid(True, alpha=0.3)

    # Training loss
    ax = axes[1,2]
    for key, col, lbl in [
        ("total", "black",   "Total"),
        ("fm",    "#2E86AB", "FM"),
        ("physics", color,   "Physics"),
    ]:
        if key in losses and losses[key]:
            ax.plot(losses[key], color=col, lw=1.2, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{out_dir}/case5{tag.lower().replace(' ','_')}_main.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_lambda_distribution(model, test_raw, device, out_dir, tag="A"):
    """Variant A only: plot the learned lambda distribution over ICs."""
    import torch
    from data import hamiltonian

    ic_enc = np.stack([
        np.sin(test_raw[:,0,0]),
        np.cos(test_raw[:,0,0]),
        test_raw[:,0,1],
    ], axis=-1)
    ic_t = torch.tensor(ic_enc, dtype=torch.float32, device=device)

    with torch.no_grad():
        lam = model.get_lambda(ic_t).cpu().numpy()

    E = hamiltonian(test_raw[:,0,0], test_raw[:,0,1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Variant A: Learned Lambda Distribution", fontsize=11)

    axes[0].hist(lam, bins=30, color="#E07B35", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("lambda(IC)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Lambda distribution over test ICs")
    axes[0].axvline(lam.mean(), color="black", linestyle="--",
                    label=f"mean={lam.mean():.4f}")
    axes[0].legend(fontsize=8)

    axes[1].scatter(E, lam, alpha=0.5, s=15, color="#E07B35")
    axes[1].set_xlabel("Initial energy H(theta0, omega0)")
    axes[1].set_ylabel("lambda(IC)")
    axes[1].set_title("Lambda vs energy level")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{out_dir}/case5{tag}_lambda_distribution.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_comparison_all(test_raw, results, out_dir):
    """Side-by-side phase portraits for all three variants + GT."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    configs = [
        ("GT",        test_raw,           "#2E86AB"),
        ("Var A\nAdaptive λ", results["A"]["gen"], "#E07B35"),
        ("Var B\nIC+physics", results["B"]["gen"], "#9B5DE5"),
        ("Var C\nIC+H0 cond",   results["C"]["gen"], "#3BB273"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Case 5: All Variants Compared to GT", fontsize=12, fontweight="bold")

    tau = np.arange(N_STEPS) * DT_PHYS

    for col, (lbl, trajs, color) in enumerate(configs):
        # Phase portrait
        ax = axes[0, col]
        for i in range(min(40, len(trajs))):
            ax.plot(trajs[i,:,0], trajs[i,:,1], alpha=0.3, lw=0.7, color=color)
        ax.set_title(lbl); ax.set_xlabel("theta"); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel("omega")

        # Energy vs tau
        ax = axes[1, col]
        H = hamiltonian(trajs[:,:,0], trajs[:,:,1])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, H[i], alpha=0.35, lw=0.7, color=color)
        ax.set_xlabel("tau"); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel("H")

    plt.tight_layout()
    fname = f"{out_dir}/case5_comparison_all.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(stats_gt, results):
    print("\n" + "=" * 72)
    print(f"{'Metric':<28} {'GT':>10}  {'Var A':>10}  {'Var B':>10}  {'Var C':>10}")
    print("-" * 72)
    for name, key in [
        ("mean std(H)",   "std"),
        ("mean |DH|",     "dh"),
        ("PDE res theta", "pde_theta"),
        ("PDE res omega", "pde_omega"),
    ]:
        row = f"  {name:<26} {stats_gt[key]:>10.3e}"
        for v in ["A", "B", "C"]:
            row += f"  {results[v]['stats'][key]:>10.3e}"
        print(row)
    print("=" * 72)
    print("  Var A: Adaptive lambda (IC-conditioned physics weight)")
    print("  Var B: FiLM IC conditioning + fixed lambda physics loss")
    print("  Var C: FiLM IC conditioning + Hamiltonian enforcement (mean + drift)")


# ── Main evaluation pipeline ──────────────────────────────────────────────────

def run_eval(models, losses_dict, test_raw, device, out_dir="outputs",
             ref_stats_4b=None):
    """
    Run evaluation for all three variants.

    models : dict with keys "A", "B", "C" -> trained model instances
    losses : dict with keys "A", "B", "C" -> loss history dicts
    """
    from train import sample_A, sample_BC, sample_C

    print("\n" + "=" * 65)
    print("EVALUATION — Case 5: Extended Experiments")
    print("=" * 65)

    n = len(test_raw)
    results = {}

    # Ground truth metrics
    print("\nGround Truth:")
    stats_gt = evaluate(test_raw, "GT")

    # Variant A
    print("\nVariant A (Adaptive Lambda):")
    gen_A = sample_A(models["A"], n, device)
    stats_A = evaluate(gen_A, "Var A: Adaptive lambda")
    results["A"] = {"gen": gen_A, "stats": stats_A}
    plot_variant(test_raw, gen_A, "Adaptive Lambda", "#E07B35",
                 stats_gt, stats_A, losses_dict["A"], out_dir, "A")
    plot_lambda_distribution(models["A"], test_raw, device, out_dir, "A")

    # Variant B
    print("\nVariant B (IC + physics loss):")
    gen_B, _ = sample_BC(models["B"], n, device)
    stats_B = evaluate(gen_B, "Var B: IC + physics loss")
    results["B"] = {"gen": gen_B, "stats": stats_B}
    plot_variant(test_raw, gen_B, "IC + Physics Loss", "#9B5DE5",
                 stats_gt, stats_B, losses_dict["B"], out_dir, "B")

    # Variant C
    print("\nVariant C (FiLM IC + H0, no physics loss):")
    gen_C, _ = sample_C(models["C"], n, device)
    stats_C = evaluate(gen_C, "Var C: FiLM IC + H0 (no physics loss)")
    results["C"] = {"gen": gen_C, "stats": stats_C}
    plot_variant(test_raw, gen_C, "IC + H0 Conditioning (no physics loss)", "#3BB273",
                 stats_gt, stats_C, losses_dict["C"], out_dir, "C")

    # Comparison plot
    plot_comparison_all(test_raw, results, out_dir)

    # Summary
    print_summary(stats_gt, results)

    return {"gt": stats_gt, "results": results}
