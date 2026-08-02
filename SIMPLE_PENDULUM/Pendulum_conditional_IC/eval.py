"""
eval.py  --  Case 4b: Conditional Generation on Initial State
--------------------------------------------------------------
Two evaluation modes:

  A) Generative mode:
       Sample random ICs from U(E_MIN, E_MAX), generate trajectories
       conditioned on those ICs. Measures whether the model can generate
       physically valid trajectories given an initial state it hasn't
       seen before. Directly comparable to Case 4a metrics.

  B) Surrogate solver mode:
       Condition on the EXACT IC of each test trajectory and try to
       reproduce the ground truth. This is the upper bound -- maximum
       information available. If this fails, the architecture is the
       bottleneck. If this succeeds but A) fails, the issue is IC
       sampling quality.

  Key comparison this eval answers:
       Case 4a (cond. on E)  vs  Case 4b (cond. on IC):
       Does knowing the exact IC make energy conservation / PDE
       fidelity significantly better than just knowing E?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch

from Simple_pendulum.Pendulum_conditional_IC.config import N_STEPS, DT_PHYS, E_MIN, E_MAX, SEED
from Simple_pendulum.Pendulum_conditional_IC.data   import hamiltonian, get_dataloaders
from Simple_pendulum.Pendulum_conditional_IC.model  import build_model
from Simple_pendulum.Pendulum_conditional_IC.train  import train, sample_generative, sample_surrogate


# ── Metrics (same as all other cases for direct comparison) ──────────────────

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
    print(f"  [{label:<30}]  std(H)={std:.3e}  "
          f"|DH|={drift:.3e}  PDE_th={r_th:.3e}  PDE_om={r_om:.3e}")
    return {"H": H, "std": std, "drift": drift,
            "pde_theta": r_th, "pde_omega": r_om}


# ── IC targeting test ─────────────────────────────────────────────────────────

def ic_targeting_test(model, test_raw, device, n_per_ic=1):
    """
    For each test trajectory, condition on its true IC and measure:
      1. How close is the generated first step to the true IC?
      2. What is the energy of the generated trajectory?
      3. What is the trajectory MSE vs ground truth?

    This directly tests whether the model has learned to respect
    the initial condition as a hard constraint.
    """
    gen = sample_surrogate(model, test_raw, device)

    # IC error: how well does the generated first step match the true first step
    ic_true = test_raw[:, 0, :]   # (n, 2)  [theta0, omega0]
    ic_gen  = gen[:,    0, :]     # (n, 2)
    ic_err  = np.sqrt(np.mean((ic_gen - ic_true)**2, axis=1))  # (n,)

    # Trajectory MSE
    traj_mse = np.mean((gen - test_raw)**2)

    # Energy of generated vs true
    E_true = hamiltonian(test_raw[:, 0, 0], test_raw[:, 0, 1])
    E_gen  = hamiltonian(gen[:,    0, 0], gen[:,    0, 1])
    E_err  = np.abs(E_gen - E_true)

    print(f"\n  IC targeting (surrogate mode):")
    print(f"    Mean IC position error : {ic_err.mean():.4e}  rad/rad/s")
    print(f"    Mean |E_gen - E_true|  : {E_err.mean():.4e}")
    print(f"    Trajectory MSE         : {traj_mse:.4e}")

    return gen, ic_err, E_err, traj_mse


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(
    test_raw, gen_generative, gen_surrogate,
    stats_gt, stats_gen, stats_surr,
    losses, ic_raw, E_ic, ic_err,
    out_dir="outputs"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tau = np.arange(N_STEPS) * DT_PHYS

    # ── Figure 1: Main comparison ──────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Case 4b — Conditioning on Initial State (theta0, omega0)\n"
        "Left: generative mode (random IC)   |   Right: surrogate mode (exact IC from test set)",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35)

    configs = [
        (test_raw,        stats_gt,   "#2E86AB", "Ground Truth"),
        (gen_generative,  stats_gen,  "#9B5DE5", "Case 4b: Generative (random IC)"),
        (gen_surrogate,   stats_surr, "#3BB273", "Case 4b: Surrogate (exact IC)"),
    ]

    # Phase portraits
    for col, (trajs, _, color, label) in enumerate(configs):
        ax = fig.add_subplot(gs[0, col])
        for i in range(min(40, len(trajs))):
            ax.plot(trajs[i,:,0], trajs[i,:,1],
                    alpha=0.3, lw=0.7, color=color)
        ax.set_title(f"Phase Portrait\n{label}", fontsize=8)
        ax.set_xlabel("theta"); ax.set_ylabel("omega"); ax.grid(True, alpha=0.3)

    # Energy vs tau
    for col, (trajs, stats, color, label) in enumerate(configs):
        ax = fig.add_subplot(gs[1, col])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, stats["H"][i], alpha=0.35, lw=0.7, color=color)
        ax.set_title(f"Energy vs tau\n{label}", fontsize=8)
        ax.set_xlabel("tau"); ax.set_ylabel("H"); ax.grid(True, alpha=0.3)

    # IC error distribution (surrogate mode)
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(ic_err, bins=25, color="#3BB273", alpha=0.8, edgecolor="white")
    ax.set_xlabel("IC position error (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Surrogate Mode\nIC Position Error Distribution", fontsize=8)
    ax.axvline(ic_err.mean(), color="black", linestyle="--", lw=1.2,
               label=f"mean = {ic_err.mean():.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Training loss
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(losses, color="#9B5DE5", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("FM Loss")
    ax.set_title("Training Loss", fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.savefig(f"{out_dir}/case4b_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/case4b_main.png")

    # ── Figure 2: Surrogate mode — 5 individual trajectory comparisons ──
    fig2, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig2.suptitle(
        "Case 4b Surrogate Mode — Individual Trajectories\n"
        "GT (blue) vs Generated given exact IC (green) with true energy contour (dotted)",
        fontsize=10
    )
    for i, ax in enumerate(axes):
        if i >= len(test_raw): break
        E_true = hamiltonian(test_raw[i,0,0], test_raw[i,0,1])
        ax.plot(test_raw[i,:,0], test_raw[i,:,1],
                color="#2E86AB", lw=1.8, label="GT" if i==0 else "")
        ax.plot(gen_surrogate[i,:,0], gen_surrogate[i,:,1],
                color="#3BB273", lw=1.2, linestyle="--",
                label="Surrogate" if i==0 else "")
        # True energy contour
        th_c = np.linspace(-np.pi*0.9, np.pi*0.9, 300)
        val  = 2*(E_true + np.cos(th_c))
        mask = val >= 0
        ax.plot(th_c,  np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4)
        ax.plot(th_c, -np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4)
        ax.set_title(f"Sample {i+1}\nE={E_true:.2f}"); ax.set_xlabel("theta")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case4b_surrogate_individual.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir}/case4b_surrogate_individual.png")

    # ── Figure 3: Generative mode — 5 individual trajectories ──────────
    fig3, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig3.suptitle(
        "Case 4b Generative Mode — Individual Trajectories\n"
        "Generated conditioned on random ICs (purple) with true contour (dotted)",
        fontsize=10
    )
    for i, ax in enumerate(axes):
        if i >= len(gen_generative): break
        th0, om0 = ic_raw[i, 0], ic_raw[i, 1]
        E_cond = hamiltonian(th0, om0)
        ax.plot(gen_generative[i,:,0], gen_generative[i,:,1],
                color="#9B5DE5", lw=1.2)
        th_c = np.linspace(-np.pi*0.9, np.pi*0.9, 300)
        val  = 2*(E_cond + np.cos(th_c))
        mask = val >= 0
        ax.plot(th_c,  np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4, label="True contour" if i==0 else "")
        ax.plot(th_c, -np.where(mask, np.sqrt(np.maximum(val,0)), np.nan),
                "k:", lw=0.8, alpha=0.4)
        ax.set_title(f"Sample {i+1}\nIC E={E_cond:.2f}")
        ax.set_xlabel("theta"); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/case4b_generative_individual.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_dir}/case4b_generative_individual.png")


# ── Summary comparison table ──────────────────────────────────────────────────

def print_comparison(stats_gt, stats_4a_ref, stats_gen, stats_surr):
    """
    Print a table comparing Case 4a vs Case 4b (both modes).
    stats_4a_ref can be passed in from Case 4a results if available,
    otherwise set to None to skip that column.
    """
    print("\n" + "=" * 75)
    print(f"{'Metric':<28} {'GT':>10}  {'4a (E)':>10}  {'4b-gen':>10}  {'4b-surr':>10}")
    print("-" * 75)
    rows = [
        ("mean std(H)",   "std"),
        ("mean |DH|",     "drift"),
        ("PDE res theta", "pde_theta"),
        ("PDE res omega", "pde_omega"),
    ]
    for name, key in rows:
        g  = f"{stats_gt[key]:.3e}"
        a  = f"{stats_4a_ref[key]:.3e}" if stats_4a_ref else "  ---"
        bg = f"{stats_gen[key]:.3e}"
        bs = f"{stats_surr[key]:.3e}"
        print(f"  {name:<26} {g:>10}  {a:>10}  {bg:>10}  {bs:>10}")
    print("=" * 75)
    print("  4a (E)    = Case 4a: conditioned on scalar energy E")
    print("  4b-gen    = Case 4b: conditioned on IC, random IC at inference")
    print("  4b-surr   = Case 4b: conditioned on IC, exact test IC at inference")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_eval(model, losses, test_raw, device, out_dir="outputs",
             stats_4a_ref=None):
    """
    Full evaluation pipeline for Case 4b.

    Args:
        model        : trained ConditionalVelocityFieldIC
        losses       : training loss history
        test_raw     : ground truth test trajectories, shape (n, N_STEPS, 2)
        device       : torch device
        out_dir      : output directory for plots
        stats_4a_ref : optional dict of Case 4a metrics for comparison table
    """
    print("\n" + "=" * 65)
    print("EVALUATION — Case 4b: Conditioning on Initial State")
    print("=" * 65)
    n = len(test_raw)

    # Mode A: generative (random IC)
    print(f"\n[Mode A] Generative: sampling {n} random ICs...")
    gen_gen, ic_raw, E_ic = sample_generative(model, n, device)

    # Mode B: surrogate (exact IC from test set)
    print(f"[Mode B] Surrogate: conditioning on exact test ICs...")
    gen_surr, ic_err, E_err, traj_mse = ic_targeting_test(model, test_raw, device)

    # Metrics
    print("\nMetrics:")
    stats_gt   = evaluate(test_raw,  "Ground Truth")
    stats_gen  = evaluate(gen_gen,   "Case 4b Generative (random IC)")
    stats_surr = evaluate(gen_surr,  "Case 4b Surrogate  (exact IC) ")

    # Comparison table
    print_comparison(stats_gt, stats_4a_ref, stats_gen, stats_surr)

    # Plots
    print("\nSaving plots...")
    plot_all(test_raw, gen_gen, gen_surr,
             stats_gt, stats_gen, stats_surr,
             losses, ic_raw, E_ic, ic_err, out_dir)

    return {
        "gen_generative": gen_gen,
        "gen_surrogate":  gen_surr,
        "stats_gt":       stats_gt,
        "stats_gen":      stats_gen,
        "stats_surr":     stats_surr,
        "ic_raw":         ic_raw,
        "ic_err":         ic_err,
        "traj_mse":       traj_mse,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_loader, _, test_raw = get_dataloaders()
    model  = build_model(device)
    losses = train(model, train_loader, device, save_dir="checkpoints")
    results = run_eval(model, losses, test_raw, device, out_dir="outputs")
    print("\nAll done.")


# ── Variant D evaluation ───────────────────────────────────────────────────────

def run_variant_D(model, test_raw, device, out_dir="outputs"):
    """
    Evaluate Variant D: Case 4b + post-hoc projection.

    Compares three things:
      - Ground truth
      - Case 4b unconstrained (plain FM with IC conditioning)
      - Variant D (FM + post-hoc projection onto energy contour)

    Key question: now that the model is already near the constraint
    manifold (unlike Case 2), does projection help or hurt?
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from Simple_pendulum.Pendulum_conditional_IC.train import sample_generative_projected
    from Simple_pendulum.Pendulum_conditional_IC.data  import hamiltonian

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n   = len(test_raw)
    tau = np.arange(N_STEPS) * DT_PHYS

    print("\n" + "=" * 65)
    print("EVALUATION — Variant D: Case 4b + Post-hoc Projection")
    print("=" * 65)

    gen_unc, gen_proj, ic_raw, E_ic, diag = sample_generative_projected(
        model, n, device
    )

    print("\nMetrics:")
    stats_gt   = evaluate(test_raw,  "Ground Truth              ")
    stats_unc  = evaluate(gen_unc,   "Case 4b unconstrained     ")
    stats_proj = evaluate(gen_proj,  "Variant D: 4b + projection")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Variant D: Case 4b + Post-hoc Projection\n"
        "Does projection help when the model is already near the constraint manifold?",
        fontsize=11, fontweight="bold"
    )

    configs = [
        (test_raw,  "#2E86AB", "Ground Truth"),
        (gen_unc,   "#9B5DE5", "Case 4b: IC conditioned"),
        (gen_proj,  "#3BB273", "Variant D: 4b + projection"),
    ]

    # Phase portraits
    for col, (trajs, color, label) in enumerate(configs):
        ax = axes[0, col]
        for i in range(min(40, len(trajs))):
            ax.plot(trajs[i,:,0], trajs[i,:,1],
                    alpha=0.3, lw=0.7, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("theta"); ax.set_ylabel("omega")
        ax.grid(True, alpha=0.3)

    # Energy vs tau
    for col, (trajs, color, label) in enumerate(configs):
        ax = axes[1, col]
        H = hamiltonian(trajs[:,:,0], trajs[:,:,1])
        for i in range(min(20, len(trajs))):
            ax.plot(tau, H[i], alpha=0.35, lw=0.7, color=color)
        ax.set_title(f"Energy vs tau — {label}", fontsize=9)
        ax.set_xlabel("tau"); ax.set_ylabel("H")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/variantD_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir}/variantD_main.png")

    # Individual samples
    fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4))
    fig2.suptitle(
        "Variant D — Individual Trajectories\n"
        "GT (blue) | 4b unconstrained (purple) | 4b + projection (green) | true contour (dotted)",
        fontsize=10
    )
    for i, ax in enumerate(axes2):
        if i >= n: break
        E_true = hamiltonian(test_raw[i,0,0], test_raw[i,0,1])
        ax.plot(test_raw[i,:,0],  test_raw[i,:,1],  color="#2E86AB", lw=1.8, label="GT" if i==0 else "")
        ax.plot(gen_unc[i,:,0],   gen_unc[i,:,1],   color="#9B5DE5", lw=1.0, linestyle="--", alpha=0.7, label="4b" if i==0 else "")
        ax.plot(gen_proj[i,:,0],  gen_proj[i,:,1],  color="#3BB273", lw=1.2, linestyle=":", label="4b+proj" if i==0 else "")
        # True energy contour
        th_c = np.linspace(-np.pi*0.9, np.pi*0.9, 300)
        val  = 2*(E_ic[i] + np.cos(th_c))
        mask = val >= 0
        ax.plot(th_c,  np.where(mask, np.sqrt(np.maximum(val,0)), np.nan), "k:", lw=0.7, alpha=0.3)
        ax.plot(th_c, -np.where(mask, np.sqrt(np.maximum(val,0)), np.nan), "k:", lw=0.7, alpha=0.3)
        ax.set_title(f"Sample {i+1}\nE_ic={E_ic[i]:.2f}", fontsize=8)
        ax.set_xlabel("theta"); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/variantD_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_dir}/variantD_individual.png")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'GT':>10}  {'4b':>10}  {'4b+proj':>10}")
    print("-" * 65)
    for name, key in [("std(H)", "std"), ("|DH|","drift"),
                       ("PDE theta","pde_theta"), ("PDE omega","pde_omega")]:
        print(f"  {name:<26} {stats_gt[key]:>10.3e}  "
              f"{stats_unc[key]:>10.3e}  {stats_proj[key]:>10.3e}")
    print("=" * 65)
    print("  If 4b+proj better than 4b: projection works when model is near manifold")
    print("  If 4b+proj worse than 4b:  projection still hurts even with good base model")

    return {"stats_gt": stats_gt, "stats_unc": stats_unc,
            "stats_proj": stats_proj, "diag": diag}