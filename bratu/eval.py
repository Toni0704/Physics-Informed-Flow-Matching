"""
eval.py  --  Bratu Evaluation
------------------------------
Metrics and plots for all four cases.

Key new evaluation: mode coverage histogram.
Since there is no branch label at inference, we evaluate whether the
model generates from both branches by plotting the u_max distribution
and checking for two peaks around the lower and upper branch u_max values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import N_X, U_CRIT, C_CRIT
from data import make_grid, exact_solution, pde_residual, bc_error


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(u_gen, C_vals, b_vals_true=None, label=""):
    """
    Compute metrics for a batch of generated solutions.

    b_vals_true is the ground-truth branch of the test set --
    used only to compute per-branch PDE residuals, NOT passed to the model.
    """
    pde_res = pde_residual(u_gen, C_vals)
    bc_err  = bc_error(u_gen)
    u_maxes = u_gen.max(axis=1)

    # Post-hoc branch classification of generated samples
    b_gen  = (u_maxes > U_CRIT).astype(int)
    n_low  = (b_gen == 0).sum()
    n_high = (b_gen == 1).sum()

    metrics = {
        "pde_residual_mean": pde_res.mean(),
        "pde_residual_std":  pde_res.std(),
        "bc_error_mean":     bc_err.mean(),
        "u_max_mean":        u_maxes.mean(),
        "u_max_std":         u_maxes.std(),
        "frac_lower":        n_low  / len(u_gen),
        "frac_upper":        n_high / len(u_gen),
    }

    # Per-branch PDE residual from generated classification
    for b, bname in [(0, "lower"), (1, "upper")]:
        mask = b_gen == b
        if mask.sum() > 0:
            metrics[f"pde_res_{bname}_gen"] = pde_res[mask].mean()

    # Exact error (where exact solution matches the generated branch)
    errs = []
    for i in range(len(u_gen)):
        b_i  = int(b_gen[i])
        u_ex = exact_solution(float(C_vals[i]), branch=b_i)
        if u_ex is not None:
            errs.append(np.max(np.abs(u_gen[i] - u_ex)))
    if errs:
        metrics["exact_error_mean"] = np.mean(errs)
        metrics["exact_error_max"]  = np.max(errs)

    print(f"\n  [{label:<38}]")
    print(f"    PDE residual (mean): {metrics['pde_residual_mean']:.4e}")
    print(f"    BC error (mean):     {metrics['bc_error_mean']:.4e}")
    print(f"    u_max (mean±std):    "
          f"{metrics['u_max_mean']:.3f} ± {metrics['u_max_std']:.3f}")
    print(f"    Generated: {n_low} lower ({100*metrics['frac_lower']:.1f}%) "
          f"| {n_high} upper ({100*metrics['frac_upper']:.1f}%)")
    if "exact_error_mean" in metrics:
        print(f"    Exact error (mean):  {metrics['exact_error_mean']:.4e}")

    return metrics


# ── Mode coverage histogram ───────────────────────────────────────────────────

def plot_mode_coverage(results_dict, C_vals, out_path):
    """
    Central diagnostic: histogram of generated u_max for each case.
    A good model shows two peaks (bimodal); a collapsed model shows one.

    results_dict: {case_name: u_gen array}
    """
    n_cases = len(results_dict)
    fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 4),
                             sharey=False)
    if n_cases == 1:
        axes = [axes]
    fig.suptitle("Mode Coverage: Distribution of Generated $u_{\\max}$\n"
                 "Two peaks = both branches learned; one peak = collapse",
                 fontsize=11, fontweight="bold")

    colors = ["#E07B35", "#2E86AB", "#3BB273", "#9B5DE5"]

    # True distribution from test set
    u_maxes_true = []
    for i in range(len(C_vals)):
        for b in [0, 1]:
            u_ex = exact_solution(float(C_vals[i]), branch=b)
            if u_ex is not None:
                u_maxes_true.append(u_ex.max())

    for ax, (name, u_gen), color in zip(
            axes, results_dict.items(), colors):
        u_max_gen = u_gen.max(axis=1)

        ax.hist(u_max_gen, bins=40, color=color, alpha=0.75,
                edgecolor="white", label="Generated", density=True)
        ax.axvline(U_CRIT, color="red", linestyle="--", lw=1.5,
                   label=f"$u_c$={U_CRIT:.3f}")

        # Mark where lower and upper branch peaks should be
        u_maxes_true_arr = np.array(u_maxes_true)
        ax.axvspan(u_maxes_true_arr[u_maxes_true_arr < U_CRIT].mean() - 0.3,
                   u_maxes_true_arr[u_maxes_true_arr < U_CRIT].mean() + 0.3,
                   alpha=0.15, color="blue", label="Expected lower peak")
        ax.axvspan(u_maxes_true_arr[u_maxes_true_arr > U_CRIT].mean() - 0.3,
                   u_maxes_true_arr[u_maxes_true_arr > U_CRIT].mean() + 0.3,
                   alpha=0.15, color="green", label="Expected upper peak")

        n_low  = (u_max_gen <= U_CRIT).sum()
        n_high = (u_max_gen  > U_CRIT).sum()
        ax.set_title(f"{name}\n"
                     f"lower: {n_low} ({100*n_low/len(u_max_gen):.0f}%)  "
                     f"upper: {n_high} ({100*n_high/len(u_max_gen):.0f}%)",
                     fontsize=9)
        ax.set_xlabel("$u_{\\max}$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Solution quality plots ────────────────────────────────────────────────────

def plot_solutions(u_gen, C_vals, title, color, out_path, n_show=5):
    """Plot n_show generated solutions vs exact (matching generated branch)."""
    x, _ = make_grid()
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    indices = np.linspace(0, len(u_gen)-1, n_show, dtype=int)
    for ax, idx in zip(axes, indices):
        C = float(C_vals[idx])
        u = u_gen[idx]
        b_gen = int(u.max() > U_CRIT)
        ax.plot(x, u, color=color, lw=2, label="Generated")
        u_ex = exact_solution(C, branch=b_gen)
        if u_ex is not None:
            ax.plot(x, u_ex, "k--", lw=1.5, alpha=0.7, label="Exact")
        bname = "upper" if b_gen == 1 else "lower"
        ax.set_title(f"C={C:.2f} ({bname})", fontsize=9)
        ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("u(x)"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_projection_effect(u_unc, u_proj, C_vals, out_path, n_show=5):
    """Case B2: before vs after PCFM projection."""
    x, _ = make_grid()
    from data import _bratu_residual, _nsfd_denominator
    _, h = make_grid(); hs = _nsfd_denominator(h)

    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    fig.suptitle("Case B2: Before vs After PCFM Projection",
                 fontsize=11, fontweight="bold")
    indices = np.linspace(0, len(u_unc)-1, n_show, dtype=int)
    for col, idx in enumerate(indices):
        C  = float(C_vals[idx])
        b  = int(u_proj[idx].max() > U_CRIT)
        u_ex = exact_solution(C, branch=b)

        ax = axes[0, col]
        ax.plot(x, u_unc[idx],  color="#9B5DE5", lw=1.5, label="Before")
        ax.plot(x, u_proj[idx], color="#3BB273", lw=1.5, label="After")
        if u_ex is not None:
            ax.plot(x, u_ex, "k--", lw=1.2, alpha=0.7, label="Exact")
        ax.set_title(f"C={C:.2f}", fontsize=9); ax.grid(True, alpha=0.3)
        if col == 0: ax.legend(fontsize=7); ax.set_ylabel("u(x)")

        ax2 = axes[1, col]
        r_unc  = np.abs(_bratu_residual(u_unc[idx],  C, hs))
        r_proj = np.abs(_bratu_residual(u_proj[idx], C, hs))
        ax2.semilogy(r_unc,  color="#9B5DE5", lw=1.5,
                     label=f"Before: {r_unc.mean():.2e}")
        ax2.semilogy(r_proj, color="#3BB273", lw=1.5,
                     label=f"After:  {r_proj.mean():.2e}")
        ax2.set_xlabel("Node"); ax2.grid(True, alpha=0.3)
        if col == 0: ax2.legend(fontsize=7); ax2.set_ylabel("|PDE residual|")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_critical_case(u_gen, u_exact, out_path):
    """Case B3: variance of generated solutions at C=Cc."""
    x, _ = make_grid()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Case B3: Critical Case $C = C_c = {C_CRIT:.4f}$\n"
                 "Unique solution — low variance expected",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    for i in range(min(30, len(u_gen))):
        ax.plot(x, u_gen[i], color="#9B5DE5", alpha=0.2, lw=0.8)
    ax.plot(x, u_gen.mean(axis=0), color="#9B5DE5", lw=2.5,
            label="Generated mean")
    if u_exact is not None:
        ax.plot(x, u_exact, "k--", lw=2, label="Exact")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Samples (purple) + mean vs exact (dashed)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    std_x = u_gen.std(axis=0)
    ax.plot(x, std_x, color="#9B5DE5", lw=2)
    ax.fill_between(x, 0, std_x, alpha=0.3, color="#9B5DE5")
    ax.set_xlabel("x"); ax.set_ylabel("std($u(x)$) across samples")
    ax.set_title(f"Pointwise std (mean={std_x.mean():.4e})\n"
                 "Should be ~0 since only one solution exists at $C_c$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_metrics):
    print("\n" + "=" * 78)
    print(f"{'Case':<22} {'PDE Res':>10}  {'BC Err':>10}  "
          f"{'Frac Lower':>10}  {'Frac Upper':>10}  {'Exact Err':>10}")
    print("-" * 78)
    for name, m in all_metrics.items():
        print(f"  {name:<20} "
              f"{m.get('pde_residual_mean', float('nan')):>10.4e}  "
              f"{m.get('bc_error_mean',     float('nan')):>10.4e}  "
              f"{m.get('frac_lower',        float('nan')):>10.3f}  "
              f"{m.get('frac_upper',        float('nan')):>10.3f}  "
              f"{m.get('exact_error_mean',  float('nan')):>10.4e}")
    print("=" * 78)
    print("  Frac Lower/Upper = fraction of generated samples on each branch")
    print("  Training split is 50/50 -- ideal model: Frac Lower ~ Frac Upper ~ 0.5")