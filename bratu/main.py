"""
main.py  --  Bratu Experiments Runner
---------------------------------------
Case B0: Vanilla FM -- no conditioning, no branch label
Case B1: FiLM on C -- model must learn bimodal p(u|C) from data
Case B2: B1 + PCFM projection at t=1 (no retraining)
Case B3: B1 evaluated at C = Cc (critical, unique solution)

Usage:
    python main.py               # all cases
    python main.py --case B0
    python main.py --case B1
    python main.py --case B2     # needs B1 checkpoint
    python main.py --case B3     # needs B1 checkpoint
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from config import SEED, N_EPOCHS, LR, C_CRIT
from data import get_dataloaders
from model import build_model_B0, build_model_B1
from train import (train, sample_B0, sample_B1,
                   sample_B2, sample_B3_critical)
from eval import (compute_metrics, plot_mode_coverage,
                  plot_solutions, plot_projection_effect,
                  plot_critical_case, print_summary)


def main(case="all"):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    print("\nLoading data...")
    train_loader, _, test_raw = get_dataloaders(seed=SEED)
    u_test  = test_raw["u"]
    C_test  = test_raw["C"]
    b_test  = test_raw["b"]    # kept for reference, NOT passed to model
    u_mean  = test_raw["u_mean"]
    u_std   = test_raw["u_std"]
    n       = len(u_test)

    all_metrics = {}
    gen_samples = {}          # store u_gen per case for mode coverage plot
    run_all = (case == "all")

    # ── B0 ────────────────────────────────────────────────────────────────────
    if run_all or case == "B0":
        print("\n" + "="*55 + "\nCASE B0: Vanilla FM\n" + "="*55)
        m = build_model_B0(device)
        train(m, train_loader, device, model_type="B0",
              save_name="best_B0.pt")
        m.load_state_dict(torch.load("checkpoints/best_B0.pt",
                                     map_location=device))
        u_gen = sample_B0(m, n, device, u_mean, u_std)
        all_metrics["B0: Vanilla"] = compute_metrics(
            u_gen, C_test, b_test, label="B0: Vanilla FM"
        )
        gen_samples["B0: Vanilla"] = u_gen
        plot_solutions(u_gen, C_test,
                       "Case B0: Vanilla FM", "#E07B35",
                       "outputs/B0_solutions.png")

    # ── B1 ────────────────────────────────────────────────────────────────────
    if run_all or case == "B1":
        print("\n" + "="*55 + "\nCASE B1: FiLM on C (no branch label)\n" + "="*55)
        m = build_model_B1(device)
        train(m, train_loader, device, model_type="B1",
              save_name="best_B1.pt")
        m.load_state_dict(torch.load("checkpoints/best_B1.pt",
                                     map_location=device))
        u_gen = sample_B1(m, C_test, device, u_mean, u_std)
        all_metrics["B1: FiLM(C)"] = compute_metrics(
            u_gen, C_test, b_test, label="B1: FiLM on C"
        )
        gen_samples["B1: FiLM(C)"] = u_gen
        plot_solutions(u_gen, C_test,
                       "Case B1: FiLM conditioned on C", "#2E86AB",
                       "outputs/B1_solutions.png")

    # Mode coverage plot (B0 vs B1 if both available)
    if len(gen_samples) > 0:
        plot_mode_coverage(gen_samples, C_test,
                           "outputs/mode_coverage.png")

    # ── B2 ────────────────────────────────────────────────────────────────────
    if run_all or case == "B2":
        print("\n" + "="*55 +
              "\nCASE B2: B1 + PCFM projection\n" + "="*55)
        ckpt = "checkpoints/best_B1.pt"
        if not Path(ckpt).exists():
            print(f"ERROR: {ckpt} not found. Run B1 first.")
        else:
            m = build_model_B1(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            u_unc, u_proj, infos = sample_B2(
                m, C_test, device, u_mean, u_std
            )
            all_metrics["B2: +PCFM"] = compute_metrics(
                u_proj, C_test, b_test,
                label="B2: B1 + PCFM projection"
            )
            gen_samples["B2: +PCFM"] = u_proj
            plot_projection_effect(u_unc, u_proj, C_test,
                                   "outputs/B2_projection.png")

    # ── B3 ────────────────────────────────────────────────────────────────────
    if run_all or case == "B3":
        print("\n" + "="*55 +
              f"\nCASE B3: Critical C = {C_CRIT:.4f}\n" + "="*55)
        ckpt = "checkpoints/best_B1.pt"
        if not Path(ckpt).exists():
            print(f"ERROR: {ckpt} not found. Run B1 first.")
        else:
            m = build_model_B1(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            u_gen, u_exact, C_crit = sample_B3_critical(
                m, device, u_mean, u_std, n_samples=200
            )
            b_crit = np.zeros(len(u_gen), dtype=np.int64)
            all_metrics["B3: Critical"] = compute_metrics(
                u_gen, C_crit, b_crit, label=f"B3: C = Cc"
            )
            std_mean = u_gen.std(axis=0).mean()
            all_metrics["B3: Critical"]["pointwise_std"] = std_mean
            print(f"    Pointwise std mean: {std_mean:.4e} "
                  f"(~0 expected for unique solution)")
            plot_critical_case(u_gen, u_exact,
                               "outputs/B3_critical.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_metrics:
        print_summary(all_metrics)

    print("\nAll done. Check outputs/ for plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all",
                        choices=["all","B0","B1","B2","B3"])
    args = parser.parse_args()
    main(args.case)