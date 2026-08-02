"""
main.py  --  Case 5: Extended Experiments
-------------------------------------------
Runs all three new variants sequentially and produces a unified comparison.

  Variant A: Adaptive lambda (IC-conditioned physics weight, standard MLP)
  Variant B: FiLM IC conditioning + fixed lambda physics loss
  Variant C: FiLM IC conditioning + Hamiltonian enforcement (drift + level)

Usage:
    python main.py                    # all three variants
    python main.py --variant A        # single variant
    python main.py --variant B
    python main.py --variant C

To compare against Case 4b (IC conditioning only, no physics loss),
run Case 4b first and note its metrics from the terminal output.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

from Simple_pendulum.Pendulum_2.config import SEED, N_EPOCHS, LR
from Simple_pendulum.Pendulum_2.data   import get_dataloaders
from Simple_pendulum.Pendulum_2.model  import build_model_A, build_model_BC, build_model_C
from Simple_pendulum.Pendulum_2.train  import train_A, train_BC, train_C, sample_C, LAMBDA_BASE
from Simple_pendulum.Pendulum_2.eval   import run_eval


def main(variant: str = "all"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("Case 5: Extended Experiments")
    print(f"Device: {device}")
    print("=" * 65)

    train_loader, _, test_raw = get_dataloaders()

    models     = {}
    losses_all = {}
    variants_to_run = ["A", "B", "C"] if variant == "all" else [variant.upper()]

    for v in variants_to_run:
        save_dir = f"checkpoints/variant_{v}"

        if v == "A":
            print(f"\n{'='*30} Variant A: Adaptive Lambda {'='*30}")
            model = build_model_A(lambda_base=LAMBDA_BASE, device=device)
            losses = train_A(model, train_loader, device,
                             n_epochs=N_EPOCHS, lr=LR, save_dir=save_dir)

        elif v == "B":
            print(f"\n{'='*30} Variant B: IC + Physics Loss {'='*30}")
            model = build_model_BC(device=device)
            losses = train_BC(model, train_loader, device,
                              lam=LAMBDA_BASE, variant="B",
                              n_epochs=N_EPOCHS, lr=LR, save_dir=save_dir)

        elif v == "C":
            print(f"\n{'='*30} Variant C: FiLM on IC + H0 (no physics loss) {'='*30}")
            model = build_model_C(device=device)
            losses = train_C(model, train_loader, device,
                             n_epochs=N_EPOCHS, lr=LR, save_dir=save_dir)

        models[v]     = model
        losses_all[v] = losses

    # Run evaluation for all trained variants
    Path("outputs").mkdir(exist_ok=True)
    if len(models) == 3:
        # Full comparison across all three
        summary = run_eval(models, losses_all, test_raw, device, out_dir="outputs")
    else:
        # Single variant — still run eval but only for that variant
        from Simple_pendulum.Pendulum_2.eval import evaluate, plot_variant
        from Simple_pendulum.Pendulum_2.train import sample_A, sample_BC, sample_C

        v = list(models.keys())[0]
        model = models[v]
        n = len(test_raw)
        if v == "A":
            gen = sample_A(model, n, device)
        elif v == "C":
            gen, _ = sample_C(model, n, device)
        else:
            gen, _ = sample_BC(model, n, device)

        print(f"\nMetrics (Variant {v}):")
        stats = evaluate(gen, f"Variant {v}")
        stats_gt_val = evaluate(test_raw, "Ground Truth")
        plot_variant(test_raw, gen, f"Variant {v}", "#9B5DE5",
                     stats_gt_val, stats, losses_all[v], "outputs", v)

    print("\nAll done. Check outputs/ for plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all",
                        choices=["all", "A", "B", "C", "a", "b", "c"])
    args = parser.parse_args()
    main(args.variant)