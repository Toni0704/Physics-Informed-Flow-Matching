"""
main.py
-------
Entry point for Case 3: Physics Inherent to Data.

Ties together data → model → train → eval in one clean run.

Usage:
    python main.py

Outputs:
    checkpoints/best_model.pt   — saved model weights
    outputs/case3_main.png      — training loss + phase portraits + energy plots
    outputs/case3_individual.png — 5 individual trajectory comparisons
    outputs/case3_energy_drift.png — energy drift scatter
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Allow imports from parent directory (for config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from Simple_pendulum.Pendulum_data_only.config import SEED
from Simple_pendulum.Pendulum_data_only.data   import get_dataloaders
from Simple_pendulum.Pendulum_data_only.model  import build_model
from Simple_pendulum.Pendulum_data_only.train  import train
from Simple_pendulum.Pendulum_data_only.eval   import run_eval


def main():
    # ── Reproducibility ────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 55)
    print("Case 3: Physics Inherent to Data")
    print("Vanilla FM  +  Störmer-Verlet training data")
    print("=" * 55)
    print(f"Device : {device}")

    # ── Data ───────────────────────────────────────────────────
    train_loader, train_trajs, test_trajs = get_dataloaders()

    # ── Model ──────────────────────────────────────────────────
    model = build_model(device)

    # ── Train ──────────────────────────────────────────────────
    losses = train(
        model,
        train_loader,
        device,
        save_dir="checkpoints",
    )

    # ── Evaluate ───────────────────────────────────────────────
    results = run_eval(
        model,
        losses,
        test_trajs,
        device,
        out_dir="outputs",
    )

    print("\nDone. Check outputs/ for plots.")
    return results


if __name__ == "__main__":
    main()
