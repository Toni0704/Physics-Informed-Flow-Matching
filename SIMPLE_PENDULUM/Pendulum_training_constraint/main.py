"""
main.py  —  Case 1: Physics Enforced in Training
-------------------------------------------------
Trains four models with λ ∈ {0.0, 0.1, 1.0, 10.0} and compares them.

λ=0.0 is identical to Case 3 (vanilla FM) — serves as the baseline.

Usage:
    python main.py

Outputs (in outputs/):
    case1_phase_portraits.png  — phase portraits for each λ
    case1_training_curves.png  — FM loss + physics loss per λ
    case1_tradeoff.png         — metric vs λ bar charts (the key plot)
    case1_energy_vs_tau.png    — energy over physical time for best λ
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Simple_pendulum.Pendulum_training_constraint.config import SEED
from Simple_pendulum.Pendulum_training_constraint.data   import get_dataloaders
from Simple_pendulum.Pendulum_training_constraint.eval   import run_eval


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("Case 1: Physics Enforced in Training")
    print("Loss = L_FM  +  λ · Var_τ[H(θ,ω)]")
    print(f"λ sweep: [0.0, 0.1, 1.0, 10.0]")
    print("=" * 65)
    print(f"Device: {device}")

    _, _, test_raw = get_dataloaders()
    results = run_eval(test_raw, device, out_dir="outputs")

    print("\nDone. Check outputs/ for plots.")
    return results


if __name__ == "__main__":
    main()
