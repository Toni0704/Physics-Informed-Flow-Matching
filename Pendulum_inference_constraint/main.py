"""
main.py  —  Case 2: Physics Enforced in Sampling
-------------------------------------------------
Entry point. Ties data → model → train → eval together.

Key point: the model here is IDENTICAL to Case 3.
The only difference is in eval.py / sample_constrained.py —
the inference procedure adds energy projection after every Euler step.

This means any difference in results between Case 2 and Case 3
is entirely attributable to the projection, not the model.

Usage:
    python main.py

Outputs (in outputs/):
    case2_main.png           — 3×3 comparison: GT | Unconstrained | Constrained
    case2_individual.png     — 5 individual phase portrait comparisons
    case2_violation_trace.png — energy violation during FM integration
    case2_training_loss.png  — training loss curve
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SEED
from data   import get_dataloaders
from model  import build_model
from train  import train
from eval   import run_eval


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("Case 2: Physics Enforced in Sampling")
    print("Model   : identical to Case 3 (vanilla FM)")
    print("Inference: Euler + energy projection at every step")
    print("=" * 65)
    print(f"Device: {device}")

    train_loader, _, test_raw = get_dataloaders()
    model  = build_model(device)
    losses = train(model, train_loader, device, save_dir="checkpoints")

    results = run_eval(model, losses, test_raw, device, out_dir="outputs")

    print("\nDone. Check outputs/ for plots.")
    return results


if __name__ == "__main__":
    main()
