"""
main.py  —  Case 4: Conditional Generation
-------------------------------------------
Usage:
    python main.py

Outputs (in outputs/):
    case4_main.png          — phase portraits, energy vs tau, targeting plot
    case4_energy_sweep.png  — phase portraits at 5 different E* values
                              with true energy contours overlaid (dashed)
    case4_individual.png    — 5 sample comparisons GT vs conditional gen
"""

import sys, torch, numpy as np
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
    print("Case 4: Conditional Generation")
    print("Architecture: FiLM-conditioned velocity field vθ(u_t, t, E)")
    print("No physics penalty. No projection. Constraint via conditioning.")
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
