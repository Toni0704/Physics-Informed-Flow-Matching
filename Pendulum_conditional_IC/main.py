"""
main.py  --  Case 4b: Conditional Generation on Initial State
--------------------------------------------------------------
Usage:
    python main.py

Outputs (in outputs/):
    case4b_main.png                   -- phase portraits + energy plots,
                                         generative vs surrogate vs GT
    case4b_surrogate_individual.png   -- 5 samples: exact IC conditioning
    case4b_generative_individual.png  -- 5 samples: random IC conditioning

To compare against Case 4a, pass the Case 4a stats dict to run_eval:
    results_4a = {...}   # from running Case 4a
    results = run_eval(model, losses, test_raw, device,
                       stats_4a_ref=results_4a["stats_gen"])
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
    print("Case 4b: Conditional Generation on Initial State")
    print("Conditioning: v_theta(u_t, t, (sin t0, cos t0, omega0))")
    print("Two eval modes: generative (random IC) + surrogate (exact IC)")
    print("=" * 65)
    print(f"Device: {device}")

    train_loader, _, test_raw = get_dataloaders()
    model  = build_model(device)
    losses = train(model, train_loader, device, save_dir="checkpoints")

    # Pass stats_4a_ref=None since we don't have Case 4a results here.
    # To compare, run Case 4a first and pass its stats dict.
    results = run_eval(model, losses, test_raw, device,
                       out_dir="outputs", stats_4a_ref=None)

    print("\nDone. Check outputs/ for plots.")
    return results


if __name__ == "__main__":
    main()


def run_variant_D():
    """
    Run Variant D: load trained Case 4b model and apply post-hoc projection.
    No retraining needed — this is purely an inference-time change.
    """
    import torch
    from data import get_dataloaders
    from model import build_model
    from eval import run_variant_D as eval_D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Variant D: Case 4b + post-hoc projection")
    print(f"Device: {device}")

    _, _, test_raw = get_dataloaders()
    model = build_model(device)

    ckpt = "checkpoints/best_model_case4b.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded weights from {ckpt}")

    results = eval_D(model, test_raw, device, out_dir="outputs")
    return results