#!/usr/bin/env python
"""
Train the UNCONDITIONED base FFM model (technique 3's backbone).

Faithful to the btp2-3 notebook, where this model was trained by the PCFM repo's
own `scripts/training/main.py` (GP-prior functional flow matching, Adam + plateau
schedule, grad clipping, 20k iters). This script orchestrates that exact training
and exports the resulting checkpoint into weights/best_fm_uncond.pt.

It does NOT reimplement the training loop, so it cannot drift from the notebook.

Prerequisite: run experiments/generate_data.py first (it writes the .h5 files into
<PCFM>/datasets/data/, which PCFM's config reads from).

    python experiments/train_fm_uncond.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import add_pcfm_to_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pcfm = Path(add_pcfm_to_path())
    out_ckpt = repo_root / "weights" / "best_fm_uncond.pt"

    train_h5 = pcfm / "datasets" / "data" / "burgers_train_nIC80_nBC80.h5"
    if not train_h5.exists():
        raise FileNotFoundError(
            f"{train_h5} not found. Run `python experiments/generate_data.py` first.")

    # Run PCFM's own training, from inside the repo with PYTHONPATH set to it.
    logdir = pcfm / "logs"
    env = dict(os.environ, PYTHONPATH=str(pcfm))
    subprocess.run(
        ["python", "scripts/training/main.py", "configs/burgers1d.yml",
         "--mode", "train", "--logdir", str(logdir), "--savename", "burgers_uncond"],
        cwd=str(pcfm), env=env, check=True,
    )

    latest = logdir / "burgers_uncond" / "latest.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out_ckpt)
    print(f"Saved unconditioned model -> {out_ckpt}  (dict with key 'model')")


if __name__ == "__main__":
    main()
