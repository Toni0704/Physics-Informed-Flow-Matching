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

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import add_pcfm_to_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="final checkpoint output path (default: weights/best_fm_uncond.pt)")
    p.add_argument("--logdir", default=None,
                   help="dir for PCFM's periodic logs/checkpoints (default: <PCFM>/logs); "
                        "point this at a persistent location (e.g. /kaggle/working/logs) "
                        "so progress survives a session restart")
    p.add_argument("--resume", default=None,
                   help="path to a latest.pt / <step>.pt to resume training from")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override configs/burgers1d.yml's train.batch_size (e.g. lower "
                        "this on GPUs with less than ~16GB to avoid CUDA OOM)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pcfm = Path(add_pcfm_to_path())
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_fm_uncond.pt"

    train_h5 = pcfm / "datasets" / "data" / "burgers_train_nIC80_nBC80.h5"
    if not train_h5.exists():
        raise FileNotFoundError(
            f"{train_h5} not found. Run `python experiments/generate_data.py` first.")

    config_path = "configs/burgers1d.yml"
    if args.batch_size is not None:
        with open(pcfm / "configs" / "burgers1d.yml") as f:
            cfg = yaml.safe_load(f)
        cfg["train"]["batch_size"] = args.batch_size
        tmp_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, dir=str(pcfm / "configs"))
        yaml.safe_dump(cfg, tmp_cfg)
        tmp_cfg.close()
        config_path = os.path.basename(tmp_cfg.name)
        print(f"[train_fm_uncond] batch_size overridden to {args.batch_size} "
              f"(temp config: configs/{config_path})")

    # Run PCFM's own training, from inside the repo with PYTHONPATH set to it.
    logdir = Path(args.logdir) if args.logdir else pcfm / "logs"
    env = dict(os.environ, PYTHONPATH=str(pcfm))
    cmd = ["python", "scripts/training/main.py", config_path,
           "--mode", "train", "--logdir", str(logdir), "--savename", "burgers_uncond"]
    if args.resume:
        cmd += ["--resume", args.resume]
    subprocess.run(cmd, cwd=str(pcfm), env=env, check=True)

    latest = logdir / "burgers_uncond" / "latest.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out_ckpt)
    print(f"Saved unconditioned model -> {out_ckpt}  (dict with key 'model')")


if __name__ == "__main__":
    main()
