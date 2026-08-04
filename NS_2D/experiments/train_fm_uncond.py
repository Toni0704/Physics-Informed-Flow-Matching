#!/usr/bin/env python
"""
Train the UNCONDITIONED base FFM model (technique 3's backbone) for 2D NS.

Delegates to the PCFM repo's own scripts/training/main.py (randn-prior
functional flow matching with a 3D FNO, configs/ns.yml), exactly as
Burgers_1D/experiments/train_fm_uncond.py does for Burgers, so the
unconditioned backbone cannot drift from the reference pipeline. The
resulting checkpoint is exported to weights/best_fm_uncond.pt (or --out).

Kaggle usage (data attached as Kaggle Datasets, outputs persisted):

    python experiments/train_fm_uncond.py \
        --data-train /kaggle/input/ns50_data/ns_nw50_nf50_s64_t50_mu0.001.h5 \
        --data-test  /kaggle/input/10_100/ns_nw10_nf100_s64_t50_mu0.001.h5 \
        --logdir /kaggle/working/ns_logs \
        --out /kaggle/working/weights/best_fm_uncond.pt \
        --batch-size 8
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
from src import add_repo_to_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train", default=None,
                   help="training .h5 (default: configs/ns.yml's path)")
    p.add_argument("--data-test", default=None,
                   help="test/validation .h5 (default: configs/ns.yml's path)")
    p.add_argument("--out", default=None,
                   help="final checkpoint output path (default: weights/best_fm_uncond.pt)")
    p.add_argument("--logdir", default=None,
                   help="dir for periodic logs/checkpoints (default: <repo>/logs); point "
                        "at a persistent location (e.g. /kaggle/working/ns_logs) so "
                        "progress survives a session restart")
    p.add_argument("--resume", default=None,
                   help="path to a latest.pt / <step>.pt to resume training from")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override configs/ns.yml's train.batch_size")
    p.add_argument("--max-iter", type=int, default=None,
                   help="override configs/ns.yml's train.max_iter")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    repo = Path(add_repo_to_path())
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_fm_uncond.pt"

    # Bake overrides into a temp config so they survive session restarts and
    # relative-path confusion (lesson learned from the Burgers Kaggle runs).
    if bool(args.data_train) != bool(args.data_test):
        p.error("--data-train and --data-test must be given together "
                "(both paths become absolute once root is cleared)")
    with open(repo / "configs" / "ns.yml") as f:
        cfg = yaml.safe_load(f)
    if args.data_train:
        cfg["datasets"]["root"] = ""
        cfg["datasets"]["train"]["data_file"] = str(Path(args.data_train).resolve())
        cfg["datasets"]["test"]["data_file"] = str(Path(args.data_test).resolve())
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_iter is not None:
        cfg["train"]["max_iter"] = args.max_iter

    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, dir=str(repo / "configs"))
    yaml.safe_dump(cfg, tmp_cfg)
    tmp_cfg.close()
    config_path = os.path.join("configs", os.path.basename(tmp_cfg.name))
    print(f"[train_fm_uncond] launching with temp config {config_path}")

    logdir = Path(args.logdir) if args.logdir else repo / "logs"
    env = dict(os.environ, PYTHONPATH=str(repo))
    cmd = ["python", "scripts/training/main.py", config_path,
           "--mode", "train", "--logdir", str(logdir), "--savename", "ns_uncond"]
    if args.resume:
        cmd += ["--resume", args.resume]
    try:
        subprocess.run(cmd, cwd=str(repo), env=env, check=True)
    finally:
        os.unlink(repo / config_path)

    latest = logdir / "ns_uncond" / "latest.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out_ckpt)
    print(f"Saved unconditioned model -> {out_ckpt}  (dict with key 'model')")


if __name__ == "__main__":
    main()
