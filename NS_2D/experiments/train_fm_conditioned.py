#!/usr/bin/env python
"""
Train the FiLM-(IC+forcing)-CONDITIONED model with pure (vanilla) flow
matching -- no physics. This single checkpoint is reused by two evaluation
techniques: FiLM conditioning + PCFM sampling, and FiLM conditioning +
vanilla sampling.

Same recipe as Burgers_1D/experiments/train_fm_conditioned.py:
  - Adam, lr 3e-4
  - gradient accumulation to an effective batch (default 32 = 4 x 8),
    grad-norm clip 1.0
  - EMA decay 0.99, started after iteration 1000
  - validation every 1000 iters (FM loss on a fixed validation batch),
    early stopping with patience 15, best (lowest val) EMA weights saved

plus interruption safety learned from the Burgers Kaggle runs: a resumable
`latest` checkpoint (model + EMA + optimizer + iteration) is written at every
validation, and --resume restarts from it.

Kaggle usage:

    python experiments/train_fm_conditioned.py \
        --data-train /kaggle/input/ns50_data/ns_nw50_nf50_s64_t50_mu0.001.h5 \
        --data-test  /kaggle/input/10_100/ns_nw10_nf100_s64_t50_mu0.001.h5 \
        --out /kaggle/working/weights/best_fm_conditioned.pt
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import REPO_ROOT
from src.dataset import NSConditionedDataset, cycle
from src.models import NSVelocityNet_FiLM_ICF
from src.losses import flow_matching_loss

CONFIG = {
    "batch_size": 8,
    "eff_batch_size": 32,
    "num_iterations": 30000,
    "lr": 3e-4,
    "eval_every": 1000,
    "patience": 15,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train",
                   default=str(REPO_ROOT / "datasets" / "data" / "ns_nw50_nf50_s64_t50_mu0.001.h5"))
    p.add_argument("--data-test",
                   default=str(REPO_ROOT / "datasets" / "data" / "ns_nw10_nf100_s64_t50_mu0.001.h5"))
    p.add_argument("--out", default=None, help="best-checkpoint output path")
    p.add_argument("--resume", default=None,
                   help="resume from a <out>.latest.pt written by a previous run")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    args = p.parse_args()

    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.iters:
        CONFIG["num_iterations"] = args.iters

    repo_root = Path(__file__).resolve().parents[1]
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_fm_conditioned.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_ckpt.with_suffix(".latest.pt")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_ds = NSConditionedDataset(args.data_train)
    test_ds = NSConditionedDataset(args.data_test, w_scale=train_ds.w_scale)
    print(f"train {len(train_ds)} / test {len(test_ds)} samples, "
          f"w_scale={train_ds.w_scale:.4f}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=2)
    data_iter = cycle(train_loader)

    val_x1, val_a, val_f = next(iter(val_loader))
    val_x1, val_a, val_f = val_x1.to(device), val_a.to(device), val_f.to(device)

    model = NSVelocityNet_FiLM_ICF(n_t=train_ds.n_t).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for q in ema_model.parameters():
        q.requires_grad = False
    print(f"model params: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    accumulation_steps = max(1, CONFIG["eff_batch_size"] // CONFIG["batch_size"])

    start_iter = 1
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        print(f"resumed from {args.resume} at iteration {start_iter}")

    model.train()
    optimizer.zero_grad()
    for iteration in range(start_iter, CONFIG["num_iterations"] + 1):
        x1, cond_a, cond_f = next(data_iter)
        loss = flow_matching_loss(model, x1.to(device), cond_a.to(device), cond_f.to(device))
        (loss / accumulation_steps).backward()

        if iteration % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if iteration > 1000:
                with torch.no_grad():
                    for ep, q in zip(ema_model.parameters(), model.parameters()):
                        ep.data.mul_(0.99).add_(q.data, alpha=0.01)

        if iteration % 100 == 0 or iteration == start_iter:
            print(f"Iter {iteration:05d} | FM loss: {loss.item():.4e}")

        if iteration % CONFIG["eval_every"] == 0:
            ema_model.eval()
            with torch.no_grad():
                v_loss = flow_matching_loss(ema_model, val_x1, val_a, val_f).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save({"model": ema_model.state_dict(),
                            "w_scale": train_ds.w_scale}, out_ckpt)
                print(f"   => [Val] FM loss {v_loss:.4e} (saved)")
            else:
                patience_counter += 1
                print(f"   => [Val] FM loss {v_loss:.4e} | best {best_val_loss:.4e} "
                      f"| patience {patience_counter}/{CONFIG['patience']}")

            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "w_scale": train_ds.w_scale,
            }, latest_ckpt)

            if patience_counter >= CONFIG["patience"]:
                print(f"[!] Early stopping at iteration {iteration}.")
                break
            model.train()

    print(f"Saved conditioned model -> {out_ckpt}")


if __name__ == "__main__":
    main()
