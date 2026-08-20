#!/usr/bin/env python
"""
Train the FiLM-CONDITIONED model with pure (vanilla) flow matching -- no physics.

This checkpoint is the vanilla-sampling / no-physics-training baseline that a
future evaluate.py compares PBFM against (same conditioning, same backbone,
so the comparison isolates the training objective, not model capacity).

Recipe (same shape as Burgers_1D/NS_2D's train_fm_conditioned.py):
  - Adam (NOT AdamW), lr = 3e-4
  - gradient accumulation: effective batch 128 = 4 x 32, grad-norm clip 1.0
  - EMA of the weights with decay 0.99, started after iteration 1000
  - validation every 1000 iters via the FM loss on a fixed validation batch,
    early stopping with patience 15, best (lowest val) EMA weights saved.

    python experiments/train_fm_conditioned.py
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
from src.dataset import Darcy3DConditionedDataset, cycle
from src.models import PatchedFNO3D, PCFM_FiLMConditionedFNO3D
from src.losses import flow_matching_loss

CONFIG = {
    "batch_size": 32,
    "eff_batch_size": 128,
    "num_iterations": 50000,
    "lr": 3e-4,
    "eval_every": 1000,
    "patience": 15,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train",
                   default=str(REPO_ROOT / "datasets" / "data" / "darcy3d_train_n5000.h5"))
    p.add_argument("--data-test",
                   default=str(REPO_ROOT / "datasets" / "data" / "darcy3d_test_n500.h5"))
    p.add_argument("--out", default=None, help="checkpoint output path")
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

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_ds = Darcy3DConditionedDataset(args.data_train)
    U_MIN, U_MAX = train_ds.u_min, train_ds.u_max
    LOGK_MIN, LOGK_MAX = train_ds.logk_min, train_ds.logk_max
    test_ds = Darcy3DConditionedDataset(args.data_test, u_min=U_MIN, u_max=U_MAX,
                                         logk_min=LOGK_MIN, logk_max=LOGK_MAX)
    print(f"train {len(train_ds)} / test {len(test_ds)} samples, "
          f"u range [{U_MIN:.3f}, {U_MAX:.3f}]")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=2)
    data_iter = cycle(train_loader)

    val_x1, val_k, val_bc, _, _ = next(iter(val_loader))
    val_x1, val_k, val_bc = val_x1.to(device), val_k.to(device), val_bc.to(device)

    base_fno = PatchedFNO3D(n_modes=[8, 8, 8], in_channels=4, emb_channels=32,
                            hidden_channels=32, proj_channels=128, n_layers=4).to(device)
    model = PCFM_FiLMConditionedFNO3D(base_fno, feature_channels=32).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for q in ema_model.parameters():
        q.requires_grad = False
    print(f"model params: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    accumulation_steps = CONFIG["eff_batch_size"] // CONFIG["batch_size"]

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    optimizer.zero_grad()
    for iteration in range(1, CONFIG["num_iterations"] + 1):
        x1, cond_k, cond_bc, _, _ = next(data_iter)
        loss = flow_matching_loss(model, x1.to(device), cond_k.to(device), cond_bc.to(device))
        (loss / accumulation_steps).backward()

        if iteration % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if iteration > 1000:
                with torch.no_grad():
                    for ep, q in zip(ema_model.parameters(), model.parameters()):
                        ep.data.mul_(0.99).add_(q.data, alpha=0.01)

        if iteration % 100 == 0 or iteration == 1:
            print(f"Iter {iteration:05d} | FM loss: {loss.item():.4e}")

        if iteration % CONFIG["eval_every"] == 0:
            ema_model.eval()
            with torch.no_grad():
                v_loss = flow_matching_loss(ema_model, val_x1, val_k, val_bc).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save({"model": ema_model.state_dict(),
                            "u_min": U_MIN, "u_max": U_MAX,
                            "logk_min": LOGK_MIN, "logk_max": LOGK_MAX}, out_ckpt)
                print(f"   => [Val] FM loss {v_loss:.4e} (saved)")
            else:
                patience_counter += 1
                print(f"   => [Val] FM loss {v_loss:.4e} | best {best_val_loss:.4e} "
                      f"| patience {patience_counter}/{CONFIG['patience']}")
                if patience_counter >= CONFIG["patience"]:
                    print(f"[!] Early stopping at iteration {iteration}.")
                    break
            model.train()

    print(f"Saved conditioned model -> {out_ckpt}")


if __name__ == "__main__":
    main()
