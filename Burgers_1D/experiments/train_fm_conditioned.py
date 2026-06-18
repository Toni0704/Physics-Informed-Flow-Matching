#!/usr/bin/env python
"""
Train the FiLM-CONDITIONED model with pure (vanilla) flow matching — no physics.

This single checkpoint is reused by two evaluation techniques:
  * technique 2 (FiLM conditioning + PCFM sampling), and
  * technique 4 (FiLM conditioning + vanilla sampling).

Faithful to the pcfm_film_ICBC notebook (cell 6):
  - Adam (NOT AdamW), lr = 3e-4
  - gradient accumulation: effective batch 128 = 2 x 64, with grad-norm clip 1.0
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
from src import add_pcfm_to_path
from src.dataset import BurgersConditionedDataset, cycle
from src.models import PatchedFNO, PCFM_FiLMConditionedFNO
from src.losses import flow_matching_loss

CONFIG = {
    "batch_size": 64,
    "eff_batch_size": 128,
    "num_iterations": 50000,
    "lr": 3e-4,
    "eval_every": 1000,
    "patience": 15,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="checkpoint output path")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(add_pcfm_to_path()) / "datasets" / "data"
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_fm_conditioned.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BurgersConditionedDataset(data_dir / "burgers_train_nIC80_nBC80.h5")
    U_MIN, U_MAX = train_ds.u_min, train_ds.u_max
    test_ds = BurgersConditionedDataset(data_dir / "burgers_test_nIC30_nBC30.h5",
                                        u_min=U_MIN, u_max=U_MAX)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=4)
    data_iter = cycle(train_loader)

    val_x0, val_ic, val_bc, _ = next(iter(val_loader))
    val_x0, val_ic, val_bc = val_x0.to(device), val_ic.to(device), val_bc.to(device)

    base_fno = PatchedFNO(n_modes=[32, 32], in_channels=3, emb_channels=32,
                          hidden_channels=64, proj_channels=256, n_layers=4).to(device)
    model = PCFM_FiLMConditionedFNO(base_fno, feature_channels=32).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for q in ema_model.parameters():
        q.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    accumulation_steps = CONFIG["eff_batch_size"] // CONFIG["batch_size"]

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    optimizer.zero_grad()
    for iteration in range(1, CONFIG["num_iterations"] + 1):
        x0, cond_ic, cond_bc, _ = next(data_iter)
        loss = flow_matching_loss(model, x0.to(device), cond_ic.to(device), cond_bc.to(device))
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
                v_loss = flow_matching_loss(ema_model, val_x0, val_ic, val_bc).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(ema_model.state_dict(), out_ckpt)
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
