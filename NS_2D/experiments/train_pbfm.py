#!/usr/bin/env python
"""
Train the FiLM-(IC+forcing)-conditioned model with Physics-Based Flow Matching
(PBFM), using the spectral vorticity-equation residual (src/physics.py) as the
physics term. Same recipe as Burgers_1D/experiments/train_pbfm.py:

  - AdamW, lr 3e-5, weight_decay 0, betas (0.5, 0.999)
  - differentiable Euler unrolling with a curriculum that ramps n: 1 -> 4
  - logit-normal importance weighting on the FM loss
  - conflict-free gradient update (ConFIG) combining the separate FM and
    physics gradients, with a NaN guard falling back to the FM-only gradient
  - EMA decay 0.999
  - validation every 1000 iters on (data_loss + phys_loss); early stopping
    with patience 20; best (lowest val sum) EMA weights saved

plus a resumable `latest` checkpoint at every validation (--resume restarts).

Kaggle usage:

    pip install conflictfree
    python experiments/train_pbfm.py \
        --data-train /kaggle/input/ns50_data/ns_nw50_nf50_s64_t50_mu0.001.h5 \
        --data-test  /kaggle/input/10_100/ns_nw10_nf100_s64_t50_mu0.001.h5 \
        --out /kaggle/working/weights/best_pbfm.pt
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
from src.losses import pbfm_loss

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector, apply_gradient_vector

CONFIG = {
    "batch_size": 4,
    "num_iterations": 10000,
    "lr": 3e-5,
    "ema_decay": 0.999,
    "eval_every": 1000,
    "patience": 20,
    "max_unroll_steps": 4,
    "use_dignorm": True,
    "use_config": True,
    "visc": 1e-3,
    "frame_dt": 1.0,
}


def curriculum_n_steps(iteration, total, max_n):
    quarter = max(1, total // max_n)
    n = (iteration - 1) // quarter + 1
    return min(n, max_n)


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
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_pbfm.pt"
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

    vb_x1, vb_a, vb_f = next(iter(val_loader))
    vb_x1, vb_a, vb_f = vb_x1.to(device), vb_a.to(device), vb_f.to(device)

    model = NSVelocityNet_FiLM_ICF(n_t=train_ds.n_t).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for q in ema_model.parameters():
        q.requires_grad = False
    print(f"model params: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=0.0, betas=(0.5, 0.999))

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

    def ema_update(decay):
        with torch.no_grad():
            for ep, q in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(decay).add_(q.data, alpha=1 - decay)

    model.train()
    for iteration in range(start_iter, CONFIG["num_iterations"] + 1):
        n_unroll = curriculum_n_steps(iteration, CONFIG["num_iterations"],
                                      CONFIG["max_unroll_steps"])

        x1, cond_a, cond_f = next(data_iter)
        x1, cond_a, cond_f = x1.to(device), cond_a.to(device), cond_f.to(device)

        data_loss, phys_loss = pbfm_loss(
            model, x1, cond_a, cond_f, n_steps=n_unroll,
            w_scale=train_ds.w_scale, visc=CONFIG["visc"],
            frame_dt=CONFIG["frame_dt"], use_dignorm=CONFIG["use_dignorm"],
        )

        if CONFIG["use_config"]:
            grads = []
            optimizer.zero_grad()
            data_loss.backward(retain_graph=True)
            grads.append(get_gradient_vector(model))
            optimizer.zero_grad()
            phys_loss.backward()
            grads.append(get_gradient_vector(model))

            if not grads[1].isnan().any():
                apply_gradient_vector(model, ConFIG_update(grads))
            else:
                if iteration % 100 == 0:
                    print(f"   [warn] NaN physics gradient at iter {iteration}; using FM grads")
                apply_gradient_vector(model, grads[0])
        else:
            optimizer.zero_grad()
            (data_loss + phys_loss).backward()

        optimizer.step()
        ema_update(CONFIG["ema_decay"])

        if iteration % 100 == 0 or iteration == start_iter:
            print(f"Iter {iteration:05d} | n_unroll={n_unroll} | "
                  f"Data {data_loss.item():.4e} | Phys {phys_loss.item():.4e}")

        if iteration % CONFIG["eval_every"] == 0:
            ema_model.eval()
            with torch.no_grad():
                v_data, v_phys = pbfm_loss(
                    ema_model, vb_x1, vb_a, vb_f, n_steps=n_unroll,
                    w_scale=train_ds.w_scale, visc=CONFIG["visc"],
                    frame_dt=CONFIG["frame_dt"], use_dignorm=CONFIG["use_dignorm"],
                )
            v_total = v_data.item() + v_phys.item()
            if v_total < best_val_loss:
                best_val_loss = v_total
                patience_counter = 0
                torch.save({"model": ema_model.state_dict(),
                            "w_scale": train_ds.w_scale}, out_ckpt)
                print(f"   => [Val] Data {v_data.item():.4e} | Phys {v_phys.item():.4e} "
                      f"| Sum {v_total:.4e} (saved)")
            else:
                patience_counter += 1
                print(f"   => [Val] Sum {v_total:.4e} | best {best_val_loss:.4e} "
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

    print(f"Saved PBFM model -> {out_ckpt}")


if __name__ == "__main__":
    main()
