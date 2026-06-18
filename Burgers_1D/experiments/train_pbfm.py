#!/usr/bin/env python
"""
Train the FiLM-conditioned model with Physics-Based Flow Matching (PBFM).

Faithful to the pbfm_fno_paper_ICBC notebook:
  - AdamW, lr = 3e-5, weight_decay 0, betas (0.5, 0.999)
  - differentiable Euler unrolling with a curriculum that ramps n: 1 -> 4
  - logit-normal importance weighting on the FM loss
  - conflict-free gradient update (ConFIG) combining the separate FM and physics
    gradients, with a NaN guard that falls back to the FM-only gradient
  - EMA decay 0.999
  - validation every 1000 iters on (data_loss + phys_loss); early stopping with
    patience 20; best (lowest val sum) EMA weights saved.

    python experiments/train_pbfm.py
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
from src.physics import calc_physics_residual_pcfm
from src.losses import pbfm_loss

# src.physics has already put PCFM on the path; conflictfree is a normal pip dep.
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector, apply_gradient_vector

CONFIG = {
    "batch_size": 64,
    "num_iterations": 10000,
    "lr": 3e-5,
    "ema_decay": 0.999,
    "eval_every": 1000,
    "patience": 20,
    "max_unroll_steps": 4,
    "use_dignorm": True,
    "use_config": True,
}


def curriculum_n_steps(iteration, total, max_n):
    quarter = total // max_n
    n = (iteration - 1) // quarter + 1
    return min(n, max_n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="checkpoint output path")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(add_pcfm_to_path()) / "datasets" / "data"
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_pbfm.pt"
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

    vb_x1, vb_ic, vb_bc, vb_bcphys = next(iter(val_loader))
    vb_x1, vb_ic, vb_bc, vb_bcphys = (vb_x1.to(device), vb_ic.to(device),
                                      vb_bc.to(device), vb_bcphys.to(device))

    base_fno = PatchedFNO(n_modes=[32, 32], in_channels=3, emb_channels=32,
                          hidden_channels=64, proj_channels=256, n_layers=4).to(device)
    model = PCFM_FiLMConditionedFNO(base_fno, feature_channels=32).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for q in ema_model.parameters():
        q.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=0.0, betas=(0.5, 0.999))

    best_val_loss = float("inf")
    patience_counter = 0

    def ema_update(decay):
        with torch.no_grad():
            for ep, q in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(decay).add_(q.data, alpha=1 - decay)

    model.train()
    for iteration in range(1, CONFIG["num_iterations"] + 1):
        n_unroll = curriculum_n_steps(iteration, CONFIG["num_iterations"], CONFIG["max_unroll_steps"])

        x1, cond_ic, cond_bc, cond_bc_phys = next(data_iter)
        x1, cond_ic, cond_bc, cond_bc_phys = (x1.to(device), cond_ic.to(device),
                                              cond_bc.to(device), cond_bc_phys.to(device))

        data_loss, phys_loss = pbfm_loss(
            model, x1, cond_ic, cond_bc, cond_bc_phys,
            calc_physics_residual_pcfm, n_steps=n_unroll,
            use_dignorm=CONFIG["use_dignorm"], U_MIN=U_MIN, U_MAX=U_MAX, device=device,
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

        if iteration % 100 == 0 or iteration == 1:
            print(f"Iter {iteration:05d} | n_unroll={n_unroll} | "
                  f"Data {data_loss.item():.4e} | Phys {phys_loss.item():.4e}")

        if iteration % CONFIG["eval_every"] == 0:
            ema_model.eval()
            with torch.no_grad():
                v_data, v_phys = pbfm_loss(
                    ema_model, vb_x1, vb_ic, vb_bc, vb_bcphys,
                    calc_physics_residual_pcfm, n_steps=n_unroll,
                    use_dignorm=CONFIG["use_dignorm"], U_MIN=U_MIN, U_MAX=U_MAX, device=device,
                )
            v_total = v_data.item() + v_phys.item()
            if v_total < best_val_loss:
                best_val_loss = v_total
                patience_counter = 0
                torch.save(ema_model.state_dict(), out_ckpt)
                print(f"   => [Val] Data {v_data.item():.4e} | Phys {v_phys.item():.4e} "
                      f"| Sum {v_total:.4e} (saved)")
            else:
                patience_counter += 1
                print(f"   => [Val] Sum {v_total:.4e} | best {best_val_loss:.4e} "
                      f"| patience {patience_counter}/{CONFIG['patience']}")
                if patience_counter >= CONFIG["patience"]:
                    print(f"[!] Early stopping at iteration {iteration}.")
                    break
            model.train()

    print(f"Saved PBFM model -> {out_ckpt}")


if __name__ == "__main__":
    main()
