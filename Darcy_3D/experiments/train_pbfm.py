#!/usr/bin/env python
"""
Train the FiLM-(permeability, BC)-conditioned model with Physics-Based Flow
Matching (PBFM), using the steady-state 3D Darcy finite-volume flux-balance
residual (pcfm.Residuals3D) as the physics term. Same recipe as
Burgers_1D/NS_2D's train_pbfm.py, minus any physical-time unroll -- Darcy has
no time axis, so "unrolling" only ever means Euler-integrating the flow-
matching ODE (in interpolation time t) from the sampled t to t=1; the model
already outputs the whole steady-state field in one shot.

  - AdamW, lr 3e-5, weight_decay 0, betas (0.5, 0.999)
  - differentiable Euler unrolling of the FM ODE with a curriculum n: 1 -> 4
  - logit-normal importance weighting on the FM loss
  - conflict-free gradient update (ConFIG) combining the separate FM and
    physics gradients, with a NaN guard falling back to the FM-only gradient
  - EMA decay 0.999
  - validation every 1000 iters; best-checkpoint selection and early-stopping
    patience computed from the DATA loss ALONE, never Data+Phys: an
    undertrained model with near-zero output amplitude trivially minimizes
    the physics residual (little dynamics = little to violate), so an
    additive Sum criterion rewards that degenerate solution over real
    data-fitting progress (see feedback_pbfm_checkpoint_selection memory,
    found and fixed in NS_2D/experiments/train_pbfm.py) -- applied here from
    the start rather than repeating that bug.

plus a resumable `latest` checkpoint at every validation (--resume restarts).

Kaggle usage:

    pip install conflictfree
    python experiments/train_pbfm.py \
        --data-train /kaggle/input/.../darcy3d_train_n5000.h5 \
        --data-test  /kaggle/input/.../darcy3d_test_n500.h5 \
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
from src.dataset import Darcy3DConditionedDataset, cycle
from src.models import PatchedFNO3D, PCFM_FiLMConditionedFNO3D
from src.physics import calc_physics_residual_pcfm
from src.losses import pbfm_loss

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector, apply_gradient_vector

CONFIG = {
    "batch_size": 32,
    "num_iterations": 10000,
    # Curriculum ramp length (1 -> max_unroll_steps), independent of
    # num_iterations/--iters -- see NS_2D/experiments/train_pbfm.py for the
    # full rationale (a run with --iters 60000 there early-stopped stuck at
    # n_unroll=2, with real quality as bad as an undertrained n_unroll=1
    # checkpoint, because patience exhausted before the curriculum -- tied
    # to num_iterations at the time -- ever reached the productive stages).
    "curriculum_iters": 10000,
    "lr": 3e-5,
    "ema_decay": 0.999,
    "eval_every": 1000,
    "patience": 20,
    "max_unroll_steps": 4,
    "use_dignorm": True,
    "use_config": True,
}


def curriculum_n_steps(iteration, total, max_n):
    quarter = max(1, total // max_n)
    n = (iteration - 1) // quarter + 1
    return min(n, max_n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train",
                   default=str(REPO_ROOT / "datasets" / "data" / "darcy3d_train_n5000.h5"))
    p.add_argument("--data-test",
                   default=str(REPO_ROOT / "datasets" / "data" / "darcy3d_test_n500.h5"))
    p.add_argument("--out", default=None, help="best-checkpoint output path")
    p.add_argument("--resume", default=None,
                   help="resume from a <out>.latest.pt written by a previous run")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--iters", type=int, default=None,
                   help="overall training-budget cap; does NOT affect curriculum ramp "
                        "speed, see --curriculum-iters")
    p.add_argument("--curriculum-iters", type=int, default=None,
                   help="iterations over which n_unroll ramps 1 -> max_unroll_steps, "
                        "independent of --iters (default 10000)")
    args = p.parse_args()

    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.curriculum_iters:
        CONFIG["curriculum_iters"] = args.curriculum_iters
    if args.iters:
        CONFIG["num_iterations"] = args.iters

    repo_root = Path(__file__).resolve().parents[1]
    out_ckpt = Path(args.out) if args.out else repo_root / "weights" / "best_pbfm.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_ckpt.with_suffix(".latest.pt")

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

    vb_x1, vb_k, vb_bc, vb_bcphys, vb_kphys = next(iter(val_loader))
    vb_x1, vb_k, vb_bc = vb_x1.to(device), vb_k.to(device), vb_bc.to(device)
    vb_bcphys, vb_kphys = vb_bcphys.to(device), vb_kphys.to(device)

    base_fno = PatchedFNO3D(n_modes=[8, 8, 8], in_channels=4, emb_channels=32,
                            hidden_channels=32, proj_channels=128, n_layers=4).to(device)
    model = PCFM_FiLMConditionedFNO3D(base_fno, feature_channels=32).to(device)
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
        n_unroll = curriculum_n_steps(iteration, CONFIG["curriculum_iters"],
                                      CONFIG["max_unroll_steps"])

        x1, cond_k, cond_bc, cond_bc_phys, k_phys = next(data_iter)
        x1, cond_k, cond_bc = x1.to(device), cond_k.to(device), cond_bc.to(device)
        cond_bc_phys, k_phys = cond_bc_phys.to(device), k_phys.to(device)

        data_loss, phys_loss = pbfm_loss(
            model, x1, cond_k, cond_bc, cond_bc_phys, k_phys,
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

        if iteration % 100 == 0 or iteration == start_iter:
            print(f"Iter {iteration:05d} | n_unroll={n_unroll} | "
                  f"Data {data_loss.item():.4e} | Phys {phys_loss.item():.4e}")

        if iteration % CONFIG["eval_every"] == 0:
            ema_model.eval()
            with torch.no_grad():
                # Always validate at the full unroll length, not the current
                # curriculum n_unroll -- otherwise the val loss gets harder
                # over the course of training (curriculum ramps 1 -> 4) and
                # "best" checkpoint selection / early stopping spuriously
                # favors early, undertrained checkpoints evaluated under the
                # easy n_unroll=1 regime.
                v_data, v_phys = pbfm_loss(
                    ema_model, vb_x1, vb_k, vb_bc, vb_bcphys, vb_kphys,
                    calc_physics_residual_pcfm, n_steps=CONFIG["max_unroll_steps"],
                    use_dignorm=CONFIG["use_dignorm"], U_MIN=U_MIN, U_MAX=U_MAX, device=device,
                )
            v_total = v_data.item()
            # Select/early-stop on Data loss alone, not Data+Phys -- see the
            # module docstring and feedback_pbfm_checkpoint_selection memory.
            if v_total < best_val_loss:
                best_val_loss = v_total
                patience_counter = 0
                torch.save({"model": ema_model.state_dict(),
                            "u_min": U_MIN, "u_max": U_MAX,
                            "logk_min": LOGK_MIN, "logk_max": LOGK_MAX}, out_ckpt)
                print(f"   => [Val] Data {v_data.item():.4e} | Phys {v_phys.item():.4e} "
                      f"(saved, best Data)")
            else:
                patience_counter += 1
                print(f"   => [Val] Data {v_data.item():.4e} | Phys {v_phys.item():.4e} "
                      f"| best Data {best_val_loss:.4e} | patience {patience_counter}/{CONFIG['patience']}")

            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "u_min": U_MIN, "u_max": U_MAX,
                "logk_min": LOGK_MIN, "logk_max": LOGK_MAX,
            }, latest_ckpt)

            if patience_counter >= CONFIG["patience"]:
                print(f"[!] Early stopping at iteration {iteration}.")
                break
            model.train()

    print(f"Saved PBFM model -> {out_ckpt}")


if __name__ == "__main__":
    main()
