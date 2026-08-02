# """
# ns2d_conditioned.py -- Training script for the FiLM-IC-conditioned NS model
# -----------------------------------------------------------------------------
# Trains NSVelocityNet_FiLM_IC (from ns_model_film_ic.py) via conditional flow
# matching on the 2D Navier-Stokes dataset (vorticity trajectories + IC).

# Usage:
#     python ns2d_conditioned.py --data datasets/data/ns_nw10_nf100_s64_t50_mu0.001.h5 \
#         --steps 10 --batch_size 4

# Run with --help to see all options. This script is verbose by design: every
# stage (data load, model build, each logged step) prints to stdout with
# flush=True, so partial progress is visible even if the run is killed early
# or stdout is being piped/redirected.
# """

# import argparse
# import importlib.util
# import os
# import sys
# import time

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.insert(0, project_root)

# # ──────────────────────────────────────────────────────────────────────────
# # Dataset
# # ──────────────────────────────────────────────────────────────────────────

# class NSDataset:
#     """
#     Loads NS vorticity trajectories from an HDF5 file.

#     This class is defensive about key names: on first load it prints every
#     top-level key and dataset shape found in the .h5 file, so a key-name
#     mismatch is immediately visible rather than failing silently or with an
#     opaque KeyError deep in __getitem__.

#     Expected (but not assumed) layout, based on the PCFM paper's NS dataset
#     convention:
#         trajectories : (N, n_t, H, W)  -- vorticity field at n_t timesteps
#         ic           : (N, H, W)       -- initial vorticity (t=0 slice),
#                                            OR simply trajectories[:, 0]
#     If a separate `ic` key is not found, the IC is taken as the first
#     timestep of the trajectory.
#     """

#     def __init__(self, h5_path: str, traj_key: str = None, ic_key: str = None):
#         import h5py

#         self.h5_path = h5_path

#         if not os.path.exists(h5_path):
#             raise FileNotFoundError(f"Dataset file not found: {h5_path}")

#         with h5py.File(h5_path, "r") as f:
#             print(f"[NSDataset] Opened {h5_path}", flush=True)
#             print(f"[NSDataset] Top-level keys: {list(f.keys())}", flush=True)
#             for k in f.keys():
#                 try:
#                     print(f"[NSDataset]   '{k}': shape={f[k].shape}, dtype={f[k].dtype}", flush=True)
#                 except AttributeError:
#                     print(f"[NSDataset]   '{k}': (group, not a dataset)", flush=True)

#             # Resolve trajectory key
#             if traj_key is None:
#                 for candidate in ["trajectories", "vorticity", "u", "data", "w"]:
#                     if candidate in f.keys():
#                         traj_key = candidate
#                         break
#             if traj_key is None or traj_key not in f.keys():
#                 raise KeyError(
#                     f"Could not find trajectory data in {h5_path}. "
#                     f"Available keys: {list(f.keys())}. "
#                     f"Pass --traj_key explicitly to specify the correct key."
#                 )
#             self.traj_key = traj_key
#             self.n_samples = f[traj_key].shape[0]

#             # Resolve IC key (optional -- fall back to trajectories[:, 0])
#             self.ic_key = ic_key
#             if self.ic_key is None:
#                 for candidate in ["ic", "a", "initial_condition", "w0"]:
#                     if candidate in f.keys():
#                         self.ic_key = candidate
#                         break

#             print(f"[NSDataset] Using traj_key='{self.traj_key}', "
#                   f"ic_key={'(derived from traj[:,0])' if self.ic_key is None else repr(self.ic_key)}",
#                   flush=True)
#             print(f"[NSDataset] Loaded {self.n_samples} samples.", flush=True)

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         import h5py
#         import numpy as np
#         import torch

#         with h5py.File(self.h5_path, "r") as f:
#             traj = f[self.traj_key][idx]            # (n_t, H, W)
#             if self.ic_key is not None:
#                 ic = f[self.ic_key][idx]             # (H, W)
#             else:
#                 ic = traj[0]                          # (H, W), derived

#         traj = torch.from_numpy(np.asarray(traj)).float()        # (n_t, H, W)
#         ic = torch.from_numpy(np.asarray(ic)).float().unsqueeze(0)  # (1, H, W)
#         return traj, ic


# # ──────────────────────────────────────────────────────────────────────────
# # Training loop
# # ──────────────────────────────────────────────────────────────────────────

# def train(args):
#     import torch
#     import torch.nn.functional as F
#     from torch.utils.data import DataLoader

#     model_path = os.path.join(project_root, "models", "ns_model_film_ic.py")
#     model_spec = importlib.util.spec_from_file_location("ns_model_film_ic", model_path)
#     model_module = importlib.util.module_from_spec(model_spec)
#     assert model_spec is not None and model_spec.loader is not None
#     model_spec.loader.exec_module(model_module)
#     build_model_film_ic = model_module.build_model_film_ic

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() and not args.cpu
#         else "mps" if torch.backends.mps.is_available() and not args.cpu
#         else "cpu"
#     )
#     print(f"[train] Using device: {device}", flush=True)

#     # ── Data ──
#     dataset = NSDataset(args.data, traj_key=args.traj_key, ic_key=args.ic_key)
#     if len(dataset) == 0:
#         print("[train] ERROR: dataset has 0 samples. Aborting.", flush=True)
#         sys.exit(1)

#     dataloader = DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers, drop_last=True,
#     )
#     print(f"[train] DataLoader ready: {len(dataset)} samples, "
#           f"batch_size={args.batch_size}, {len(dataloader)} batches/epoch", flush=True)

#     # Peek one batch to confirm shapes before training starts
#     sample_traj, sample_ic = next(iter(dataloader))
#     print(f"[train] Sample batch shapes: traj={tuple(sample_traj.shape)}, "
#           f"ic={tuple(sample_ic.shape)}", flush=True)
#     n_t = sample_traj.shape[1]
#     H, W = sample_traj.shape[2], sample_traj.shape[3]
#     if H != 64 or W != 64:
#         print(f"[train] WARNING: expected 64x64 grid, got {H}x{W}. "
#               f"Model's spectral conv assumes 64x64 -- verify compatibility.", flush=True)

#     # ── Model ──
#     model = build_model_film_ic(
#         n_t=n_t, modes=args.modes, width=args.width,
#         n_layers=args.n_layers, ic_emb_dim=args.ic_emb_dim, device=device,
#     )
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=args.lr_patience,
#     )

#     os.makedirs(args.out_dir, exist_ok=True)
#     log_path = os.path.join(args.out_dir, "train_log.csv")
#     with open(log_path, "w") as f:
#         f.write("step,loss,lr,elapsed_sec\n")

#     print(f"[train] Starting training for {args.steps} steps. "
#           f"Logging to {log_path}", flush=True)

#     model.train()
#     step = 0
#     t_start = time.time()
#     data_iter = iter(dataloader)

#     while step < args.steps:
#         try:
#             u1, a = next(data_iter)
#         except StopIteration:
#             data_iter = iter(dataloader)
#             u1, a = next(data_iter)

#         u1 = u1.to(device)          # (B, n_t, H, W) -- ground-truth trajectory
#         a = a.to(device)            # (B, 1, H, W)   -- initial condition

#         u0 = torch.randn_like(u1)   # base distribution sample
#         t = torch.rand(u1.shape[0], device=device)

#         t_broadcast = t[:, None, None, None]
#         u_t = (1 - t_broadcast) * u0 + t_broadcast * u1
#         target_v = u1 - u0

#         pred_v = model(u_t, a, t)
#         loss = F.mse_loss(pred_v, target_v)

#         optimizer.zero_grad()
#         loss.backward()

#         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
#         optimizer.step()
#         scheduler.step(loss)

#         elapsed = time.time() - t_start
#         lr_now = optimizer.param_groups[0]["lr"]

#         if step % args.log_every == 0 or step == args.steps - 1:
#             print(f"[step {step:5d}/{args.steps}] loss={loss.item():.6f}  "
#                   f"grad_norm={grad_norm:.4f}  lr={lr_now:.2e}  "
#                   f"elapsed={elapsed:.1f}s", flush=True)
#             with open(log_path, "a") as f:
#                 f.write(f"{step},{loss.item():.6f},{lr_now:.6e},{elapsed:.2f}\n")

#         if (step + 1) % args.ckpt_every == 0 or step == args.steps - 1:
#             ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step+1}.pt")
#             torch.save({
#                 "step": step + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "loss": loss.item(),
#                 "args": vars(args),
#             }, ckpt_path)
#             print(f"[train] Saved checkpoint: {ckpt_path}", flush=True)

#         step += 1

#     print(f"[train] Done. Total time: {time.time() - t_start:.1f}s", flush=True)


# # ──────────────────────────────────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────────────────────────────────

# def build_argparser():
#     p = argparse.ArgumentParser(
#         description="Train the FiLM-IC-conditioned NS flow-matching model."
#     )
#     # Data
#     p.add_argument("--data", type=str, required=True,
#                     help="Path to the .h5 NS dataset file.")
#     p.add_argument("--traj_key", type=str, default=None,
#                     help="HDF5 key for trajectory data (auto-detected if omitted).")
#     p.add_argument("--ic_key", type=str, default=None,
#                     help="HDF5 key for IC data (derived from traj[:,0] if omitted).")
#     p.add_argument("--num_workers", type=int, default=0,
#                     help="DataLoader worker processes (0 = main process only).")

#     # Model
#     p.add_argument("--modes", type=int, default=12)
#     p.add_argument("--width", type=int, default=48)
#     p.add_argument("--n_layers", type=int, default=4)
#     p.add_argument("--ic_emb_dim", type=int, default=64)

#     # Training
#     p.add_argument("--steps", type=int, default=10,
#                     help="Number of training steps (not epochs).")
#     p.add_argument("--batch_size", type=int, default=4)
#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--lr_patience", type=int, default=10)
#     p.add_argument("--grad_clip", type=float, default=1.0)
#     p.add_argument("--cpu", action="store_true",
#                     help="Force CPU even if CUDA/MPS is available.")

#     # Logging / checkpointing
#     p.add_argument("--log_every", type=int, default=1,
#                     help="Print/log every N steps.")
#     p.add_argument("--ckpt_every", type=int, default=50,
#                     help="Save a checkpoint every N steps.")
#     p.add_argument("--out_dir", type=str, default="runs/ns_film_ic",
#                     help="Directory for logs and checkpoints.")
#     return p


# if __name__ == "__main__":
#     parser = build_argparser()
#     args = parser.parse_args()
#     print(f"[main] Parsed args: {vars(args)}", flush=True)
#     train(args)


import h5py
with h5py.File("datasets/data/ns_nw10_nf100_s64_t50_mu0.001.h5", "r") as f:
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)
    # also check for any attrs that explain the dims
    for k in f.keys():
        print(k, "attrs:", dict(f[k].attrs))