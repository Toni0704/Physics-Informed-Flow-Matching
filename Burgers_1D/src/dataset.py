"""
Dataset loader for the 1D Burgers' data produced by the PCFM generator.

The HDF5 file holds:
  u  : (n_ic, n_bc, nx, nt)  PDE solutions
  ic : (n_ic,)               initial-condition parameter (front location p_loc)
  bc : (n_bc,)               left Dirichlet boundary value u_bc

`BurgersConditionedDataset` flattens the (IC x BC) grid into a linear, IC-major
sequence (sample k = i_ic * n_bc + i_bc), normalises the field to [-1, 1] for the
flow-matching model, and exposes BOTH the normalised scalar BC (for the network)
and the physical scalar BC (for the exact PDE residual evaluator).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BurgersConditionedDataset(Dataset):
    def __init__(self, path, u_min=None, u_max=None):
        with h5py.File(path, "r") as f:
            u_raw = f["u"][:]     # (n_ic, n_bc, nx, nt)
            bc_arr = f["bc"][:]   # (n_bc,)

        n_ic, n_bc, nx, nt = u_raw.shape

        # Flatten in IC-major order: sample k = i_ic * n_bc + i_bc
        self.u_native = torch.from_numpy(
            u_raw.reshape(n_ic * n_bc, nx, nt).astype(np.float32)
        )
        # Tile BCs to match the flattened (IC-major) layout
        bc_per_sample = torch.from_numpy(np.tile(bc_arr, n_ic).astype(np.float32))

        # Dataset-wide bounds (passed from the train split into the test split so
        # both share the same normalisation).
        if u_min is None:
            u_min = self.u_native.min()
        if u_max is None:
            u_max = self.u_native.max()
        self.u_min, self.u_max = u_min, u_max

        # Normalise the field to [-1, 1]
        self.data = 2.0 * (self.u_native - u_min) / (u_max - u_min + 1e-8) - 1.0

        # Initial condition slice u(x, t=0), already normalised
        self.cond_ic = self.data[:, :, 0]                       # [N, nx]

        # Normalised scalar BC for the network ...
        bc_norm = 2.0 * (bc_per_sample - u_min) / (u_max - u_min + 1e-8) - 1.0
        self.cond_bc = bc_norm.unsqueeze(-1)                    # [N, 1]
        # ... and the physical scalar BC for the PDE residual
        self.cond_bc_phys = bc_per_sample.unsqueeze(-1)         # [N, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.cond_ic[idx],
                self.cond_bc[idx], self.cond_bc_phys[idx])


def cycle(dl):
    """Infinite iterator for step-based training loops."""
    while True:
        for batch in dl:
            yield batch
