"""
Conditioned dataset for steady-state 3D Darcy flow.

Reads the h5 files written by datasets/generate_darcy3d_data.py:

    u  : (N, nx, ny, nz)   steady-state pressure fields
    k  : (N, nx, ny, nz)   log-normal permeability fields
    bc : (N, 2)            (p_left, p_right) Dirichlet boundary values

Each item is one (k, bc) -> p sample:

    x1            : (nx, ny, nz)  pressure field, normalized to [-1, 1]
                                   (min/max over the TRAIN split, passed into
                                   the test split so both share identical units)
    cond_k        : (1, nx, ny, nz)  log-normalized permeability field (model
                                      conditioning input; log because k is
                                      log-normal and spans ~20x)
    cond_bc       : (2,)  normalized (p_left, p_right), same affine map as x1
    cond_bc_phys  : (2,)  physical (p_left, p_right), for the PDE residual
    k_phys        : (nx, ny, nz)  physical permeability field, for the residual

h5py handles are opened lazily per-process (never in __init__): handles are
not fork-safe, and DataLoader workers fork after construction.
"""

import h5py
import torch
from torch.utils.data import Dataset


class Darcy3DConditionedDataset(Dataset):
    def __init__(self, path, u_min=None, u_max=None, logk_min=None, logk_max=None):
        self.path = str(path)
        self.file = None  # opened lazily per-process, see _ensure_open

        with h5py.File(self.path, "r") as fh:
            self.n_data, self.nx, self.ny, self.nz = fh["u"].shape
            if u_min is None or u_max is None or logk_min is None or logk_max is None:
                u_all = fh["u"][:]
                k_all = fh["k"][:]
                if u_min is None:
                    u_min = float(u_all.min())
                if u_max is None:
                    u_max = float(u_all.max())
                logk_all = torch.from_numpy(k_all).log()
                if logk_min is None:
                    logk_min = float(logk_all.min())
                if logk_max is None:
                    logk_max = float(logk_all.max())

        self.u_min, self.u_max = u_min, u_max
        self.logk_min, self.logk_max = logk_min, logk_max

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
            self.u = self.file["u"]
            self.k = self.file["k"]
            self.bc = self.file["bc"]

    def __getstate__(self):
        # Drop the h5py handle when pickling (spawn-based DataLoader workers
        # pickle the dataset); each worker reopens lazily via _ensure_open.
        state = self.__dict__.copy()
        state["file"] = None
        state.pop("u", None)
        state.pop("k", None)
        state.pop("bc", None)
        return state

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        self._ensure_open()
        u_phys = torch.from_numpy(self.u[index])   # (nx, ny, nz)
        k_phys = torch.from_numpy(self.k[index])   # (nx, ny, nz)
        p_left, p_right = [float(v) for v in self.bc[index]]

        x1 = 2.0 * (u_phys - self.u_min) / (self.u_max - self.u_min + 1e-8) - 1.0

        logk = torch.log(k_phys)
        cond_k = (2.0 * (logk - self.logk_min) / (self.logk_max - self.logk_min + 1e-8) - 1.0).unsqueeze(0)

        bc_phys = torch.tensor([p_left, p_right], dtype=torch.float32)
        bc_norm = 2.0 * (bc_phys - self.u_min) / (self.u_max - self.u_min + 1e-8) - 1.0

        return x1, cond_k, bc_norm, bc_phys, k_phys

    def denormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * (self.u_max - self.u_min) + self.u_min


def cycle(dl):
    """Infinite iterator for step-based training loops."""
    while True:
        for batch in dl:
            yield batch
