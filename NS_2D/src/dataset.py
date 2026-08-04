"""
Conditioned 2D Navier-Stokes dataset.

Reads the h5 files written by datasets/generate_ns_2d.py:

    a : (nw, s, s)          initial vorticity fields
    f : (nf, s, s)          forcing fields
    u : (nw, nf, s, s, t)   vorticity trajectories, one per (IC, forcing) pair

Each item is one (IC, forcing) pair:

    x1     : (T, H, W)  trajectory, normalized to [-1, 1] (symmetric max-abs)
    cond_a : (1, H, W)  IC vorticity field, same normalization as x1
    cond_f : (1, H, W)  forcing field, PHYSICAL units (the physics residual
                        needs f unscaled; the model's forcing encoder can
                        absorb the scale)

Normalization: w_scale = max |u| over the TRAIN split, passed into the test
split so both use identical units (mirrors Burgers' u_min/u_max handoff).

h5py handles are opened lazily per-process (never in __init__): handles are
not fork-safe, and DataLoader workers fork after construction.
"""

import h5py
import torch
from torch.utils.data import Dataset


class NSConditionedDataset(Dataset):
    def __init__(self, path, w_scale=None):
        self.path = str(path)
        self.file = None  # opened lazily per-process, see _ensure_open

        with h5py.File(self.path, "r") as fh:
            self.nw, self.nf = fh["u"].shape[:2]
            self.n_t = fh["u"].shape[-1]
            if w_scale is None:
                # chunked max-abs scan (train file can be multi-GB)
                m = 0.0
                for i in range(self.nw):
                    m = max(m, abs(fh["u"][i]).max())
                w_scale = float(m)
        self.w_scale = w_scale

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
            self.u = self.file["u"]
            self.a = self.file["a"]
            self.f = self.file["f"]

    def __getstate__(self):
        # Drop the h5py handle when pickling (spawn-based DataLoader workers
        # pickle the dataset); each worker reopens lazily via _ensure_open.
        state = self.__dict__.copy()
        state["file"] = None
        state.pop("u", None)
        state.pop("a", None)
        state.pop("f", None)
        return state

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __len__(self):
        return self.nw * self.nf

    def __getitem__(self, index):
        self._ensure_open()
        i, j = divmod(index, self.nf)
        w = torch.from_numpy(self.u[i, j]).permute(2, 0, 1)  # (T, H, W)
        a = torch.from_numpy(self.a[i][()]).unsqueeze(0)     # (1, H, W)
        f = torch.from_numpy(self.f[j][()]).unsqueeze(0)     # (1, H, W)
        return w / self.w_scale, a / self.w_scale, f

    def denormalize(self, w_norm):
        return w_norm * self.w_scale


def cycle(loader):
    while True:
        for batch in loader:
            yield batch
