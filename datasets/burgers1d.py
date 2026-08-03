# Burgers dataset

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ._base import register_dataset

@register_dataset('burgers1d')
class Burgers1DDataset(Dataset):
    def __init__(self, root, split, data_file):
        self.root = root
        self.split = split
        self.path = os.path.join(root, data_file)
        self.file = None  # opened lazily per-process, see _ensure_open

        with h5py.File(self.path, 'r') as f:
            self.N_ic, self.N_bc, self.Nx, self.nt = f['u'].shape
        self.n_data = self.N_ic * self.N_bc

    def _ensure_open(self):
        # h5py file handles aren't fork-safe; opening here (instead of in
        # __init__) means each DataLoader worker process opens its own handle
        # after forking, rather than inheriting one opened by the main process.
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
            self.u = self.file['u']

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        self._ensure_open()
        i_ic, i_bc = divmod(index, self.N_bc)
        arr = self.u[i_ic, i_bc]
        return torch.from_numpy(arr.astype(np.float32))
