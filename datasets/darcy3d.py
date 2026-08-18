# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# Steady-state 3D Darcy flow dataset

import os

import h5py
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('darcy3d')
class Darcy3DDataset(Dataset):
    """
    Yields steady-state pressure fields p(x, y, z), shape (nx, ny, nz), from
    the 3D Darcy flow FV solves in generate_darcy3d_data.py. The flow-matching
    model is trained unconditionally on this marginal distribution of pressure
    fields (across many random permeability fields/BC values); the matching
    permeability field ('k') and Dirichlet BC values ('bc') needed to build a
    Residuals3D constraint at PCFM sampling time are read directly from the
    h5 file (see scripts/training/run_pcfm_hpc.py-style eval scripts), not
    through this Dataset.
    """
    def __init__(self, root, split, data_file):
        self.root = root
        self.split = split
        self.data_file = data_file
        self.path = os.path.join(root, data_file)
        self.file = None  # opened lazily per-process, see _ensure_open

        with h5py.File(self.path, 'r') as f:
            self.n_data, self.nx, self.ny, self.nz = f['u'].shape

    def _ensure_open(self):
        # h5py file handles aren't fork-safe; opening here (instead of in
        # __init__) means each DataLoader worker process opens its own handle
        # after forking, rather than inheriting one opened by the main process.
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
            self.data = self.file['u']

    def __getstate__(self):
        # Drop the h5py handle when pickling (spawn-based DataLoader workers
        # pickle the dataset); each worker reopens lazily via _ensure_open.
        state = self.__dict__.copy()
        state['file'] = None
        state.pop('data', None)
        return state

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        self._ensure_open()
        return torch.from_numpy(self.data[index])
