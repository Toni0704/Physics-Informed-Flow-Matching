# script to solve the steady-state 3D Darcy flow equation numerically to construct the
# PDE solution datasets for training and sampling.
#
# Governing equation (cell-centered finite-volume discretization):
#   -div(K grad p) = 0   on the unit cube [0,1]^3
# Dirichlet BC: p = p_left on the x=0 face, p = p_right on the x=1 face.
# No-flow (Neumann) BC on all other faces.
# K is a log-normal random field (exp of a GaussianRF sample), representing
# heterogeneous permeability.

import argparse
import multiprocessing as mp
import os

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from datasets.random_fields import GaussianRF

N = 16  # grid cells per axis
h = 1.0 / N
xc = np.linspace(0, 1, N, endpoint=False) + h * 0.5


def harmonic(k1, k2):
    return 2 * k1 * k2 / (k1 + k2)


def solve_single(K, p_left, p_right):
    """
    K: permeability field, shape (N, N, N)
    Returns pressure field p, shape (N, N, N), solving the cell-centered FV
    discretization of -div(K grad p) = 0 with Dirichlet BC on the x=0/x=1
    faces and no-flow elsewhere.
    """
    idx = np.arange(N ** 3).reshape(N, N, N)
    diag = np.zeros(N ** 3)
    b = np.zeros(N ** 3)
    rows, cols, vals = [], [], []

    def add_face(left_idx, right_idx, T):
        rows.append(left_idx); cols.append(right_idx); vals.append(-T)
        rows.append(right_idx); cols.append(left_idx); vals.append(-T)
        np.add.at(diag, left_idx, T)
        np.add.at(diag, right_idx, T)

    Tx = (harmonic(K[:-1, :, :], K[1:, :, :]) * h).ravel()
    add_face(idx[:-1, :, :].ravel(), idx[1:, :, :].ravel(), Tx)

    Ty = (harmonic(K[:, :-1, :], K[:, 1:, :]) * h).ravel()
    add_face(idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel(), Ty)

    Tz = (harmonic(K[:, :, :-1], K[:, :, 1:]) * h).ravel()
    add_face(idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel(), Tz)

    # Dirichlet boundary faces (half-cell distance h/2 to the face)
    Tw = (2 * K[0, :, :] * h).ravel()
    west_idx = idx[0, :, :].ravel()
    np.add.at(diag, west_idx, Tw)
    np.add.at(b, west_idx, Tw * p_left)

    Te = (2 * K[-1, :, :] * h).ravel()
    east_idx = idx[-1, :, :].ravel()
    np.add.at(diag, east_idx, Te)
    np.add.at(b, east_idx, Te * p_right)

    rows.append(np.arange(N ** 3)); cols.append(np.arange(N ** 3)); vals.append(diag)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N ** 3, N ** 3)).tocsr()
    p = spla.spsolve(A, b)
    return p.reshape(N, N, N)


def generate_permeability(alpha=2.5, tau=5, log_range=(-1.5, 1.5)):
    """
    Samples a log-normal permeability field via a 3D GaussianRF, then rescales
    the log-field into log_range to control the permeability contrast ratio.
    """
    grf = GaussianRF(3, N, alpha=alpha, tau=tau)
    logK = grf.sample(1)[0]
    logK = logK - logK.mean()
    span = logK.abs().max().clamp_min(1e-6)
    lo, hi = log_range
    logK = logK / span * max(abs(lo), abs(hi))
    K = torch.exp(logK).numpy().astype(np.float64)
    return K


def worker(args):
    i, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    K = generate_permeability()
    p_left = float(np.round(np.random.uniform(0.5, 1.5), 3))
    p_right = float(np.round(np.random.uniform(-1.5, -0.5), 3))
    p = solve_single(K, p_left, p_right)
    print(f"Finished: sample #{i}, seed={seed}, p_left={p_left:.3f}, p_right={p_right:.3f}")
    return i, K.astype(np.float32), p.astype(np.float32), p_left, p_right


def run_parallel(root, n_data=500, nproc=8, seed=42, filename="darcy3d_train"):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{filename}_n{n_data}.h5')

    tasks = [(i, seed + i) for i in range(n_data)]
    with h5py.File(path, 'w') as f:
        f.create_dataset('u', shape=(n_data, N, N, N), dtype=np.float32)
        f.create_dataset('k', shape=(n_data, N, N, N), dtype=np.float32)
        f.create_dataset('bc', shape=(n_data, 2), dtype=np.float32)
        f.create_dataset('x', data=xc, dtype=np.float32)
        f.create_dataset('y', data=xc, dtype=np.float32)
        f.create_dataset('z', data=xc, dtype=np.float32)
        f.attrs['h'] = h
        f.attrs['N'] = N

        with mp.Pool(processes=nproc) as pool:
            for i, K, p, p_left, p_right in pool.imap_unordered(worker, tasks):
                f['u'][i] = p
                f['k'][i] = K
                f['bc'][i] = [p_left, p_right]
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/data')
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    run_parallel(root=args.root, n_data=5000, nproc=args.nproc, seed=42, filename="darcy3d_train")
    run_parallel(root=args.root, n_data=500, nproc=args.nproc, seed=10_000, filename="darcy3d_test")
