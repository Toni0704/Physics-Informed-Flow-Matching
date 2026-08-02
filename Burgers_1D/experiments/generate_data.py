#!/usr/bin/env python
"""
Regenerate the exact 1D Burgers' datasets used in these experiments.

The datasets themselves are NOT committed; this script reproduces them from the
PCFM repo's numerical Godunov solver. It loads ``solve_burgers`` from the vendored
``datasets/generate_burgers1d_data.py`` (without executing that file's trailing
module-level calls) and fills the (IC x BC) grid SINGLE-PROCESS.

Why single-process: the upstream generator uses ``multiprocessing.Pool``, which
breaks on Windows (spawn start method cannot pickle a function loaded via exec).
Running the deterministic solver in a plain loop produces a bit-identical dataset
on every platform. The arrays, sampling, and seeds match the upstream generator:

    train : N_ic = 80, N_bc = 80, seed = 42  -> burgers_train_nIC80_nBC80.h5
    test  : N_ic = 30, N_bc = 30, seed = 0   -> burgers_test_nIC30_nBC30.h5

Output goes to <PCFM>/datasets/data/ (the canonical location every other script
reads from, and where PCFM's own config expects it).

Run from anywhere:
    python experiments/generate_data.py
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import h5py

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional; fall back to a no-op
    def tqdm(x=None, **kwargs):
        return x if x is not None else None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable
from src import add_pcfm_to_path


def load_solver(pcfm_dir):
    """Exec the generator module's defs only (drop top-level call statements),
    and return its ``solve_burgers`` function (pure NumPy, deterministic)."""
    src_path = Path(pcfm_dir) / "datasets" / "generate_burgers1d_data.py"
    source = src_path.read_text()
    tree = ast.parse(source)
    # Keep imports + function/class defs; drop bare expression statements
    # (the trailing generate_*(...) calls are ast.Expr nodes).
    tree.body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    ns = {}
    exec(compile(tree, str(src_path), "exec"), ns)  # noqa: S102 (trusted vendored code)
    return ns["solve_burgers"]


def generate_split(solve_burgers, out_dir, N_ic, N_bc, seed, filename,
                   Nx=100, Nt=100, T=1.0):
    """Single-process re-implementation of the upstream generate_burgers_dataset.

    Identical RNG (one seed before sampling), identical parameter draws, identical
    HDF5 layout -> identical output, just without multiprocessing.
    """
    np.random.seed(seed)
    p_locs = np.random.uniform(0.2, 0.8, N_ic)   # IC front locations
    u_bcs = np.random.uniform(0.0, 1.0, N_bc)    # left Dirichlet values
    x = np.linspace(0, 1.0, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    full_path = out_dir / f"{filename}_nIC{N_ic}_nBC{N_bc}.h5"
    with h5py.File(full_path, "w") as f:
        u_ds = f.create_dataset("u", shape=(N_ic, N_bc, Nx + 1, Nt + 1), dtype=np.float32)
        f.create_dataset("ic", data=p_locs.astype(np.float32))
        f.create_dataset("bc", data=u_bcs.astype(np.float32))
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("t", data=t.astype(np.float32))

        pbar = tqdm(total=N_ic * N_bc, desc=filename)
        for i in range(N_ic):
            for j in range(N_bc):
                u_ds[i, j] = solve_burgers(p_locs[i], u_bcs[j], Nx=Nx, Nt=Nt).astype(np.float32)
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
    return full_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, default=1,
                   help="accepted for compatibility; generation is single-process "
                        "for cross-platform determinism")
    p.add_argument("--train-seed", type=int, default=42,
                   help="RNG seed for the training split (generator default: 42)")
    p.add_argument("--test-seed", type=int, default=0,
                   help="RNG seed for the test split (generator default: 0)")
    args = p.parse_args()

    pcfm_dir = add_pcfm_to_path()
    out_dir = Path(pcfm_dir) / "datasets" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    solve_burgers = load_solver(pcfm_dir)

    print(f"[generate] training set (80x80, seed={args.train_seed}) -> {out_dir}")
    generate_split(solve_burgers, out_dir, N_ic=80, N_bc=80,
                   seed=args.train_seed, filename="burgers_train")

    print(f"[generate] test set (30x30, seed={args.test_seed}) -> {out_dir}")
    generate_split(solve_burgers, out_dir, N_ic=30, N_bc=30,
                   seed=args.test_seed, filename="burgers_test")

    print("[generate] done:",
          out_dir / "burgers_train_nIC80_nBC80.h5", "and",
          out_dir / "burgers_test_nIC30_nBC30.h5")


if __name__ == "__main__":
    main()
