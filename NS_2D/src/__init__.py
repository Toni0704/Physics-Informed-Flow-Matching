"""
Core library for the 2D Navier-Stokes flow-matching experiments.

Unlike Burgers_1D (which vendors PCFM), NS_2D lives INSIDE the PCFM repo, so
`add_repo_to_path()` simply puts the repo root on sys.path so that the top-level
`pcfm`, `models`, and `datasets` packages import correctly no matter where the
experiment scripts are launched from.
"""

import sys
from pathlib import Path

__all__ = ["add_repo_to_path", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def add_repo_to_path():
    """Prepend the PCFM repo root to sys.path. Returns the resolved path."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
