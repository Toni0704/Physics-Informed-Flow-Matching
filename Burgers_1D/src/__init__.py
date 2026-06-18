"""
Core library for the 1D Burgers' equation flow-matching experiments.

Contains the shared dataset loader, model architecture (FiLM-conditioned FNO),
physics residual evaluator, and loss functions used by every experiment script.

It also exposes `add_pcfm_to_path()`, which locates the (separate, public) PCFM
repository and puts it on `sys.path` so that the top-level `pcfm`, `models`, and
`scripts.training.utils` modules import correctly. PCFM is NOT a pip-installable
package (it has no setup.py / pyproject.toml); it is used by adding its root to
the path, exactly as the original notebooks did with `sys.path.append('./PCFM-main')`.
"""

import os
import sys
from pathlib import Path

__all__ = ["add_pcfm_to_path"]


def add_pcfm_to_path():
    """Find the PCFM repo and prepend it to sys.path. Returns the resolved path.

    Search order:
      1. $PCFM_REPO_PATH (if set)
      2. <Burgers_1D>/external/PCFM-main  (recommended: vendored copy)
      3. <Burgers_1D>/external/PCFM
      4. <repo-parent>/PCFM-main
      5. <repo-parent>/PCFM
    A directory qualifies only if it contains `pcfm/__init__.py`.
    """
    candidates = []
    env = os.environ.get("PCFM_REPO_PATH")
    if env:
        candidates.append(Path(env).expanduser())

    repo_root = Path(__file__).resolve().parents[1]   # .../Burgers_1D
    candidates += [
        repo_root / "external" / "PCFM-main",
        repo_root / "external" / "PCFM",
        repo_root.parent / "PCFM-main",
        repo_root.parent / "PCFM",
    ]

    for c in candidates:
        if (c / "pcfm" / "__init__.py").exists():
            cpath = str(c)
            if cpath not in sys.path:
                sys.path.insert(0, cpath)
            return cpath

    raise FileNotFoundError(
        "Could not locate the PCFM repository. Either set the PCFM_REPO_PATH "
        "environment variable, or place the repo at 'external/PCFM-main' inside "
        "Burgers_1D/. Searched: " + ", ".join(str(c) for c in candidates)
    )
