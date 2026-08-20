"""
Physics residual evaluator (shared with PCFM).

Wraps the PCFM repo's `Residuals3D.full_residual_darcy3d` -- the finite-volume
flux-balance residual for steady-state 3D Darcy flow (see pcfm/constraints.py).
Darcy_3D lives inside the PCFM repo (src/__init__.py's add_repo_to_path), so
importing `pcfm` needs no vendoring/path tricks beyond that.
"""

import torch

from . import add_repo_to_path
add_repo_to_path()                        # ensure the repo root is importable
from pcfm import Residuals3D               # noqa: E402  (import after path injection)


def calc_physics_residual_pcfm(p_phys_fp64, k_phys_fp64, bc_phys_fp64):
    """Darcy PDE residual on a batch.

    p_phys_fp64  : [B, nx, ny, nz] predicted pressure field, physical units
                   (float64 to avoid catastrophic cancellation in the flux
                   differences)
    k_phys_fp64  : [B, nx, ny, nz] permeability field, physical units
    bc_phys_fp64 : [B, 2] (p_left, p_right), physical units

    Returns the stacked per-sample residual vectors (used as the PBFM
    training target). Unlike Burgers'/NS's residuals (fixed PDE parameters),
    the permeability field is sample-specific data, so a fresh Residuals3D
    object is built per sample.
    """
    B, nx, ny, nz = p_phys_fp64.shape
    R_list = []
    for i in range(B):
        res_obj = Residuals3D(
            k=k_phys_fp64[i],
            p_left=bc_phys_fp64[i, 0].item(),
            p_right=bc_phys_fp64[i, 1].item(),
            nx=nx, ny=ny, nz=nz,
        )
        R_list.append(res_obj.full_residual_darcy3d(p_phys_fp64[i].flatten()))
    return torch.stack(R_list)
