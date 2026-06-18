"""
Physics residual evaluator (shared with PCFM).

Wraps the PCFM repo's `Residuals.burgers_local_multistep_residual`. Because this
module imports from the (separate) PCFM repo, importing it triggers the path
resolver in `src/__init__.py` so that `from pcfm import Residuals` succeeds
regardless of the current working directory.
"""

import torch

from . import add_pcfm_to_path
add_pcfm_to_path()                       # ensure the vendored PCFM repo is importable
from pcfm import Residuals               # noqa: E402  (import after path injection)


def calc_physics_residual_pcfm(u_phys_fp64, bc_phys_fp64):
    """Explicit Burgers PDE residual on a batch.

    u_phys_fp64 : [B, 1, nx, nt] in physical units (float64 to avoid catastrophic
                  cancellation in the finite-difference gradients)
    bc_phys_fp64: [B, 1] left Dirichlet value per sample, physical units

    Returns the stacked per-sample residual vectors (used as the PBFM training
    target; k=100 sets the multistep stiffness weight).
    """
    B = u_phys_fp64.shape[0]
    nx_dim, nt_dim = u_phys_fp64.shape[2], u_phys_fp64.shape[3]
    x_grid = torch.linspace(0, 1.0, nx_dim, dtype=torch.float64, device=u_phys_fp64.device)
    t_grid = torch.linspace(0, 1.0, nt_dim, dtype=torch.float64, device=u_phys_fp64.device)

    R_list = []
    for i in range(B):
        u_sample = u_phys_fp64[i, 0]
        left_bc_i = bc_phys_fp64[i].to(torch.float64).view(1)
        res_obj = Residuals(
            data=u_sample.unsqueeze(0),
            x=x_grid, t_grid=t_grid, nx=nx_dim, nt=nt_dim,
            left_bc=left_bc_i,
        )
        phys_res = res_obj.burgers_local_multistep_residual(u_sample.flatten(), k=100)
        R_list.append(phys_res)

    return torch.stack(R_list)
