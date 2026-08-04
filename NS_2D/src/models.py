"""
FiLM-(IC+forcing)-conditioned CFM velocity network for 2D Navier-Stokes.

Reuses the backbone pieces from models/ns_model_film_ic.py (SpectralConv2d /
FNOBlock2d / ICEncoder2d) unchanged, and extends the conditioning from IC-only
to (IC, forcing): each conditioning field gets its own encoder + zero-initialized
per-layer FiLM branch, applied after the time-FiLM at every FNO block. This is
the NS analog of Burgers' (ic, bc) FiLM pair -- forcing plays the role the
boundary condition played there (the second external input that, with the IC,
determines the trajectory).
"""

import importlib.util

import torch
import torch.nn as nn

from . import REPO_ROOT

# Load ns_model_film_ic.py directly by file path: importing it as
# `models.ns_model_film_ic` would execute models/__init__.py, which pulls in
# neuralop -- a heavy dependency this module doesn't need.
_spec = importlib.util.spec_from_file_location(
    "_ns_model_film_ic", REPO_ROOT / "models" / "ns_model_film_ic.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FNOBlock2d, ICEncoder2d = _mod.FNOBlock2d, _mod.ICEncoder2d


class NSVelocityNet_FiLM_ICF(nn.Module):
    """
    Inputs:  u_t (B, n_t, H, W), t (B,), a (B, 1, H, W), f (B, 1, H, W)
    Output:  velocity (B, n_t, H, W)

    Backbone (input_proj / FNO blocks / output_proj / time-FiLM) identical to
    NSVelocityNet_FiLM_IC so that conditioned-vs-unconditioned comparisons
    isolate the conditioning mechanism, not capacity.
    """

    def __init__(self, n_t: int = 50, modes: int = 12, width: int = 48,
                 n_layers: int = 4, cond_emb_dim: int = 64):
        super().__init__()
        self.n_t = n_t

        self.input_proj = nn.Conv2d(n_t, width, 1)
        self.fno_blocks = nn.ModuleList(
            [FNOBlock2d(width, modes, modes) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, n_t, 1),
        )

        # Time-FiLM (unchanged from the IC-only model)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width),
        )
        self.film_scale_t = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.film_shift_t = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])

        # Conditioning FiLM branches: one per conditioning field, zero-init so
        # the model starts conditioning-blind and learns them in gradually.
        self.ic_encoder = ICEncoder2d(cond_emb_dim)
        self.f_encoder = ICEncoder2d(cond_emb_dim)
        self.film_scale_ic = nn.ModuleList([nn.Linear(cond_emb_dim, width) for _ in range(n_layers)])
        self.film_shift_ic = nn.ModuleList([nn.Linear(cond_emb_dim, width) for _ in range(n_layers)])
        self.film_scale_f = nn.ModuleList([nn.Linear(cond_emb_dim, width) for _ in range(n_layers)])
        self.film_shift_f = nn.ModuleList([nn.Linear(cond_emb_dim, width) for _ in range(n_layers)])
        for branch in (self.film_scale_ic, self.film_shift_ic,
                       self.film_scale_f, self.film_shift_f):
            for layer in branch:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def time_embedding(self, t):
        import math
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, u_t, t, a, f):
        t_emb = self.time_mlp(self.time_embedding(t))  # (B, width)
        ic_emb = self.ic_encoder(a)                    # (B, cond_emb_dim)
        f_emb = self.f_encoder(f)                      # (B, cond_emb_dim)

        x = self.input_proj(u_t)
        for i, blk in enumerate(self.fno_blocks):
            x = blk(x)

            scale = self.film_scale_t[i](t_emb)[:, :, None, None]
            shift = self.film_shift_t[i](t_emb)[:, :, None, None]
            x = x * (1 + scale) + shift

            scale = self.film_scale_ic[i](ic_emb)[:, :, None, None]
            shift = self.film_shift_ic[i](ic_emb)[:, :, None, None]
            x = x * (1 + scale) + shift

            scale = self.film_scale_f[i](f_emb)[:, :, None, None]
            shift = self.film_shift_f[i](f_emb)[:, :, None, None]
            x = x * (1 + scale) + shift

        return self.output_proj(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
