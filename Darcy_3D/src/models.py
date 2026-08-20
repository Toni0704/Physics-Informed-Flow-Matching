"""
Model architecture for FiLM-conditioned steady-state 3D Darcy flow.

A neuraloperator FNO backbone (`PatchedFNO3D`) produces a base velocity field
over the 3D pressure volume, modulated by a spatially-aware FiLM generator
conditioned on the (log-normalized) permeability field and the two scalar
Dirichlet BC values (p_left, p_right) -- the 3D/steady-state analog of
Burgers_1D's (ic, bc) FiLM pair: permeability plays the role the initial
condition played there, and the (p_left, p_right) pair plays the role the
single scalar BC played there.

Forward signature convention for the conditioned model is STATE-FIRST:
    model(xt, time, cond_k, cond_bc)
This differs from the PCFM repo's FFM/FNO, which is TIME-FIRST: model(t, x).
"""

import math
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO as _FNO


class SinusoidalPositionEmbeddings(nn.Module):
    """Continuous time embedding for flow-matching trajectories."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time.float()
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class RandomFourierFeatures3D(nn.Module):
    """Maps low-dimensional inputs to high-frequency space to reduce spectral bias."""

    def __init__(self, in_channels, out_channels=32, scale=10.0):
        super().__init__()
        self.register_buffer("weight", torch.randn(out_channels // 2, in_channels) * scale)

    def forward(self, x):
        proj = F.conv3d(x, self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        return torch.cat([torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=1)


class SpatiallyAwareFiLM3D(nn.Module):
    """Generates FiLM scale (gamma) and shift (beta) from k, time, and BC paths."""

    def __init__(self, feature_channels=32, time_emb_dim=128):
        super().__init__()
        self.fourier_k = RandomFourierFeatures3D(in_channels=1, out_channels=32)
        self.spatial_net_k = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, feature_channels * 2, kernel_size=3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, feature_channels * 2),
            nn.SiLU(),
        )
        # BC path: 2 scalar BCs (p_left, p_right) broadcast to constant
        # volumes, concatenated with the xyz coordinate grid.
        self.spatial_net_bc = nn.Sequential(
            nn.Conv3d(2 + 3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, feature_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, cond_k_3d, cond_bc_scalar, t, grid_xyz):
        B, _, nx, ny, nz = cond_k_3d.shape

        k_params = self.spatial_net_k(self.fourier_k(cond_k_3d))
        gamma_k, beta_k = torch.chunk(k_params, 2, dim=1)

        t_params = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma_t, beta_t = torch.chunk(t_params, 2, dim=1)

        bc_tile = cond_bc_scalar.view(B, 2, 1, 1, 1).expand(-1, -1, nx, ny, nz)
        bc_in = torch.cat([bc_tile, grid_xyz], dim=1)
        bc_params = self.spatial_net_bc(bc_in)
        gamma_bc, beta_bc = torch.chunk(bc_params, 2, dim=1)

        gamma = gamma_k + gamma_t + gamma_bc
        beta = beta_k + beta_t + beta_bc
        return gamma, beta


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """Sinusoidal time embedding injected directly into the FNO input tensor."""
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class PatchedFNO3D(nn.Module):
    """neuraloperator FNO wrapper that appends time + positional channels.

    Forward is TIME-FIRST: forward(t, x_in).
    """

    def __init__(self, n_modes, in_channels=4, emb_channels=32, hidden_channels=32,
                 proj_channels=128, n_layers=4):
        super().__init__()
        self.emb_channels = emb_channels
        fno_kwargs = {
            "n_modes": n_modes,
            "hidden_channels": hidden_channels,
            "in_channels": in_channels + len(n_modes) + emb_channels,
            "out_channels": 1,
        }
        # Tolerate different neuraloperator versions by inspecting the constructor.
        sig = inspect.signature(_FNO.__init__)
        accepted = sig.parameters.keys()
        if "n_layers" in accepted:
            fno_kwargs["n_layers"] = n_layers
        if "lifting_channels" in accepted:
            fno_kwargs["lifting_channels"] = proj_channels
        if "proj_channels" in accepted:
            fno_kwargs["proj_channels"] = proj_channels
        if "projection_channels" in accepted:
            fno_kwargs["projection_channels"] = proj_channels
        self.model = _FNO(**fno_kwargs)

    def forward(self, t, x_in):
        batch_size, channels, *dims = x_in.size()
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(batch_size, device=t.device) * t

        t_emb = get_time_embedding(t, self.emb_channels).view(
            [batch_size, -1] + [1] * len(dims)
        ).expand(-1, -1, *dims)
        pos_emb = torch.stack(torch.meshgrid(
            [torch.linspace(0, 1, n, device=x_in.device) for n in dims], indexing="ij"
        ), dim=0).unsqueeze(0).expand(batch_size, -1, *dims)

        u = torch.cat([x_in, t_emb, pos_emb], dim=1)
        return self.model(u).squeeze(1)


class PCFM_FiLMConditionedFNO3D(nn.Module):
    """FNO base field + FiLM modulation conditioned on permeability and BCs.

    Forward is STATE-FIRST: forward(xt, time, cond_k, cond_bc).
    """

    def __init__(self, base_fno, feature_channels=32):
        super().__init__()
        self.fno = base_fno
        self.film_gen = SpatiallyAwareFiLM3D(feature_channels=feature_channels, time_emb_dim=128)
        # feat_in channels: xt, fno_out, k, bc_left, bc_right, x, y, z = 8
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(8, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv3d(feature_channels, 1, kernel_size=1)

    def forward(self, xt, time, cond_k, cond_bc):
        B, nx, ny, nz = xt.shape
        grid_x = torch.linspace(0, 1, nx, device=xt.device).view(1, 1, nx, 1, 1).expand(B, 1, nx, ny, nz)
        grid_y = torch.linspace(0, 1, ny, device=xt.device).view(1, 1, 1, ny, 1).expand(B, 1, nx, ny, nz)
        grid_z = torch.linspace(0, 1, nz, device=xt.device).view(1, 1, 1, 1, nz).expand(B, 1, nx, ny, nz)
        grid_xyz = torch.cat([grid_x, grid_y, grid_z], dim=1)

        bc_left = cond_bc[:, 0].view(B, 1, 1, 1, 1).expand(-1, 1, nx, ny, nz)
        bc_right = cond_bc[:, 1].view(B, 1, 1, 1, 1).expand(-1, 1, nx, ny, nz)

        xt_1 = xt.unsqueeze(1)
        k_1 = cond_k if cond_k.dim() == 5 else cond_k.unsqueeze(1)

        # Base FNO field (note time-first call into PatchedFNO3D)
        x_in = torch.cat([xt_1, k_1, bc_left, bc_right], dim=1)
        fno_out = self.fno(time, x_in)

        feat_in = torch.cat([
            xt_1, fno_out.unsqueeze(1), k_1, bc_left, bc_right,
            grid_x, grid_y, grid_z,
        ], dim=1)
        features = self.feature_extractor(feat_in)

        gamma, beta = self.film_gen(k_1, cond_bc, time, grid_xyz)
        modulated = features * (gamma + 1.0) + beta
        return self.out_conv(modulated).squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
