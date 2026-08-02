"""
ns_model_film_ic.py -- FiLM-IC-conditioned NS velocity network
-----------------------------------------------------------------
Built directly on top of the existing FNO backbone (SpectralConv2d,
FNOBlock2d) used for the PCFM baseline, with one architectural change:

  PCFM baseline (NSVelocityNet):
      IC field `a` is concatenated as a raw input channel at input_proj.
      FiLM is used only for flow-matching time tau.

  Ours (NSVelocityNet_FiLM_IC):
      IC field `a` is removed from the raw input concatenation and instead
      encoded through a small CNN -> embedding -> FiLM branch, applied at
      every FNO block IN ADDITION TO the existing time-FiLM. This mirrors
      the FiLM-IC pattern used for the Pendulum case (ICEncoder + FiLMLayer
      per hidden layer), so that "conditioning vs. no conditioning" is a
      clean, switchable architectural difference rather than a confound.

Everything else (spectral conv, FNO block structure, mode count, width,
n_layers, time embedding) is UNCHANGED from the PCFM baseline, so that any
performance difference between this model and NSVelocityNet is attributable
to the conditioning mechanism, not to backbone capacity.
"""

import math
import torch
import torch.nn as nn


# ── Spectral conv + FNO block: UNCHANGED from PCFM baseline ──────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w1r = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w1i = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2r = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2i = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))

    def compl_mul2d(self, xr, xi, wr, wi):
        real = torch.einsum('bixy,ioxy->boxy', xr, wr) - \
               torch.einsum('bixy,ioxy->boxy', xi, wi)
        imag = torch.einsum('bixy,ioxy->boxy', xr, wi) + \
               torch.einsum('bixy,ioxy->boxy', xi, wr)
        return real, imag

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_r = torch.zeros(B, self.out_ch, H, W // 2 + 1, device=x.device)
        out_i = torch.zeros(B, self.out_ch, H, W // 2 + 1, device=x.device)

        xr1 = x_ft[:, :, :self.modes1, :self.modes2].real
        xi1 = x_ft[:, :, :self.modes1, :self.modes2].imag
        r1, i1 = self.compl_mul2d(xr1, xi1, self.w1r, self.w1i)
        out_r[:, :, :self.modes1, :self.modes2] = r1
        out_i[:, :, :self.modes1, :self.modes2] = i1

        xr2 = x_ft[:, :, -self.modes1:, :self.modes2].real
        xi2 = x_ft[:, :, -self.modes1:, :self.modes2].imag
        r2, i2 = self.compl_mul2d(xr2, xi2, self.w2r, self.w2i)
        out_r[:, :, -self.modes1:, :self.modes2] = r2
        out_i[:, :, -self.modes1:, :self.modes2] = i2

        out_ft = torch.view_as_complex(torch.stack([out_r, out_i], dim=-1))
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass   = nn.Conv2d(width, width, 1)
        self.norm     = nn.GroupNorm(min(8, width), width)
        self.act      = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


# ── IC encoder: NEW, mirrors the Pendulum ICEncoder pattern ──────────────────

class ICEncoder2d(nn.Module):
    """
    Encodes the initial vorticity field a (B, 1, 64, 64) -> a global
    embedding (B, emb_dim), analogous to the Pendulum case's ICEncoder
    (3 -> 32D MLP), but operating on a spatial field instead of a vector.

    Small CNN + global average pool, kept lightweight relative to the FNO
    backbone since its only job is to summarize "which IC" is being
    conditioned on, not to do heavy spatial processing itself.
    """
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # (B, 32, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, 1, 64, 64) -> (B, emb_dim)
        return self.head(self.net(a))


# ── FiLM-IC-conditioned velocity network ──────────────────────────────────────

class NSVelocityNet_FiLM_IC(nn.Module):
    """
    FiLM-IC-conditioned CFM velocity network for 2D Navier-Stokes.

    Identical backbone to the PCFM baseline (NSVelocityNet): same
    SpectralConv2d / FNOBlock2d, same mode count, width, n_layers, and
    time-FiLM mechanism. The only architectural change is the IC pathway:

      PCFM baseline : a concatenated as raw input channel at input_proj
      Ours          : a encoded via ICEncoder2d -> separate FiLM branch,
                      applied at every FNO block alongside time-FiLM

    Inputs:  u_t (B, n_t, 64, 64) + a (B, 1, 64, 64) + t scalar (B,)
    Output:  velocity (B, n_t, 64, 64)
    """
    def __init__(self, n_t: int = 50, modes: int = 12, width: int = 48,
                 n_layers: int = 4, ic_emb_dim: int = 64):
        super().__init__()
        self.n_t = n_t

        # NOTE: input is exactly n_t channels (down from n_t + 2 in the PCFM
        # baseline), since `a` is removed from the raw concatenation and
        # routed through FiLM instead, and the old time-broadcast channel is
        # also no longer concatenated here (time enters only via FiLM below,
        # consistent with how IC now enters).
        self.input_proj = nn.Conv2d(n_t, width, 1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes, modes) for _ in range(n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, n_t, 1),
        )

        # Time-FiLM: UNCHANGED from PCFM baseline.
        self.time_mlp = nn.Sequential(
            nn.Linear(64, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width),
        )
        self.film_scale_t = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.film_shift_t = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])

        # IC-FiLM: NEW. Separate scale/shift heads per layer, zero-initialized
        # so that at initialization this branch is the identity transform
        # (consistent with the Pendulum FiLMLayer init convention) and the
        # model starts equivalent to an "IC-blind" network, with conditioning
        # learned in gradually over training.
        self.ic_encoder = ICEncoder2d(ic_emb_dim)
        self.film_scale_ic = nn.ModuleList([nn.Linear(ic_emb_dim, width) for _ in range(n_layers)])
        self.film_shift_ic = nn.ModuleList([nn.Linear(ic_emb_dim, width) for _ in range(n_layers)])
        for layer in self.film_scale_ic:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        for layer in self.film_shift_ic:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def time_embedding(self, t):
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, u_t: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u_t: (B, n_t, 64, 64), a: (B, 1, 64, 64), t: (B,)
        B, _, H, W = u_t.shape

        t_emb = self.time_mlp(self.time_embedding(t))   # (B, width)
        ic_emb = self.ic_encoder(a)                      # (B, ic_emb_dim)

        # `a` is NOT concatenated here -- it only enters via FiLM below.
        x = self.input_proj(u_t)
        for i, blk in enumerate(self.fno_blocks):
            x = blk(x)

            scale_t = self.film_scale_t[i](t_emb)[:, :, None, None]
            shift_t = self.film_shift_t[i](t_emb)[:, :, None, None]
            x = x * (1 + scale_t) + shift_t

            scale_ic = self.film_scale_ic[i](ic_emb)[:, :, None, None]
            shift_ic = self.film_shift_ic[i](ic_emb)[:, :, None, None]
            x = x * (1 + scale_ic) + shift_ic

        return self.output_proj(x)   # (B, n_t, 64, 64)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_film_ic(n_t=50, modes=12, width=48, n_layers=4,
                         ic_emb_dim=64, device: str = "cpu"):
    model = NSVelocityNet_FiLM_IC(n_t, modes, width, n_layers, ic_emb_dim).to(device)
    print(f"NS FiLM-IC model: {model.count_parameters():,} params")
    return model


