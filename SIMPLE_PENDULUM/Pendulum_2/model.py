"""
model.py  --  Case 5: Extended Experiments
-------------------------------------------
Three model variants sharing the same FiLM+IC base architecture:

  Variant A  --  Adaptive lambda (IC-conditioned physics weight)
      Physics loss weight is a learned function of the IC rather than a
      fixed scalar. The model still uses the standard MLP (no FiLM) but
      the lambda network outputs a per-sample weight conditioned on the IC.

  Variant B  --  IC conditioning + physics loss during training
      FiLM conditioning on IC (identical to Case 4b) PLUS a rollout-based
      physics penalty during training. Tests whether explicit enforcement
      on top of complete information helps further.

  Variant C  --  IC conditioning + Hamiltonian enforcement (strongest)
      Same as Variant B but with a stronger Hamiltonian penalty that also
      penalises the mean energy level deviation (not just variance/drift).
      This is the most constrained model and should give the best energy
      conservation achievable without intrinsic Hamiltonicity.

All variants share:
  - The IC encoder (3D -> 32D embedding)
  - FiLM modulation of hidden layers (Variants B, C)
  - The rollout-based physics loss infrastructure
  - Standard FM loss
"""

import numpy as np
import torch
import torch.nn as nn

from Simple_pendulum.Pendulum_2.config import N_STEPS, STATE_DIM

HIDDEN_DIM  = 512
N_LAYERS    = 4
T_EMB_DIM   = 16
IC_EMB_DIM  = 32
from Simple_pendulum.Pendulum_2.config import N_FM_STEPS_TRAIN


# ── Sinusoidal FM-time embedding ──────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        freqs = torch.exp(
            torch.arange(dim // 2) * -(np.log(10000.0) / (dim // 2 - 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.view(-1, 1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ── IC Encoder (shared across all variants) ───────────────────────────────────

class ICEncoder(nn.Module):
    """Encodes (sin theta0, cos theta0, omega0) -> 32D embedding."""
    def __init__(self, emb_dim: int = IC_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        return self.net(ic)


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, hidden_dim)
        self.to_beta  = nn.Linear(emb_dim, hidden_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.to_gamma(emb) * h + self.to_beta(emb)


# ── Lambda network for Variant A ──────────────────────────────────────────────

class AdaptiveLambdaNet(nn.Module):
    """
    Learns a per-sample physics loss weight conditioned on the IC.
    Output is passed through softplus to ensure positivity, then scaled
    by lambda_base. The network is initialised so that it outputs
    approximately lambda_base for all ICs at the start of training.

    lambda(ic) = lambda_base * softplus(net(ic))  / softplus(0)
    """
    def __init__(self, lambda_base: float = 0.1, emb_dim: int = IC_EMB_DIM):
        super().__init__()
        self.lambda_base = lambda_base
        self.encoder = ICEncoder(emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # initialise to output 0 -> softplus(0) ~ 0.693

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ic : shape (B, 3)
        Returns:
            lam : shape (B,)  -- per-sample lambda values
        """
        emb = self.encoder(ic)               # (B, 32)
        raw = self.head(emb).squeeze(-1)     # (B,)
        # softplus ensures positivity; divide by softplus(0) to normalise
        sp0 = torch.nn.functional.softplus(torch.zeros(1, device=ic.device))
        lam = self.lambda_base * torch.nn.functional.softplus(raw) / sp0
        return lam                           # (B,)


# ── Variant A: Standard MLP + adaptive lambda network ─────────────────────────

class VelocityFieldMLP_AdaptiveLambda(nn.Module):
    """
    Standard MLP velocity field (no FiLM conditioning).
    The lambda network is a separate module trained jointly.

    At training: lambda = AdaptiveLambdaNet(ic)  -> per-sample physics weight
    At inference: plain Euler FM, no conditioning
    """
    def __init__(self, lambda_base: float = 0.1):
        super().__init__()
        self.t_embed    = SinusoidalEmbedding(T_EMB_DIM)
        self.lambda_net = AdaptiveLambdaNet(lambda_base)

        input_dim = N_STEPS * STATE_DIM + T_EMB_DIM
        layers = [nn.Linear(input_dim, HIDDEN_DIM), nn.SiLU()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU()]
        layers.append(nn.Linear(HIDDEN_DIM, N_STEPS * STATE_DIM))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = u_t.shape[0]
        t_emb  = self.t_embed(t)
        x      = torch.cat([u_t.reshape(B, -1), t_emb], dim=1)
        out    = self.mlp(x)
        return out.reshape(B, N_STEPS, STATE_DIM)

    def get_lambda(self, ic: torch.Tensor) -> torch.Tensor:
        """Return per-sample lambda values given encoded IC."""
        return self.lambda_net(ic)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Variants B & C: FiLM-conditioned on IC + physics loss ─────────────────────

class VelocityFieldFiLM_IC_Physics(nn.Module):
    """
    FiLM-conditioned velocity field (same as Case 4b) with optional
    physics loss enforcement during training.

    Used for:
      Variant B -- IC conditioning + fixed lambda physics loss
      Variant C -- IC conditioning + stronger Hamiltonian penalty
    """
    def __init__(self):
        super().__init__()
        self.t_embed  = SinusoidalEmbedding(T_EMB_DIM)
        self.ic_embed = ICEncoder(IC_EMB_DIM)

        input_dim = N_STEPS * STATE_DIM + T_EMB_DIM
        self.input_proj = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.SiLU())

        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(IC_EMB_DIM, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.act = nn.SiLU()
        self.output_proj = nn.Linear(HIDDEN_DIM, N_STEPS * STATE_DIM)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        u_t: torch.Tensor,   # (B, N_STEPS, 3)
        t:   torch.Tensor,   # (B,)
        ic:  torch.Tensor,   # (B, 3)
    ) -> torch.Tensor:
        B = u_t.shape[0]
        t_emb  = self.t_embed(t)
        ic_emb = self.ic_embed(ic)
        x = torch.cat([u_t.reshape(B, -1), t_emb], dim=1)
        h = self.input_proj(x)
        for linear, film in zip(self.hidden_layers, self.film_layers):
            h = self.act(linear(h))
            h = film(h, ic_emb)
        return self.output_proj(h).reshape(B, N_STEPS, STATE_DIM)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory functions ─────────────────────────────────────────────────────────

def build_model_A(lambda_base: float = 0.1, device: str = "cpu"):
    model = VelocityFieldMLP_AdaptiveLambda(lambda_base).to(device)
    print(f"Variant A (adaptive lambda): {model.count_parameters():,} params")
    return model

def build_model_BC(device: str = "cpu"):
    model = VelocityFieldFiLM_IC_Physics().to(device)
    print(f"Variant B/C (FiLM+IC+physics): {model.count_parameters():,} params")
    return model


# ── Variant C: FiLM conditioned on IC + H0 (no physics loss) ─────────────────

class HEncoder(nn.Module):
    """Encodes scalar H0 -> 32D embedding."""
    def __init__(self, emb_dim: int = IC_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, H0: torch.Tensor) -> torch.Tensor:
        # H0 shape: (B,) -> (B, emb_dim)
        return self.net(H0.unsqueeze(-1))


class VelocityFieldFiLM_IC_H(nn.Module):
    """
    FiLM conditioning on BOTH IC and H0.

    Conditioning signal: concat(ic_emb, H0_emb) in R^(32+32) = R^64
    This gives the model:
      - IC embedding: where on the contour to start (phase information)
      - H0 embedding: which contour (energy level)
    Both are encoded separately and concatenated before FiLM projection.

    No physics loss. Just standard FM loss.
    This tests whether redundant information (IC already implies H0)
    helps the model lock onto the correct trajectory more precisely.

    Variant C in Case 5 extended experiments.
    """
    def __init__(self):
        super().__init__()
        self.t_embed  = SinusoidalEmbedding(T_EMB_DIM)
        self.ic_embed = ICEncoder(IC_EMB_DIM)        # IC  -> 32D
        self.h0_embed = HEncoder(IC_EMB_DIM)         # H0  -> 32D
        cond_dim = IC_EMB_DIM * 2                    # 64D combined

        input_dim = N_STEPS * STATE_DIM + T_EMB_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM), nn.SiLU()
        )
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.act = nn.SiLU()
        self.output_proj = nn.Linear(HIDDEN_DIM, N_STEPS * STATE_DIM)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        u_t: torch.Tensor,   # (B, N_STEPS, 3)
        t:   torch.Tensor,   # (B,)
        ic:  torch.Tensor,   # (B, 3)  encoded IC
        H0:  torch.Tensor,   # (B,)    initial Hamiltonian value
    ) -> torch.Tensor:
        B = u_t.shape[0]
        t_emb   = self.t_embed(t)                    # (B, T_EMB_DIM)
        ic_emb  = self.ic_embed(ic)                  # (B, 32)
        h0_emb  = self.h0_embed(H0)                  # (B, 32)
        cond    = torch.cat([ic_emb, h0_emb], dim=1) # (B, 64)

        x = torch.cat([u_t.reshape(B, -1), t_emb], dim=1)
        h = self.input_proj(x)
        for linear, film in zip(self.hidden_layers, self.film_layers):
            h = self.act(linear(h))
            h = film(h, cond)
        return self.output_proj(h).reshape(B, N_STEPS, STATE_DIM)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_C(device: str = "cpu"):
    model = VelocityFieldFiLM_IC_H().to(device)
    print(f"Variant C (FiLM IC+H0, no physics loss): {model.count_parameters():,} params")
    return model
