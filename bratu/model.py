"""
model.py  --  Bratu FM Models
------------------------------
Two model variants:

  ModelB0  --  Vanilla MLP, no conditioning          [Case B0]
  ModelB1  --  FiLM conditioned on scalar C only     [Cases B1, B2, B3]

The key design decision: NO branch label b in the conditioning signal.
The model must learn from data that there are two solution branches for
each C < Cc and generate from either branch depending on the noise
realisation u0 ~ N(0,I). This tests whether FM can learn a bimodal
conditional distribution p(u | C).

FiLM conditioning:  h_l <- gamma_l(C) * h_l + beta_l(C)
Initialised to identity: gamma=1, beta=0
"""

import numpy as np
import torch
import torch.nn as nn

from config import N_X, HIDDEN_DIM, N_LAYERS, T_EMB_DIM, C_EMB_DIM


# ── Sinusoidal FM-time embedding ──────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        freqs = torch.exp(
            torch.arange(dim // 2) * -(np.log(10000.0) / (dim // 2 - 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t.view(-1, 1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ── C Encoder ─────────────────────────────────────────────────────────────────

class CEncoder(nn.Module):
    """Encodes scalar C -> C_EMB_DIM embedding."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, C_EMB_DIM), nn.SiLU(),
            nn.Linear(C_EMB_DIM, C_EMB_DIM), nn.SiLU(),
            nn.Linear(C_EMB_DIM, C_EMB_DIM),
        )

    def forward(self, C):
        return self.net(C.unsqueeze(-1).float())   # (B,) -> (B, C_EMB_DIM)


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, hidden_dim)
        self.to_beta  = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, cond):
        return self.to_gamma(cond) * h + self.to_beta(cond)


# ── Case B0: Vanilla MLP ──────────────────────────────────────────────────────

class ModelB0(nn.Module):
    """
    Vanilla FM. No conditioning — sees only (u_t, t).
    Expected to fail: bimodal target for same input creates conflicting
    gradients; model likely collapses to the dominant lower branch.
    """
    def __init__(self):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(T_EMB_DIM)
        input_dim = N_X + T_EMB_DIM
        layers = [nn.Linear(input_dim, HIDDEN_DIM), nn.SiLU()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU()]
        layers.append(nn.Linear(HIDDEN_DIM, N_X))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u_t, t):
        t_emb = self.t_embed(t)
        return self.mlp(torch.cat([u_t, t_emb], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Cases B1 / B2 / B3: FiLM conditioned on C only ───────────────────────────

class ModelB1(nn.Module):
    """
    FiLM-conditioned on scalar C only. No branch label.

    The model receives the same C for both branches of the same solution.
    Branch selection must emerge from the noise realisation u0 ~ N(0,I):
    different noise samples at the same C should map to different branches
    if the model correctly learns the bimodal structure of p(u | C).

    This is the central scientific question: can FM learn a bimodal
    conditional distribution from data alone?

    Used for:
        B1 -- evaluate mode coverage and PDE quality (no projection)
        B2 -- same weights, add PCFM projection at t=1
        B3 -- same weights, evaluate at C = Cc
    """
    def __init__(self):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(T_EMB_DIM)
        self.c_embed = CEncoder()

        input_dim = N_X + T_EMB_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM), nn.SiLU()
        )
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(C_EMB_DIM, HIDDEN_DIM) for _ in range(N_LAYERS - 1)
        ])
        self.act = nn.SiLU()
        self.output_proj = nn.Linear(HIDDEN_DIM, N_X)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, u_t, t, C):
        """
        Args:
            u_t : (B, N_X)
            t   : (B,)
            C   : (B,)    float
        Returns:
            v   : (B, N_X)
        """
        t_emb = self.t_embed(t)
        c_emb = self.c_embed(C)
        x = torch.cat([u_t, t_emb], dim=1)
        h = self.input_proj(x)
        for linear, film in zip(self.hidden_layers, self.film_layers):
            h = self.act(linear(h))
            h = film(h, c_emb)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model_B0(device="cpu"):
    model = ModelB0().to(device)
    print(f"ModelB0 (vanilla):  {model.count_parameters():,} parameters")
    return model

def build_model_B1(device="cpu"):
    model = ModelB1().to(device)
    print(f"ModelB1 (FiLM C):   {model.count_parameters():,} parameters")
    return model