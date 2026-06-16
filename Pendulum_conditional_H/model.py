"""
model.py  —  Case 4: Conditional Generation
--------------------------------------------
The velocity field is conditioned on the target energy level E:

    vθ(u_t, t, E) : ℝ^(N*3) × ℝ × ℝ  →  ℝ^(N*3)

Conditioning mechanism: FiLM (Feature-wise Linear Modulation)
--------------------------------------------------------------
Instead of simply concatenating E to the input, FiLM modulates
the internal activations of each hidden layer:

    h_conditioned = γ(E) ⊙ h  +  β(E)

where γ(E) and β(E) are learned linear projections of the energy
embedding. This is strictly more expressive than concatenation
because it can scale and shift each hidden unit independently,
allowing the energy level to reorganise the internal representation
rather than just append information at the input.

Intuition: the energy level tells the network WHICH part of phase
space to generate in. FiLM lets the network fundamentally change
its behaviour based on E, not just use E as one extra feature.

Why not just concatenate E?
    With concatenation, E competes with the 90-dimensional trajectory
    signal at the first layer. With FiLM, E directly controls the
    gain and bias of every hidden layer — a much stronger signal.

Architecture:
    1. Sinusoidal embedding of FM time t  →  ℝ^T_EMB_DIM
    2. Energy embedding:  E (scalar)  →  MLP  →  ℝ^E_EMB_DIM
    3. Flatten trajectory: (B, N, 3)  →  (B, N*3)
    4. Input projection: (B, N*3 + T_EMB_DIM)  →  (B, hidden_dim)
    5. For each hidden layer:
           h = SiLU(Linear(h))
           γ, β = FiLM_layer(energy_emb)   shape: (B, hidden_dim) each
           h = γ ⊙ h + β
    6. Output: (B, hidden_dim)  →  (B, N*3)  →  (B, N, 3)
"""

import numpy as np
import torch
import torch.nn as nn

from config import N_STEPS, HIDDEN_DIM, N_LAYERS, T_EMB_DIM, STATE_DIM

# Energy embedding dimension
E_EMB_DIM = 32


# ─────────────────────────────────────────────────────────────
# Sinusoidal time embedding (same as Cases 1-3)
# ─────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        freqs = torch.exp(
            torch.arange(dim // 2) * -(np.log(10000.0) / (dim // 2 - 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t    = t.view(-1, 1)
        args = t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────────────────────────────────────────────────
# Energy embedding
# ─────────────────────────────────────────────────────────────

class EnergyEmbedding(nn.Module):
    """
    Maps scalar energy E ∈ [E_MIN, E_MAX] to a dense embedding.

    Uses a small MLP rather than a sinusoidal embedding because
    energy is a continuous physical quantity with smooth structure —
    nearby energy levels should produce similar embeddings, which
    a learned MLP respects better than fixed frequencies.

    Args:
        emb_dim : output embedding dimension
    """
    def __init__(self, emb_dim: int = E_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E : shape (B,)  — energy values

        Returns:
            emb : shape (B, emb_dim)
        """
        return self.net(E.view(-1, 1))   # (B, emb_dim)


# ─────────────────────────────────────────────────────────────
# FiLM layer
# ─────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Given an energy embedding, produces scale (γ) and shift (β)
    to modulate a hidden state h:

        h_out = γ ⊙ h  +  β

    Args:
        emb_dim    : energy embedding dimension (input)
        hidden_dim : hidden state dimension (output γ and β)
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, hidden_dim)
        self.to_beta  = nn.Linear(emb_dim, hidden_dim)

        # Initialise to identity modulation: γ=1, β=0
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, e_emb: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            h     : shape (B, hidden_dim)  — hidden state
            e_emb : shape (B, emb_dim)     — energy embedding

        Returns:
            h_modulated : shape (B, hidden_dim)
        """
        gamma = self.to_gamma(e_emb)   # (B, hidden_dim)
        beta  = self.to_beta(e_emb)    # (B, hidden_dim)
        return gamma * h + beta


# ─────────────────────────────────────────────────────────────
# Conditional velocity field
# ─────────────────────────────────────────────────────────────

class ConditionalVelocityField(nn.Module):
    """
    FiLM-conditioned Flow Matching velocity field.

        vθ(u_t, t, E) : ℝ^(N*3) × ℝ × ℝ  →  ℝ^(N*3)

    Args:
        n_steps    : physical time steps per trajectory
        hidden_dim : MLP hidden width
        n_layers   : number of hidden layers (each with FiLM)
        t_emb_dim  : sinusoidal FM time embedding dim
        state_dim  : state dimension (3: sin θ, cos θ, ω)
        e_emb_dim  : energy embedding dimension
    """
    def __init__(
        self,
        n_steps:    int = N_STEPS,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
        t_emb_dim:  int = T_EMB_DIM,
        state_dim:  int = STATE_DIM,
        e_emb_dim:  int = E_EMB_DIM,
    ):
        super().__init__()
        self.n_steps   = n_steps
        self.state_dim = state_dim

        # Embeddings
        self.t_embed = SinusoidalEmbedding(t_emb_dim)
        self.e_embed = EnergyEmbedding(e_emb_dim)

        # Input projection: trajectory + time embedding → hidden
        input_dim = n_steps * state_dim + t_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        # Hidden layers + FiLM conditioning
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(e_emb_dim, hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.act = nn.SiLU()

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_steps * state_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        u_t: torch.Tensor,
        t:   torch.Tensor,
        E:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            u_t : shape (B, N_STEPS, STATE_DIM)  — interpolated trajectory
            t   : shape (B,)                      — FM time
            E   : shape (B,)                      — target energy level

        Returns:
            v   : shape (B, N_STEPS, STATE_DIM)   — predicted velocity
        """
        B = u_t.shape[0]

        # Embed FM time and energy
        t_emb = self.t_embed(t)    # (B, t_emb_dim)
        e_emb = self.e_embed(E)    # (B, e_emb_dim)

        # Flatten trajectory and concatenate time embedding
        u_flat = u_t.reshape(B, -1)                     # (B, N*state_dim)
        x      = torch.cat([u_flat, t_emb], dim=1)      # (B, N*3 + t_emb)

        # Input projection
        h = self.input_proj(x)                           # (B, hidden_dim)

        # Hidden layers with FiLM conditioning
        for linear, film in zip(self.hidden_layers, self.film_layers):
            h = self.act(linear(h))                      # (B, hidden_dim)
            h = film(h, e_emb)                           # FiLM modulation

        # Output
        out = self.output_proj(h)                        # (B, N*state_dim)
        return out.reshape(B, self.n_steps, self.state_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> ConditionalVelocityField:
    model = ConditionalVelocityField().to(device)
    print(f"Conditional model built: {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    m = build_model()
    u = torch.randn(4, N_STEPS, STATE_DIM)
    t = torch.rand(4)
    E = torch.rand(4) * 0.8 + 0.1   # E ∈ [0.1, 0.9]
    v = m(u, t, E)
    assert v.shape == u.shape
    print(f"Input  shape : {u.shape}")
    print(f"Output shape : {v.shape}")
    print("model.py OK")
