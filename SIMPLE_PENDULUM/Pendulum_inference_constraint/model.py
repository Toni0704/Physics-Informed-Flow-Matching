"""
model.py
--------
Flow Matching velocity field network for the pendulum.

The network learns:
    vθ(u_t, t) : ℝ^(N*2) × ℝ  →  ℝ^(N*2)

where:
    u_t  is an interpolated pendulum trajectory, shape (B, N_STEPS, 2)
    t    is the FM time scalar ∈ [0, 1],         shape (B,)

Important: t here is *FM time*, not physical time τ.
    - FM time t : the artificial interpolation parameter (0=noise, 1=data)
    - Physical τ : the actual pendulum time, implicit in the trajectory index

Architecture:
    1. Flatten trajectory:  (B, N, 2) → (B, N*2)
    2. Sinusoidal embedding of t:  (B,) → (B, T_EMB_DIM)
    3. Concatenate: (B, N*2 + T_EMB_DIM)
    4. MLP with SiLU activations
    5. Reshape output: (B, N*2) → (B, N, 2)
"""

import numpy as np
import torch
import torch.nn as nn

from Simple_pendulum.Pendulum_inference_constraint.config import N_STEPS, HIDDEN_DIM, N_LAYERS, T_EMB_DIM, STATE_DIM


# ─────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Maps a scalar FM time t ∈ [0, 1] to a fixed-size embedding.

    Uses sin and cos at geometrically spaced frequencies — the same
    idea as positional encodings in transformers, adapted for a
    continuous scalar input.

    This lets the network distinguish FM time steps without having
    to learn positional structure from scratch.

    Args:
        dim : embedding dimension (must be even; outputs sin + cos halves)
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "T_EMB_DIM must be even"
        self.dim = dim
        # Precompute frequencies — not learned, just fixed
        freqs = torch.exp(
            torch.arange(dim // 2) * -(np.log(10000.0) / (dim // 2 - 1))
        )
        self.register_buffer("freqs", freqs)  # (dim/2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : shape (B,) or (B, 1)  — FM time values in [0, 1]

        Returns:
            emb : shape (B, dim)
        """
        t    = t.view(-1, 1)                                      # (B, 1)
        args = t * self.freqs.unsqueeze(0)                        # (B, dim/2)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], -1) # (B, dim)
        return emb


# ─────────────────────────────────────────────────────────────
# Velocity field MLP
# ─────────────────────────────────────────────────────────────

class VelocityFieldMLP(nn.Module):
    """
    The core Flow Matching network.

    Takes a noisy/interpolated trajectory and FM time t, outputs
    the predicted velocity direction in trajectory space.

    Args:
        n_steps    : number of physical time steps in each trajectory
        hidden_dim : MLP hidden layer width
        n_layers   : number of hidden layers
        t_emb_dim  : sinusoidal embedding dimension for FM time t
    """
    def __init__(
        self,
        n_steps:    int = N_STEPS,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
        t_emb_dim:  int = T_EMB_DIM,
        state_dim:  int = STATE_DIM,   # 3: (sin θ, cos θ, ω)
    ):
        super().__init__()
        self.n_steps  = n_steps
        self.state_dim = state_dim
        self.t_embed  = SinusoidalEmbedding(t_emb_dim)

        # Input: flattened trajectory + time embedding
        input_dim = n_steps * state_dim + t_emb_dim

        # Build MLP
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, n_steps * state_dim)]

        self.net = nn.Sequential(*layers)

        # Weight initialisation — small output scale for stable early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_t : shape (B, N_STEPS, STATE_DIM)  — interpolated trajectory
            t   : shape (B,)                      — FM time

        Returns:
            v   : shape (B, N_STEPS, STATE_DIM)   — predicted velocity
        """
        B = u_t.shape[0]

        u_flat = u_t.reshape(B, -1)                # (B, N*STATE_DIM)
        t_emb  = self.t_embed(t)                   # (B, t_emb_dim)
        x      = torch.cat([u_flat, t_emb], dim=1) # (B, N*STATE_DIM + t_emb_dim)
        out    = self.net(x)                        # (B, N*STATE_DIM)
        return out.reshape(B, self.n_steps, self.state_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> VelocityFieldMLP:
    model = VelocityFieldMLP().to(device)
    print(f"Model built: {model.count_parameters():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Smoke test
    m = build_model()
    u  = torch.randn(4, N_STEPS, 2)
    t  = torch.rand(4)
    v  = m(u, t)
    print(f"Input  shape: {u.shape}")
    print(f"Output shape: {v.shape}")
    assert v.shape == u.shape, "Output shape mismatch!"
    print("model.py OK")
