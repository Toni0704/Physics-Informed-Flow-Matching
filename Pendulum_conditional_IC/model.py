"""
model.py  --  Case 4b: Conditional Generation on Initial State
--------------------------------------------------------------
The velocity field is conditioned on the initial state (theta0, omega0)
instead of the scalar energy E:

    v_theta(u_t, t, ic) :  R^(N*3) x R x R^2  ->  R^(N*3)

where ic = (sin(theta0), cos(theta0), omega0)  -- shape (3,)

Why encode ic as (sin, cos, omega) instead of raw (theta, omega)?
    Same reason as the trajectory encoding -- raw theta has a branch
    cut at +-pi. Using (sin, cos) keeps the conditioning signal smooth
    and consistent with how the trajectory itself is represented.

Why is this a stronger signal than energy E?
    E is a many-to-one summary: infinitely many (theta0, omega0) pairs
    map to the same energy level. Given only E, the model must also
    decide WHERE on the energy contour to start and which DIRECTION
    to traverse it -- it learns a distribution over trajectories.

    Given (theta0, omega0) exactly, the physics fully determines the
    trajectory. There is only one physical path from that initial state.
    The model is essentially learning a surrogate ODE solver.

Conditioning mechanism: FiLM (same as Case 4a)
    h_l <- gamma_l(ic) * h_l + beta_l(ic)

The only change from Case 4a is the IC encoder replaces the energy
encoder -- everything else (architecture, training, FiLM layers)
is identical. This makes Case 4a vs 4b a clean ablation over the
conditioning signal.
"""

import numpy as np
import torch
import torch.nn as nn

from config import N_STEPS, HIDDEN_DIM, N_LAYERS, T_EMB_DIM, STATE_DIM

IC_EMB_DIM = 32   # same as E_EMB_DIM in Case 4a for fair comparison


# ── Sinusoidal time embedding (unchanged from all other cases) ────────────────

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


# ── Initial condition encoder ─────────────────────────────────────────────────

class ICEncoder(nn.Module):
    """
    Encodes the initial state ic = (sin theta0, cos theta0, omega0)
    into a dense embedding vector.

    Input:  (B, 3)   -- encoded initial state
    Output: (B, IC_EMB_DIM)

    Uses a small MLP. A learned encoder is preferred over a fixed
    sinusoidal one because the IC lives in a smooth 2D manifold
    (the phase space) and nearby ICs should produce similar embeddings.

    Note: we encode theta0 as (sin, cos) before passing to this encoder,
    so the raw input is 3-dimensional, not 2.
    """
    def __init__(self, emb_dim: int = IC_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ic : shape (B, 3)  -- (sin theta0, cos theta0, omega0)
        Returns:
            emb : shape (B, IC_EMB_DIM)
        """
        return self.net(ic)


# ── FiLM layer (unchanged from Case 4a) ──────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    h_out = gamma(cond) * h  +  beta(cond)

    Args:
        emb_dim    : conditioning embedding dimension
        hidden_dim : hidden state dimension
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, hidden_dim)
        self.to_beta  = nn.Linear(emb_dim, hidden_dim)
        # Initialise to identity: gamma=1, beta=0
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.to_gamma(emb) * h + self.to_beta(emb)


# ── Conditional velocity field ────────────────────────────────────────────────

class ConditionalVelocityFieldIC(nn.Module):
    """
    FiLM-conditioned velocity field using initial condition (theta0, omega0).

        v_theta(u_t, t, ic) :  R^(N*3) x R x R^3  ->  R^(N*3)

    Identical to Case 4a except:
        - ICEncoder replaces EnergyEmbedding
        - Input ic has shape (B, 3) instead of (B,)

    Args:
        n_steps    : physical time steps per trajectory
        hidden_dim : MLP hidden width
        n_layers   : number of hidden layers (each with FiLM)
        t_emb_dim  : sinusoidal FM-time embedding dim
        state_dim  : state dimension (3: sin theta, cos theta, omega)
        ic_emb_dim : IC embedding dimension
    """
    def __init__(
        self,
        n_steps:    int = N_STEPS,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
        t_emb_dim:  int = T_EMB_DIM,
        state_dim:  int = STATE_DIM,
        ic_emb_dim: int = IC_EMB_DIM,
    ):
        super().__init__()
        self.n_steps   = n_steps
        self.state_dim = state_dim

        # Embeddings
        self.t_embed  = SinusoidalEmbedding(t_emb_dim)
        self.ic_embed = ICEncoder(ic_emb_dim)

        # Input projection
        input_dim = n_steps * state_dim + t_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        # Hidden layers with FiLM conditioning on IC
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(ic_emb_dim, hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.act = nn.SiLU()

        # Output
        self.output_proj = nn.Linear(hidden_dim, n_steps * state_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        u_t: torch.Tensor,   # (B, N_STEPS, STATE_DIM)  interpolated trajectory
        t:   torch.Tensor,   # (B,)                      FM time
        ic:  torch.Tensor,   # (B, 3)                    (sin t0, cos t0, omega0)
    ) -> torch.Tensor:
        """
        Returns:
            v : shape (B, N_STEPS, STATE_DIM)  predicted velocity
        """
        B = u_t.shape[0]

        t_emb  = self.t_embed(t)                         # (B, t_emb_dim)
        ic_emb = self.ic_embed(ic)                       # (B, ic_emb_dim)

        u_flat = u_t.reshape(B, -1)                      # (B, N*state_dim)
        x      = torch.cat([u_flat, t_emb], dim=1)       # (B, N*3 + t_emb)

        h = self.input_proj(x)                           # (B, hidden_dim)

        for linear, film in zip(self.hidden_layers, self.film_layers):
            h = self.act(linear(h))
            h = film(h, ic_emb)                          # FiLM modulation

        out = self.output_proj(h)                        # (B, N*state_dim)
        return out.reshape(B, self.n_steps, self.state_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> ConditionalVelocityFieldIC:
    model = ConditionalVelocityFieldIC().to(device)
    print(f"Case 4b model: {model.count_parameters():,} parameters")
    print(f"  Conditioning on: (sin theta0, cos theta0, omega0)  -- shape (3,)")
    return model


if __name__ == "__main__":
    m = build_model()
    u  = torch.randn(4, N_STEPS, STATE_DIM)
    t  = torch.rand(4)
    ic = torch.randn(4, 3)    # (sin t0, cos t0, omega0)
    v  = m(u, t, ic)
    assert v.shape == u.shape
    print(f"Input u shape : {u.shape}")
    print(f"IC    shape   : {ic.shape}")
    print(f"Output shape  : {v.shape}")
    print("model.py OK")