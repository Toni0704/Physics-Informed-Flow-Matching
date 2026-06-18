# Burgers' 1D — Imposing Physics, IC & BC in Flow-Matching PDE Surrogates

This directory is the **1D Burgers'** dataset module of the wider study on enforcing
physics and initial/boundary conditions (IC/BC) in generative flow-matching PDE
surrogates. Each dataset in the parent project lives in its own subdirectory; this
one is self-contained.

## The four techniques compared

| # | Technique | Where physics enters | Backbone | Checkpoint |
|---|-----------|----------------------|----------|------------|
| 1 | **PBFM** — Physics-Based Flow Matching | **training** (residual loss, ConFIG) + FiLM | conditioned FNO | `weights/best_pbfm.pt` |
| 2 | **PCFM** sampling + FiLM | **sampling** (hard projection) | conditioned FNO | `weights/best_fm_conditioned.pt` |
| 3 | **Pure PCFM** | **sampling** (physics+IC+BC as hard constraints) | *unconditioned* FFM | `weights/best_fm_uncond.pt` |
| 4 | **Vanilla** + FiLM | none (baseline) | conditioned FNO | `weights/best_fm_conditioned.pt` |

Techniques 2 and 4 share a single conditioned checkpoint (they differ only in the
sampler). Technique 3 uses the PCFM repo's own unconditioned model.

## Directory layout

```
Burgers_1D/
├── README.md
├── requirements.txt
├── src/                         # shared library
│   ├── __init__.py              # add_pcfm_to_path(): locates the PCFM repo
│   ├── dataset.py               # BurgersConditionedDataset (normalises to [-1,1])
│   ├── models.py                # FiLM-conditioned FNO (PatchedFNO + FiLM)
│   ├── physics.py               # Burgers residual via PCFM's Residuals
│   └── losses.py                # flow_matching_loss, pbfm_loss, logit_normal_pdf
├── experiments/
│   ├── generate_data.py         # regenerate the .h5 datasets (not committed)
│   ├── train_fm_uncond.py       # technique 3 backbone (runs PCFM's main.py)
│   ├── train_fm_conditioned.py  # techniques 2 & 4 backbone
│   ├── train_pbfm.py            # technique 1
│   └── evaluate.py              # all four techniques -> results/
├── weights/                     # checkpoints (created by the training scripts)
├── results/
│   ├── metrics/                 # conditioned_pbfm.txt, conditioned_pcfm.txt,
│   │                            #   conditioned_vanilla.txt, pure_pcfm.txt
│   └── figures/                 # matching .png comparison plots
└── external/
    ├── README.md                # attribution + license for the vendored dep
    └── PCFM-main/               # the PCFM repo, vendored verbatim (MIT/Apache-2.0)
```

## Setup

```bash
pip install -r requirements.txt
```

The **PCFM** repository is a dependency but is *not* pip-installable (it has no
`setup.py`). It is **already vendored** in this directory at `external/PCFM-main/`
(a verbatim, unmodified copy under its MIT / Apache-2.0 licenses — see
`external/README.md` for attribution), so no extra step is needed.

If you prefer not to ship it, delete `external/` and point an environment variable
at your own clone instead:

```bash
export PCFM_REPO_PATH=/path/to/PCFM-main
```

`src/add_pcfm_to_path()` resolves either location, so the experiment scripts can be
run from anywhere. PCFM source: <https://github.com/cpfpengfei/PCFM>.

## Step 1 — Generate the data (datasets are not committed)

The HDF5 datasets are reproduced from PCFM's numerical solver rather than shipped:

```bash
python experiments/generate_data.py --nproc 4
```

This writes both splits into `<PCFM>/datasets/data/` (the canonical location every
other script reads from, and where PCFM's own config expects them):

* `burgers_train_nIC80_nBC80.h5` — 80 initial conditions × 80 boundary conditions (`seed = 42`)
* `burgers_test_nIC30_nBC30.h5`  — 30 × 30 (`seed = 0`)

**What the data is.** Each sample solves the **inviscid Burgers' equation**
`u_t + (u²/2)_x = 0` on `x,t ∈ [0,1]` with a **Godunov** finite-volume scheme
(`Nx = Nt = 100`, giving 101×101 grids). The initial condition is a smoothed
descending front `u(x,0) = 1 / (1 + exp((x − p)/ε))` with width `ε = 0.02` and front
location `p ∼ U(0.2, 0.8)`; the left boundary is a Dirichlet value `u(0,t) = u_bc`
with `u_bc ∼ U(0,1)`, and the right boundary is outflow (zero-gradient). The arrays
stored are `u (n_ic, n_bc, 101, 101)`, `ic (n_ic,)`, `bc (n_bc,)`, plus grids `x`, `t`.

## Step 2 — Train

```bash
# Technique 3 backbone — delegates to PCFM's main.py (GP-prior FFM), then exports
#   the checkpoint to weights/best_fm_uncond.pt
python experiments/train_fm_uncond.py

# Techniques 2 & 4 backbone — FiLM-conditioned vanilla flow matching
#   -> weights/best_fm_conditioned.pt
python experiments/train_fm_conditioned.py

# Technique 1 — PBFM (physics-in-training, ConFIG)
#   -> weights/best_pbfm.pt
python experiments/train_pbfm.py
```

Training faithfully reproduces the notebook configurations: the conditioned model
uses Adam (lr 3e-4), gradient accumulation to an effective batch of 128, grad-norm
clipping at 1.0, and an EMA (decay 0.99) started after 1000 iterations, with
validation/early-stopping (patience 15). PBFM uses AdamW (lr 3e-5, betas 0.5/0.999),
a 1→4 unrolling curriculum, logit-normal FM weighting, ConFIG gradient updates with
a NaN fallback, EMA decay 0.999, and validation/early-stopping (patience 20). The
unconditioned model is trained by PCFM's own pipeline so it cannot drift from the
reference.

## Step 3 — Evaluate

```bash
python experiments/evaluate.py                     # all four techniques
python experiments/evaluate.py --technique cond_pcfm   # or one at a time
```

For each technique this writes a metrics table to `results/metrics/<technique>.txt`
and a comparison figure to `results/figures/<technique>.png`. The conditioned
techniques report per-sample **Data**, **Physics**, **IC** and **BC** MSE/NRMSE;
pure PCFM reports the constraint-abidement table (Full Constraint / Original Burgers
/ IC / Mass) for Ground Truth vs. Vanilla vs. PCFM.

> The metric and figure files committed under `results/` are the recorded outputs
> from the original notebook runs. `evaluate.py` regenerates files with the same
> names, but the numbers will not match exactly: it draws the first `N` test
> samples deterministically (the notebooks used a randomly shuffled validation
> batch) and the generative prior (Gaussian noise / GP sample) is stochastic. The
> `pure_pcfm.txt` it writes contains one constraint table per sample; the committed
> file shows a single representative sample. Schema and columns are identical.


