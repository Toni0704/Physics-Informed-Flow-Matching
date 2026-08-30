# NS2D HPC Run — the 2 experiments blocked on Kaggle

Completes the two experiments described in [`HPC_HANDOFF.md`](../../HPC_HANDOFF.md): `cond_pcfm`
(blocked on T4 VRAM) and the `pure_pcfm` divergence debug. Run 2026-08-30 against commit `5792c96`
on 2 × NVIDIA A40 (47.6 GB each), one experiment per GPU.

**Both ran to completion. Both surfaced defects that need fixing before the numbers mean anything** —
see [Findings](#findings). The metrics below record *what the current code does*, and are not results
to put in the paper table.

## Reproducing

```bash
export PCFM_DATA=$PWD/ns_data/ns_nw10_nf100_s64_t50_mu0.001.h5
export PCFM_CKPT_COND=$PWD/NS_2D/best_fm_conditioned.pt     # w_scale = 2.1572
export PCFM_CKPT_UNCOND=$PWD/NS_2D/latest.pt                # step 33400
export PCFM_OUTDIR=$PWD/runs/hpc_eval

CUDA_VISIBLE_DEVICES=0 bash ./run_cond_pcfm.sh        # ~2.5 h, peak 20.2 GB
CUDA_VISIBLE_DEVICES=1 bash ./run_pure_pcfm_debug.sh  # ~1 h,   peak 13.9 GB
```

Two things to know before running:

- **`requirements.txt`'s `neuraloperator==0.3.0` pin is load-bearing for `pure_pcfm`.** Under 1.0.2 the
  unconditioned checkpoint does not load, and not because of key renaming — the architecture differs
  (channel MLPs added to the FNO blocks, lifting width 256 → 64, `in_channels` 20 → 23, 21 vs 45
  tensors). A key-remapping shim would silently build a different model. `cond_pcfm` is unaffected:
  `NSVelocityNet_FiLM_ICF` loads `ns_model_film_ic.py` by file path precisely to avoid importing
  neuralop.
- **`evaluate.py` skips rather than fails on a missing checkpoint.** Watch for
  `[skip] <technique>: checkpoint not found`, or a run exits 0 having done nothing.

Checkpoints and datasets are gitignored — get the checkpoints from the team. `run_pure_pcfm_debug.sh`
writes its trace to `NS_2D/pcfm_debug_trace.log`; the copy kept here was moved into this directory
(`*.log` is gitignored repo-wide, so it carries an explicit un-ignore rule).

### Data

`ns_nw10_nf100_s64_t50_mu0.001.h5` (821 MB) — `a (10,64,64)`, `f (100,64,64)`,
`u (10,100,64,64,50)`, float32. **Regenerable bit-exactly** from the in-repo generator:

```bash
python -m datasets.generate_ns_2d --root ns_data --nw 10 --nf 100 \
    --s 64 --t 49 --steps 50 --mu 1e-3 --seed 42 --delta 1e-3
```

Verified: ICs and forcing both reproduce with `max|diff| = 0.000e+00` against the file the team
supplied. The train split (`--nw 50 --nf 50`, 2.0 GB, ~24 min on one A40) regenerates the same way;
neither experiment needs it, as both are eval-only.

## Results

### `cond_pcfm` — N=20, 200 steps (`metrics/conditioned_pcfm.txt`)

| Metric | Mean |
| --- | --- |
| Data MSE | 2.663e+04 |
| Data NRMSE (%) | 4021.7 |
| Phys MSE (spectral) | 1.464e+08 |
| IC MSE | 1.985e-15 |
| Mass drift MSE | 2.130e-10 |

Constraints are enforced to machine precision; the trajectory is nonetheless garbage. Per-sample Data
MSE spans only 26,234–26,928 — **a spread under 3% across 20 independent noise seeds.** That
uniformity says systematic defect, not stochastic divergence. Cause identified and confirmed —
see Finding 2.

Reference: ground truth's own spectral residual floor is **Phys MSE ≈ 2.2–3.0e-04** (recorded frames
are 1000× coarser than the solver's internal step, so GT is not zero). Quote model Phys MSE against
that floor.

### `pure_pcfm` — N=10, 200 steps, `--interp none` (`debug_guards/metrics/pure_pcfm_none.txt`)

**7 of 10 samples diverge.** Guard trips: **184 MAGNITUDE, 0 RESIDUAL.**

| Sample | Data MSE | divergence onset | 1st guard trip | trips |
| --- | --- | --- | --- | --- |
| 1, 3, 8 | 0.88 – 4.36 | — | — | 0 |
| 5 | 3.29e+01 | 0.960 | — | 0 |
| 9 | 6.41e+04 | 0.890 | 0.935 | 13 |
| 7 | 3.64e+05 | 0.880 | 0.920 | 16 |
| 2 | 1.37e+12 | 0.805 | 0.830 | 34 |
| 6 | 1.13e+13 | 0.795 | 0.820 | 36 |
| 4 | 7.87e+15 | 0.770 | 0.790 | 42 |
| 10 | 2.11e+16 | 0.765 | 0.785 | 43 |

Vanilla sampling on the *same* checkpoint is healthy (mean Data MSE 0.694, NRMSE 20.1%,
`debug_guards/metrics/pure_pcfm_vanilla.txt`) — so this is PCFM-specific, not a bad checkpoint.

## Findings

### 1. `cond_pcfm`'s OOM is a `float64` cast, not a hardware ceiling

`evaluate.py:pcfm_sample_with_physics_one`'s `hfunc_single` casts to float64 before the residual, so
`jacrev` builds the 4145 × 204,800 Jacobian in double. Measured with the same residual and state,
only the `hfunc` differing:

| `hfunc` | Peak VRAM | Time |
| --- | --- | --- |
| denormalize → **float64** → float32 (as shipped) | **20.45 GB** | 2.81 s |
| float32 throughout (what `pure_pcfm` does) | **10.27 GB** | 1.73 s |

Exactly 2× — 3.40 GB → 6.80 GB for `J`, matching the ~6.33 GB allocation that was failing on the T4.
That is the whole reason `pure_pcfm` fits in 14.56 GB and `cond_pcfm` does not (confirmed in this run:
13.9 GB vs 20.2 GB actual). Nothing to do with conditioning, model size, or GC.

**Better than downgrading precision: cache `J`.** It is provably constant — computed at two states
differing by 100× it is bit-identical (`max|J1-J2| = 0.000e+00`), because the IC+mass residual is
affine in `u`. It is also **0.0478% dense** (405,504 nonzeros of 848.9M) with only **3 distinct nonzero
values** (`±w_s·Δx·Δy` and `w_s`): 3.40 GB dense vs 1.6 MB sparse. Per Newton step only the Jacobian is
expensive — the residual itself is a slice and a sum. Computing it once per sample (and reusing the
`JJᵀ` factorization, also constant) removes the memory wall and the 200× recompute **with identical
numerics**, so results stay comparable with the 10 already-completed ones.

### 2. `cond_pcfm` passes the wrong `u0` to the projection — confirmed, ~2000× error

`NS_2D/experiments/evaluate.py:218`:

```python
proj_v = pcfm_2d_batched(ut=ut_hwt, ..., u0=ut_hwt, dt=dt, ...)
                                          # ^^^^^^^ current state, not the noise seed
```

`pcfm_sample` uses `u0_flat` for the final interpolation
`ut_interp = (1-t_next)*u0_flat + t_next*u_corr` **and** for the guard's
`ref_scale = max(u0_flat.norm(), 1e-8)`. It must be the fixed original noise seed — which
`pcfm_sample_with_physics_one` never retains from `i=0`. `FFM_NS_sampler.pcfm_sample` (the `pure_pcfm`
path) correctly passes `u0=u0`.

`pcfm_sampling.py`'s own comment warns against exactly this: anchoring on the running state means
"if a prior step let contamination through, ut1 is already inflated, and folding it into the reference
would raise the threshold along with the contamination."

A/B on 2 samples, identical seeds, only `u0` changed:

| | Data MSE | Data NRMSE (%) | Phys MSE | IC MSE |
| --- | --- | --- | --- | --- |
| shipped (`u0` = running state) | 26720.1 / 26782.8 | 4012 / 4111 | 1.37e+08 / 6.83e+07 | 5.3e-15 / 4.3e-15 |
| **fixed** (`u0` = noise seed) | **13.03 / 13.97** | **88.6 / 93.9** | **1.27e+04 / 1.25e+04** | 1.2e-15 / 1.4e-15 |

**~2000× reduction in Data MSE, ~10,000× in Phys MSE**, with IC and mass still at machine precision.
This accounts for essentially all of the 26,600. The one-line fix is to capture the initial noise
before the loop and pass it every step.

Note the fixed run is still worse than vanilla sampling (NRMSE ~89% vs ~20% for the unconditioned
vanilla baseline), so PCFM is not yet *helping* here — but it is no longer catastrophically broken.
That residual gap is the next thing to chase, and Finding 3 is the likely lead.

### 3. The divergence guards fire far too late, and cannot stop a runaway

Two independent problems, both visible in `debug_guards/pcfm_debug_trace.log`:

**(a) An absolute threshold against a multiplicative runaway.** The guard trips at
`‖u‖ > 50·‖u₀‖ ≈ 22,600`. Sample 10 begins diverging at t=0.765 but does not cross that until t=0.785 —
by which point it is growing ~10× every 4 steps. Onset precedes the first trip by 20–45 steps in every
diverged sample.

**(b) Tripping the guard does not stop the divergence.** With `newtonsteps=1`, `break` leaves
`u_corr = ut1` *unprojected*, so the already-exploding velocity goes straight into the Euler step. The
guard only prevents PCFM from making things worse. **184 trips produced zero saves.**

**The root cause is not the linear algebra** — `RESIDUAL` never tripped once, so the projection drives
`h` down correctly every time. It is a **model–state feedback runaway**: `‖v‖/‖u‖` sits flat at ~1.85
through t=0.70, then climbs 3.7 → 5.6 → 7.5 as the state leaves the training distribution and the
network extrapolates. Onset is always t ≈ 0.77–0.96, consistent with the mechanism: near t=1 the
interpolation weights `u_corr` almost fully, so the `u0` term no longer damps error.

Suggested direction: guard on the *growth rate* of `‖u‖` between steps rather than absolute magnitude,
and make the response actually damp (revert to the vanilla step, or halt) rather than merely skipping
the projection.

## Files

```text
metrics/conditioned_pcfm.txt                 cond_pcfm, N=20
figures/conditioned_pcfm.png                 GT / prediction / error triptych
debug_guards/metrics/pure_pcfm_none.txt      pure_pcfm, N=10, interp=none
debug_guards/metrics/pure_pcfm_vanilla.txt   vanilla baseline, same checkpoint
debug_guards/figures/*.png                   triptychs for both
debug_guards/pcfm_debug_trace.log            422 KB, PCFM_DEBUG_GUARDS=1 per-step trace
test_u0_fix.py                               Finding 2 A/B; standalone, edits no repo code
u0_ab_test.txt                               its raw output (the table in Finding 2)
```
