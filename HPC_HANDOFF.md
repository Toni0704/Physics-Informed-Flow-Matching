# HPC Handoff — 2 Remaining NS2D Experiments

Two experiments from the PCFM study are blocked on Kaggle's free-tier GPU (T4, 14.56GB VRAM) and
need to run somewhere with more memory. Everything else in the study (10 of 12 core results,
NS2D + Darcy3D) is already done — see `RESULTS.md` for the full table.

## Compute needed

- **1 GPU, ≥24GB VRAM** (A100 40GB/80GB preferred — real headroom, not just barely clearing it)
- No multi-GPU needed — nothing in this codebase shards across GPUs
- A few hours of uninterrupted runtime is enough for both; no special scheduling needs

## Why these two need more memory

1. **`cond_pcfm`** — the PCFM Newton-projection step computes a dense Jacobian of the physics
   residual w.r.t. the full flattened state (64×64×50 grid). That computation peaks at ~19-20GB,
   which a 14.56GB T4 can't hold. Confirmed via repeated identical `CUDA out of memory` errors
   (same ~6.33GB allocation failing against a ~12.9GB baseline, every attempt, with the GPU
   otherwise confirmed idle) — this isn't a code bug or GPU contention, it's a genuine VRAM
   ceiling. The equivalent unconditioned technique (`pure_pcfm`) runs fine at the same problem
   size, for reasons not yet understood — worth keeping in mind if you have spare
   cycles/curiosity, but not required to unblock this run.
2. **`pure_pcfm` divergence debug** — not a memory issue, but benefits from a stable, longer
   session (no Kaggle tunnel disconnects) to iterate on the trace output if the first pass doesn't
   immediately explain the bug.

## Ingredients

### 1. Code
```bash
git clone https://github.com/Toni0704/Physics-Informed-Flow-Matching.git
cd Physics-Informed-Flow-Matching
git log -1 --oneline   # sanity-check: confirm this is the commit you expect
```

### 2. Environment
```bash
conda env create -f pcfm_env.yml
conda activate pcfm
```

### 3. Data
Only the NS2D **test** file is needed (both experiments are eval-only, no training data required):

- `ns_nw10_nf100_s64_t50_mu0.001.h5` — currently on Kaggle at
  `/kaggle/input/datasets/rishabhj74/10-100/ns_nw10_nf100_s64_t50_mu0.001.h5`

### 4. Checkpoints
These only exist on the Kaggle account's `/kaggle/working` right now — download them from the
Kaggle notebook's Output/Files panel (or `kaggle kernels output` via the Kaggle API) and transfer
to the HPC:

- `ns/weights/best_fm_conditioned.pt` — needed for `cond_pcfm`
- `ns/logs/ns_uncond/latest.pt` (226.8MB) — needed for the `pure_pcfm` debug run

## Running it

Set these three paths, then run either script from the repo's `NS_2D/` directory:

```bash
export PCFM_DATA=/path/to/ns_nw10_nf100_s64_t50_mu0.001.h5
export PCFM_CKPT_COND=/path/to/best_fm_conditioned.pt
export PCFM_CKPT_UNCOND=/path/to/latest.pt
export PCFM_OUTDIR=/path/to/output/dir   # anywhere writable

bash ../run_cond_pcfm.sh
bash ../run_pure_pcfm_debug.sh
```

(Scripts live at the repo root, alongside this file — `run_cond_pcfm.sh` and
`run_pure_pcfm_debug.sh`.)

## What to send back

- **`cond_pcfm`**: the full terminal output (prints per-sample Data MSE/NRMSE/Phys MSE/IC MSE/mass
  drift for N=20), plus `$PCFM_OUTDIR/metrics/conditioned_pcfm.txt`.
- **`pure_pcfm` debug**: the count from
  `grep -c "MAGNITUDE guard tripped\|RESIDUAL guard tripped" pcfm_debug_trace.log`, and — only if
  any sample's Data MSE in `$PCFM_OUTDIR/debug_guards/metrics/pure_pcfm_none.txt` comes back
  absurdly large (>100, say) — the full `pcfm_debug_trace.log` file itself. That log is what lets
  us diagnose why the existing magnitude/residual guards in `pcfm/pcfm_sampling.py` aren't
  catching the divergence.

## Questions

If anything here is unclear or a path doesn't resolve, the fastest path is to ask directly rather
than guess — these commands were validated against the code as of the commit above, but not
run end-to-end on a non-Kaggle machine yet.
