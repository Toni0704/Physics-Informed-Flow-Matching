# HPC Handoff — pure_pcfm final N=20 run

You've already done two rounds on this repo (`runs/hpc_eval/`, `runs/hpc_eval_32bb58a/`) — thank
you, both were extremely useful. This round is much smaller: **one script, no new ingredients.**

## What's needed

Just `run_pure_pcfm_n20.sh` — the final, paper-standard N=20 run of `pure_pcfm` (vanilla + PCFM)
against the guard-fallback fix from `32bb58a`. Your last run (N=10, debug) confirmed the fix works
well: diverged samples dropped 7/10 → 2/10, guard trips 184 → 39. We now need the same thing at
N=20 so the number is actually paper-standard, with any guard-tripped samples identified so they
can be excluded from the reported mean rather than silently averaged in.

**`cond_pcfm` is not part of this ask.** Your `32bb58a` re-run confirmed both fixes work exactly as
predicted (VRAM 20.2GB → 10.3GB, Data MSE 26,600 → 13.48), but also surfaced a *third*, still
unidentified systematic defect (the ~10% cross-seed uniformity, Phys MSE still 7 orders above the
GT floor). That needs root-causing before another run is useful — we'll come back to you for that
separately once we've made some headway locally.

## Steps

```bash
cd Physics-Informed-Flow-Matching   # your existing clone
git pull                             # picks up run_pure_pcfm_n20.sh (5820e1b) and nothing else new
git log -1 --oneline                 # sanity-check
```

Same checkpoint, same data, same env as last time — nothing new to fetch. Just:

```bash
export PCFM_DATA=/path/to/ns_nw10_nf100_s64_t50_mu0.001.h5      # same file as before
export PCFM_CKPT_UNCOND=/path/to/latest.pt                      # same ns_uncond checkpoint (step 33400)
export PCFM_OUTDIR=/path/to/output/dir                          # anywhere writable

bash run_pure_pcfm_n20.sh
```

(No `PCFM_CKPT_COND` needed — this script only touches `pure_pcfm`.) Should take well under an
hour: it's the same per-sample cost as your N=10 debug run, just twice the samples, and `pure_pcfm`
was never the memory-bound one (peaked at ~13.9GB on your A40 last time).

## What to send back

- `$PCFM_OUTDIR/pure_pcfm_n20/metrics/pure_pcfm_vanilla.txt` and `pure_pcfm_none.txt` (both N=20)
- The script's own trip-count line at the end (`grep -c "... guard tripped" pure_pcfm_n20_trace.log`)
- If it's convenient: which sample indices (if any) show a Data MSE far above the rest in
  `pure_pcfm_none.txt` (last time the cutoff was roughly two orders of magnitude — clean samples
  sat under ~35, diverged ones were 1e3+) — saves us a round-trip identifying them ourselves, but
  not essential, we can do it from the raw numbers either way.

## Questions

Same as before — ask directly rather than guess if anything doesn't resolve.
