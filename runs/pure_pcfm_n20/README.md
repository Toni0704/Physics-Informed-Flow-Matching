# `pure_pcfm` final N=20 run (`ec25807`)

The paper-standard N=20 run requested in [`HPC_HANDOFF.md`](../../HPC_HANDOFF.md), via
`run_pure_pcfm_n20.sh`. Same checkpoint (ns_uncond, step 33400), same data, same environment as the
two previous rounds. Run 2026-09-02 on one A40; peak 13,945 MiB, ~1 h wall clock.

**Guard-tripped samples: 1, 2, 18 (3 of 20). Total guard trips: 70.**

> **Note on timing for a Kaggle re-run:** the handoff estimated "well under an hour." On an A40 this
> took ~60 min (~3 min/sample). A T4 will be slower — budget accordingly.

## ⚠️ Read this before computing the reported mean

The handoff's criterion is *guard-tripped samples excluded*. The script's suggested shortcut —
"any sample with Data MSE >> the rest, cutoff around 100" — **does not implement that criterion**,
and on this run the two disagree in a way that changes the headline by 15×.

**Sample 13 has Data MSE 218.9 but zero guard trips.** Its IC MSE is 7.45e-13 and mass drift
3.01e-08 — the constraint *was* enforced. It is a legitimately PCFM-processed sample that performed
badly, not a contained blow-up, and it belongs in the mean.

Guard-trip status was determined from `pure_pcfm_n20_trace.log`, not inferred from Data MSE:

| Sample | Data MSE | IC MSE | guard trips | 1st trip | classification |
| --- | --- | --- | --- | --- | --- |
| 1 | 4305.2 | 1.21e+03 | 15 | t=0.925 | guard-tripped — exclude |
| 2 | 7361.5 | 6.04e+03 | 44 | t=0.780 | guard-tripped — exclude |
| 18 | 3050.6 | 5.33e+02 | 11 | t=0.945 | guard-tripped — exclude |
| **13** | **218.9** | **7.45e-13** | **0** | — | **clean projection, bad result — keep** |
| all others | 0.275 – 2.19 | 7e-16 – 7e-14 | 0 | — | clean |

The IC MSE column is the reliable discriminator: ~1e-15 on clean samples, 1e+2–1e+3 on tripped ones.
Sample 13 sits at 7e-13 — about 1000× the clean samples but fifteen orders below the tripped ones.

## Results

| Policy | Data MSE | NRMSE (%) | Phys MSE | IC MSE | Mass drift |
| --- | --- | --- | --- | --- | --- |
| all 20 | 747.6 | 289.7 | 1.332e+06 | 389.3 | 1.058 |
| **17 clean — exclude 1, 2, 18 (the stated criterion)** | **13.76** | **42.3** | **1129** | **5.31e-14** | **1.85e-09** |
| 16 — also dropping #13 (what the shortcut gives) | 0.935 | 23.06 | 0.0552 | 9.86e-15 | 7.78e-11 |

**The reported number should be 13.76 over 17 of 20.** Dropping sample 13 as well would be
cherry-picking — it alone accounts for essentially the entire gap between the last two rows.

### PCFM vs vanilla, N=20, same checkpoint

| | PCFM (17 clean) | Vanilla (all 20) | |
| --- | --- | --- | --- |
| Data MSE | 13.76 | 0.579 | ~24× worse |
| Data NRMSE (%) | 42.3 | 18.45 | worse |
| Phys MSE (spectral) | 1129 | 0.00688 | far worse |
| **IC MSE** | **5.31e-14** | 0.151 | **~12 orders better** |
| **Mass drift MSE** | **1.85e-09** | 1.99e-03 | **~6 orders better** |

Reference: ground truth's own spectral residual floor is 2.007e-04
(`../hpc_eval_32bb58a/metrics/ground_truth.txt`).

## How this compares to the N=10 debug run

| | N=10 (`32bb58a`) | N=20 (this run) |
| --- | --- | --- |
| Diverged / guard-tripped | 2 / 10 (20%) | 3 / 20 (15%) |
| Guard trips | 39 (3.9 per sample) | 70 (3.5 per sample) |
| Clean-sample Data MSE vs vanilla | ~2× worse | ~24× worse (17 clean) / ~1.6× (16) |

Divergence rate and trip density are consistent — the guard fix holds up at N=20.

**But the accuracy picture is worse than N=10 suggested**, and that is the main new information here.
N=10 showed clean samples at ~2× vanilla's Data MSE. That was optimistic: N=10 happened to contain no
sample-13-type case. The 16 non-tripped, non-degenerate samples do reproduce the ~2× figure
(0.935 vs 0.579), so nothing contradicts the earlier run — N=10 was simply too small to surface this
third mode.

## The finding worth acting on: a silent failure mode

Sample 13 is a failure with **no guard signature at all** — zero trips, constraint satisfied to
7e-13, and yet Data MSE 218.9 and Phys MSE 1.9e+04 (seven orders above the GT floor). No growth-rate
guard would catch it either, because nothing diverges: the projection converges, the constraints
hold, and the trajectory is simply wrong.

That is arguably more concerning than the divergences, which at least announce themselves. It means
the current QA story for `pure_pcfm` — "trust a sample unless the guard tripped" — is not sufficient,
and 1 in 20 samples can pass every internal check while being ~400× worse than vanilla. Worth
deciding whether the paper's reported mean needs an independent quality criterion (e.g. the spectral
residual against the GT floor) rather than only guard status.

## Files

```text
metrics/pure_pcfm_none.txt        PCFM, N=20, interp=none
metrics/pure_pcfm_vanilla.txt     vanilla baseline, N=20, same checkpoint
figures/*.png                     GT / prediction / error triptychs
pure_pcfm_n20_trace.log           801 KB, PCFM_DEBUG_GUARDS=1 per-step trace
```
