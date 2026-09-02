# NS2D re-run on `32bb58a` — verifying the three fixes

Re-runs `cond_pcfm` (N=20) and the `pure_pcfm` divergence debug (N=10) against commit `32bb58a`,
which fixes the three defects found in the `5792c96` run. Same hardware (2 × A40), same checkpoints
(`w_scale = 2.1572`, uncond step 33400), same data, same commands — see
[`../hpc_eval/README.md`](../hpc_eval/README.md) for the reproduction recipe and the baseline numbers
this is compared against.

**All three fixes work. Two of them work very well. `cond_pcfm` is still not usable, and the
un-addressed half of Finding 3 is now the dominant remaining problem.**

Reference floor for Phys MSE: ground truth's own spectral residual is **2.007e-04**
(`metrics/ground_truth.txt`, N=20). It is not zero because recorded frames are 1000× coarser than the
solver's internal step.

## Finding 1 — float64 → float32: confirmed

| | `5792c96` | `32bb58a` |
| --- | --- | --- |
| `cond_pcfm` peak VRAM | 20,183 MiB | **10,337 MiB** |

Exactly the predicted 2× drop. **10.3 GB fits a T4's 14.56 GB**, so `cond_pcfm` was never actually
blocked on hardware — the entire premise of the HPC handoff was one `.to(torch.float64)`. Step time
also improved (~2.5 s → ~1.7 s), cutting the N=20 run from ~2.5 h to ~1.9 h.

## Finding 2 — correct `u0`: confirmed, ~1975×

`cond_pcfm`, N=20:

| Metric | `5792c96` | `32bb58a` | |
| --- | --- | --- | --- |
| Data MSE | 2.663e+04 | **13.48** | ~1975× |
| Data NRMSE (%) | 4021.7 | **90.5** | |
| Phys MSE (spectral) | 1.464e+08 | 1.249e+04 | ~11700× |
| IC MSE | 1.985e-15 | 1.229e-15 | (already exact) |
| Mass drift MSE | 2.130e-10 | 1.443e-10 | (already exact) |

The 2-sample A/B in the previous run predicted 13.03 / 13.97; the full N=20 mean landed at **13.48**.

### But `cond_pcfm` is still broken

Data MSE now spans 12.95–14.22 — **a ~10% spread across 20 independent seeds.** That is the same
suspicious uniformity that flagged the `u0` bug, so a *second* systematic defect remains. Phys MSE
1.249e+04 is still seven orders above the 2.007e-04 floor, and NRMSE ~90% means the prediction is
still roughly as wrong as the field is large. **Do not put this in the results table.** The `u0` fix
removed a 2000× error sitting on top of a problem that is still there underneath.

## Finding 3 — guard falls back to vanilla: large improvement, and a clear remaining gap

`pure_pcfm`, N=10:

| | `5792c96` | `32bb58a` |
| --- | --- | --- |
| Samples diverged | 7 / 10 | **2 / 10** |
| Guard trips | 184 | 39 |
| Mean Data MSE | 2.894e+15 | 998.7 |

### The 8 healthy samples are now a real result

Against vanilla sampling on the *same* checkpoint (`debug_guards/metrics/pure_pcfm_vanilla.txt`):

| | PCFM (8 non-diverged) | Vanilla (all 10) | |
| --- | --- | --- | --- |
| Data MSE | 1.073 | 0.511 | ~2× worse |
| Data NRMSE (%) | 25.30 | 17.30 | worse |
| Phys MSE (spectral) | 0.0612 | 0.0061 | ~10× worse |
| **IC MSE** | **~1e-14** | 0.129 | **~13 orders better** |
| **Mass drift MSE** | **~1e-11** | 2.03e-3 | **~8 orders better** |

This is the method behaving as advertised: the constraints it targets are enforced to machine
precision, at roughly 2× in data accuracy.

Worth discussing: the **independent** spectral residual gets ~10× *worse*. Enforcing IC + global mass
exactly is pushing the trajectory further off the PDE, not closer to it. That is a substantive result
about the method rather than a bug, and it is a fair question for the paper — PCFM's projection target
(IC + 49 mass rows) is not the same thing as PDE consistency.

### The 2 that still diverge — the deferred timing problem

| Sample | Data MSE | IC MSE | onset | 1st trip | trips |
| --- | --- | --- | --- | --- | --- |
| 4 | 7092.5 | 4.747e+03 | 0.815 | 0.840 | 32 |
| 9 | 2885.7 | 2.518e+02 | 0.910 | 0.965 | 7 |

Both improved by 12 orders of magnitude (sample 4 was 7.87e+15), so the new fallback genuinely
contains the explosion. Two things to note:

1. **The fallback costs the constraint.** These two samples have IC MSE 4747 and 252, while every
   healthy sample sits at ~1e-14. Returning `v_flat` skips the projection, so on a sample that trips
   repeatedly the constraint is simply never enforced — the guard stops the blow-up by abandoning the
   method for that sample. That is the right emergency behaviour, but it means a tripped sample is
   not a PCFM sample and should not be averaged in as one.
2. **The timing gap the commit deliberately deferred is still the binding constraint.** Onset 0.815 vs
   first trip 0.840; onset 0.910 vs 0.965. The guard still fires 5–25 steps after divergence starts,
   because it compares against an absolute threshold (`50·‖u₀‖`) while the runaway is multiplicative.
   Drift accumulates through those steps before anything reacts.

Suggested next step, unchanged from the prior report: trigger on the *growth rate* of `‖u‖` between
consecutive steps rather than absolute magnitude. That would fire near onset instead of 20+ steps
later, and would let the fallback preserve the projection on the many steps that are still healthy.
It needs per-PDE tuning since the guard is shared with Darcy3D/1D, which is why `32bb58a` scoped it
out.

## Files

```text
metrics/ground_truth.txt                     GT spectral-residual floor, N=20 (2.007e-04)
metrics/conditioned_pcfm.txt                 cond_pcfm, N=20
figures/conditioned_pcfm.png                 GT / prediction / error triptych
debug_guards/metrics/pure_pcfm_none.txt      pure_pcfm, N=10, interp=none
debug_guards/metrics/pure_pcfm_vanilla.txt   vanilla baseline, same checkpoint
debug_guards/figures/*.png                   triptychs for both
debug_guards/pcfm_debug_trace.log            402 KB, PCFM_DEBUG_GUARDS=1 per-step trace
```
