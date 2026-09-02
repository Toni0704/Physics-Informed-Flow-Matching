#!/usr/bin/env bash
# Final N=20 pure_pcfm evaluation against the guard-fallback fix (32bb58a).
# Run this on Kaggle (or anywhere with the ns_uncond checkpoint + T4-class GPU) --
# pure_pcfm already fits comfortably (~13.9GB peak per the HPC report), no OOM
# concern here unlike cond_pcfm. See RESULTS.md footnote 3 / EXPERIMENTS_STATUS.md
# item 3 for why this re-run is needed: the earlier debug run was only N=10.
set -euo pipefail

: "${PCFM_DATA:?Set PCFM_DATA to the path of ns_nw10_nf100_s64_t50_mu0.001.h5}"
: "${PCFM_CKPT_UNCOND:?Set PCFM_CKPT_UNCOND to the path of latest.pt (ns_uncond backbone)}"
: "${PCFM_OUTDIR:?Set PCFM_OUTDIR to a writable output directory (use /kaggle/working/... on Kaggle)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/NS_2D"

PCFM_DEBUG_GUARDS=1 python experiments/evaluate.py --technique pure_pcfm --num-samples 20 \
    --ckpt-uncond "$PCFM_CKPT_UNCOND" \
    --data-test "$PCFM_DATA" \
    --n-step 200 --interp none \
    --outdir "$PCFM_OUTDIR/pure_pcfm_n20" \
    2>&1 | tee pure_pcfm_n20_trace.log

echo
echo "Done. Two files matter:"
echo "  $PCFM_OUTDIR/pure_pcfm_n20/metrics/pure_pcfm_vanilla.txt   (vanilla baseline, N=20)"
echo "  $PCFM_OUTDIR/pure_pcfm_n20/metrics/pure_pcfm_none.txt      (PCFM, N=20)"
echo
echo "In pure_pcfm_none.txt, any sample with Data MSE >> the rest (earlier runs put the"
echo "cutoff around 100 -- clean samples were < 35, diverged ones were 1e3+) is a"
echo "guard-tripped sample: the fallback contained the blow-up but never enforced the"
echo "constraint on it (check its IC MSE column -- ~1e-14 on clean samples, orders of"
echo "magnitude higher on tripped ones). Exclude those from the reported mean and note"
echo "how many/which indices, mirroring the 'N of 20 clean' convention used before this fix."
echo
echo "Guard trip count for reference (should be << the pre-fix 184, per-N=10):"
grep -c "MAGNITUDE guard tripped\|RESIDUAL guard tripped" pure_pcfm_n20_trace.log || echo "0"
