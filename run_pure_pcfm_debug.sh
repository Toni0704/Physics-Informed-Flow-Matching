#!/usr/bin/env bash
# See HPC_HANDOFF.md for full context. Run from anywhere; cd's into NS_2D itself.
set -euo pipefail

: "${PCFM_DATA:?Set PCFM_DATA to the path of ns_nw10_nf100_s64_t50_mu0.001.h5}"
: "${PCFM_CKPT_UNCOND:?Set PCFM_CKPT_UNCOND to the path of latest.pt (ns_uncond backbone)}"
: "${PCFM_OUTDIR:?Set PCFM_OUTDIR to a writable output directory}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/NS_2D"

PCFM_DEBUG_GUARDS=1 python experiments/evaluate.py --technique pure_pcfm --num-samples 10 \
    --ckpt-uncond "$PCFM_CKPT_UNCOND" \
    --data-test "$PCFM_DATA" \
    --n-step 200 --interp none \
    --outdir "$PCFM_OUTDIR/debug_guards" \
    > pcfm_debug_trace.log 2>&1

echo "Done. Guard trip count:"
grep -c "MAGNITUDE guard tripped\|RESIDUAL guard tripped" pcfm_debug_trace.log || echo "0"

echo
echo "Check $PCFM_OUTDIR/debug_guards/metrics/pure_pcfm_none.txt for any sample with an"
echo "absurdly large Data MSE (>100). If one exists, send back pcfm_debug_trace.log in full"
echo "along with the guard trip count above."
