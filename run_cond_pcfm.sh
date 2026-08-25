#!/usr/bin/env bash
# See HPC_HANDOFF.md for full context. Run from anywhere; cd's into NS_2D itself.
set -euo pipefail

: "${PCFM_DATA:?Set PCFM_DATA to the path of ns_nw10_nf100_s64_t50_mu0.001.h5}"
: "${PCFM_CKPT_COND:?Set PCFM_CKPT_COND to the path of best_fm_conditioned.pt}"
: "${PCFM_OUTDIR:?Set PCFM_OUTDIR to a writable output directory}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/NS_2D"

python experiments/evaluate.py --technique cond_pcfm --num-samples 20 \
    --ckpt-cond "$PCFM_CKPT_COND" \
    --data-test "$PCFM_DATA" \
    --outdir "$PCFM_OUTDIR"

echo
echo "Done. Send back the terminal output above plus:"
echo "  $PCFM_OUTDIR/metrics/conditioned_pcfm.txt"
