#!/usr/bin/env bash
# Run pretrained DiffDock-PP on all 4 clean stratified bins (skips clean_le200
# because it only has 1 sample) at the paper's K=40 sampling protocol.
#
# This is the headline "DPP cliff" eval. Expected wall: 4-10 hours total on
# a 4070 Ti SUPER, dominated by the larger bins.
#
# Outputs land at benchmarks/predictions/diffdock_pp/clean_{bin}/{sample_id}.npz.
# Score with: python benchmarks/scripts/compute_metrics.py --model diffdock_pp.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"

K="${K:-40}"   # override with K=N env var if you want a faster trial run

echo "=== DiffDock-PP pretrained eval, K=$K samples per complex ==="
echo "=== Sequential over 4 clean stratified bins ==="
echo

for BIN in clean_200_400 clean_400_600 clean_600_1000 clean_ge1000; do
  echo
  echo "############################################################"
  echo "# $BIN  ($(date '+%H:%M:%S'))"
  echo "############################################################"
  VIRTUAL_ENV="$REPO_ROOT/.venv" "$PYTHON" \
    "$HERE/run_pretrained.py" \
      --split "$REPO_ROOT/benchmarks/splits/${BIN}.json" \
      --num_samples "$K"
done

echo
echo "=== All bins complete ($(date '+%H:%M:%S')) ==="
echo "Score with: python benchmarks/scripts/compute_metrics.py --model diffdock_pp"
