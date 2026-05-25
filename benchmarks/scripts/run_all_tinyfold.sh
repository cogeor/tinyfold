#!/usr/bin/env bash
# Run the headline TinyFold checkpoints on all 4 clean stratified bins.
# Fast: ~5 s/sample at residue level; 30 samples * 4 bins * 3 checkpoints ~= 30 min total.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"

BINS="clean_200_400 clean_400_600 clean_600_1000 clean_ge1000"

# (ckpt_path, model_tag) per checkpoint
declare -A CHECKPOINTS=(
  ["tinyfold_phase_d"]="outputs/resfold/phase_d_n8600_full/resfold_s1_8K_20260524_085326/best_model.pt"
  ["tinyfold_f_medium"]="outputs/resfold/phase_f_medium/resfold_s1_80_20260525_013420/best_model.pt"
  ["tinyfold_f_medium_esm"]="outputs/resfold/phase_f_medium_esm/resfold_s1_80_20260525_024844/best_model.pt"
)

for TAG in "${!CHECKPOINTS[@]}"; do
  CKPT="${CHECKPOINTS[$TAG]}"
  echo
  echo "############################################################"
  echo "# $TAG  ($(date '+%H:%M:%S'))"
  echo "#   ckpt: $CKPT"
  echo "############################################################"
  # shellcheck disable=SC2086
  VIRTUAL_ENV="$REPO_ROOT/.venv" "$PYTHON" \
    "$HERE/eval_tinyfold.py" \
      --checkpoint "$CKPT" \
      --model_tag "$TAG" \
      --splits $BINS \
      --skip_existing
done

echo
echo "=== All TinyFold checkpoints done ($(date '+%H:%M:%S')) ==="
for TAG in "${!CHECKPOINTS[@]}"; do
  echo "Scoring $TAG..."
  VIRTUAL_ENV="$REPO_ROOT/.venv" "$PYTHON" \
    "$HERE/compute_metrics.py" --model "$TAG"
done
