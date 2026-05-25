#!/usr/bin/env bash
# Set up an isolated Python venv for running DiffDock-PP inference.
#
# DiffDock-PP's published deps target torch 1.13 + CUDA 11.6 + Linux. We pin
# to torch 2.4 + CUDA 12.1 (works on Windows with prebuilt PyG wheels) and
# apply small Windows portability patches (see apply_patches.py).
#
# Idempotent: re-running with the venv already present prints a no-op message.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
VENV="$REPO_ROOT/.venv_diffdock_pp"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  echo "ERROR: 'uv' not found on PATH. Install uv first (https://astral.sh/uv)."
  exit 1
fi

if [ -d "$VENV" ]; then
  echo "Reusing existing venv: $VENV"
else
  echo "Creating venv: $VENV"
  "$UV_BIN" venv "$VENV" --python=3.10
fi

export VIRTUAL_ENV="$VENV"
echo "Installing torch 2.4 + CUDA 12.1 wheels..."
"$UV_BIN" pip install torch==2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "Pinning numpy<2 for torch 2.4 compat..."
"$UV_BIN" pip install "numpy<2"

echo "Installing PyG extensions..."
"$UV_BIN" pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

echo "Installing remaining DiffDock-PP deps..."
"$UV_BIN" pip install torch-geometric e3nn biopython biopandas tqdm pyyaml pandas \
    scikit-learn matplotlib tensorboard tensorboardX dill transformers fair-esm wandb

echo "Applying Windows portability patches..."
"$VENV/Scripts/python.exe" "$HERE/apply_patches.py"

echo
echo "Done. Activate the env with:"
echo "  source $VENV/Scripts/activate    # bash"
echo "  $VENV/Scripts/activate           # cmd"
echo
echo "Verify with:"
echo "  $VENV/Scripts/python.exe -c 'import torch, torch_geometric, e3nn; print(torch.cuda.is_available())'"
