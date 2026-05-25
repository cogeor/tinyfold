#!/usr/bin/env bash
# Clone DiffDock-PP at a pinned commit into benchmarks/baselines/diffdock_pp/repo/.
# Idempotent: if repo/ already exists at the pinned commit, exits silently.
set -euo pipefail

# Pin to a specific upstream commit so re-clones are reproducible.
# After first clone, run `git -C repo rev-parse HEAD` and paste the hash here.
PINNED_COMMIT="25a28900736c0730821e45265ee8e409751c358a"   # tip of main, 2024-2025 era
REPO_URL="https://github.com/ketatam/DiffDock-PP.git"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/repo"

if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  if [ -n "$PINNED_COMMIT" ]; then
    current=$(git rev-parse HEAD)
    if [ "$current" = "$PINNED_COMMIT" ]; then
      echo "DiffDock-PP already at pinned commit $PINNED_COMMIT"
      exit 0
    fi
    git fetch origin
    git checkout "$PINNED_COMMIT"
  else
    echo "DiffDock-PP repo already cloned at $(git rev-parse --short HEAD); pin commit in clone.sh"
  fi
  exit 0
fi

echo "Cloning DiffDock-PP into $REPO_DIR ..."
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
if [ -n "$PINNED_COMMIT" ]; then
  git checkout "$PINNED_COMMIT"
fi
echo "Done. Commit: $(git rev-parse --short HEAD)"
echo "REMEMBER: paste the full commit hash into clone.sh PINNED_COMMIT."
