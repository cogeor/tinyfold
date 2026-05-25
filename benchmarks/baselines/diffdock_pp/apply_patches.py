"""Patch DiffDock-PP source to run on Windows + modern dependencies.

Idempotent: re-running won't double-apply. Each patch is checked for an
in-place marker before applying.

Run AFTER clone.sh:
    python benchmarks/baselines/diffdock_pp/apply_patches.py

Currently applied patches:
  1. Replace bare `import resource` (Unix-only) with a try/except wrapper
     plus a no-op rlimit fallback. Affects main.py, main_confidence.py,
     main_generate_samples.py, main_inf.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE / "repo"

MARKER = "# WIN_PATCH:resource"

RESOURCE_PATCH_OLD = "import resource\n"
RESOURCE_PATCH_NEW = (
    f"{MARKER}\n"
    "try:\n"
    "    import resource  # Unix only\n"
    "except ImportError:\n"
    "    resource = None  # Windows: rlimit is N/A, default fd cap is high\n"
)

RESOURCE_BLOCK_OLD = (
    "    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
    "    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))\n"
)
RESOURCE_BLOCK_NEW = (
    "    if resource is not None:\n"
    "        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
    "        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))\n"
)


PYG_TRANSFORM_MARKER = "# WIN_PATCH:pyg_transform_forward"

PYG_TRANSFORM_OLD = (
    "class NoiseTransform(BaseTransform):\n"
    "    \"\"\"\n"
    "        Apply translation, rotation, torsional noise\n"
    "    \"\"\"\n"
    "    def __init__(self, args):"
)
PYG_TRANSFORM_NEW = (
    "class NoiseTransform(BaseTransform):\n"
    "    \"\"\"\n"
    "        Apply translation, rotation, torsional noise\n"
    "    \"\"\"\n"
    f"    {PYG_TRANSFORM_MARKER}\n"
    "    # torch_geometric>=2.4 made BaseTransform abstract w/ required forward().\n"
    "    # DPP defines __call__; alias forward to it for new-PyG compatibility.\n"
    "    def forward(self, data):\n"
    "        return self.__call__(data)\n"
    "\n"
    "    def __init__(self, args):"
)


def patch_pyg_transform(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if PYG_TRANSFORM_MARKER in text:
        return False
    if PYG_TRANSFORM_OLD not in text:
        return False
    text = text.replace(PYG_TRANSFORM_OLD, PYG_TRANSFORM_NEW, 1)
    file_path.write_text(text, encoding="utf-8")
    print(f"  patched NoiseTransform.forward: {file_path.relative_to(REPO)}")
    return True


def patch_resource(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if MARKER in text:
        return False  # already patched

    n_imports = text.count(RESOURCE_PATCH_OLD)
    n_blocks = text.count(RESOURCE_BLOCK_OLD)
    if n_imports == 0 and n_blocks == 0:
        return False  # nothing to patch
    new_text = text.replace(RESOURCE_PATCH_OLD, RESOURCE_PATCH_NEW, 1)
    new_text = new_text.replace(RESOURCE_BLOCK_OLD, RESOURCE_BLOCK_NEW, 1)
    file_path.write_text(new_text, encoding="utf-8")
    print(f"  patched resource import + rlimit block: {file_path.relative_to(REPO)}")
    return True


def main() -> int:
    if not REPO.exists():
        print(f"ERROR: {REPO} does not exist. Run clone.sh first.")
        return 1

    resource_targets = [
        REPO / "src" / "main.py",
        REPO / "src" / "main_confidence.py",
        REPO / "src" / "main_generate_samples.py",
        REPO / "src" / "main_inf.py",
    ]
    n_resource = sum(patch_resource(p) for p in resource_targets)
    n_pyg = patch_pyg_transform(REPO / "src" / "geom_utils" / "transform.py")
    if n_resource or n_pyg:
        print(f"Applied {n_resource + n_pyg} patches "
              f"({n_resource} resource, {n_pyg} pyg-transform).")
    else:
        print("All targets already patched (or nothing to patch).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
