#!/usr/bin/env python
"""Build a split.json that adopts DiffDock-PP's exact train/val/test
partition of DIPS-Plus.

DiffDock-PP (Ketata et al. 2023 ICLR-MLDD) inherits a family-clustered
split from EquiDock (Ganea et al. 2022 ICLR). The split is hardcoded as
a CSV (`datasets/DIPS/data_file.csv`) in the DiffDock-PP repo:

    <prefix>/<sample_id>.dill,<train|val|test>

This script consumes the (already-downloaded) list, intersects it with
our local parquet, and writes `split.json` files for the trainer.

Three split variants are produced:

  diffdock_pp_full.json
    All DiffDock-PP train/test IDs that exist in our parquet.
    Train: ~40,290 (whatever's available locally).
    Test: 100-complex headline subset (from data_file_100_test.csv).

  diffdock_pp_8k.json
    Random subset of the full DiffDock-PP train (seed=42), capped at 8000
    samples — matches Phase D's training budget for a clean "same data
    size, different split" comparison.

  diffdock_pp_smoke.json
    1000-train / 100-test for quick architecture iteration.

All three use the 100-complex eval set as the headline test split (the
same one DiffDock-PP Table 1 reports on).

Run once after the DiffDock-PP test file has been pulled (see
`data/processed/splits/diffdock_pp_100.txt`).
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DIFFDOCK_PP_FULL_URL = (
    "https://raw.githubusercontent.com/ketatam/DiffDock-PP/main/"
    "datasets/DIPS/data_file.csv"
)
DIFFDOCK_PP_100_URL = (
    "https://raw.githubusercontent.com/ketatam/DiffDock-PP/main/"
    "datasets/DIPS/data_file_100_test.csv"
)


def _strip_id(raw: str) -> str:
    """Convert `eb/1ebo.pdb2_1.dill,test` -> `1ebo.pdb2_1`."""
    csv_lhs = raw.split(",")[0].strip()
    # Drop two-letter prefix dir and `.dill` suffix.
    stem = csv_lhs.split("/")[-1]
    if stem.endswith(".dill"):
        stem = stem[:-5]
    return stem


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())


def _load_full_split(path: Path) -> dict[str, list[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} from data_file.csv."""
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        sample_id = _strip_id(parts[0])
        split = parts[1].strip()
        if split in buckets:
            buckets[split].append(sample_id)
    return buckets


def _build_split_json(
    train_ids: list[str],
    test_ids: list[str],
    sample_to_idx: dict[str, int],
    sample_to_atoms: dict[str, int],
) -> dict:
    """Build the split-json schema the trainer expects."""
    train_idx = [sample_to_idx[s] for s in train_ids if s in sample_to_idx]
    test_idx = [sample_to_idx[s] for s in test_ids if s in sample_to_idx]
    kept_train = [s for s in train_ids if s in sample_to_idx]
    kept_test = [s for s in test_ids if s in sample_to_idx]

    train_atoms = [sample_to_atoms[s] for s in kept_train]
    test_atoms = [sample_to_atoms[s] for s in kept_test]

    return {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_ids": kept_train,
        "test_ids": kept_test,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_atom_range": [min(train_atoms), max(train_atoms)] if train_atoms else [0, 0],
        "test_atom_range": [min(test_atoms), max(test_atoms)] if test_atoms else [0, 0],
        "config": {
            "source": "DiffDock-PP family-clustered split (EquiDock origin)",
            "headline_test": "100-complex subset from DiffDock-PP Table 1",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--out-dir", default="data/processed/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading parquet sample IDs + atom counts...")
    table = pq.read_table(args.parquet, columns=["sample_id", "atom_type", "LA", "LB"])
    sample_to_idx: dict[str, int] = {}
    sample_to_atoms: dict[str, int] = {}
    for i in range(len(table)):
        sid = table["sample_id"][i].as_py()
        sample_to_idx[sid] = i
        sample_to_atoms[sid] = len(table["atom_type"][i].as_py())
    print(f"  parquet has {len(sample_to_idx):,} samples")

    print("[2/4] Fetching DiffDock-PP split files (idempotent)...")
    full_csv = out_dir / "diffdock_pp_full.csv"
    test100 = out_dir / "diffdock_pp_100.txt"
    _download(DIFFDOCK_PP_FULL_URL, full_csv)
    if not test100.exists():
        _download(DIFFDOCK_PP_100_URL, test100.with_suffix(".csv"))
        # Convert the 100-file CSV (just one ID per line) into a plain txt.
        ids = [_strip_id(l) for l in test100.with_suffix(".csv").read_text().splitlines() if l.strip()]
        test100.write_text("\n".join(ids) + "\n")

    print("[3/4] Parsing splits + intersecting with local parquet...")
    full = _load_full_split(full_csv)
    headline_test = test100.read_text().strip().split()
    for k, v in full.items():
        n_in = sum(1 for s in v if s in sample_to_idx)
        print(f"  {k}: upstream {len(v):,} -> {n_in:,} present locally")
    headline_present = sum(1 for s in headline_test if s in sample_to_idx)
    print(f"  headline-100: {headline_present}/100 present locally")

    print("[4/4] Writing split.json variants...")
    rng = random.Random(args.seed)

    # Headline test list is what we report on regardless of which train pool.
    test_ids = [s for s in headline_test if s in sample_to_idx]

    train_full = [s for s in full["train"] if s in sample_to_idx]
    rng.shuffle(train_full)  # for the 8k / smoke variants

    # For the smoke variant, restrict to small-to-medium complexes (LA+LB <= 600
    # total residues == max 2400 atoms) so per-step wall time stays under 1s
    # and a 3K-step smoke completes in ~5 min. The Boltz-vs-baseline hypothesis
    # doesn't need huge complexes to validate.
    small_train = [s for s in train_full if sample_to_atoms[s] <= 2400]
    small_test = [s for s in test_ids if sample_to_atoms[s] <= 2400]

    variants = {
        "diffdock_pp_full.json": (train_full, test_ids),
        "diffdock_pp_8k.json": (train_full[:8000], test_ids),
        "diffdock_pp_smoke.json": (small_train[:500], small_test[:30]),
    }
    for name, (train_ids, t_ids) in variants.items():
        path = out_dir / name
        sj = _build_split_json(train_ids, t_ids, sample_to_idx, sample_to_atoms)
        path.write_text(json.dumps(sj, indent=2))
        print(f"  {name}: n_train={sj['n_train']}, n_test={sj['n_test']}, "
              f"atom_range_train={sj['train_atom_range']}, "
              f"atom_range_test={sj['test_atom_range']}")
    print("Done.")


if __name__ == "__main__":
    main()
