"""Check overlap between a baseline's training-set sample IDs and our test bins.

DiffDock-PP and our test bins both derive from DIPS, so non-trivial overlap
is plausible. Run this BEFORE publishing comparison numbers.

Usage:
    python benchmarks/scripts/check_leakage.py \
        --baseline_train_ids benchmarks/baselines/diffdock_pp/train_ids.txt

If the baseline's training IDs file lists `1abc.pdb1_2` per line, this prints
the intersection per stratified bin and writes a JSON report to
`benchmarks/results/{baseline}_leakage.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["le200", "200_400", "400_600", "600_1000", "ge1000"]


def _load_ids(path: Path) -> set[str]:
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_train_ids", required=True, type=Path,
                   help="One sample_id per line.")
    p.add_argument("--baseline_name", default=None,
                   help="Defaults to parent dir name of --baseline_train_ids.")
    p.add_argument("--splits_root", default="benchmarks/splits")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    name = args.baseline_name or args.baseline_train_ids.parent.name
    out_path = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "results" / f"{name}_leakage.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.baseline_train_ids.exists():
        sys.exit(f"ERROR: train IDs file not found at {args.baseline_train_ids}")
    train_ids = _load_ids(args.baseline_train_ids)
    print(f"Baseline '{name}': {len(train_ids)} train sample IDs")

    splits_root = REPO_ROOT / args.splits_root
    report = {"baseline": name, "n_baseline_train": len(train_ids), "per_bin": {}}
    total_overlap = 0
    total_test = 0
    for bin_name in SPLITS:
        split_file = splits_root / f"{bin_name}.json"
        if not split_file.exists():
            print(f"  [skip] {bin_name}: missing {split_file}")
            continue
        d = json.load(open(split_file))
        test_ids = set(d["test_ids"])
        overlap = sorted(test_ids & train_ids)
        report["per_bin"][bin_name] = {
            "n_test": len(test_ids),
            "n_overlap": len(overlap),
            "overlap_ids": overlap,
        }
        total_overlap += len(overlap)
        total_test += len(test_ids)
        print(f"  {bin_name}: {len(overlap)}/{len(test_ids)} test IDs in baseline train")
    report["total"] = {"n_test": total_test, "n_overlap": total_overlap}

    json.dump(report, open(out_path, "w"), indent=2)
    print(f"Wrote leakage report -> {out_path}")
    if total_overlap == 0:
        print("\nNo overlap. Safe to compare directly.")
    else:
        pct = 100 * total_overlap / total_test if total_test else 0.0
        print(f"\nWARNING: {total_overlap}/{total_test} ({pct:.1f}%) overlap. "
              "Either retrain baseline on our split, filter test bins, or disclose.")


if __name__ == "__main__":
    main()
