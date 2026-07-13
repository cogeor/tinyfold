"""C2 CLI: build a leakage-clean cluster-holdout split file.

Produces a split JSON (train_indices/test_indices) consumable by
``train_resfold.py --load_split``. One split per size regime keeps E0..E4 on the
SAME clean holdout.

Usage:
    uv run python scripts/data/make_cluster_split.py \
        --clusters data/processed/clusters.json \
        --max-atoms 960 --n-train 5000 --n-test 200 \
        --out data/processed/splits/clean_le240.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pyarrow.parquet as pq

from tinyfold.training.cluster_split import (
    cluster_holdout_indices,
    load_clusters,
    save_split,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--clusters", default="data/processed/clusters.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-atoms", type=int, default=0)
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--forced-test-clusters", default=None,
                    help="Optional JSON file with a list of cluster ids to force "
                         "test-side (C3 PINDER yardstick).")
    args = ap.parse_args()

    # Only sample_id + atom_type are needed (atom counts + ids). Reading the
    # full table (atom_coords etc.) can exhaust RAM while a training run holds
    # its preloaded samples.
    table = pq.read_table(args.parquet, columns=["sample_id", "atom_type"])
    clusters = load_clusters(args.clusters)

    forced = None
    if args.forced_test_clusters:
        import json
        forced = set(json.load(open(args.forced_test_clusters)))

    train_idx, test_idx, info = cluster_holdout_indices(
        table, clusters,
        n_train=args.n_train, n_test=args.n_test,
        min_atoms=args.min_atoms, max_atoms=args.max_atoms,
        seed=args.seed, forced_test_clusters=forced,
    )
    print("Split info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    assert not info["cluster_overlap"], f"LEAKAGE: clusters in both sides: {info['cluster_overlap']}"
    save_split(table, train_idx, test_idx, info, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
