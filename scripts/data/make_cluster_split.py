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

import pyarrow.parquet as pq

from tinyfold.training.cluster_split import (
    cluster_holdout_indices,
    load_clusters,
    save_cluster_split,
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
    ap.add_argument("--per-cluster-cap", type=int, default=None,
                    help="Max test samples taken from any one cluster. Use 1 so "
                         "n_test samples carry n_test INDEPENDENT clusters. "
                         "Omitting it reproduces the historical uncapped splits, "
                         "where one cluster could supply most of the test set "
                         "(clean_le600: 137 of 200 samples from a single cluster).")
    ap.add_argument("--n-test-clusters", type=int, default=None,
                    help="Stop after this many test clusters instead of after "
                         "--n-test samples. Use with --per-cluster-cap > 1.")
    ap.add_argument("--train-cluster-uniform", action="store_true",
                    help="Also write per-train-sample weights of 1/|cluster| so "
                         "every training cluster carries equal total mass.")
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
        per_cluster_cap=args.per_cluster_cap,
        n_test_clusters=args.n_test_clusters,
        train_cluster_uniform=args.train_cluster_uniform,
    )
    print("Split info:")
    for k, v in info.items():
        # train_weights is one float per training sample; print its shape only.
        if k == "train_weights":
            print(f"  {k}: [{len(v)} weights]")
        elif k == "test_cluster_sizes":
            print(f"  {k}: {v[:10]}{' ...' if len(v) > 10 else ''}")
        else:
            print(f"  {k}: {v}")
    assert not info["cluster_overlap"], f"LEAKAGE: clusters in both sides: {info['cluster_overlap']}"
    if info["max_test_cluster_share"] > 0.1:
        print(
            f"  WARNING: one cluster is {100 * info['max_test_cluster_share']:.0f}% of "
            f"the test set ({info['n_test_clusters']} clusters for {info['n_test']} "
            f"samples). Pass --per-cluster-cap 1 for an informative test set."
        )
    save_cluster_split(table, train_idx, test_idx, info, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
