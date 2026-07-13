"""C1: cluster complexes by interface for leakage-clean splits.

Deviation D1 (BUILD-REPORT): SPEC calls for MMseqs2 + Foldseek; neither is
installed and Foldseek has no clean Windows build. We cluster by SEQUENCE
identity in pure Python (tinyfold.retrieval.seq_clustering) — the dominant
leakage axis. Two dimers share a cluster iff BOTH chains are homologous.

Pipeline:
  1. Read data/processed/samples.parquet (seq, chain_id_res per sample).
  2. Extract per-chain integer sequences; dedup to UNIQUE chains.
  3. MinHash+LSH+union-find cluster the unique chains at a k-mer Jaccard
     threshold.
  4. Complex cluster = canonical unordered pair of its two chain-cluster ids.
  5. Write data/processed/clusters.json:
       { "sample_to_cluster": {sample_id: complex_cluster_id},
         "sample_to_chain_clusters": {sample_id: [cA, cB]},
         "params": {...} }

Usage:
    uv run python scripts/data/cluster_interfaces.py \
        --parquet data/processed/samples.parquet \
        --out data/processed/clusters.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pyarrow.parquet as pq

from tinyfold.retrieval.seq_clustering import cluster_sequences


def extract_chain_seqs(table):
    """Return list of (sample_id, seqA_tuple, seqB_tuple) for every sample."""
    sample_ids = table["sample_id"].to_pylist()
    seqs = table["seq"].to_pylist()
    chain_ids = table["chain_id_res"].to_pylist()
    out = []
    for sid, seq, cid in zip(sample_ids, seqs, chain_ids):
        a = tuple(s for s, c in zip(seq, cid) if c == 0)
        b = tuple(s for s, c in zip(seq, cid) if c == 1)
        out.append((sid, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--out", default="data/processed/clusters.json")
    ap.add_argument("--k", type=int, default=3, help="k-mer length")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="exact k-mer Jaccard to merge two chains")
    ap.add_argument("--num_perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=64)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"Reading {args.parquet} ...")
    table = pq.read_table(args.parquet, columns=["sample_id", "seq", "chain_id_res"])
    rows = extract_chain_seqs(table)
    print(f"  {len(rows)} samples")

    # Dedup chains -> unique sequence list; map each chain to its unique index.
    uniq_index = {}
    uniq_seqs = []
    def uidx(seq):
        if seq not in uniq_index:
            uniq_index[seq] = len(uniq_seqs)
            uniq_seqs.append(seq)
        return uniq_index[seq]

    chainA_u = []
    chainB_u = []
    for _sid, a, b in rows:
        chainA_u.append(uidx(a))
        chainB_u.append(uidx(b))
    print(f"  {len(uniq_seqs)} unique chains (from {2*len(rows)} chain slots)")

    print(f"Clustering unique chains (k={args.k}, thr={args.threshold}, "
          f"bands={args.bands}, rows={args.rows}) ...")
    chain_labels = cluster_sequences(
        uniq_seqs, k=args.k, threshold=args.threshold,
        num_perm=args.num_perm, bands=args.bands, rows=args.rows, seed=args.seed,
    )
    n_chain_clusters = len(set(chain_labels))
    print(f"  {n_chain_clusters} chain clusters")

    # Complex cluster = canonical unordered pair of the two chain clusters.
    pair_to_id = {}
    sample_to_cluster = {}
    sample_to_chain_clusters = {}
    for (sid, _a, _b), ai, bi in zip(rows, chainA_u, chainB_u):
        cA = chain_labels[ai]
        cB = chain_labels[bi]
        key = (min(cA, cB), max(cA, cB))
        if key not in pair_to_id:
            pair_to_id[key] = len(pair_to_id)
        sample_to_cluster[sid] = pair_to_id[key]
        sample_to_chain_clusters[sid] = [int(cA), int(cB)]

    n_complex_clusters = len(pair_to_id)
    # Cluster-size distribution sanity.
    from collections import Counter
    sizes = Counter(sample_to_cluster.values())
    size_hist = Counter(sizes.values())
    biggest = max(sizes.values())
    singletons = sum(1 for v in sizes.values() if v == 1)
    print(f"  {n_complex_clusters} complex clusters over {len(rows)} samples")
    print(f"  largest cluster: {biggest} samples; singletons: {singletons} "
          f"({100*singletons/n_complex_clusters:.1f}% of clusters)")

    out = {
        "sample_to_cluster": sample_to_cluster,
        "sample_to_chain_clusters": sample_to_chain_clusters,
        "params": {
            "method": "seq_minhash_lsh",
            "k": args.k, "threshold": args.threshold,
            "num_perm": args.num_perm, "bands": args.bands, "rows": args.rows,
            "seed": args.seed,
            "n_samples": len(rows),
            "n_unique_chains": len(uniq_seqs),
            "n_chain_clusters": n_chain_clusters,
            "n_complex_clusters": n_complex_clusters,
            "largest_cluster": biggest,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
