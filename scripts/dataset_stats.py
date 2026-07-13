#!/usr/bin/env python
"""Characterise the DIPS-Plus parquet so the full-DIPS Phase E config
is built with eyes open.

Reports the LA / LB / (LA+LB) histogram, what fraction falls inside the
Phase D filter [200, 1200], where the upper tail lives, and a few
percentiles. Also computes how many bioassembly groups exist so the
"family-filtered split" question can be reasoned about cheaply.

No GPU, no model — just a parquet scan.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq


def _hist_line(label: str, values: np.ndarray, bins: list[int]) -> None:
    counts = np.zeros(len(bins) - 1, dtype=int)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        counts[i] = int(((values >= lo) & (values < hi)).sum())
    total = counts.sum()
    pct = counts / max(total, 1) * 100
    print(f"\n{label} histogram (n={total}):")
    for i in range(len(bins) - 1):
        bar = "#" * int(pct[i] / 2)
        print(f"  [{bins[i]:>5}, {bins[i+1]:>5}): {counts[i]:>5}  {pct[i]:>5.1f}%  {bar}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--lo", type=int, default=200, help="Phase D filter low bound (LA+LB)")
    p.add_argument("--hi", type=int, default=1200, help="Phase D filter high bound (LA+LB)")
    args = p.parse_args()

    print(f"Parquet: {args.parquet}")
    table = pq.read_table(args.parquet, columns=["sample_id", "pdb_id", "LA", "LB"])
    df = table.to_pandas()
    n = len(df)
    print(f"Total samples: {n}")

    LA = df["LA"].to_numpy()
    LB = df["LB"].to_numpy()
    L_total = LA + LB

    print(f"\nUnique pdb_id (parent structures): {df['pdb_id'].nunique()}")
    print(f"Unique sample_id (bioassemblies):  {df['sample_id'].nunique()}")
    sample_per_pdb = df.groupby("pdb_id").size().to_numpy()
    print(f"Bioassemblies per pdb_id: median {int(np.median(sample_per_pdb))}, "
          f"p95 {int(np.percentile(sample_per_pdb, 95))}, "
          f"max {int(np.max(sample_per_pdb))}")

    print("\n--- Chain length distributions (per chain) ---")
    for label, arr in (("LA", LA), ("LB", LB)):
        print(f"  {label}: min {arr.min()}, p25 {int(np.percentile(arr, 25))}, "
              f"med {int(np.median(arr))}, p75 {int(np.percentile(arr, 75))}, "
              f"p95 {int(np.percentile(arr, 95))}, p99 {int(np.percentile(arr, 99))}, "
              f"max {arr.max()}")

    print("\n--- Complex size (LA + LB) ---")
    print(f"  min {L_total.min()}, p10 {int(np.percentile(L_total, 10))}, "
          f"p25 {int(np.percentile(L_total, 25))}, med {int(np.median(L_total))}, "
          f"p75 {int(np.percentile(L_total, 75))}, p90 {int(np.percentile(L_total, 90))}, "
          f"p95 {int(np.percentile(L_total, 95))}, p99 {int(np.percentile(L_total, 99))}, "
          f"max {L_total.max()}")

    _hist_line(
        "Complex size (LA+LB)",
        L_total,
        [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 5000, 100000],
    )

    in_filter = ((L_total >= args.lo) & (L_total <= args.hi))
    print(
        f"\nPhase D filter [LA+LB in [{args.lo}, {args.hi}]] retains "
        f"{int(in_filter.sum()):,} / {n:,} samples ({100*in_filter.mean():.1f}%)."
    )
    above_filter = L_total > args.hi
    below_filter = L_total < args.lo
    print(f"  Above {args.hi}: {int(above_filter.sum()):,} "
          f"({100*above_filter.mean():.1f}%)")
    print(f"  Below {args.lo}: {int(below_filter.sum()):,} "
          f"({100*below_filter.mean():.1f}%)")

    print("\n--- Implications for full-DIPS training ---")
    p99 = int(np.percentile(L_total, 99))
    p95 = int(np.percentile(L_total, 95))
    p90 = int(np.percentile(L_total, 90))
    print(f"  - To cover 90% of DIPS: filter <= {p90} residues")
    print(f"  - To cover 95% of DIPS: filter <= {p95} residues")
    print(f"  - To cover 99% of DIPS: filter <= {p99} residues")
    print(f"  - Max is {L_total.max()} (very few samples; will OOM at any reasonable batch)")
    print()
    print("  Dynamic batching (max_tokens) is the right mechanism: fixed token budget,")
    print("  variable batch size by sample length. Phase E config below uses that.")


if __name__ == "__main__":
    main()
