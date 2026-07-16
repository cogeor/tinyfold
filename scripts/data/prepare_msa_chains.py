"""Build the deduped per-chain MSA work-list (FASTA) from the complex parquet.

MSAs are per unique chain SEQUENCE. Measured 2026-07-16:

    clean_le600 : 5,200 complexes -> 10,400 chain instances -> 4,750 unique
    full        : 41,883 complexes -> 83,766 chain instances -> 22,293 unique

Emitting the deduped set is a 2.2x (le600) / 3.8x (full) saving on the search,
and lets the a3m cache be keyed by sequence hash so complexes sharing a chain
share its MSA.

Usage:
    uv run python scripts/data/prepare_msa_chains.py \
        --split data/processed/splits/clean_le600.json \
        --out data/processed/msa/chains_le600.fasta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.msa.dataset import split_complex, unique_chains, write_fasta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=Path("data/processed/samples.parquet"))
    ap.add_argument("--split", type=Path, default=None,
                    help="split json; omit to use the whole dataset (22,293 chains)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index-out", type=Path, default=None,
                    help="also write {sample_id: [key_a, key_b]} json for the feature step")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["sample_id", "seq", "LA", "LB"])
    if args.split:
        split = json.loads(args.split.read_text())
        ids = set(split["train_ids"]) | set(split["test_ids"])
        df = df[df.sample_id.isin(ids)]

    complexes = [
        split_complex(seq, int(la), int(lb), sid)
        for sid, seq, la, lb in zip(df.sample_id, df.seq, df.LA, df.LB)
    ]
    chains = unique_chains(complexes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write_fasta(chains, args.out)

    lens = np.array([len(s) for s in chains.values()])
    print(f"complexes        : {len(complexes)}")
    print(f"chain instances  : {2 * len(complexes)}")
    print(f"UNIQUE chains    : {n}  ({2 * len(complexes) / max(n, 1):.1f}x dedup)")
    print(f"chain length     : mean {lens.mean():.0f} | max {lens.max()} | total {lens.sum() / 1e6:.2f} M res")
    print(f"wrote            : {args.out}")

    if args.index_out:
        index = {c.sample_id: [c.key_a, c.key_b] for c in complexes}
        args.index_out.parent.mkdir(parents=True, exist_ok=True)
        args.index_out.write_text(json.dumps(index))
        print(f"wrote            : {args.index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
