"""M5 -- precompute + cache coevolution pair features per complex.

Mirrors the ESM-cache pattern: one .npz per sample_id, loaded by the dataloader.

WHY THIS IS CACHED AND THE TEMPLATE FEATURES ARE NOT
----------------------------------------------------
Template pair features are built IN-MODEL from a compact per-residue tensor
(retrieval BUILD-REPORT D3) precisely to avoid an O(L^2) cache. Coevolution
cannot do that: the features need the whole MSA, which is far bigger than the
[L, L, F] it reduces to. So we pay the O(L^2) cache here -- and it is affordable
only because F is small:

    L=600, F=4, fp16  ->  2.9 MB/complex  ->  ~15 GB for a 5,200-complex split

Do NOT cache MSA-Transformer row attention raw: 12 layers x 12 heads of [L, L]
is ~158 MB per complex (~800 GB per split). Any richer feature source must be
reduced to a handful of channels BEFORE it reaches this cache.

Usage:
    uv run python scripts/data/prepare_msa_features.py \
        --a3m-dir data/processed/msa/a3m \
        --index data/processed/msa/index_le600.json \
        --out-dir data/processed/msa_feats_le600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.msa.a3m import read_a3m
from tinyfold.msa.features import build_msa_pair_features
from tinyfold.msa.pairing import pair_msas, pairing_stats


def find_a3m(a3m_dir: Path, key: str) -> Path | None:
    for suffix in (".a3m", ".a3m.gz"):
        p = a3m_dir / f"{key}{suffix}"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a3m-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True,
                    help="{sample_id: [key_a, key_b]} from prepare_msa_chains.py")
    ap.add_argument("--parquet", type=Path, default=Path("data/processed/samples.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=4096,
                    help="cap paired rows (reweighting is O(N^2 L))")
    ap.add_argument("--max-per-species", type=int, default=4,
                    help="cap paired rows per species (bounds paralog explosion)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    index = json.loads(args.index.read_text())
    df = pd.read_parquet(args.parquet, columns=["sample_id", "LA", "LB"])
    lens = {s: (int(a), int(b)) for s, a, b in zip(df.sample_id, df.LA, df.LB)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    total_bytes = 0

    for sample_id, (key_a, key_b) in index.items():
        out = args.out_dir / f"{sample_id}.npz"
        if out.exists() and not args.overwrite:
            done += 1
            continue
        pa, pb = find_a3m(args.a3m_dir, key_a), find_a3m(args.a3m_dir, key_b)
        if pa is None or pb is None:
            skipped += 1
            continue

        msa_a = read_a3m(pa)[: args.max_rows]
        msa_b = read_a3m(pb)[: args.max_rows]
        la, lb = lens[sample_id]
        if len(msa_a[0].seq) != la or len(msa_b[0].seq) != lb:
            # The query row MUST match our residue count or the [L,L] features
            # would be indexed against the wrong residues -- the exact silent
            # failure the retrieval debugging playbook calls out.
            print(f"  {sample_id}: query length mismatch "
                  f"({len(msa_a[0].seq)},{len(msa_b[0].seq)}) vs ({la},{lb}) -- skipped")
            skipped += 1
            continue

        stats = pairing_stats(msa_a, msa_b)
        paired_a, paired_b = pair_msas(msa_a, msa_b, max_per_species=args.max_per_species)
        feats = build_msa_pair_features(paired_a, paired_b)

        np.savez_compressed(
            out,
            msa_feats=feats.astype(np.float16),   # fp16: the model upcasts
            n_paired=np.int32(stats["n_paired"]),
            n_species_shared=np.int32(stats["n_species_shared"]),
        )
        total_bytes += out.stat().st_size
        done += 1
        if done % 200 == 0:
            print(f"  {done} cached ({total_bytes / 1e9:.2f} GB)")

    print(f"\ncached : {done}")
    print(f"skipped: {skipped}")
    print(f"size   : {total_bytes / 1e9:.2f} GB in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
