"""STEP 0 -- the MSA-depth de-risk. Run this BEFORE buying a database.

From notes/2026-07-14-msa-coevolution-pair-prior-SPEC.md §8 and the plan §2.2:

    "Before any pipeline build: measure paired-MSA depth (Neff / #paired seqs)
     per complex. If most pair to only a handful of sequences, coevolution is
     thin -> reconsider before investing."

This script answers that for a handful of complexes and prints the distribution.
It reads a3m files that you have already generated (see --a3m-dir); it does NOT
generate them, precisely so Step 0 can run against the ColabFold PUBLIC server
output for ~200 test complexes (fair use, zero storage, zero download) instead of
the ~99 GB local UniRef30.

THE GATE
--------
Look at the paired-depth distribution, NOT the per-chain depth. Per-chain depth
is what ESM already internalises; paired depth is the only thing that carries
cross-chain coevolution.

  * median n_paired in the single digits  -> coevolution is thin on DIPS.
    STOP. Do not buy the database. Reconsider or drop Phase 2.
  * frac_no_taxid near 1.0                -> your database has no taxonomy
    (the BFD/DIPS-Plus case) -> pairing is IMPOSSIBLE regardless of depth.

Usage:
    uv run python scripts/data/measure_msa_depth.py \
        --a3m-dir data/processed/msa/a3m \
        --split data/processed/splits/clean_le600.json \
        --subset test \
        --out benchmarks/results/msa_depth_step0.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.msa.a3m import neff, read_a3m
from tinyfold.msa.dataset import split_complex
from tinyfold.msa.pairing import pairing_stats


def find_a3m(a3m_dir: Path, key: str) -> Path | None:
    for suffix in (".a3m", ".a3m.gz"):
        p = a3m_dir / f"{key}{suffix}"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a3m-dir", type=Path, required=True, help="dir of <chain_key>.a3m[.gz]")
    ap.add_argument("--parquet", type=Path, default=Path("data/processed/samples.parquet"))
    ap.add_argument("--split", type=Path, default=None, help="split json (limits which complexes)")
    ap.add_argument("--subset", default="test", choices=["train", "test"])
    ap.add_argument("--out", type=Path, default=None, help="write per-complex CSV here")
    ap.add_argument("--max-rows", type=int, default=4096,
                    help="cap MSA rows before Neff (Neff is O(N^2 L))")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["sample_id", "seq", "LA", "LB"])
    if args.split:
        ids = set(json.loads(args.split.read_text())[f"{args.subset}_ids"])
        df = df[df.sample_id.isin(ids)]
    print(f"complexes: {len(df)}")

    rows, missing = [], 0
    for sample_id, seq, la, lb in zip(df.sample_id, df.seq, df.LA, df.LB):
        cc = split_complex(seq, int(la), int(lb), sample_id)
        pa, pb = find_a3m(args.a3m_dir, cc.key_a), find_a3m(args.a3m_dir, cc.key_b)
        if pa is None or pb is None:
            missing += 1
            continue
        msa_a, msa_b = read_a3m(pa)[: args.max_rows], read_a3m(pb)[: args.max_rows]
        st = pairing_stats(msa_a, msa_b)
        st["sample_id"] = sample_id
        # Neff of the PAIRED alignment is the number that matters: raw paired
        # depth overstates signal when the paired rows are near-duplicates.
        from tinyfold.msa.pairing import pair_msas

        qa, qb = pair_msas(msa_a, msa_b)
        st["neff_paired"] = neff([a.seq + b.seq for a, b in zip(qa, qb)])
        st["neff_a"] = neff([r.seq for r in msa_a])
        st["neff_b"] = neff([r.seq for r in msa_b])
        rows.append(st)

    if not rows:
        print(f"NO a3m pairs found (missing for {missing} complexes) -- nothing to report.")
        return 1

    out = pd.DataFrame(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    def q(col):
        v = out[col].to_numpy()
        return f"median {np.median(v):8.1f} | p10 {np.percentile(v, 10):8.1f} | p90 {np.percentile(v, 90):8.1f}"

    print(f"\nskipped (a3m missing): {missing}")
    print("\n=== PER-CHAIN depth (what ESM already covers) ===")
    print(f"  n_a        : {q('n_a')}")
    print(f"  neff_a     : {q('neff_a')}")
    print("\n=== PAIRED depth (THE number -- cross-chain coevolution) ===")
    print(f"  n_paired   : {q('n_paired')}")
    print(f"  neff_paired: {q('neff_paired')}")
    print(f"  shared spp : {q('n_species_shared')}")
    print(f"\n  unpairable hits (no taxonomy): chain A mean {out.frac_no_taxid_a.mean():.1%}")

    med = float(np.median(out.n_paired))
    print("\n=== GATE ===")
    if out.frac_no_taxid_a.mean() > 0.9:
        print("  FAIL: hits carry no taxonomy -> pairing impossible (BFD-style db).")
        print("        Use a taxonomy-bearing db (UniRef30/90/50). Do NOT proceed.")
    elif med < 10:
        print(f"  FAIL: median paired depth {med:.0f} -- coevolution is thin on this set.")
        print("        Do NOT buy the database. Reconsider or drop Phase 2.")
    elif med < 100:
        print(f"  MARGINAL: median paired depth {med:.0f}. Expect a weak prior;")
        print("        consider the cheap UniRef50 path before the 99 GB UniRef30.")
    else:
        print(f"  PASS: median paired depth {med:.0f} -- proceed to the database build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
