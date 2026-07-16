"""S1/S3 -- re-acquire raw DIPS and build the atom14 (sidechain) cache.

THE PHASE-3 BLOCKER (notes/2026-07-16-three-phase-plan.md §3.0)
--------------------------------------------------------------
``data/raw/`` was deleted and ``samples.parquet`` is backbone-only: the
sidechains were discarded at parse time by
``dips_loader.extract_backbone_from_dataframe``, which filters the DIPS
atom-level DataFrame down to N/CA/C/O. The atoms DO exist in the source dill.
So stage 3 cannot start from the parquet -- the raw source must come back.

This is re-extraction of the SAME source, not new data collection:
    final_raw_dips.tar.gz   14.62 GB   (Zenodo 8140981)

We deliberately fetch ONLY that file. The record also ships
``interim_external_feats_dips_msas.tar.gz`` (11.17 GB of HHblits-vs-BFD MSAs),
which is useless to us: BFD carries no taxonomy, so those MSAs cannot be paired
across chains (plan §B4). Downloading it would waste 11 GB.

Output: one .npz per sample_id with coords [L,14,3] fp16 + mask [L,14].
Cheap -- ~84 B/residue, ~1.9 GB for the full 22.03 M residues.

START HERE (costs nothing, validates the premise before the 14.6 GB download):
    uv run python scripts/data/prepare_atom14.py --verify-only --data-dir data/raw

Then:
    uv run python scripts/data/prepare_atom14.py --download --data-dir data/raw
    uv run python scripts/data/prepare_atom14.py --data-dir data/raw \
        --out-dir data/processed/atom14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.data.parsing.dips_loader import (
    extract_atom14_from_dataframe,
    load_dips_pair,
)
from tinyfold.data.sources.dips_plus import (
    download_file,
    extract_archive,
    find_dips_dill_files,
    get_zenodo_files,
    parse_dips_dill_filename,
)

RAW_ARCHIVE = "final_raw_dips.tar.gz"


def download_raw_only(dest: Path) -> None:
    """Fetch ONLY final_raw_dips.tar.gz (skip the 11.17 GB unpairable MSAs)."""
    dest.mkdir(parents=True, exist_ok=True)
    for info in get_zenodo_files():
        if info["key"] != RAW_ARCHIVE:
            print(f"  skipping {info['key']} ({info['size'] / 1e9:.2f} GB) -- not needed")
            continue
        out = dest / info["key"]
        if out.exists():
            print(f"  {out} already present")
        else:
            print(f"  downloading {info['key']} ({info['size'] / 1e9:.2f} GB)...")
            download_file(info["links"]["self"], out)
        print(f"  extracting {out.name}...")
        extract_archive(out, dest)   # path-traversal hardened (commit 2c6b6ec)


def verify_sidechains(data_dir: Path, n: int = 3) -> int:
    """Assert the source really carries sidechains. The premise check.

    If this fails, Phase 3 is dead as designed and no amount of cache-building
    will help -- so it is worth running BEFORE the 14.6 GB download completes,
    against any single dill you already have.
    """
    dills = list(find_dips_dill_files(data_dir))
    if not dills:
        print(f"No .dill files under {data_dir} -- nothing to verify. "
              f"Run with --download first.")
        return 1

    print(f"found {len(dills)} dill files; inspecting {min(n, len(dills))}\n")
    ok = True
    for path in dills[:n]:
        pair = load_dips_pair(path)
        names = set(pair.df0["atom_name"].unique())
        sidechain = names - {"N", "CA", "C", "O"}
        a14 = extract_atom14_from_dataframe(pair.df0)
        n_sc = int(a14.mask[:, 4:].sum())
        print(f"  {path.name}")
        print(f"    distinct atom names : {len(names)}")
        print(f"    non-backbone names  : {len(sidechain)}  e.g. {sorted(sidechain)[:8]}")
        print(f"    residues            : {len(a14.sequence)}")
        print(f"    sidechain atoms kept: {n_sc}")
        if n_sc == 0:
            print("    *** NO SIDECHAIN ATOMS -- premise violated ***")
            ok = False
    print("\nVERDICT:", "PASS -- sidechains present, Phase 3 can proceed."
          if ok else "FAIL -- source has no sidechains; Phase 3 needs another source.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/atom14"))
    ap.add_argument("--download", action="store_true",
                    help=f"fetch + extract {RAW_ARCHIVE} (14.62 GB) first")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the source carries sidechains, then stop")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.download:
        download_raw_only(args.data_dir)

    if args.verify_only:
        return verify_sidechains(args.data_dir)

    dills = list(find_dips_dill_files(args.data_dir))
    if not dills:
        print(f"No .dill files under {args.data_dir}. Run with --download first.")
        return 1
    if args.limit:
        dills = dills[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    total_res = total_sc = 0
    total_bytes = 0

    for path in dills:
        meta = parse_dips_dill_filename(path)
        if meta is None:
            skipped += 1
            continue
        sample_id = meta["sample_id"]
        out = args.out_dir / f"{sample_id}.npz"
        if out.exists() and not args.overwrite:
            done += 1
            continue
        try:
            pair = load_dips_pair(path)
            a = extract_atom14_from_dataframe(pair.df0)
            b = extract_atom14_from_dataframe(pair.df1)
        except Exception as exc:
            print(f"  {sample_id}: {type(exc).__name__}: {exc} -- skipped")
            skipped += 1
            continue

        coords = np.concatenate([a.coords, b.coords], axis=0).astype(np.float16)
        mask = np.concatenate([a.mask, b.mask], axis=0)
        np.savez_compressed(
            out,
            coords_atom14=coords,                       # [L, 14, 3] fp16
            mask_atom14=mask,                           # [L, 14] bool
            seq_indices=np.concatenate([a.seq_indices, b.seq_indices]),
            LA=np.int32(len(a.sequence)),
            LB=np.int32(len(b.sequence)),
        )
        total_res += mask.shape[0]
        total_sc += int(mask[:, 4:].sum())
        total_bytes += out.stat().st_size
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(dills)} ({total_bytes / 1e9:.2f} GB)")

    print(f"\ncached  : {done}")
    print(f"skipped : {skipped}")
    print(f"residues: {total_res / 1e6:.2f} M")
    print(f"sidechain atoms: {total_sc / 1e6:.2f} M "
          f"({total_sc / max(total_res, 1):.1f} per residue)")
    print(f"size    : {total_bytes / 1e9:.2f} GB in {args.out_dir}")
    if total_sc == 0 and done:
        print("\n*** WARNING: zero sidechain atoms cached -- the source is "
              "backbone-only. Do not train stage 3 on this. ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
