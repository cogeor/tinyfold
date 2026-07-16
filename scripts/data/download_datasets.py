"""Download the Phase-2 (MSA db) and Phase-3 (raw DIPS) datasets.

    uv run python scripts/data/download_datasets.py --phase3      # raw DIPS, 15.69 GB
    uv run python scripts/data/download_datasets.py --phase2      # uniref50,  8.80 GB
    uv run python scripts/data/download_datasets.py --all
    uv run python scripts/data/download_datasets.py --plan        # sizes only, no transfer

Resumable (HTTP Range) and checksum-verified, because these are multi-GB
transfers that will get interrupted.

WHAT GETS DOWNLOADED, AND WHAT DOES NOT
---------------------------------------
Phase 3: ``final_raw_dips.tar.gz`` (15.69 GB, Zenodo 8140981, md5-verified).
  The SAME record also holds ``interim_external_feats_dips_msas.tar.gz``
  (11.99 GB of HHblits-vs-BFD MSAs). We deliberately DO NOT fetch it: BFD has no
  taxonomy, so those MSAs cannot be paired across chains and carry zero
  cross-chain coevolution signal (plan §B4). Skipping it saves 12 GB.

  NOTE: the tar is NOT extracted. Measured compression is ~6x, so extracting
  would land ~90 GB of dills on a disk with ~244 GB free. Instead
  ``prepare_atom14.py --from-tar`` streams members straight out of the archive
  and writes only the ~2 GB atom14 cache.

Phase 2: ``uniref50.fasta.gz`` (8.80 GB) -- the taxonomy-bearing sequence db.
  UniRef50 rather than UniRef30 (99.4 GB, extracted size unverified, ColabFold
  states ~1 TB for uniref30+envdb) because it fits comfortably and its headers
  carry ``OX=<taxid>``, which is all pairing needs. Move up the ladder only if
  Step 0 says depth is the limiter.

DISK BUDGET (measured 2026-07-16: 244 GB free)
    raw DIPS tar         15.7 GB   (kept; stream-processed, never extracted)
    uniref50.fasta.gz     8.8 GB   (deleted after createdb)
    uniref50 fasta       ~30   GB  (deleted after createdb)
    uniref50 mmseqs db   ~45   GB
    atom14 cache          ~2   GB
    -> steady ~63 GB, peak ~84 GB during createdb. Comfortable.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.data.sources.dips_plus import get_zenodo_files

RAW_ARCHIVE = "final_raw_dips.tar.gz"
UNIREF50_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
)

CHUNK = 8 << 20  # 8 MiB


def human(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        n = r.headers.get("content-length")
        return int(n) if n else None
    except requests.RequestException:
        return None


def md5_of(path: Path, expect: str | None = None) -> str:
    """Stream the md5 of a large file (never loads it into memory)."""
    h = hashlib.md5()
    total = path.stat().st_size
    seen = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            seen += len(chunk)
            if time.time() - t0 > 5:
                print(f"\r    md5 {100 * seen / total:5.1f}%", end="", flush=True)
                t0 = time.time()
    print("\r" + " " * 24 + "\r", end="")
    digest = h.hexdigest()
    if expect and digest != expect:
        raise ValueError(f"checksum mismatch for {path.name}: {digest} != {expect}")
    return digest


def download_resumable(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Download with HTTP Range resume. Safe to re-run; skips a complete file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    size = expected_size or remote_size(url)

    if size and have == size:
        print(f"  {dest.name}: already complete ({human(size)})")
        return dest
    if have and size and have > size:
        print(f"  {dest.name}: local file larger than remote -- restarting")
        dest.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"  {dest.name}: resuming at {human(have)} / {human(size or 0)}")
    else:
        print(f"  {dest.name}: starting ({human(size or 0)})")

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        # 206 = partial (resume honoured), 200 = full body (server ignored Range).
        if have and r.status_code == 200:
            print("    server ignored Range -- restarting from 0")
            have = 0
        elif have and r.status_code != 206:
            r.raise_for_status()
        r.raise_for_status()

        mode = "ab" if have and r.status_code == 206 else "wb"
        seen = have if mode == "ab" else 0
        t0 = start = time.time()
        with open(dest, mode) as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                seen += len(chunk)
                if time.time() - t0 > 10:
                    rate = (seen - have) / max(time.time() - start, 1e-9)
                    eta = (size - seen) / rate if size and rate else 0
                    pct = f"{100 * seen / size:5.1f}%" if size else "  ?  "
                    print(f"\r    {pct}  {human(seen)}  {rate / 1e6:5.1f} MB/s  "
                          f"eta {eta / 60:5.1f} min", end="", flush=True)
                    t0 = time.time()
    print()
    return dest


def need_space(dest: Path, want: int, label: str) -> bool:
    free = free_bytes(dest)
    if free < want:
        print(f"  REFUSING {label}: needs {human(want)}, only {human(free)} free")
        return False
    print(f"  disk ok for {label}: need {human(want)}, have {human(free)} free")
    return True


def do_phase3(dest: Path, plan_only: bool) -> int:
    print("\n=== PHASE 3: raw DIPS (sidechains) ===")
    files = {f["key"]: f for f in get_zenodo_files()}
    info = files.get(RAW_ARCHIVE)
    if info is None:
        print(f"  {RAW_ARCHIVE} not found in the Zenodo record")
        return 1

    size = info["size"]
    md5 = (info.get("checksum") or "").removeprefix("md5:") or None
    for key, other in files.items():
        if key != RAW_ARCHIVE:
            print(f"  skipping {key} ({human(other['size'])}) -- "
                  f"BFD MSAs have no taxonomy, unusable for pairing")
    print(f"  target: {RAW_ARCHIVE} ({human(size)}) md5={md5}")
    print("  note: tar is NOT extracted (~6x -> ~90 GB); use "
          "prepare_atom14.py --from-tar")
    if plan_only:
        return 0
    # Only the tar itself: we stream-process it rather than extracting.
    if not need_space(dest, int(size * 1.05), "raw DIPS tar"):
        return 1

    out = download_resumable(info["links"]["self"], dest / RAW_ARCHIVE, size)
    if md5:
        print("  verifying md5...")
        md5_of(out, md5)
        print(f"  md5 OK ({md5})")
    print(f"  done: {out}")
    return 0


def do_phase2(dest: Path, plan_only: bool) -> int:
    print("\n=== PHASE 2: UniRef50 (taxonomy-bearing MSA db) ===")
    size = remote_size(UNIREF50_URL)
    print(f"  target: uniref50.fasta.gz ({human(size or 0)})")
    print("  (UniRef30 is 99.4 GB with unverified extracted size; UniRef50 "
          "headers carry OX=<taxid>, which is all pairing needs)")
    if plan_only:
        return 0
    # Download + later createdb needs room for fasta (~30 GB) + db (~45 GB).
    if not need_space(dest, int((size or 9e9) + 80e9), "uniref50 + createdb"):
        return 1

    out = download_resumable(UNIREF50_URL, dest / "uniref50.fasta.gz", size)
    print("  verifying gzip integrity...")
    import gzip

    try:
        with gzip.open(out, "rb") as fh:
            while fh.read(CHUNK):
                pass
    except OSError as exc:
        print(f"  CORRUPT: {exc} -- delete and re-run")
        return 1
    print("  gzip OK")
    print(f"  done: {out}")
    print("\n  next: build the mmseqs db (needs mmseqs in WSL):")
    print("    bash scripts/data/setup_msa_db.sh uniref50 data/msa_db")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase2", action="store_true", help="UniRef50 sequence db")
    ap.add_argument("--phase3", action="store_true", help="raw DIPS (sidechains)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print sizes, download nothing")
    ap.add_argument("--dips-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--msa-dir", type=Path, default=Path("data/msa_db"))
    args = ap.parse_args()

    if not (args.phase2 or args.phase3 or args.all or args.plan):
        ap.error("pick --phase2, --phase3, --all, or --plan")
    want2 = args.phase2 or args.all or args.plan
    want3 = args.phase3 or args.all or args.plan

    print(f"disk free: {human(free_bytes(Path('.')))}")
    rc = 0
    if want3:
        rc |= do_phase3(args.dips_dir, args.plan)
    if want2:
        rc |= do_phase2(args.msa_dir, args.plan)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
