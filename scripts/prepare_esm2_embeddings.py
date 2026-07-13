#!/usr/bin/env python
"""Precompute frozen ESM-2 per-residue embeddings for the DIPS-Plus parquet cache.

ESM-2 is treated as a frozen feature extractor: we run it once over every
sample's per-chain AA1 string, slice off ``[CLS]``/``[EOS]``, concatenate
chain A and chain B along the residue axis (to match the parquet's
``seq`` / ``chain_id_res`` ordering), and write one ``.npz`` per sample.

Output layout (per sample):
    ``{output_dir}/{sample_id}.npz`` containing
        embeddings: float16 [LA+LB, esm_dim]
        LA: int   (chain A residue count)
        LB: int   (chain B residue count)
        sample_id: str
        esm_dim: int

The script is idempotent: existing cache files are skipped unless
``--overwrite`` is given. A KeyboardInterrupt cleanly resumes on the next
invocation (only fully written files are kept; partial writes are deleted
on the next save).

Variants (model id, hidden dim):
    35M  -> facebook/esm2_t12_35M_UR50D   (480)
    150M -> facebook/esm2_t30_150M_UR50D  (640)

Typical wall-time on a 4070 Ti SUPER:
    ~25-35 min for the full 28352-sample cache at variant=35M.
    ~10-15 min for the in-filter subset (LA+LB in [200, 1200]).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from tinyfold.constants import IDX_TO_AA

ESM_VARIANTS = {
    "35M":  ("facebook/esm2_t12_35M_UR50D",  480),
    "150M": ("facebook/esm2_t30_150M_UR50D", 640),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parquet", type=str,
                   default="data/processed/samples.parquet",
                   help="Path to samples.parquet")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write per-sample .npz files")
    p.add_argument("--variant", type=str, default="35M",
                   choices=sorted(ESM_VARIANTS.keys()),
                   help="ESM-2 variant: 35M (480d) or 150M (640d)")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device for ESM forward pass")
    p.add_argument("--batch-size", type=int, default=1,
                   help="(Reserved) per-chain batch size; kept at 1 for simplicity")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Process only the first N samples (0 = all). "
                        "Filters apply BEFORE the cap.")
    p.add_argument("--filter-residues", type=str, default=None,
                   help="Inclusive residue-length filter as 'min-max' (e.g. "
                        "'200-1200'). Only process samples where LA+LB falls "
                        "in the range. Useful to align the cache with the "
                        "Phase D training filter.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode samples even if their .npz already exists")
    p.add_argument("--fp16-model", action="store_true",
                   help="Load ESM in float16 (saves VRAM; outputs still cast "
                        "to float16 on disk regardless of this flag)")
    return p.parse_args()


def _parse_filter_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    lo, hi = spec.split("-", 1)
    lo, hi = int(lo), int(hi)
    if lo > hi:
        raise ValueError(f"--filter-residues lo>hi: {spec!r}")
    return (lo, hi)


def _seq_to_aa1(seq_idx: np.ndarray) -> str:
    """Map integer AA indices back to a 1-letter string. Index 20 -> 'X'."""
    return "".join(IDX_TO_AA[int(i)] for i in seq_idx)


def _embed_chain(model, tokenizer, seq: str, device: str) -> np.ndarray:
    """Run ESM-2 on one chain and return [L, esm_dim] float16 numpy array.

    Strips [CLS] (position 0) and [EOS] (position -1) from the model output.
    """
    tok = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**tok).last_hidden_state  # [1, L+2, esm_dim]
    emb = out[0, 1:-1].detach().to(torch.float16).cpu().numpy()
    return emb


def main() -> int:
    args = parse_args()

    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA requested but unavailable; falling back to CPU.", flush=True)
        args.device = "cpu"

    model_id, esm_dim = ESM_VARIANTS[args.variant]
    print(f"ESM variant: {args.variant} ({model_id}, dim={esm_dim})", flush=True)
    print(f"Output dir : {output_dir}", flush=True)
    print(f"Device     : {args.device}", flush=True)

    # Lazy import so the script gives a clear error if transformers is missing.
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError as exc:
        print(f"ERROR: transformers is not installed: {exc}", file=sys.stderr)
        print("Install with: uv pip install 'transformers>=4.40'", file=sys.stderr)
        return 2

    print("Loading tokenizer + model... (first run downloads ~150-600 MB)", flush=True)
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model_kwargs = {}
    if args.fp16_model and args.device == "cuda":
        # `torch_dtype` is the modern HF kwarg; fall back to manual cast if HF complains.
        try:
            model = EsmModel.from_pretrained(model_id, dtype=torch.float16)
        except TypeError:
            model = EsmModel.from_pretrained(model_id)
            model = model.to(torch.float16)
    else:
        model = EsmModel.from_pretrained(model_id, **model_kwargs)
    model = model.eval().to(args.device)
    torch.set_grad_enabled(False)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Read parquet (only the columns we need).
    print(f"Reading parquet: {parquet_path}", flush=True)
    table = pq.read_table(parquet_path,
                          columns=["sample_id", "seq", "chain_id_res", "LA", "LB"])
    n_total = table.num_rows
    print(f"  {n_total} samples in parquet", flush=True)

    # Build a filtered index list.
    sample_ids = table["sample_id"].to_pylist()
    LAs = table["LA"].to_pylist()
    LBs = table["LB"].to_pylist()
    seq_col = table["seq"]
    chain_col = table["chain_id_res"]

    residue_filter = _parse_filter_range(args.filter_residues)
    indices = []
    for i in range(n_total):
        if residue_filter is not None:
            n_res = int(LAs[i]) + int(LBs[i])
            if not (residue_filter[0] <= n_res <= residue_filter[1]):
                continue
        indices.append(i)
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]
    n_target = len(indices)
    print(f"  after filters: processing {n_target} samples "
          f"(residue_filter={residue_filter}, max_samples={args.max_samples})",
          flush=True)

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    pbar = tqdm(indices, total=n_target, desc=f"ESM-2 {args.variant}", unit="sample")
    t_start = time.time()
    for i in pbar:
        sample_id = sample_ids[i]
        cache_path = output_dir / f"{sample_id}.npz"

        if cache_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        LA = int(LAs[i])
        LB = int(LBs[i])
        n_res = LA + LB
        seq_idx = np.asarray(seq_col[i].as_py(), dtype=np.int64)
        chain_idx = np.asarray(chain_col[i].as_py(), dtype=np.int64)

        if seq_idx.shape[0] != n_res or chain_idx.shape[0] != n_res:
            n_failed += 1
            failures.append((sample_id, f"seq/chain length mismatch with LA+LB ({seq_idx.shape[0]} vs {n_res})"))
            continue

        try:
            seq_a = _seq_to_aa1(seq_idx[:LA])
            seq_b = _seq_to_aa1(seq_idx[LA:LA + LB])
            assert len(seq_a) == LA and len(seq_b) == LB

            emb_a = _embed_chain(model, tokenizer, seq_a, args.device)
            emb_b = _embed_chain(model, tokenizer, seq_b, args.device)

            if emb_a.shape != (LA, esm_dim) or emb_b.shape != (LB, esm_dim):
                raise RuntimeError(
                    f"shape mismatch after slice: emb_a={emb_a.shape}, "
                    f"emb_b={emb_b.shape}, expected ({LA}, {esm_dim}) / ({LB}, {esm_dim})"
                )

            emb_all = np.concatenate([emb_a, emb_b], axis=0)  # [LA+LB, esm_dim]
            assert emb_all.dtype == np.float16

            # Write atomically: stage to a tmp file, then rename. This makes the
            # script safe to Ctrl-C mid-write (no partially written .npz files
            # survive an interrupt). NOTE: `np.savez_compressed` always appends
            # `.npz` to the path it's given, so we pass it the path with the
            # trailing ".npz" stripped and rename the resulting "<base>.npz".
            tmp_base = output_dir / f".{sample_id}.tmp"   # leading dot = hidden on POSIX
            tmp_file = Path(str(tmp_base) + ".npz")        # what numpy actually writes
            np.savez_compressed(
                str(tmp_base),                             # numpy appends .npz
                embeddings=emb_all,
                LA=np.int32(LA),
                LB=np.int32(LB),
                sample_id=np.array(sample_id),
                esm_dim=np.int32(esm_dim),
            )
            tmp_file.replace(cache_path)
            n_processed += 1
        except Exception as exc:
            n_failed += 1
            failures.append((sample_id, repr(exc)))
            # Best-effort cleanup of any partial tmp file.
            try:
                tmp_base = output_dir / f".{sample_id}.tmp"
                tmp_file = Path(str(tmp_base) + ".npz")
                if tmp_file.exists():
                    tmp_file.unlink()
            except OSError:
                pass

        if (n_processed + n_skipped + n_failed) % 500 == 0:
            elapsed = time.time() - t_start
            done = n_processed + n_skipped + n_failed
            rate = done / max(elapsed, 1e-6)
            remaining = (n_target - done) / max(rate, 1e-6)
            pbar.set_postfix(rate=f"{rate:.2f}/s", eta_min=f"{remaining/60:.1f}")

    pbar.close()

    # Disk usage of the output dir (best-effort; ignores nested dirs).
    total_bytes = 0
    n_files = 0
    for p in output_dir.glob("*.npz"):
        try:
            total_bytes += p.stat().st_size
            n_files += 1
        except OSError:
            pass

    elapsed = time.time() - t_start
    print("", flush=True)
    print(f"Done in {elapsed/60:.2f} min", flush=True)
    print(f"  processed : {n_processed}", flush=True)
    print(f"  skipped   : {n_skipped} (already cached)", flush=True)
    print(f"  failed    : {n_failed}", flush=True)
    print(f"  cache dir : {n_files} files, {total_bytes/(1024**3):.2f} GB", flush=True)

    if failures:
        print(f"\nFirst {min(10, len(failures))} failures:", flush=True)
        for sid, reason in failures[:10]:
            print(f"  {sid}: {reason}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
