"""Static crop auditor (C8) -- no training required.

The full-dataset plan rests entirely on ``InterfaceCrop``, and the only prior
data point (``phase_g_crop_smoke``, DockQ 0.022) was discouraging (n=80, a
smoke). Before spending a training run to discover the cropper is unsuitable,
this samples N crops across the dataset and reports what they actually keep:

- fraction of crops retaining BOTH chains,
- inter-chain CA-CA contact (<8 A) retention vs the parent complex,
- chain balance (min-chain fraction of the crop),
- fraction that fell back to SpatialCrop (no interface / single chain),
- crop-size distribution.

This makes the E1 gate in the experiment plan executable without a training run.

Usage:
    uv run python scripts/data/audit_crops.py \
        --parquet data/processed/samples.parquet \
        --n 500 --crop_size 256 --strategy interface
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq
import torch

from tinyfold.training.cropping import (
    _interface_residue_indices,
    _is_two_chain,
    build_cropper,
)
from tinyfold.training.data import load_sample


def _inter_chain_contacts(sample, cutoff: float = 8.0) -> int:
    """Number of inter-chain CA-CA pairs within ``cutoff`` A."""
    chain = sample['chain_ids']
    ca = sample['coords_res'][:, 1, :]
    a = chain == 0
    b = chain == 1
    if not (a.any() and b.any()):
        return 0
    d = torch.cdist(ca[a], ca[b])
    return int((d < cutoff).sum().item())


def _chain_balance(sample) -> float:
    """min-chain fraction of the crop (0.0 if a chain is absent)."""
    chain = sample['chain_ids']
    n = chain.numel()
    if n == 0:
        return 0.0
    na = int((chain == 0).sum().item())
    nb = int((chain == 1).sum().item())
    return min(na, nb) / n


def _would_fall_back(sample, cutoff: float) -> bool:
    """InterfaceCrop falls back to SpatialCrop for single-chain samples or when
    no inter-chain contacts exist."""
    if not _is_two_chain(sample):
        return True
    return _interface_residue_indices(sample, cutoff=cutoff).numel() == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default=None,
                    help="Optional ESM cache dir (not needed for crop auditing).")
    ap.add_argument("--n", type=int, default=500, help="Samples to audit.")
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--strategy", default="interface",
                    choices=["contiguous", "spatial", "interface"])
    ap.add_argument("--interface_cutoff", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_res", type=int, default=0,
                    help="Only audit complexes with > crop_size residues "
                         "(0 = derive from crop_size, so every audited sample is "
                         "actually cropped).")
    args = ap.parse_args()

    table = pq.read_table(args.parquet)
    n_rows = table.num_rows
    min_res = args.min_res or (args.crop_size + 1)

    rng = np.random.default_rng(args.seed)
    crop_rng = torch.Generator().manual_seed(args.seed)
    cropper = build_cropper(args.strategy, interface_cutoff=args.interface_cutoff)

    # LA/LB give the residue count without decoding coords.
    la = table["LA"].to_numpy(zero_copy_only=False)
    lb = table["LB"].to_numpy(zero_copy_only=False)
    eligible = np.nonzero((la + lb) > min_res)[0]
    if eligible.size == 0:
        raise SystemExit(
            f"No complexes with > {min_res} residues in {args.parquet}."
        )
    pick = rng.choice(eligible, size=min(args.n, eligible.size), replace=False)

    both_chains = []
    contact_retention = []
    balance = []
    fell_back = []
    crop_sizes = []
    n_audited = 0

    for idx in pick:
        sample = load_sample(table, int(idx), esm_cache_dir=args.esm_dir)
        if not _is_two_chain(sample):
            continue
        parent_contacts = _inter_chain_contacts(sample, args.interface_cutoff)
        fb = _would_fall_back(sample, args.interface_cutoff)
        crop = cropper(sample, args.crop_size, crop_rng)

        n_audited += 1
        both_chains.append(1.0 if _is_two_chain(crop) else 0.0)
        balance.append(_chain_balance(crop))
        crop_sizes.append(int(crop['n_res']))
        fell_back.append(1.0 if fb else 0.0)
        if parent_contacts > 0:
            crop_contacts = _inter_chain_contacts(crop, args.interface_cutoff)
            contact_retention.append(crop_contacts / parent_contacts)

    if n_audited == 0:
        raise SystemExit("No two-chain complexes were audited.")

    def _pct(xs):
        return 100.0 * float(np.mean(xs)) if xs else float("nan")

    def _quantiles(xs):
        if not xs:
            return "n/a"
        q = np.quantile(xs, [0.1, 0.5, 0.9])
        return f"p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}"

    print(f"Crop auditor: strategy={args.strategy} crop_size={args.crop_size} "
          f"n_audited={n_audited} (of {len(pick)} sampled, "
          f"{n_rows} total rows)")
    print(f"  retains both chains:      {_pct(both_chains):.1f}%")
    print(f"  fell back to SpatialCrop: {_pct(fell_back):.1f}%")
    print(f"  inter-chain contact retention (crop/parent): {_quantiles(contact_retention)}")
    print(f"  chain balance (min-chain fraction):          {_quantiles(balance)}")
    print(f"  crop size (residues):                        {_quantiles(crop_sizes)}")


if __name__ == "__main__":
    main()
