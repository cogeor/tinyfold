"""Shared metric pipeline for all baselines.

Reads predictions from `benchmarks/predictions/{model}/{split}/{sample_id}.npz`
(each NPZ carrying `pred_atoms` of shape [L, 4, 3] in Angstroms and a string
`sample_id`), joins against the master parquet to recover ground-truth
coordinates, and writes per-sample CSV to `benchmarks/results/{model}.csv`.

The point of this script: NO baseline gets to compute its own DockQ or
C-RMSD. They feed us coordinates; we score.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from tinyfold.model.metrics.dockq import compute_dockq

SPLITS = [
    "le200", "200_400", "400_600", "600_1000", "ge1000",
    "clean_le200", "clean_200_400", "clean_400_600", "clean_600_1000", "clean_ge1000",
]


def _kabsch_align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rigidly align ``pred`` onto ``gt`` over valid CA atoms; return aligned pred."""
    p_ca = pred[valid, 1]
    g_ca = gt[valid, 1]
    if p_ca.shape[0] < 3:
        return pred.copy()
    p_c = p_ca.mean(0)
    g_c = g_ca.mean(0)
    P = p_ca - p_c
    G = g_ca - g_c
    H = P.T @ G
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    centered = pred.reshape(-1, 3) - p_c
    return (centered @ R.T + g_c).reshape(pred.shape)


def _complex_rmsd_atoms(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-atom RMSD over the whole complex (no alignment — uses native frame)."""
    diff = pred - gt
    return float(np.sqrt((diff ** 2).sum(-1).mean()))


def _interface_rmsd_ca(
    pred: np.ndarray, gt: np.ndarray, chain_ids: np.ndarray, cutoff: float = 8.0
) -> float:
    """CA-only RMSD over interface residues (GT-defined contacts within cutoff Å)."""
    ca_pred = pred[:, 1]
    ca_gt = gt[:, 1]
    a = chain_ids == 0
    b = chain_ids == 1
    if not a.any() or not b.any():
        return float("nan")
    d = np.linalg.norm(ca_gt[a][:, None, :] - ca_gt[b][None, :, :], axis=-1)
    a_iface = (d.min(axis=1) < cutoff)
    b_iface = (d.min(axis=0) < cutoff)
    mask = np.zeros(chain_ids.shape[0], dtype=bool)
    mask[np.where(a)[0][a_iface]] = True
    mask[np.where(b)[0][b_iface]] = True
    if mask.sum() == 0:
        return float("nan")
    diff = ca_pred[mask] - ca_gt[mask]
    return float(np.sqrt((diff ** 2).sum(-1).mean()))


def _load_gt(table, sample_id: str, _id_cache: dict | None = None) -> dict | None:
    """Look up ground-truth atoms/aa/chains for a sample in the master parquet.

    Atom layout per residue: (N, CA, C, O) — verified via atom_type = [0,1,2,3]
    repeating in samples.parquet. Coords are returned in REAL Angstroms (no
    centering, no normalisation): matches what the baselines emit, and what
    DockQ / RMSD want.
    """
    if _id_cache is not None and sample_id in _id_cache:
        i = _id_cache[sample_id]
    else:
        ids = table["sample_id"].to_pylist()
        if sample_id not in ids:
            return None
        i = ids.index(sample_id)
    coords_flat = np.asarray(table["atom_coords"][i].as_py(), dtype=np.float32)
    n_atoms = coords_flat.shape[0] // 3
    n_res = n_atoms // 4
    atoms = coords_flat.reshape(n_res, 4, 3)
    aa = np.asarray(table["seq"][i].as_py(), dtype=np.int64)
    chains = np.asarray(table["chain_id_res"][i].as_py(), dtype=np.int64)
    return {"atoms": atoms, "aa": aa, "chains": chains}


_NAN_RESULT = {
    "c_rmsd_raw_A": float("nan"),
    "c_rmsd_aligned_A": float("nan"),
    "interface_rmsd_ca_A": float("nan"),
    "dockq": None,
    "fnat": None,
    "irms_dockq_A": None,
    "lrms_dockq_A": None,
}


def score_one(
    pred_atoms: np.ndarray,
    gt_atoms: np.ndarray,
    aa: np.ndarray,
    chains: np.ndarray,
) -> dict:
    """Run the full metric suite on one sample.

    All inputs in real Angstroms (NOT normalized). Returns a flat dict ready
    to write as a CSV row. Returns NaN-valued metrics when ``pred_atoms``
    contains NaN (used by external baselines that mark divergent predictions).
    """
    if not np.isfinite(pred_atoms).all():
        return dict(_NAN_RESULT)
    aligned = _kabsch_align(pred_atoms, gt_atoms, valid=np.ones(len(aa), dtype=bool))
    c_rmsd_raw = _complex_rmsd_atoms(pred_atoms, gt_atoms)
    c_rmsd_aligned = _complex_rmsd_atoms(aligned, gt_atoms)
    irmsd = _interface_rmsd_ca(aligned, gt_atoms, chains)

    dq = compute_dockq(
        torch.from_numpy(pred_atoms).float(),
        torch.from_numpy(gt_atoms).float(),
        torch.from_numpy(aa).long(),
        torch.from_numpy(chains).long(),
        std=1.0,
    )

    return {
        "c_rmsd_raw_A": c_rmsd_raw,
        "c_rmsd_aligned_A": c_rmsd_aligned,
        "interface_rmsd_ca_A": irmsd,
        "dockq": dq.get("dockq"),
        "fnat": dq.get("fnat"),
        "irms_dockq_A": dq.get("irms"),
        "lrms_dockq_A": dq.get("lrms"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Subdir under benchmarks/predictions/ (e.g. 'diffdock_pp').")
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--predictions_root", default="benchmarks/predictions")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default benchmarks/results/{model}.csv).")
    p.add_argument("--splits", nargs="*", default=SPLITS,
                   help=f"Subset of stratified bins to score (default: all {SPLITS}).")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Match the value passed to eval_tinyfold.py. K=1 reads "
                        "{sample_id}.npz (existing behavior). K>1 reads "
                        "{sample_id}_k{i}.npz, emits one CSV row per (sample, k), "
                        "and the per-bin summary prints top-1 / mean / oracle.")
    args = p.parse_args()

    pred_root = REPO_ROOT / args.predictions_root / args.model
    if not pred_root.exists():
        sys.exit(f"ERROR: no predictions at {pred_root}")

    out_path = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "results" / f"{args.model}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading parquet: {args.parquet}")
    table = pq.read_table(args.parquet)
    print(f"  {len(table)} samples in master parquet")
    # Build an O(1) sample_id -> row index map once; the per-call list scan was
    # quadratic on a 40K-row parquet.
    id_cache = {sid: i for i, sid in enumerate(table["sample_id"].to_pylist())}

    K = args.n_samples
    k_pat = re.compile(r"^(.+)_k(\d+)\.npz$")
    rows = []
    for split in args.splits:
        split_dir = pred_root / split
        if not split_dir.exists():
            print(f"  [skip] {split}: no predictions at {split_dir}")
            continue
        if K > 1:
            npzs = sorted(split_dir.glob("*_k*.npz"))
        else:
            npzs = sorted(p_ for p_ in split_dir.glob("*.npz") if not k_pat.match(p_.name))
        print(f"  {split}: {len(npzs)} predictions")
        for npz_path in npzs:
            if K > 1:
                m = k_pat.match(npz_path.name)
                if not m:
                    continue
                stem_id, k_idx = m.group(1), int(m.group(2))
            else:
                stem_id, k_idx = npz_path.stem, 0
            with np.load(npz_path, allow_pickle=True) as data:
                pred = np.asarray(data["pred_atoms"], dtype=np.float32)
                sample_id = (
                    str(data["sample_id"]) if "sample_id" in data.files
                    else stem_id
                )
            gt = _load_gt(table, sample_id, _id_cache=id_cache)
            if gt is None:
                print(f"    [skip] {sample_id}: not in parquet")
                continue
            if pred.shape != gt["atoms"].shape:
                print(f"    [skip] {sample_id}: shape mismatch pred={pred.shape} gt={gt['atoms'].shape}")
                continue
            scores = score_one(pred, gt["atoms"], gt["aa"], gt["chains"])
            row = {
                "model": args.model,
                "bin": split,
                "sample_id": sample_id,
                "n_res": len(gt["aa"]),
                **scores,
            }
            if K > 1:
                row["k_idx"] = k_idx
            rows.append(row)

    if not rows:
        sys.exit("ERROR: no scored samples")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

    # Quick per-bin summary so the user sees something useful immediately.
    # nan-safe: external baselines (DPP) occasionally write NaN-valued NPZs
    # for samples where the model diverged; filter those out so they don't
    # poison the bin mean.
    import math
    by_bin: dict[str, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(r["bin"], []).append(r)

    def _finite(xs):
        return [x for x in xs if x is not None and not math.isnan(x)]

    if K == 1:
        print("\nPer-bin mean C-RMSD (aligned) / mean DockQ:")
        print(f"  {'bin':>14} {'n':>4} {'n_ok':>5} {'c_rmsd_A':>10} {'dockq':>8}")
        for b in SPLITS:
            if b not in by_bin:
                continue
            rs = by_bin[b]
            c = _finite([r["c_rmsd_aligned_A"] for r in rs])
            dq = _finite([r["dockq"] for r in rs])
            c_mean = sum(c) / len(c) if c else float("nan")
            d_mean = sum(dq) / len(dq) if dq else float("nan")
            print(f"  {b:>14} {len(rs):>4} {len(c):>5} {c_mean:>10.2f} {d_mean:>8.3f}")
    else:
        # K-sample readout: per sample, take top-1 (k=0), mean, oracle (min c_rmsd).
        # Per bin, average each across samples. The oracle is the ceiling that a
        # perfect reranker could reach; the gap vs top-1 is the Cycle 02 lever.
        print(f"\nPer-bin K={K} mean across samples of [top1 / mean / oracle] (C-RMSD A, DockQ):")
        print(f"  {'bin':>14} {'n_smp':>5}"
              f" {'c_top1':>8} {'c_mean':>8} {'c_oracle':>9}"
              f" {'dq_top1':>8} {'dq_mean':>8} {'dq_oracle':>9}")
        for b in SPLITS:
            if b not in by_bin:
                continue
            per_sample: dict[str, list[dict]] = defaultdict(list)
            for r in by_bin[b]:
                per_sample[r["sample_id"]].append(r)
            top1_c, mean_c, oracle_c = [], [], []
            top1_d, mean_d, oracle_d = [], [], []
            for sid, rs in per_sample.items():
                rs_sorted = sorted(rs, key=lambda r: r["k_idx"])
                cs = _finite([r["c_rmsd_aligned_A"] for r in rs_sorted])
                ds = _finite([r["dockq"] for r in rs_sorted])
                if cs:
                    top1_c.append(rs_sorted[0]["c_rmsd_aligned_A"])
                    mean_c.append(sum(cs) / len(cs))
                    oracle_c.append(min(cs))
                if ds:
                    top1_d.append(rs_sorted[0]["dockq"])
                    mean_d.append(sum(ds) / len(ds))
                    # For DockQ higher is better, so "oracle" is max.
                    oracle_d.append(max(ds))
            n_smp = len(per_sample)
            avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
            print(f"  {b:>14} {n_smp:>5}"
                  f" {avg(top1_c):>8.2f} {avg(mean_c):>8.2f} {avg(oracle_c):>9.2f}"
                  f" {avg(top1_d):>8.3f} {avg(mean_d):>8.3f} {avg(oracle_d):>9.3f}")


if __name__ == "__main__":
    main()
