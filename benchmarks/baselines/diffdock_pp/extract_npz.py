"""Read a DPP prediction_storage pickle and emit NPZ predictions.

Runs INSIDE the DPP venv (the pickle contains torch_geometric HeteroData
objects, which require torch_geometric to unpickle).

Reconstruction: DPP outputs CA-only positions in its own centered frame.
We need full-backbone in the parquet (Angstrom) frame for compute_metrics:
  1. Kabsch(DPP_gt_receptor_CA -> parquet_receptor_CA) gives the global
     frame transform.
  2. Apply that transform to DPP_pred_ligand_CA to bring it into parquet
     frame.
  3. Kabsch(parquet_ligand_CA -> pred_ligand_CA_in_parquet_frame) gives
     the ligand's rigid motion.
  4. Apply that motion to the parquet ligand backbone (N, CA, C, O).
  5. Receptor backbone stays as-is (DPP rigid-docks, receptor unchanged).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Treat predictions as failed if any CA coord exceeds this (Angstroms).
# Real complexes fit comfortably in ~100 Å; DPP occasionally diverges to
# numbers in the 10^4-10^5 range — those samples become NaN in NPZ.
DIVERGENCE_THRESHOLD_A = 1000.0


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find R, t such that (R @ P + t) ~= Q."""
    p_c = P.mean(0)
    q_c = Q.mean(0)
    A = P - p_c
    B = Q - q_c
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_c - R @ p_c
    return R.astype(np.float32), t.astype(np.float32)


def reconstruct(
    parq_atoms: np.ndarray,   # [L, 4, 3] real Angstroms
    parq_chains: np.ndarray,  # [L] {0, 1}
    gt_rec_ca: np.ndarray,    # [LR, 3] DPP frame
    gt_lig_ca: np.ndarray,    # [LL, 3] DPP frame
    pred_lig_ca: np.ndarray,  # [LL, 3] DPP frame
) -> np.ndarray:
    chain_a = parq_chains == 0
    chain_b = parq_chains == 1
    assert chain_a.sum() == gt_rec_ca.shape[0]
    assert chain_b.sum() == gt_lig_ca.shape[0]
    parq_rec_ca = parq_atoms[chain_a, 1, :]
    parq_lig_ca = parq_atoms[chain_b, 1, :]

    R_g, t_g = kabsch(gt_rec_ca, parq_rec_ca)
    pred_lig_ca_pframe = pred_lig_ca @ R_g.T + t_g
    R_l, t_l = kabsch(parq_lig_ca, pred_lig_ca_pframe)

    lig_bb_in = parq_atoms[chain_b]
    lig_bb_out = lig_bb_in @ R_l.T + t_l
    out = parq_atoms.copy()
    out[chain_b] = lig_bb_out
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.pickle}")
    with open(args.pickle, "rb") as f:
        results = pickle.load(f)
    print(f"  {len(results)} complexes in pickle")

    split = json.load(open(args.split))
    test_ids = split["test_ids"][:args.limit] if args.limit > 0 else split["test_ids"]
    if len(results) != len(test_ids):
        sys.exit(f"ERROR: results len {len(results)} != test_ids len {len(test_ids)}")

    table = pq.read_table(args.parquet)
    id_to_idx = {sid: i for i, sid in enumerate(table["sample_id"].to_pylist())}

    n_ok = 0
    n_diverged = 0
    for i, sid in enumerate(test_ids):
        if sid not in id_to_idx:
            print(f"  [skip] {sid}: not in parquet")
            continue
        idx = id_to_idx[sid]
        coords_flat = np.asarray(table["atom_coords"][idx].as_py(), dtype=np.float32)
        n_atoms = coords_flat.shape[0] // 3
        n_res = n_atoms // 4
        parq_atoms = coords_flat.reshape(n_res, 4, 3)
        parq_chains = np.asarray(table["chain_id_res"][idx].as_py(), dtype=np.int64)

        gt_entry, _ = results[i][0]
        pred_entry, conf = results[i][1]

        gt_rec_ca = gt_entry["receptor"].pos.detach().cpu().numpy().astype(np.float32)
        gt_lig_ca = gt_entry["ligand"].pos.detach().cpu().numpy().astype(np.float32)
        pred_lig_ca = pred_entry["ligand"].pos.detach().cpu().numpy().astype(np.float32)

        # Catastrophic divergence — DPP rarely produces ligand coords > 1000 Å.
        # Record the sample but mark it as failed so the metric is still
        # computable across the bin.
        if np.abs(pred_lig_ca).max() > DIVERGENCE_THRESHOLD_A:
            pred_atoms = np.full_like(parq_atoms, np.nan)
            n_diverged += 1
            print(f"  [diverged] {sid}: pred ligand range "
                  f"[{pred_lig_ca.min():.0f}, {pred_lig_ca.max():.0f}] Å — NaN")
        else:
            try:
                pred_atoms = reconstruct(parq_atoms, parq_chains, gt_rec_ca, gt_lig_ca, pred_lig_ca)
            except AssertionError as e:
                print(f"  [skip] {sid}: {e}")
                continue
            n_ok += 1

        np.savez(
            args.out_dir / f"{sid}.npz",
            pred_atoms=pred_atoms.astype(np.float32),
            sample_id=sid,
            dpp_confidence=float(conf),
            diverged=bool(np.isnan(pred_atoms).any()),
        )

    print(f"[done] {n_ok} reconstructed, {n_diverged} diverged, "
          f"wrote {n_ok + n_diverged} NPZs to {args.out_dir}")


if __name__ == "__main__":
    main()
