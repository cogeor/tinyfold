#!/usr/bin/env python
"""Build the web-light showcase from a trained ResFoldOneStep checkpoint.

Zero-friction showcase pipeline for the small-specialist result: takes a
checkpoint + a split JSON, runs the real K-sample one-shot inference on each
held-out test complex, picks the confidence-ranked pose, Kabsch-aligns the
predicted complex onto the ground truth (so the overlay is meaningful), scores
DockQ / C-RMSD, and writes the top-N by DockQ to ``assets/showcase_samples.json``
in the exact schema ``web-light`` consumes.

Unlike the legacy ``web/predict_all.py`` path (built for the old af3_style /
two-stage models), this is onestep-native and self-contained.

Usage:
    python scripts/web/build_showcase.py \
        --checkpoint outputs/resfold/small_specialist_le200/resfold_s1_3K_20260613_161213/best_model.pt \
        --split      outputs/resfold/small_specialist_le200/resfold_s1_3K_20260613_161213/split.json \
        --top 6
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from tinyfold.inference import load_onestep_run, sample_k_centroids
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.geometry import kabsch_rigid
from tinyfold.model.losses import compute_c_rmsd, compute_rmse
from tinyfold.model.metrics import compute_dockq
from tinyfold.training import collate_batch, load_sample_raw
from tinyfold.viz.io.structure_writer import coords_to_pdb_string


def capri_band(dq: float) -> str:
    if dq < 0.23:
        return "incorrect"
    if dq < 0.49:
        return "acceptable"
    if dq < 0.80:
        return "medium"
    return "high"


def coords_res_to_pdb(coords_res: np.ndarray, chain_ids: np.ndarray,
                      res_idx: np.ndarray, aa_seq: np.ndarray) -> str:
    """[L,4,3] (N,CA,C,O per residue) -> PDB string via the canonical writer."""
    L = coords_res.shape[0]
    xyz = coords_res.reshape(L * 4, 3)
    atom_to_res = np.repeat(np.arange(L), 4)
    atom_type = np.tile(np.arange(4), L)  # 0=N,1=CA,2=C,3=O
    return coords_to_pdb_string(xyz, atom_to_res, atom_type, chain_ids, res_idx, seq=aa_seq)


def build(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split = json.loads(Path(args.split).read_text())
    test_indices = split["test_indices"]
    print(f"Loaded {len(test_indices)} test indices from {args.split}")

    print("Loading parquet...")
    table = pq.read_table(args.parquet)

    # Architecture read from the run's config.json (next to the checkpoint).
    model, _ = load_onestep_run(args.checkpoint, device)
    print(f"Loaded checkpoint {args.checkpoint}")

    schedule = KarrasSchedule(n_steps=50, sigma_min=0.002, sigma_max=10.0, rho=7.0)
    noiser = VENoiser(schedule, sigma_data=1.0).to(device)

    rows = []
    for pos, idx in enumerate(test_indices):
        s = load_sample_raw(table, idx, normalize=True,
                            esm_cache_dir=args.esm_dir, per_chain_res_idx=True)
        batch = collate_batch([s], device)
        n_res = s["n_res"]
        std = s["std"]

        t0 = time.time()
        cents, atoms, lddts = sample_k_centroids(
            model, batch, noiser, device, K=args.K, base_seed=42,
            target_idx=pos, is_onestep=True, one_shot=True,
        )
        infer_s = (time.time() - t0) / args.K  # per-sample wall-clock

        pick = int(torch.argmax(lddts[:, 0]).item()) if lddts is not None else 0
        pred_atoms = atoms[pick][0, :n_res]              # [L,4,3] normalized
        gt_atoms = batch["coords_res"][0, :n_res]

        dq = compute_dockq(
            pred_atoms, gt_atoms,
            batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res], std=std,
        )["dockq"]
        if dq is None:
            continue
        c_rmsd = compute_c_rmsd(
            pred_ca=cents[pick][:, :n_res], gt_ca=batch["centroids"][:, :n_res],
            chain_ids=batch["chain_ids"][:, :n_res], mask=batch["mask_res"][:, :n_res],
        ).item() * std
        atom_rmse = compute_rmse(
            pred_atoms.reshape(1, n_res * 4, 3), gt_atoms.reshape(1, n_res * 4, 3),
            batch["mask_atom"][:, : n_res * 4],
        ).item() * std

        # To Angstroms, then rigid-align the whole predicted complex onto GT so
        # the overlay is a true superposition (DockQ is alignment-invariant; the
        # viewer is not).
        gt_a = (gt_atoms * std)                           # [L,4,3]
        pred_a = (pred_atoms * std)
        flat_mask = torch.ones(1, n_res * 4, device=device)
        _, _, pred_aln = kabsch_rigid(
            pred_a.reshape(1, n_res * 4, 3), gt_a.reshape(1, n_res * 4, 3), flat_mask,
        )
        pred_aln = pred_aln.reshape(n_res, 4, 3).cpu().numpy()
        gt_np = gt_a.cpu().numpy()

        chain_ids = batch["chain_ids"][0, :n_res].cpu().numpy()
        res_idx = batch["res_idx"][0, :n_res].cpu().numpy()
        aa_seq = batch["aa_seq"][0, :n_res].cpu().numpy()

        rows.append({
            "sample_id": s["sample_id"],
            "split": "test",
            "n_atoms": int(n_res * 4),
            "n_residues": int(n_res),
            "dockq": round(float(dq), 3),
            "capri": capri_band(float(dq)),
            "c_rmsd": round(float(c_rmsd), 2),
            "rmsd": round(float(atom_rmse), 2),
            "inference_time": round(float(infer_s), 3),
            "ground_truth_pdb": coords_res_to_pdb(gt_np, chain_ids, res_idx, aa_seq),
            "prediction_pdb": coords_res_to_pdb(pred_aln, chain_ids, res_idx, aa_seq),
        })
        if (pos + 1) % 25 == 0:
            print(f"  scored {pos + 1}/{len(test_indices)}")

    rows.sort(key=lambda r: r["dockq"], reverse=True)
    selected = rows[: args.top]
    bands = {}
    for r in selected:
        bands[r["capri"]] = bands.get(r["capri"], 0) + 1
    print(f"\nSelected top {len(selected)} of {len(rows)} by DockQ. Bands: {bands}")
    for r in selected:
        print(f"  {r['sample_id']:>16}  DockQ {r['dockq']:.3f} ({r['capri']})  "
              f"C-RMSD {r['c_rmsd']:.1f} A  {r['n_residues']} res")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "model": "small-specialist (le200, onestep, 11.8M params)",
        "checkpoint": str(args.checkpoint),
        "note": "Held-out TEST complexes (never seen in training), ranked by DockQ. "
                "Blue/green = ground truth, red/orange = prediction.",
        "samples": selected,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nWrote {len(selected)} showcase samples -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, help="run split.json (uses test_indices)")
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default="data/processed/esm2_35M")
    ap.add_argument("--top", type=int, default=6, help="number of best-DockQ samples to keep")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out", default="assets/showcase_samples.json")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
