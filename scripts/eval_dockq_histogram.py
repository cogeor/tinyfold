#!/usr/bin/env python
"""Per-complex DockQ + CAPRI-band histogram for a trained ResFoldOneStep ckpt.

Standalone read-out for the Small-Specialist Confirmation (SSC) experiment.
Loads ONE checkpoint and scores it on several test splits (the small test set
plus the large OOD size bins) WITHOUT the 8,600-sample train preload that the
training script's --eval_only path triggers.

For each split it prints the CAPRI band counts:
    incorrect  DockQ < 0.23
    acceptable 0.23 <= DockQ < 0.49
    medium     0.49 <= DockQ < 0.80
    high       DockQ >= 0.80
plus mean DockQ, success% (>=0.23), and mean C-RMSD.

Usage:
    python scripts/eval_dockq_histogram.py --checkpoint <best_model.pt> \
        --small_split <run>/split.json
"""
import argparse
import json
import sys
import os

import numpy as np
import torch
import pyarrow.parquet as pq

from tinyfold.inference import sample_k_centroids, load_onestep_run
from tinyfold.training import load_sample_raw, collate_batch
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.metrics import compute_dockq
from tinyfold.model.losses import compute_c_rmsd


def band(dq):
    if dq is None:
        return None
    if dq < 0.23:
        return "incorrect"
    if dq < 0.49:
        return "acceptable"
    if dq < 0.80:
        return "medium"
    return "high"


def eval_split(model, noiser, table, test_indices, device, esm_dir,
               per_chain, K, seed, label, global_scale=None):
    bands = {"incorrect": 0, "acceptable": 0, "medium": 0, "high": 0}
    dockqs, c_rmsds = [], []
    n_skip = 0
    for pos, idx in enumerate(test_indices):
        s = load_sample_raw(table, idx, normalize=True,
                            esm_cache_dir=esm_dir, per_chain_res_idx=per_chain,
                            global_scale=global_scale)
        batch = collate_batch([s], device)
        n_res = s["n_res"]
        cents, atoms, lddts = sample_k_centroids(
            model, batch, noiser, device, K=K, base_seed=seed,
            target_idx=pos, is_onestep=True, one_shot=True,
        )
        # rank_by confidence: pick argmax predicted lDDT (matches training eval)
        if lddts is not None:
            pick = int(torch.argmax(lddts[:, 0]).item())
        else:
            pick = 0
        pred_atoms = atoms[pick][0, :n_res]              # [L,4,3] normalized
        gt_atoms = batch["coords_res"][0, :n_res]
        res = compute_dockq(
            pred_atoms, gt_atoms,
            batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res],
            std=s["std"],
        )
        dq = res["dockq"]
        if dq is None:
            n_skip += 1
        else:
            dockqs.append(dq)
            bands[band(dq)] += 1
        c = compute_c_rmsd(
            pred_ca=cents[pick][:, :n_res],
            gt_ca=batch["centroids"][:, :n_res],
            chain_ids=batch["chain_ids"][:, :n_res],
            mask=batch["mask_res"][:, :n_res],
        ).item() * s["std"]
        c_rmsds.append(c)
    n = len(dockqs)
    mean_dq = float(np.mean(dockqs)) if n else float("nan")
    succ = 100.0 * sum(1 for d in dockqs if d >= 0.23) / n if n else float("nan")
    mean_c = float(np.mean(c_rmsds)) if c_rmsds else float("nan")
    print(f"\n=== {label}  (n={n}{f', skipped {n_skip}' if n_skip else ''}) ===")
    print(f"  mean DockQ: {mean_dq:.4f}   success(>=0.23): {succ:.1f}%   "
          f"mean C-RMSD: {mean_c:.2f} A")
    print(f"  CAPRI bands: incorrect={bands['incorrect']}  "
          f"acceptable={bands['acceptable']}  medium={bands['medium']}  "
          f"high={bands['high']}")
    return {"label": label, "n": n, "mean_dockq": mean_dq, "success": succ,
            "mean_c_rmsd": mean_c, "bands": bands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default="data/processed/esm2_35M")
    ap.add_argument("--small_split", required=True,
                    help="run split.json (uses its test_indices as the small test set)")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading parquet...")
    table = pq.read_table(args.parquet)

    # Architecture is read from the run's config.json (next to the checkpoint),
    # so this matches whatever the checkpoint was trained with.
    model, _ = load_onestep_run(args.checkpoint, device)
    print(f"Loaded checkpoint {args.checkpoint}")

    # Match the training-time coordinate normalization. If the run used
    # fixed-scale (global_scale in its config.json), eval MUST divide by the
    # same constant or the model sees coords at the wrong scale.
    global_scale = None
    cfg_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
    if os.path.exists(cfg_path):
        global_scale = json.load(open(cfg_path)).get("global_scale")
    if global_scale is not None:
        print(f"Using fixed-scale normalization: coords / {global_scale:.2f} A")

    schedule = KarrasSchedule(n_steps=50, sigma_min=0.002, sigma_max=10.0, rho=7.0)
    noiser = VENoiser(schedule, sigma_data=1.0).to(device)

    # Splits: small test set (in-distribution) + large OOD bins.
    splits = [("small_le200 (test)", args.small_split)]
    for b in ["200_400", "400_600", "600_1000", "ge1000"]:
        p = f"data/processed/splits/phase_d_bin_{b}.json"
        if os.path.exists(p):
            splits.append((f"OOD {b}", p))

    results = []
    for label, path in splits:
        d = json.load(open(path))
        test_idx = d["test_indices"]
        results.append(eval_split(
            model, noiser, table, test_idx, device, args.esm_dir,
            per_chain=True, K=args.K, seed=args.seed, label=label,
            global_scale=global_scale,
        ))

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
