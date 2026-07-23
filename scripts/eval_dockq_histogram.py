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
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from tinyfold.inference import load_onestep_run, sample_k_centroids
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses import compute_c_rmsd
from tinyfold.model.metrics import capri_band as band
from tinyfold.model.metrics import compute_dockq
from tinyfold.training import collate_batch, load_sample_raw
from tinyfold.training.cluster_split import load_clusters
from tinyfold.training.leakage_report import annotate_leakage, format_report


def eval_split(model, noiser, table, test_indices, device, esm_dir,
               per_chain, K, seed, label, global_scale=None, sample_ids=None):
    bands = {"incorrect": 0, "acceptable": 0, "medium": 0, "high": 0}
    dockqs, c_rmsds = [], []
    rows = []
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
        if sample_ids is not None:
            rows.append({"sample_id": sample_ids[pos], "dockq": dq,
                         "n_res": int(n_res), "c_rmsd_A": c})
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
            "mean_c_rmsd": mean_c, "bands": bands, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default="data/processed/esm2_35M")
    ap.add_argument("--small_split", required=True,
                    help="run split.json (uses its test_indices as the small test set)")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clusters", default="data/processed/clusters.json",
                    help="clusters.json for the leakage-stratified read-out")
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

    # Leakage stratification needs the cluster map; without it an aggregate
    # mean cannot distinguish generalization from memorization.
    clusters = None
    if os.path.exists(args.clusters):
        clusters = load_clusters(args.clusters)
    else:
        print(f"WARNING: no clusters at {args.clusters}; "
              "skipping the leakage-stratified read-out")

    results = []
    for label, path in splits:
        d = json.load(open(path))
        test_idx = d["test_indices"]
        res = eval_split(
            model, noiser, table, test_idx, device, args.esm_dir,
            per_chain=True, K=args.K, seed=args.seed, label=label,
            global_scale=global_scale, sample_ids=d.get("test_ids"),
        )
        if clusters is not None and res["rows"] and d.get("train_ids"):
            annotate_leakage(res["rows"], d["train_ids"], clusters)
            print(format_report(res["rows"]))
        results.append(res)

    if args.out:
        # Per-target rows stay out of the aggregate JSON; use
        # scripts/eval_leakage_split.py when the per-target CSV is wanted.
        json.dump([{k: v for k, v in r.items() if k != "rows"} for r in results],
                  open(args.out, "w"), indent=2)
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
