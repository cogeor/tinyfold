#!/usr/bin/env python
"""Per-target DockQ for one checkpoint, stratified by train/test cluster leakage.

Answers the question an aggregate mean cannot: is the score generalization or
memorization? Writes one CSV row per test complex, annotated with its sequence
cluster, how many training complexes share that cluster, and whether it leaked.

This is the instrument that showed the le200 headline (DockQ 0.233 / 41% succ /
23% medium) was an artifact:

    cluster-LEAKED (n=183): DockQ 0.251, 44.3% succ, 25.1% medium
    cluster-CLEAN  (n= 17): DockQ 0.044,  5.9% succ,  0.0% medium

Usage:
    uv run python scripts/eval_leakage_split.py \
        --checkpoint outputs/.../best_model.pt \
        --split outputs/.../split.json \
        --out benchmarks/results/<name>_pertarget.csv

Column layout follows the benchmarks/results/tinyfold_*.csv convention for the
shared fields (sample_id, n_res, dockq, fnat, ...) and appends the leakage
columns. Fields those benchmark CSVs carry but this script does not compute
(c_rmsd_aligned_A, interface_rmsd_ca_A) are omitted rather than filled with
placeholder values.
"""
import argparse
import csv
import json
import os

import pyarrow.parquet as pq
import torch

from tinyfold.inference import load_onestep_run, sample_k_centroids
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses import compute_c_rmsd
from tinyfold.model.metrics import capri_band, compute_dockq
from tinyfold.training import collate_batch, load_sample_raw
from tinyfold.training.cluster_split import load_clusters
from tinyfold.training.leakage_report import annotate_leakage, format_report

CSV_FIELDS = [
    "sample_id", "n_res", "dockq", "capri_band", "fnat",
    "irms_dockq_A", "lrms_dockq_A", "c_rmsd_A", "chain_perm_used",
    "cluster", "n_same_cluster_train", "cluster_leaked",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True,
                    help="split.json providing test_indices/test_ids and train_ids")
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default="data/processed/esm2_35M")
    ap.add_argument("--clusters", default="data/processed/clusters.json")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="per-target CSV path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    table = pq.read_table(args.parquet)
    model, _ = load_onestep_run(args.checkpoint, device)
    print(f"Loaded checkpoint {args.checkpoint}")

    # Match the training-time coordinate normalization, or the model sees
    # coords at the wrong scale.
    global_scale = None
    cfg_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
    if os.path.exists(cfg_path):
        global_scale = json.load(open(cfg_path)).get("global_scale")
    if global_scale is not None:
        print(f"Using fixed-scale normalization: coords / {global_scale:.2f} A")

    schedule = KarrasSchedule(n_steps=50, sigma_min=0.002, sigma_max=10.0, rho=7.0)
    noiser = VENoiser(schedule, sigma_data=1.0).to(device)

    split = json.load(open(args.split))
    clusters = load_clusters(args.clusters)

    rows = []
    test_indices = split["test_indices"]
    for pos, idx in enumerate(test_indices):
        s = load_sample_raw(table, idx, normalize=True, esm_cache_dir=args.esm_dir,
                            per_chain_res_idx=True, global_scale=global_scale)
        batch = collate_batch([s], device)
        n_res = s["n_res"]
        cents, atoms, lddts = sample_k_centroids(
            model, batch, noiser, device, K=args.K, base_seed=args.seed,
            target_idx=pos, is_onestep=True, one_shot=True,
        )
        pick = int(torch.argmax(lddts[:, 0]).item()) if lddts is not None else 0
        dq = compute_dockq(
            atoms[pick][0, :n_res], batch["coords_res"][0, :n_res],
            batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res], std=s["std"],
        )
        c_rmsd = compute_c_rmsd(
            pred_ca=cents[pick][:, :n_res],
            gt_ca=batch["centroids"][:, :n_res],
            chain_ids=batch["chain_ids"][:, :n_res],
            mask=batch["mask_res"][:, :n_res],
        ).item() * s["std"]

        rows.append({
            "sample_id": split["test_ids"][pos],
            "n_res": int(n_res),
            "dockq": dq["dockq"],
            "capri_band": capri_band(dq["dockq"]),
            "fnat": dq["fnat"],
            "irms_dockq_A": dq["irms"],
            "lrms_dockq_A": dq["lrms"],
            "c_rmsd_A": c_rmsd,
            "chain_perm_used": dq.get("chain_perm_used", False),
        })
        if (pos + 1) % 25 == 0:
            print(f"{pos + 1}/{len(test_indices)}", flush=True)

    annotate_leakage(rows, split.get("train_ids", []), clusters)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k) for k in CSV_FIELDS} for r in rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    print()
    print(format_report(rows, title=os.path.basename(args.checkpoint)))


if __name__ == "__main__":
    main()
