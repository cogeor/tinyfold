#!/usr/bin/env python
"""Predict a protein-protein complex for one dataset sample and write it to PDB.

A thin standalone CLI composing the deduplicated inference primitives:
``tinyfold.inference.load_onestep_run`` (architecture from the run's config.json)
+ ``load_sample_raw`` (cached ESM-2 embeddings) + ``sample_k_centroids`` (K-sample
one-shot inference, confidence-ranked) + the canonical PDB writer.

It operates on a **dataset sample** (DIPS-Plus parquet row) because the headline
model is ESM-conditioned (needs per-residue ESM-2 embeddings) and predicts in
normalized coordinates (absolute scale comes from the sample's ``std``). Predicting
from a raw FASTA/PDB pair additionally requires live ESM-2 inference and a de-novo
scale convention — tracked as future work (see the repo README roadmap).

Usage:
    python scripts/predict.py \
        --checkpoint outputs/resfold/small_specialist_le200/<run>/best_model.pt \
        --sample_id 3lz0.pdb1_5 --out pred.pdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from tinyfold.inference import load_onestep_run, sample_k_centroids
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.metrics import compute_dockq
from tinyfold.training import collate_batch, load_sample_raw
from tinyfold.viz.io.structure_writer import coords_to_pdb_string


def _coords_res_to_pdb(coords_res: np.ndarray, chain_ids, res_idx, aa_seq) -> str:
    L = coords_res.shape[0]
    xyz = coords_res.reshape(L * 4, 3)
    atom_to_res = np.repeat(np.arange(L), 4)
    atom_type = np.tile(np.arange(4), L)  # 0=N,1=CA,2=C,3=O
    return coords_to_pdb_string(xyz, atom_to_res, atom_type, chain_ids, res_idx, seq=aa_seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample_id", help="parquet sample_id (e.g. 3lz0.pdb1_5)")
    ap.add_argument("--index", type=int, help="parquet row index (alternative to --sample_id)")
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--esm_dir", default="data/processed/esm2_35M")
    ap.add_argument("--out", default="prediction.pdb")
    ap.add_argument("--K", type=int, default=5, help="samples to draw; confidence-best is kept")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write_gt", action="store_true", help="also write <out>.gt.pdb")
    ap.add_argument("--ema", action="store_true",
                    help="load the EMA weights (ema_state_dict) instead of raw (C3)")
    args = ap.parse_args()

    if not args.sample_id and args.index is None:
        ap.error("provide --sample_id or --index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_onestep_run(args.checkpoint, device, ema=args.ema)
    table = pq.read_table(args.parquet)

    if args.index is not None:
        idx = args.index
    else:
        ids = table["sample_id"].to_pylist()
        if args.sample_id not in ids:
            raise SystemExit(f"sample_id {args.sample_id!r} not found in {args.parquet}")
        idx = ids.index(args.sample_id)

    s = load_sample_raw(table, idx, normalize=True, esm_cache_dir=args.esm_dir,
                        per_chain_res_idx=True)
    batch = collate_batch([s], device)
    n_res, std = s["n_res"], s["std"]

    noiser = VENoiser(KarrasSchedule(n_steps=50, sigma_min=0.002, sigma_max=10.0, rho=7.0),
                      sigma_data=1.0).to(device)
    cents, atoms, lddts = sample_k_centroids(
        model, batch, noiser, device, K=args.K, base_seed=args.seed,
        target_idx=0, is_onestep=True, one_shot=True,
    )
    pick = int(torch.argmax(lddts[:, 0]).item()) if lddts is not None else 0
    pred_atoms = atoms[pick][0, :n_res]  # [L,4,3] normalized

    chain_ids = batch["chain_ids"][0, :n_res].cpu().numpy()
    res_idx = batch["res_idx"][0, :n_res].cpu().numpy()
    aa_seq = batch["aa_seq"][0, :n_res].cpu().numpy()

    dq = compute_dockq(pred_atoms, batch["coords_res"][0, :n_res],
                       batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res], std=std)["dockq"]

    pred_pdb = _coords_res_to_pdb((pred_atoms * std).cpu().numpy(), chain_ids, res_idx, aa_seq)
    out = Path(args.out)
    out.write_text(pred_pdb, encoding="utf-8")
    print(f"Wrote prediction -> {out}  (sample {s['sample_id']}, "
          f"{n_res} residues, DockQ vs GT {dq:.3f})" if dq is not None
          else f"Wrote prediction -> {out}  (sample {s['sample_id']}, {n_res} residues)")

    if args.write_gt:
        gt_pdb = _coords_res_to_pdb((batch["coords_res"][0, :n_res] * std).cpu().numpy(),
                                    chain_ids, res_idx, aa_seq)
        gt_out = out.with_suffix(out.suffix + ".gt.pdb")
        gt_out.write_text(gt_pdb, encoding="utf-8")
        print(f"Wrote ground truth -> {gt_out}")


if __name__ == "__main__":
    main()
