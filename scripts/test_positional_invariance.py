#!/usr/bin/env python
"""Quick diagnostic: does Phase D's prediction depend on res_idx OFFSET?

For any protein, residue 5 of chain B should have the same biological
role regardless of whether chain A is 50 residues or 500 — same
chemistry, same neighbors, same expected structure. Our positional
encoding uses absolute `res_idx = torch.arange(L_total)` for the whole
complex, with no chain-reset, so chain B's residues get sin/cos
features that depend on chain A's length.

If positional encoding is the load-bearing bug behind Phase D's OOD
cliff, then SHIFTING the res_idx vector by a constant should change
the model's predictions even though the protein is identical.

This script runs Phase D's checkpoint forward THREE times on the same
sample with:
    A: res_idx = [0,   1, ..., L-1]    (in-distribution, what training saw)
    B: res_idx = [100, 101, ..., L+99] (still in training range)
    C: res_idx = [500, 501, ..., L+499] (way out of training range)

If predictions are identical (or near-identical), positional encoding
is NOT the load-bearing issue.

If predictions diverge as the shift grows, positional encoding IS a
real bug and the v2 fix (relative position encoding clipped to +/-32)
is justified.

Output: per-residue centroid CA-CA distance (in Angstroms) between
each pair of runs, averaged over the L residues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.training.data import load_sample


def _load_phase_d(ckpt_path: Path, device: str = "cuda") -> ResFoldOneStep:
    # Phase D was trained with default learned aa embeddings (no ESM, no
    # confidence head — those were added in later loops). Verify against
    # the checkpoint config.json.
    model = ResFoldOneStep(
        c_token=256,
        trunk_layers=6,
        denoiser_blocks=6,
        atom_head_layers=2,
        atom_head_heads=4,
        n_timesteps=50,
        aa_embed="learned",
        confidence_head=False,
        sigma_data=1.0,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing {len(missing)} keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"  WARN unexpected {len(unexpected)} keys (e.g. {unexpected[:2]})")
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt")
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--sample-id", default="1ebo.pdb2_1",
                   help="Which sample to test (default: a small one from DIPS).")
    p.add_argument("--sigma", type=float, default=10.0,
                   help="Diffusion sigma to evaluate at (sigma_max in Phase D config).")
    p.add_argument("--shifts", type=str, default="0,100,500,1000",
                   help="CSV of res_idx shifts to test.")
    p.add_argument("--per_chain_res_idx", action="store_true",
                   help="Build the BASE res_idx as [0..LA-1, 0..LB-1] (the new "
                        "per-chain-reset encoding) instead of arange(L_total). "
                        "Use this when testing a model trained with the fix.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading Phase D checkpoint: {args.checkpoint}")
    model = _load_phase_d(Path(args.checkpoint), device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params loaded")

    print(f"Loading sample {args.sample_id}...")
    table = pq.read_table(args.parquet)
    idx = None
    for i in range(len(table)):
        if table["sample_id"][i].as_py() == args.sample_id:
            idx = i
            break
    if idx is None:
        print(f"  ERROR: sample {args.sample_id} not in parquet")
        return
    sample = load_sample(table, idx, normalize=True, esm_cache_dir=None)
    L = sample["n_res"]
    print(f"  L={L} residues, LA+LB={L} (parquet field)")

    # Build batch tensors on device, shape [1, L, *].
    aa_seq = sample["aa_seq"].unsqueeze(0).to(device)
    chain_ids = sample["chain_ids"].unsqueeze(0).to(device)
    mask_res = torch.ones(1, L, dtype=torch.bool, device=device)
    esm_embed = None  # Phase D uses learned embeddings, no ESM cache

    # Fixed noisy input: same x_t and same noise sample across runs.
    torch.manual_seed(42)
    centroids_gt = sample["centroids"].unsqueeze(0).to(device)
    noise = torch.randn_like(centroids_gt)
    sigma = torch.tensor([args.sigma], device=device)
    # x_t at the given sigma (VE schedule: x_t = x_0 + sigma * eps).
    x_t = centroids_gt + sigma.view(-1, 1, 1) * noise

    shifts = [int(s) for s in args.shifts.split(",")]
    if args.per_chain_res_idx:
        # Base: parquet's per-chain reset values.
        base_res_idx_cpu = torch.tensor(table["res_idx"][idx].as_py(), dtype=torch.long)
        print(f"\nBASE res_idx = per-chain reset (chain A then B); "
              f"first 5 = {base_res_idx_cpu[:5].tolist()}, "
              f"last 5 = {base_res_idx_cpu[-5:].tolist()}")
    else:
        base_res_idx_cpu = torch.arange(L)
        print("\nBASE res_idx = arange(L_total) (legacy absolute encoding)")
    base_res_idx = base_res_idx_cpu.to(device)
    print(f"Running forward_sigma at sigma={args.sigma} with res_idx shifts: {shifts}")
    preds_centroid = {}
    preds_atoms = {}
    with torch.no_grad():
        for s in shifts:
            res_idx = (base_res_idx + s).unsqueeze(0)
            out = model.forward_sigma(
                x_t=x_t,
                sigma=sigma,
                aa_seq=aa_seq,
                chain_ids=chain_ids,
                res_idx=res_idx,
                mask=mask_res,
                esm_embed=esm_embed,
            )
            centroid_pred, atoms_pred, _pred_lddt = out
            preds_centroid[s] = centroid_pred.cpu().numpy().reshape(L, 3)
            preds_atoms[s] = atoms_pred.cpu().numpy().reshape(L, 4, 3)

    std = float(sample["std"])
    print(f"\nSample std (un-normalize multiplier): {std:.2f} A\n")

    print("=== Per-residue CA displacement between shift=0 and shift=X (Angstroms) ===")
    print(f"{'shift':>8} {'mean':>10} {'median':>10} {'max':>10} {'L':>6}")
    base = preds_centroid[0]
    for s in shifts:
        diff = preds_centroid[s] - base
        per_res_dist = np.linalg.norm(diff, axis=-1) * std  # back to A
        print(f"{s:>8} {per_res_dist.mean():>10.4f} {np.median(per_res_dist):>10.4f} "
              f"{per_res_dist.max():>10.4f} {L:>6}")

    print("\n=== Per-atom RMSD between shift=0 and shift=X (Angstroms) ===")
    print(f"{'shift':>8} {'rmsd':>10}")
    base_a = preds_atoms[0]
    for s in shifts:
        diff = preds_atoms[s] - base_a
        rmsd = float(np.sqrt((diff ** 2).sum(axis=-1).mean()) * std)
        print(f"{s:>8} {rmsd:>10.4f}")

    print("\n=== Interpretation ===")
    big_shift = shifts[-1]
    base_displacement = float(np.linalg.norm(preds_centroid[shifts[-1]] - base, axis=-1).mean()) * std
    if base_displacement < 0.1:
        print(f"  shift={big_shift} mean displacement {base_displacement:.3f} A < 0.1 A.")
        print("  -> Model is essentially invariant to res_idx shifts. Positional")
        print("     encoding is NOT the OOD-failure culprit.")
    elif base_displacement < 1.0:
        print(f"  shift={big_shift} mean displacement {base_displacement:.3f} A in [0.1, 1.0).")
        print("  -> Mild sensitivity. Probably not the dominant bug, but worth fixing.")
    else:
        print(f"  shift={big_shift} mean displacement {base_displacement:.3f} A >= 1.0 A.")
        print("  -> STRONG sensitivity. The model's prediction changes meaningfully")
        print("     when the SAME residues get different absolute positions. This is")
        print("     a load-bearing bug; v2 should use AF-Multimer relative encoding")
        print("     clipped to +/-32 + same-chain bit.")


if __name__ == "__main__":
    main()
