#!/usr/bin/env python
"""Batch prediction script for TinyFold web visualizer.

Generates predictions for all samples and saves them for fast loading in the frontend.

Supports:
- af3_style: Single-stage atom-level diffusion
- resfold: Two-stage (residue -> atom) with separate execution for memory efficiency

Usage:
    # AF3-style predictions
    python predict_all.py --model af3_style --checkpoint ../outputs/af3_15M_gaussian_5K/best_model.pt

    # ResFold stage 1 only (residue centroids)
    python predict_all.py --model resfold --stage 1 --checkpoint ../outputs/resfold_s1_5K_50K/best_model.pt

    # ResFold stage 2 only (atoms from cached centroids)
    python predict_all.py --model resfold --stage 2 --checkpoint ../outputs/resfold_s2/best_model.pt

    # ResFold both stages (if memory allows)
    python predict_all.py --model resfold --stage both --checkpoint ../outputs/resfold_full/best_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# Add paths
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent / "scripts"))
sys.path.insert(0, str(script_dir.parent / "src"))

from models import create_model, create_schedule, create_noiser
from data_split import DataSplitConfig, get_train_test_indices


# =============================================================================
# Data Loading
# =============================================================================

def load_sample_for_af3(table, idx: int, device: torch.device) -> dict:
    """Load a sample for AF3-style model (atom-level)."""
    coords = np.array(table['atom_coords'][idx].as_py(), dtype=np.float32)
    atom_types = np.array(table['atom_type'][idx].as_py(), dtype=np.int64)
    atom_to_res = np.array(table['atom_to_res'][idx].as_py(), dtype=np.int64)
    seq_res = np.array(table['seq'][idx].as_py(), dtype=np.int64)
    chain_res = np.array(table['chain_id_res'][idx].as_py(), dtype=np.int64)
    res_idx = np.array(table['res_idx'][idx].as_py(), dtype=np.int64)

    n_atoms = len(atom_types)
    coords = coords.reshape(n_atoms, 3)

    # Per-atom sequence and chain
    aa_seq = seq_res[atom_to_res]
    chain_ids = chain_res[atom_to_res]

    # Center and normalize
    centroid = coords.mean(axis=0, keepdims=True)
    coords_centered = coords - centroid
    std = float(coords_centered.std())
    coords_norm = coords_centered / std

    return {
        'sample_id': table['sample_id'][idx].as_py(),
        'coords': coords,
        'coords_norm': torch.tensor(coords_norm, dtype=torch.float32, device=device),
        'atom_types': torch.tensor(atom_types, dtype=torch.long, device=device),
        'atom_to_res': torch.tensor(atom_to_res, dtype=torch.long, device=device),
        'aa_seq': torch.tensor(aa_seq, dtype=torch.long, device=device),
        'chain_ids': torch.tensor(chain_ids, dtype=torch.long, device=device),
        'seq_res': seq_res,
        'chain_res': chain_res,
        'res_idx': res_idx,
        'std': std,
        'centroid': centroid,
        'n_atoms': n_atoms,
        'n_residues': len(seq_res),
    }


def load_sample_for_resfold(table, idx: int, device: torch.device) -> dict:
    """Load a sample for ResFold model (residue-level stage 1)."""
    coords = np.array(table['atom_coords'][idx].as_py(), dtype=np.float32)
    atom_types = np.array(table['atom_type'][idx].as_py(), dtype=np.int64)
    atom_to_res = np.array(table['atom_to_res'][idx].as_py(), dtype=np.int64)
    seq_res = np.array(table['seq'][idx].as_py(), dtype=np.int64)
    chain_res = np.array(table['chain_id_res'][idx].as_py(), dtype=np.int64)
    res_idx_arr = np.array(table['res_idx'][idx].as_py(), dtype=np.int64)

    n_atoms = len(atom_types)
    n_residues = len(seq_res)
    coords = coords.reshape(n_atoms, 3)

    # Compute residue centroids (CA positions, atom_type=1)
    ca_mask = atom_types == 1
    ca_coords = coords[ca_mask]
    assert len(ca_coords) == n_residues, f"CA count mismatch: {len(ca_coords)} vs {n_residues}"

    # Center and normalize
    centroid = ca_coords.mean(axis=0, keepdims=True)
    ca_centered = ca_coords - centroid
    std = float(ca_centered.std())
    ca_norm = ca_centered / std

    # Also store full atom coords for stage 2 / evaluation
    coords_centered = coords - centroid
    coords_norm = coords_centered / std

    return {
        'sample_id': table['sample_id'][idx].as_py(),
        'coords': coords,  # Full atom coords (original)
        'ca_coords': ca_coords,  # CA coords (original)
        'ca_norm': torch.tensor(ca_norm, dtype=torch.float32, device=device),
        'coords_norm': coords_norm,  # Full atom coords (normalized)
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': torch.tensor(seq_res, dtype=torch.long, device=device),
        'chain_ids': torch.tensor(chain_res, dtype=torch.long, device=device),
        'res_idx': torch.tensor(res_idx_arr, dtype=torch.long, device=device),
        'seq_res': seq_res,
        'chain_res': chain_res,
        'res_idx_np': res_idx_arr,
        'std': std,
        'centroid': centroid,
        'n_atoms': n_atoms,
        'n_residues': n_residues,
    }


# =============================================================================
# Kabsch Alignment
# =============================================================================

def kabsch_align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch alignment - align pred to target."""
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    P = pred - pred_mean
    Q = target - target_mean

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.linalg.det(V @ U.T)
    if d < 0:
        V[:, 2] *= -1

    R = V @ U.T
    P_aligned = P @ R.T
    pred_aligned = P_aligned + target_mean

    diff = pred_aligned - target
    rmsd = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    return pred_aligned, rmsd


# =============================================================================
# Inference Functions
# =============================================================================

@torch.no_grad()
def run_af3_inference(model, sample: dict, noiser, noise_type: str, clamp_val: float = 3.0) -> np.ndarray:
    """Run AF3-style inference on a single sample."""
    device = sample['coords_norm'].device
    B, N = 1, sample['n_atoms']

    atom_types = sample['atom_types'].unsqueeze(0)
    atom_to_res = sample['atom_to_res'].unsqueeze(0)
    aa_seq = sample['aa_seq'].unsqueeze(0)
    chain_ids = sample['chain_ids'].unsqueeze(0)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)

    # Initialize
    if noise_type in ("linear_chain", "linear_flow"):
        from models.diffusion import generate_extended_chain
        x_linear = generate_extended_chain(
            n_atoms=N,
            atom_to_res=sample['atom_to_res'],
            atom_type=sample['atom_types'],
            chain_ids=sample['chain_ids'],
            device=device,
            apply_rotation=False,
        ).unsqueeze(0)
        x = x_linear.clone()
        t_range = reversed(range(noiser.T + 1))
    else:
        x_linear = None
        x = torch.randn(B, N, 3, device=device)
        t_range = reversed(range(noiser.T))

    # Diffusion loop
    for t in t_range:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        if noise_type == "linear_chain" and hasattr(model, 'forward_direct'):
            x0_pred = model.forward_direct(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        else:
            x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)

        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if noise_type in ("linear_chain", "linear_flow"):
            x = noiser.reverse_step(x, x0_pred, t, x_linear)
        else:
            if t > 0:
                ab_t = noiser.alpha_bar[t]
                ab_prev = noiser.alpha_bar[t - 1]
                beta = noiser.betas[t]
                alpha = noiser.alphas[t]
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

    # Denormalize
    pred_norm = x[0].cpu().numpy()
    pred = pred_norm * sample['std'] + sample['centroid']
    return pred


@torch.no_grad()
def run_resfold_stage1(model, sample: dict, noiser, clamp_val: float = 3.0) -> np.ndarray:
    """Run ResFold stage 1 (residue centroids)."""
    device = sample['ca_norm'].device
    B, L = 1, sample['n_residues']

    aa_seq = sample['aa_seq'].unsqueeze(0)
    chain_ids = sample['chain_ids'].unsqueeze(0)
    res_idx = sample['res_idx'].unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    # DDPM sampling for centroids
    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model.forward_stage1(x, aa_seq, chain_ids, res_idx, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if t > 0:
            ab_t = noiser.alpha_bar[t]
            ab_prev = noiser.alpha_bar[t - 1]
            beta = noiser.betas[t]
            alpha = noiser.alphas[t]
            coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
            coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
            mean = coef1 * x0_pred + coef2 * x
            var = beta * (1 - ab_prev) / (1 - ab_t)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = x0_pred

    # Denormalize centroids
    centroids_norm = x[0].cpu().numpy()
    centroids = centroids_norm * sample['std'] + sample['centroid']
    return centroids


@torch.no_grad()
def run_resfold_stage2(model, sample: dict, centroids: np.ndarray) -> np.ndarray:
    """Run ResFold stage 2 (atom positions from centroids)."""
    device = sample['aa_seq'].device
    B, L = 1, sample['n_residues']

    # Normalize centroids
    centroid_mean = centroids.mean(axis=0, keepdims=True)
    centroids_centered = centroids - centroid_mean
    std = centroids_centered.std()
    centroids_norm = torch.tensor(centroids_centered / std, dtype=torch.float32, device=device).unsqueeze(0)

    aa_seq = sample['aa_seq'].unsqueeze(0)
    chain_ids = sample['chain_ids'].unsqueeze(0)
    res_idx = sample['res_idx'].unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    # Run stage 2
    atoms_pred = model.forward_stage2(centroids_norm, aa_seq, chain_ids, res_idx, mask)

    # Denormalize: [B, L, 4, 3] -> [L*4, 3]
    atoms_norm = atoms_pred[0].cpu().numpy().reshape(-1, 3)
    atoms = atoms_norm * std + centroid_mean
    return atoms


# =============================================================================
# Main Prediction Loop
# =============================================================================

def predict_af3(args, table, indices, device):
    """Generate predictions for AF3-style model."""
    print(f"Loading AF3 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Infer n_timesteps from checkpoint
    for k in checkpoint["model_state_dict"]:
        if "time_embed.weight" in k:
            embed_size = checkpoint["model_state_dict"][k].shape[0]
            n_timesteps = embed_size - 1
            break
    else:
        n_timesteps = 50

    print(f"Creating model with n_timesteps={n_timesteps}")
    model = create_model("af3_style", n_timesteps=n_timesteps)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    schedule = create_schedule("cosine", T=n_timesteps)
    noiser = create_noiser(args.noise_type, schedule)

    predictions = {}
    for idx in tqdm(indices, desc="Predicting"):
        sample = load_sample_for_af3(table, idx, device)
        sample_id = sample['sample_id']

        start = time.time()
        pred_coords = run_af3_inference(model, sample, noiser, args.noise_type, args.clamp_val)
        inference_time = time.time() - start

        # Align to ground truth
        pred_aligned, rmsd = kabsch_align(pred_coords, sample['coords'])

        predictions[sample_id] = {
            'coords': pred_aligned.tolist(),
            'rmsd': rmsd,
            'inference_time': inference_time,
        }

    return predictions


def predict_resfold_stage1(args, table, indices, device):
    """Generate stage 1 predictions (centroids) for ResFold."""
    print(f"Loading ResFold stage 1 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Infer n_timesteps
    for k in checkpoint["model_state_dict"]:
        if "time_embed.weight" in k:
            n_timesteps = checkpoint["model_state_dict"][k].shape[0]
            break
    else:
        n_timesteps = 50

    print(f"Creating ResFold model with n_timesteps={n_timesteps}")
    model = create_model("resfold", n_timesteps=n_timesteps)

    # Load only stage 1 weights
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    schedule = create_schedule("cosine", T=n_timesteps)
    noiser = create_noiser("gaussian", schedule)

    predictions = {}
    for idx in tqdm(indices, desc="Stage 1"):
        sample = load_sample_for_resfold(table, idx, device)
        sample_id = sample['sample_id']

        start = time.time()
        pred_centroids = run_resfold_stage1(model, sample, noiser, args.clamp_val)
        inference_time = time.time() - start

        # Align centroids to ground truth CA
        pred_aligned, rmsd = kabsch_align(pred_centroids, sample['ca_coords'])

        predictions[sample_id] = {
            'centroids': pred_aligned.tolist(),
            'rmsd_ca': rmsd,
            'inference_time': inference_time,
            'std': sample['std'],
            'centroid': sample['centroid'].tolist(),
        }

    return predictions


def predict_resfold_stage2(args, table, indices, device, stage1_predictions: dict):
    """Generate stage 2 predictions (atoms) from cached centroids."""
    print(f"Loading ResFold stage 2 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = create_model("resfold", n_timesteps=50)  # n_timesteps doesn't matter for stage 2
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    predictions = {}
    for idx in tqdm(indices, desc="Stage 2"):
        sample = load_sample_for_resfold(table, idx, device)
        sample_id = sample['sample_id']

        if sample_id not in stage1_predictions:
            print(f"Warning: No stage 1 prediction for {sample_id}, skipping")
            continue

        # Load cached centroids
        centroids = np.array(stage1_predictions[sample_id]['centroids'], dtype=np.float32)

        start = time.time()
        pred_atoms = run_resfold_stage2(model, sample, centroids)
        inference_time = time.time() - start

        # Align to ground truth atoms
        pred_aligned, rmsd = kabsch_align(pred_atoms, sample['coords'])

        predictions[sample_id] = {
            'coords': pred_aligned.tolist(),
            'rmsd': rmsd,
            'rmsd_ca': stage1_predictions[sample_id]['rmsd_ca'],
            'inference_time': inference_time + stage1_predictions[sample_id]['inference_time'],
        }

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Batch prediction for TinyFold")
    parser.add_argument("--model", type=str, required=True, choices=["af3_style", "resfold"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"],
                        help="For resfold: which stage to run")
    parser.add_argument("--stage1_cache", type=str, default=None,
                        help="Path to stage 1 predictions (required for stage 2 only)")
    parser.add_argument("--noise_type", type=str, default="gaussian",
                        choices=["gaussian", "linear_chain"],
                        help="Noise type for AF3")
    parser.add_argument("--clamp_val", type=float, default=3.0, help="Clamp value")
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Output file path")
    parser.add_argument("--data_path", type=str, default="../data/processed/samples.parquet",
                        help="Path to samples parquet")
    parser.add_argument("--n_train", type=int, default=5000, help="Number of train samples")
    parser.add_argument("--n_test", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--min_atoms", type=int, default=100, help="Min atoms filter")
    parser.add_argument("--max_atoms", type=int, default=1600, help="Max atoms filter")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"],
                        help="Which split to predict")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_path}")
    table = pq.read_table(args.data_path)
    print(f"Loaded {len(table)} samples")

    # Get train/test split
    split_config = DataSplitConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
    )
    train_indices, test_indices = get_train_test_indices(table, split_config)
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Select indices based on split
    if args.split == "train":
        indices = train_indices
    elif args.split == "test":
        indices = test_indices
    else:
        indices = train_indices + test_indices

    # Run predictions
    if args.model == "af3_style":
        predictions = predict_af3(args, table, indices, device)

    elif args.model == "resfold":
        if args.stage == "1":
            predictions = predict_resfold_stage1(args, table, indices, device)

        elif args.stage == "2":
            if args.stage1_cache is None:
                raise ValueError("--stage1_cache required for stage 2 only")
            with open(args.stage1_cache) as f:
                stage1_preds = json.load(f)
            predictions = predict_resfold_stage2(args, table, indices, device, stage1_preds)

        else:  # both
            stage1_preds = predict_resfold_stage1(args, table, indices, device)
            # Save intermediate
            stage1_path = args.output.replace(".json", "_stage1.json")
            with open(stage1_path, "w") as f:
                json.dump(stage1_preds, f)
            print(f"Saved stage 1 predictions to {stage1_path}")

            predictions = predict_resfold_stage2(args, table, indices, device, stage1_preds)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {output_path}")

    # Print summary
    rmsd_values = [p['rmsd'] for p in predictions.values() if 'rmsd' in p]
    if rmsd_values:
        print(f"RMSD: mean={np.mean(rmsd_values):.2f}, median={np.median(rmsd_values):.2f}, "
              f"min={np.min(rmsd_values):.2f}, max={np.max(rmsd_values):.2f}")


if __name__ == "__main__":
    main()
