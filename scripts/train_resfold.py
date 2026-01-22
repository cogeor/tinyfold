#!/usr/bin/env python
"""
ResFold training script.

Two-stage architecture:
- Stage 1: Residue-level diffusion on centroids
- Stage 2: Atom refinement from centroids

Training modes:
- stage1_only: Train residue diffusion only
- stage2_only: Train atom refinement only (using GT centroids)
- end_to_end: Train both stages together

Usage:
    python train_resfold.py --mode stage1_only --n_train 80 --n_steps 10000
    python train_resfold.py --mode stage2_only --n_train 80 --n_steps 5000
    python train_resfold.py --mode end_to_end --n_train 80 --n_steps 15000
"""

import sys
import math
import random
import torch
import torch.nn as nn
import pyarrow.parquet as pq
import argparse
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model imports - use new tinyfold.model paths where available
from models import create_schedule, create_noiser, kabsch_align_to_target
from models.resfold_pipeline import ResFoldPipeline
from models.dockq_utils import compute_dockq

# Loss imports - use consolidated losses from tinyfold.model.losses
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
    ContactLoss,
    compute_lddt_metrics,
)
from data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info, save_split, load_split,
    LengthBucketSampler, DynamicBatchSampler
)


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Dual output to console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = open(log_path, 'w', buffering=1)

    def log(self, msg: str = ""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


@torch.no_grad()
def sample_centroids(model, batch, noiser, device, clamp_val=3.0,
                     align_per_step=False, recenter=False):
    """DDPM sampling for Stage 1 centroids only.

    Args:
        model: ResFoldPipeline model
        batch: Batch dict with aa_seq, chain_ids, res_idx, mask_res
        noiser: Diffusion noiser with schedule
        device: torch device
        clamp_val: Value to clamp predictions
        align_per_step: If True, Kabsch-align x0_pred to current x before update.
                        This fixes frame drift (Boltz-1 style). Default False for
                        backward compatibility.
        recenter: If True, re-center coordinates each step (avoids translation drift).
                  Default False for backward compatibility.

    Returns:
        centroids: [B, L, 3] predicted centroids
    """
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Start from noise
    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict x0 (clean centroids)
        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t_batch, mask
        )
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # NEW: Kabsch-align x0_pred to current x's frame (fixes drift)
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        # DDPM reverse step
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

        # NEW: Re-center to avoid translation drift
        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    return x


@torch.no_grad()
def generate_stage1_predictions(
    model, samples, indices, noiser, device, batch_size=1, logger=None,
    align_per_step=False, recenter=False
):
    """Generate Stage 1 centroid predictions for all samples.

    Args:
        model: ResFoldPipeline model with trained Stage 1
        samples: dict of sample_idx -> sample dict
        indices: list of sample indices to process
        noiser: DiffusionNoiser
        device: torch device
        batch_size: batch size for inference (1 for variable lengths)
        logger: optional logger
        align_per_step: Kabsch-align x0_pred each step (fixes drift)
        recenter: Re-center each step (avoids translation drift)

    Returns:
        dict of sample_idx -> predicted centroids tensor [L, 3] (normalized)
    """
    model.eval()
    predictions = {}

    total = len(indices)
    for i, idx in enumerate(indices):
        s = samples[idx]
        batch = collate_batch([s], device)

        # Run Stage 1 diffusion sampling
        centroids_pred = sample_centroids(
            model, batch, noiser, device,
            align_per_step=align_per_step, recenter=recenter
        )

        # Store prediction (trim to actual length)
        n_res = s['n_res']
        predictions[idx] = centroids_pred[0, :n_res].cpu()

        if logger and (i + 1) % 500 == 0:
            logger.log(f"    Generated {i + 1}/{total} predictions...")

    return predictions


def save_stage1_predictions(predictions, path):
    """Save Stage 1 predictions to file."""
    torch.save(predictions, path)


def load_stage1_predictions(path):
    """Load Stage 1 predictions from file."""
    return torch.load(path)


# =============================================================================
# Data Loading
# =============================================================================

def load_sample_raw(table, i):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Normalize
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    # Compute residue centroids (mean of 4 backbone atoms)
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)  # [L, 3]

    # Residue-level features
    aa_seq = seq_res  # [L]
    chain_ids = chain_res  # [L]
    res_idx = torch.arange(n_res)  # [L]

    return {
        'coords': coords,  # [N_atoms, 3]
        'coords_res': coords_res,  # [L, 4, 3]
        'centroids': centroids,  # [L, 3]
        'atom_types': atom_types,  # [N_atoms]
        'atom_to_res': atom_to_res,  # [N_atoms]
        'aa_seq': aa_seq,  # [L]
        'chain_ids': chain_ids,  # [L]
        'res_idx': res_idx,  # [L]
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    """Collate samples into a padded batch."""
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    # Residue-level tensors
    centroids = torch.zeros(B, max_res, 3)
    centroids_pred = torch.zeros(B, max_res, 3)  # Stage 1 predictions (if available)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors (for evaluation)
    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)

    stds = []
    has_predictions = False

    for i, s in enumerate(samples):
        L = s['n_res']
        N = s['n_atoms']

        centroids[i, :L] = s['centroids']
        coords_res[i, :L] = s['coords_res']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True

        coords[i, :N] = s['coords']
        atom_types[i, :N] = s['atom_types']
        mask_atom[i, :N] = True

        stds.append(s['std'])

        # Stage 1 predictions (if available)
        if 'centroids_pred' in s:
            centroids_pred[i, :L] = s['centroids_pred']
            has_predictions = True

    result = {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }

    if has_predictions:
        result['centroids_pred'] = centroids_pred.to(device)

    return result


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Plot prediction vs ground truth."""
    fig = plt.figure(figsize=(12, 5))

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    mask_a = chain_ids == 0
    mask_b = chain_ids == 1
    if mask_a.any():
        ax1.scatter(target[mask_a, 0], target[mask_a, 1], target[mask_a, 2], c='blue', s=10, alpha=0.7)
    if mask_b.any():
        ax1.scatter(target[mask_b, 0], target[mask_b, 1], target[mask_b, 2], c='red', s=10, alpha=0.7)
    ax1.set_title(f'{sample_id}\nGround Truth')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2], c='cyan', s=10, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2], c='orange', s=10, alpha=0.7)
    ax2.set_title(f'Prediction\nRMSE: {rmse:.2f} A')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ResFold training")

    # Training mode
    parser.add_argument("--mode", type=str, default="end_to_end",
                        choices=["stage1_only", "stage2_only", "end_to_end"],
                        help="Training mode")

    # Data
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=14)
    parser.add_argument("--n_eval_train", type=int, default=200)
    parser.add_argument("--min_atoms", type=int, default=200)
    parser.add_argument("--max_atoms", type=int, default=400)
    parser.add_argument("--select_smallest", action="store_true",
                        help="Select N smallest proteins instead of filtering by atom range")
    parser.add_argument("--load_split", type=str, default=None,
                        help="Load train/test split from JSON (for Stage 2 to reuse Stage 1 split)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_bucketing", action="store_true",
                        help="Use length bucketing for efficient batching")
    parser.add_argument("--n_buckets", type=int, default=8,
                        help="Number of length buckets (default: 8)")
    parser.add_argument("--dynamic_batch", action="store_true",
                        help="Use dynamic batch sizing based on sequence length")
    parser.add_argument("--max_tokens", type=int, default=30000,
                        help="Max tokens per batch for dynamic batching (batch_size * max_seq_len)")

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_accum", type=int, default=1)

    # Model - Stage 1
    parser.add_argument("--c_token_s1", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=9)
    parser.add_argument("--denoiser_blocks", type=int, default=7)

    # Model - Stage 2 (AtomRefinerV2: ~15M params with defaults)
    parser.add_argument("--c_token_s2", type=int, default=256)
    parser.add_argument("--s2_layers", type=int, default=18)
    parser.add_argument("--s2_heads", type=int, default=8)
    parser.add_argument("--centroid_noise", type=float, default=0.0,
                        help="Noise std for centroid augmentation in Stage 2 (normalized units)")

    # Diffusion
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="linear")

    # Loss weights
    parser.add_argument("--dist_weight", type=float, default=0.1,
                        help="Weight for distance consistency loss")

    # Geometry loss weights (set to 0 to disable)
    parser.add_argument("--geom_weight", type=float, default=0.1,
                        help="Overall weight for geometry loss (0 to disable all)")
    parser.add_argument("--bond_length_weight", type=float, default=1.0,
                        help="Weight for bond length loss within geometry loss")
    parser.add_argument("--bond_angle_weight", type=float, default=0.1,
                        help="Weight for bond angle loss within geometry loss")
    parser.add_argument("--omega_weight", type=float, default=0.1,
                        help="Weight for omega dihedral loss within geometry loss")
    parser.add_argument("--o_chirality_weight", type=float, default=0.1,
                        help="Weight for O chirality loss within geometry loss")
    parser.add_argument("--cb_chirality_weight", type=float, default=0.0,
                        help="Weight for virtual CB chirality loss (default 0, experimental)")

    # Contact loss
    parser.add_argument("--contact_weight", type=float, default=0.0,
                        help="Weight for contact-based loss (0 to disable)")
    parser.add_argument("--contact_threshold", type=float, default=1.0,
                        help="Contact distance threshold in normalized units (1.0 = 10A)")
    parser.add_argument("--contact_min_seq_sep", type=int, default=5,
                        help="Minimum sequence separation for intra-chain contacts")
    parser.add_argument("--contact_inter_weight", type=float, default=2.0,
                        help="Weight multiplier for inter-chain contacts")
    parser.add_argument("--contact_stage", type=str, default="stage1",
                        choices=["stage1", "stage2", "both"],
                        help="Which stage(s) to apply contact loss")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load (e.g., Stage 1 checkpoint for Stage 2 training)")

    # Sampling (evaluation)
    parser.add_argument("--align_per_step", action="store_true",
                        help="Kabsch-align x0_pred to x_t each step (fixes drift, Boltz-1 style)")
    parser.add_argument("--recenter", action="store_true",
                        help="Re-center coordinates each step (avoids translation drift)")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/resfold")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(args.output_dir, 'train.log')
    logger = Logger(log_path)

    # Log header
    logger.log("=" * 70)
    logger.log("ResFold Training")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Script: {os.path.abspath(__file__)}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Log config
    logger.log("Configuration:")
    logger.log(f"  mode:          {args.mode}")
    logger.log(f"  output_dir:    {args.output_dir}")
    logger.log(f"  n_train:       {args.n_train}")
    logger.log(f"  n_test:        {args.n_test}")
    logger.log(f"  batch_size:    {args.batch_size}")
    logger.log(f"  grad_accum:    {args.grad_accum}")
    logger.log(f"  eff_batch:     {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:       {args.n_steps}")
    logger.log(f"  eval_every:    {args.eval_every}")
    logger.log(f"  lr:            {args.lr}")
    logger.log(f"  T:             {args.T}")
    logger.log(f"  centroid_noise:{args.centroid_noise}")
    logger.log(f"  dist_weight:   {args.dist_weight}")
    logger.log(f"  geom_weight:   {args.geom_weight}")
    if args.geom_weight > 0:
        logger.log(f"    bond_length: {args.bond_length_weight}")
        logger.log(f"    bond_angle:  {args.bond_angle_weight}")
        logger.log(f"    omega:       {args.omega_weight}")
        logger.log(f"    o_chirality: {args.o_chirality_weight}")
        logger.log(f"    cb_chirality:{args.cb_chirality_weight}")
    logger.log("")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.log("")

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    table = pq.read_table(data_path)

    # Deterministic split (either load from file or create new)
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        train_indices, test_indices, loaded_info = load_split(args.load_split)
        logger.log(f"Data split (loaded from file):")
        logger.log(f"  Training: {len(train_indices)} samples, atoms {loaded_info['train_atom_range'][0]}-{loaded_info['train_atom_range'][1]}")
        logger.log(f"  Test: {len(test_indices)} samples, atoms {loaded_info['test_atom_range'][0]}-{loaded_info['test_atom_range'][1]}")
        split_info = loaded_info
    else:
        split_config = DataSplitConfig(
            n_train=args.n_train,
            n_test=args.n_test,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            select_smallest=args.select_smallest,
            seed=42,
        )
        train_indices, test_indices = get_train_test_indices(table, split_config)
        split_info = get_split_info(table, split_config)

        logger.log(f"Data split (seed={split_config.seed}):")
        if args.select_smallest:
            logger.log(f"  Selected {split_info['eligible_samples']} smallest proteins")
        else:
            logger.log(f"  Eligible samples ({args.min_atoms}-{args.max_atoms} atoms): {split_info['eligible_samples']}")
        logger.log(f"  Training: {len(train_indices)} samples")
        logger.log(f"  Test: {len(test_indices)} samples")

        # Save split for Stage 2 reuse
        split_path = os.path.join(args.output_dir, "split.json")
        save_split(split_info, split_path)
    logger.log("")

    # Preload samples
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx) for idx in test_indices}
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples")

    # For Stage 2: load or generate Stage 1 predictions
    if args.mode == "stage2_only" and args.load_split:
        stage1_dir = os.path.dirname(args.load_split)
        predictions_path = os.path.join(stage1_dir, "stage1_predictions.pt")

        if os.path.exists(predictions_path):
            logger.log(f"  Loading Stage 1 predictions from: {predictions_path}")
            s1_predictions = load_stage1_predictions(predictions_path)
            logger.log(f"    Loaded {len(s1_predictions)} predictions")
        else:
            # Generate predictions using Stage 1 checkpoint
            s1_checkpoint = os.path.join(stage1_dir, "best_model.pt")
            if not os.path.exists(s1_checkpoint):
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_checkpoint}")

            logger.log(f"  Generating Stage 1 predictions (this may take a while)...")
            logger.log(f"    Loading Stage 1 checkpoint: {s1_checkpoint}")

            # Create temporary model for Stage 1
            s1_model = ResFoldPipeline(
                c_token_s1=args.c_token_s1,
                trunk_layers=args.trunk_layers,
                denoiser_blocks=args.denoiser_blocks,
                c_token_s2=args.c_token_s2,
                s2_layers=args.s2_layers,
                s2_heads=args.s2_heads,
                n_timesteps=args.T,
                dropout=0.0,
            ).to(device)

            ckpt = torch.load(s1_checkpoint, map_location=device)
            s1_model.load_state_dict(ckpt['model_state_dict'], strict=False)

            # Create noiser for sampling
            schedule = create_schedule("linear", T=args.T)
            s1_noiser = create_noiser("gaussian", schedule)

            # Generate predictions for train and test
            all_samples = {**train_samples, **test_samples}
            all_indices = train_indices + test_indices
            s1_predictions = generate_stage1_predictions(
                s1_model, all_samples, all_indices, s1_noiser, device, logger=logger
            )

            # Save predictions
            save_stage1_predictions(s1_predictions, predictions_path)
            logger.log(f"    Saved predictions to: {predictions_path}")

            # Clean up
            del s1_model
            torch.cuda.empty_cache()

        # Inject predictions into samples
        for idx, pred in s1_predictions.items():
            if idx in train_samples:
                train_samples[idx]['centroids_pred'] = pred
            if idx in test_samples:
                test_samples[idx]['centroids_pred'] = pred
        logger.log(f"  Injected Stage 1 predictions into samples")

    # Create sampler for efficient batching
    train_sampler = None
    if args.dynamic_batch:
        train_sampler = DynamicBatchSampler(
            train_samples,
            base_batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            n_buckets=args.n_buckets,
            seed=42,
        )
        logger.log(f"  Using dynamic batch sampler (max_tokens={args.max_tokens})")
        for info in train_sampler.get_batch_sizes():
            logger.log(f"    Bucket {info['bucket']}: max_res={info['max_res']}, batch_size={info['batch_size']}")
    elif args.use_bucketing:
        train_sampler = LengthBucketSampler(
            train_samples,
            n_buckets=args.n_buckets,
            seed=42,
        )
        logger.log(f"  Using length bucketing ({args.n_buckets} buckets)")

    # Create model
    model = ResFoldPipeline(
        c_token_s1=args.c_token_s1,
        trunk_layers=args.trunk_layers,
        denoiser_blocks=args.denoiser_blocks,
        c_token_s2=args.c_token_s2,
        s2_layers=args.s2_layers,
        s2_heads=args.s2_heads,
        n_timesteps=args.T,
        dropout=0.0,
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.log(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict']
        # Filter out Stage 2 keys if architecture changed (for stage2_only training with new Stage 2)
        if args.mode == "stage2_only":
            stage2_keys = [k for k in state_dict.keys() if k.startswith('stage2.')]
            model_stage2_keys = [k for k in model.state_dict().keys() if k.startswith('stage2.')]
            # Check if Stage 2 architectures match
            if len(stage2_keys) > 0 and len(model_stage2_keys) > 0:
                sample_ckpt = state_dict[stage2_keys[0]].shape
                sample_model = model.state_dict()[model_stage2_keys[0]].shape
                if sample_ckpt != sample_model:
                    logger.log(f"  Stage 2 architecture changed, loading only Stage 1 weights")
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('stage2.')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.log(f"  Missing keys: {len(missing)} (expected for new Stage 2)")
        if unexpected:
            logger.log(f"  Unexpected keys: {len(unexpected)}")
        logger.log(f"  Checkpoint loaded from step {ckpt.get('step', 'unknown')}")
        logger.log("")

    # Set training mode (freeze/unfreeze stages)
    model.set_training_mode(args.mode)

    param_counts = model.count_parameters()
    logger.log(f"Model: ResFold ({args.mode})")
    logger.log(f"  Stage 1 params: {param_counts['stage1']:,} ({param_counts['stage1_pct']:.1f}%)")
    logger.log(f"  Stage 2 params: {param_counts['stage2']:,} ({param_counts['stage2_pct']:.1f}%)")
    logger.log(f"  Total params:   {param_counts['total']:,}")
    logger.log("")

    # Create diffusion components (for Stage 1)
    schedule = create_schedule(args.schedule, T=args.T)
    noiser = create_noiser("gaussian", schedule)
    noiser = noiser.to(device)

    logger.log(f"Diffusion:")
    logger.log(f"  Schedule: {args.schedule}")
    logger.log(f"  Noise type: gaussian")
    logger.log("")

    # Geometry loss (for Stage 2 only - not used in stage1_only)
    geom_loss_fn = None
    if args.geom_weight > 0 and args.mode != "stage1_only":
        geom_loss_fn = GeometryLoss(
            bond_length_weight=args.bond_length_weight,
            bond_angle_weight=args.bond_angle_weight,
            omega_weight=args.omega_weight,
            o_chirality_weight=args.o_chirality_weight,
            cb_chirality_weight=args.cb_chirality_weight,
        )
        logger.log(f"Geometry loss: {geom_loss_fn}")
        logger.log("")

    # Contact loss (can be used in Stage 1 and/or Stage 2)
    contact_loss_fn = None
    if args.contact_weight > 0:
        contact_loss_fn = ContactLoss(
            threshold=args.contact_threshold,
            min_seq_sep=args.contact_min_seq_sep,
            inter_chain_weight=args.contact_inter_weight,
            stage=args.contact_stage,
        )
        logger.log(f"Contact loss: {contact_loss_fn}")
        logger.log("")

    # Optimizer (only on trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=1e-5)

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    start_time = time.time()

    model.train()
    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(args.grad_accum):
            # Sample batch (using bucketing if enabled)
            if train_sampler is not None:
                if args.dynamic_batch:
                    batch_indices, current_batch_size = train_sampler.sample_batch()
                else:
                    batch_indices = train_sampler.sample_batch(args.batch_size)
                    current_batch_size = args.batch_size
            else:
                batch_indices = random.choices(train_indices, k=args.batch_size)
                current_batch_size = args.batch_size

            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            # Sample timesteps
            t = torch.randint(0, noiser.T, (current_batch_size,), device=device)

            # Add noise to centroids (for Stage 1)
            noise = torch.randn_like(batch['centroids'])
            sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            x_t = sqrt_ab * batch['centroids'] + sqrt_one_minus_ab * noise

            # Forward pass
            if args.mode == "stage1_only":
                # Only Stage 1
                centroids_pred = model.forward_stage1(
                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                    t, batch['mask_res']
                )
                # Loss: MSE on centroids + distance consistency
                loss_mse = compute_mse_loss(centroids_pred, batch['centroids'], batch['mask_res'])
                loss_dist = compute_distance_consistency_loss(
                    centroids_pred, batch['centroids'], batch['mask_res']
                )
                loss = loss_mse + args.dist_weight * loss_dist

                # Contact loss for Stage 1
                loss_contact = 0.0
                if contact_loss_fn is not None and args.contact_stage in ["stage1", "both"]:
                    contact_losses = contact_loss_fn(
                        pred_centroids=centroids_pred,
                        gt_centroids=batch['centroids'],
                        chain_ids=batch['chain_ids'],
                        mask=batch['mask_res']
                    )
                    loss_contact = contact_losses['stage1'].item()
                    loss = loss + args.contact_weight * contact_losses['stage1']

                # Track loss components for logging
                if accum_step == args.grad_accum - 1:
                    loss_components = {
                        'mse': loss_mse.item(),
                        'dist': loss_dist.item(),
                        'contact': loss_contact,
                    }

            elif args.mode == "stage2_only":
                # Only Stage 2
                # Use Stage 1 predicted centroids if available, else GT
                if 'centroids_pred' in batch:
                    centroids_for_s2 = batch['centroids_pred']
                else:
                    centroids_for_s2 = batch['centroids']

                # Compute trunk tokens from centroids
                trunk_tokens = model.get_trunk_tokens(
                    centroids_for_s2, batch['aa_seq'], batch['chain_ids'],
                    batch['res_idx'], batch['mask_res']
                )
                # Add noise augmentation to centroids input (optional)
                centroids_input = centroids_for_s2
                if args.centroid_noise > 0:
                    centroids_input = centroids_for_s2 + args.centroid_noise * torch.randn_like(centroids_for_s2)
                # Forward stage 2 with trunk tokens
                atoms_pred = model.forward_stage2(
                    centroids_input, batch['aa_seq'], batch['chain_ids'],
                    batch['res_idx'], batch['mask_res'], trunk_tokens=trunk_tokens
                )
                # Loss: MSE on atom positions
                # Reshape coords_res to [B, L*4, 3] for comparison
                B, L = batch['centroids'].shape[:2]
                atoms_target = batch['coords_res'].view(B, L * 4, 3)
                atoms_pred_flat = atoms_pred.view(B, L * 4, 3)
                loss_mse = compute_mse_loss(atoms_pred_flat, atoms_target, batch['mask_atom'])
                loss = loss_mse

                # Add geometry loss if enabled
                loss_geom = 0.0
                loss_bond = 0.0
                loss_angle = 0.0
                loss_omega = 0.0
                if geom_loss_fn is not None:
                    # Pass GT coords to detect chain breaks (don't penalize valid structural gaps)
                    geom_losses = geom_loss_fn(atoms_pred, batch['mask_res'], gt_coords=batch['coords_res'])
                    loss_geom = geom_losses['total'].item()
                    loss_bond = geom_losses['bond_length'].item()
                    loss_angle = geom_losses['bond_angle'].item()
                    loss_omega = geom_losses['omega'].item()
                    loss = loss + args.geom_weight * geom_losses['total']

                # Add contact loss for Stage 2
                loss_contact = 0.0
                if contact_loss_fn is not None and args.contact_stage in ["stage2", "both"]:
                    contact_losses = contact_loss_fn(
                        gt_centroids=batch['centroids'],
                        pred_atoms=atoms_pred,
                        gt_atoms=batch['coords_res'],
                        chain_ids=batch['chain_ids'],
                        mask=batch['mask_res']
                    )
                    loss_contact = contact_losses['stage2'].item()
                    loss = loss + args.contact_weight * contact_losses['stage2']

                # Track loss components for logging
                if accum_step == args.grad_accum - 1:
                    loss_components = {
                        'mse': loss_mse.item(),
                        'geom': loss_geom,
                        'bond': loss_bond,
                        'angle': loss_angle,
                        'omega': loss_omega,
                        'contact': loss_contact,
                    }

            else:  # end_to_end
                # Full pipeline
                result = model(
                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                    t, batch['mask_res'], mode="end_to_end"
                )
                centroids_pred = result['centroids_pred']
                atoms_pred = result['atoms_pred']

                # Loss: centroid loss + distance consistency + atom loss
                loss_centroid = compute_mse_loss(centroids_pred, batch['centroids'], batch['mask_res'])
                loss_dist = compute_distance_consistency_loss(
                    centroids_pred, batch['centroids'], batch['mask_res']
                )

                B, L = batch['centroids'].shape[:2]
                atoms_target = batch['coords_res'].view(B, L * 4, 3)
                atoms_pred_flat = atoms_pred.view(B, L * 4, 3)
                loss_atoms = compute_mse_loss(atoms_pred_flat, atoms_target, batch['mask_atom'])

                loss = loss_centroid + args.dist_weight * loss_dist + loss_atoms

                # Add geometry loss if enabled
                loss_geom = 0.0
                if geom_loss_fn is not None:
                    # Pass GT coords to detect chain breaks (don't penalize valid structural gaps)
                    geom_losses = geom_loss_fn(atoms_pred, batch['mask_res'], gt_coords=batch['coords_res'])
                    loss_geom = geom_losses['total'].item()
                    loss = loss + args.geom_weight * geom_losses['total']

                # Add contact loss for end-to-end (both stages)
                loss_contact = 0.0
                if contact_loss_fn is not None:
                    contact_losses = contact_loss_fn(
                        pred_centroids=centroids_pred,
                        gt_centroids=batch['centroids'],
                        pred_atoms=atoms_pred,
                        gt_atoms=batch['coords_res'],
                        chain_ids=batch['chain_ids'],
                        mask=batch['mask_res']
                    )
                    loss_contact = contact_losses['total'].item()
                    loss = loss + args.contact_weight * contact_losses['total']

                # Track individual losses (for last accumulation step)
                if accum_step == args.grad_accum - 1:
                    loss_components = {
                        'centroid': loss_centroid.item(),
                        'dist': loss_dist.item(),
                        'atoms': loss_atoms.item(),
                        'geom': loss_geom,
                        'contact': loss_contact,
                    }

            # Backward
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        loss = accum_loss

        if step % 100 == 0:
            elapsed = time.time() - start_time
            if args.mode == "end_to_end" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" | cnt: {lc.get('contact', 0):.4f}" if args.contact_weight > 0 else ""
                logger.log(f"Step {step:5d} | loss: {loss:.4f} | ctr: {lc['centroid']:.4f} | atm: {lc['atoms']:.4f} | dst: {lc['dist']:.4f}{contact_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            elif args.mode == "stage2_only" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" cnt:{lc.get('contact', 0):.3f}" if args.contact_weight > 0 else ""
                if args.geom_weight > 0:
                    logger.log(f"Step {step:5d} | loss: {loss:.4f} | mse: {lc['mse']:.4f} | geom: {lc['geom']:.4f} (bnd:{lc['bond']:.3f} ang:{lc['angle']:.3f} omg:{lc['omega']:.3f}{contact_str}) | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
                else:
                    logger.log(f"Step {step:5d} | loss: {loss:.6f} | mse: {lc['mse']:.6f}{contact_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            elif args.mode == "stage1_only" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" | cnt: {lc.get('contact', 0):.4f}" if args.contact_weight > 0 else ""
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | mse: {lc['mse']:.4f} | dst: {lc['dist']:.4f}{contact_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            else:
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on train set
                n_eval = min(args.n_eval_train, len(train_indices))
                eval_train_indices = random.sample(train_indices, n_eval)
                train_rmses = []
                for idx in eval_train_indices:
                    s = train_samples[idx]
                    batch = collate_batch([s], device)

                    if args.mode == "stage1_only":
                        # For Stage 1: evaluate centroid RMSE via diffusion sampling
                        centroids_pred = sample_centroids(
                            model, batch, noiser, device,
                            align_per_step=args.align_per_step,
                            recenter=args.recenter
                        )
                        rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']
                    elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                        # For Stage 2 with cached predictions: use Stage 1 predictions directly
                        atoms_pred = model.forward_stage2(
                            batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                            batch['res_idx'], batch['mask_res']
                        )
                        atoms_pred = atoms_pred.view(1, -1, 3)
                        rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']
                    else:
                        # For end_to_end or stage2 without cached: full sampling
                        atoms_pred = model.sample(
                            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                            noiser, batch['mask_res']
                        )
                        rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']
                    train_rmses.append(rmse)
                train_avg = sum(train_rmses) / len(train_rmses)

                # Evaluate on test set
                test_rmses = []
                test_dockq_scores = []
                test_lddt_scores = []
                test_ilddt_scores = []
                for idx in test_indices:
                    s = test_samples[idx]
                    batch = collate_batch([s], device)

                    if args.mode == "stage1_only":
                        centroids_pred = sample_centroids(
                            model, batch, noiser, device,
                            align_per_step=args.align_per_step,
                            recenter=args.recenter
                        )
                        rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']
                    elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                        # For Stage 2 with cached predictions: use Stage 1 predictions directly
                        atoms_pred = model.forward_stage2(
                            batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                            batch['res_idx'], batch['mask_res']
                        )
                        atoms_pred_flat = atoms_pred.view(1, -1, 3)
                        rmse = compute_rmse(atoms_pred_flat, batch['coords'], batch['mask_atom']).item() * s['std']

                        # Compute DockQ, lDDT/ilDDT for Stage 2
                        n_res = s['n_res']
                        pred_coords_res = atoms_pred[:, :n_res]  # [1, L, 4, 3]
                        gt_coords_res = batch['coords_res'][:, :n_res]

                        # DockQ
                        dockq_result = compute_dockq(
                            pred_coords_res[0], gt_coords_res[0],
                            batch['aa_seq'][0, :n_res], batch['chain_ids'][0, :n_res],
                            std=s['std']
                        )
                        if dockq_result['dockq'] is not None:
                            test_dockq_scores.append(dockq_result['dockq'])

                        # lDDT/ilDDT
                        lddt_result = compute_lddt_metrics(
                            pred_coords_res, gt_coords_res,
                            batch['chain_ids'][:, :n_res],
                            batch['mask_res'][:, :n_res],
                            coord_scale=s['std']
                        )
                        test_lddt_scores.append(lddt_result['lddt'])
                        if lddt_result['n_interface'] > 0:
                            test_ilddt_scores.append(lddt_result['ilddt'])
                    else:
                        # For end_to_end or stage2 without cached: full sampling
                        atoms_pred = model.sample(
                            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                            noiser, batch['mask_res']
                        )
                        rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']

                        # Compute DockQ and lDDT/ilDDT for Stage 2 / end-to-end
                        n_res = s['n_res']
                        pred_coords_res = atoms_pred[0].view(n_res, 4, 3)
                        gt_coords_res = batch['coords_res'][0, :n_res]
                        aa_seq = batch['aa_seq'][0, :n_res]
                        chain_ids = batch['chain_ids'][0, :n_res]
                        dockq_result = compute_dockq(pred_coords_res, gt_coords_res, aa_seq, chain_ids, std=s['std'])
                        if dockq_result['dockq'] is not None:
                            test_dockq_scores.append(dockq_result['dockq'])

                        # lDDT/ilDDT
                        lddt_result = compute_lddt_metrics(
                            pred_coords_res.unsqueeze(0), gt_coords_res.unsqueeze(0),
                            chain_ids.unsqueeze(0),
                            batch['mask_res'][:, :n_res],
                            coord_scale=s['std']
                        )
                        test_lddt_scores.append(lddt_result['lddt'])
                        if lddt_result['n_interface'] > 0:
                            test_ilddt_scores.append(lddt_result['ilddt'])

                    test_rmses.append(rmse)
                test_avg = sum(test_rmses) / len(test_rmses)

                metric_name = "Centroid RMSE" if args.mode == "stage1_only" else "Atom RMSE"
                log_msg = f"         >>> Train {metric_name} ({n_eval}): {train_avg:.4f} A | Test {metric_name} ({len(test_indices)}): {test_avg:.4f} A"
                if test_dockq_scores:
                    dockq_avg = sum(test_dockq_scores) / len(test_dockq_scores)
                    log_msg += f" | DockQ: {dockq_avg:.4f}"
                if test_lddt_scores:
                    lddt_avg = sum(test_lddt_scores) / len(test_lddt_scores)
                    log_msg += f" | lDDT: {lddt_avg:.4f}"
                if test_ilddt_scores:
                    ilddt_avg = sum(test_ilddt_scores) / len(test_ilddt_scores)
                    log_msg += f" | ilDDT: {ilddt_avg:.4f}"
                logger.log(log_msg)

                # Plot first sample
                s = train_samples[train_indices[0]]
                batch = collate_batch([s], device)

                if args.mode == "stage1_only":
                    # Plot centroids for Stage 1
                    centroids_pred = sample_centroids(
                        model, batch, noiser, device,
                        align_per_step=args.align_per_step,
                        recenter=args.recenter
                    )
                    n_res = s['n_res']
                    pred = centroids_pred[0, :n_res] * s['std']
                    target = batch['centroids'][0, :n_res] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item()
                    chain_ids_plot = batch['chain_ids'][0, :n_res]
                elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                    # Plot atoms for Stage 2 with cached predictions
                    atoms_pred = model.forward_stage2(
                        batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                        batch['res_idx'], batch['mask_res']
                    )
                    n = s['n_atoms']
                    pred = atoms_pred[0].view(-1, 3)[:n] * s['std']
                    target = batch['coords'][0, :n] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item()
                    chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]
                else:
                    # Plot atoms for end_to_end (or stage2 without cached)
                    atoms_pred = model.sample(
                        batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                        noiser, batch['mask_res']
                    )
                    n = s['n_atoms']
                    pred = atoms_pred[0, :n] * s['std']
                    target = batch['coords'][0, :n] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item()
                    chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]

                plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                plot_prediction(pred_aligned[0], target_c[0], chain_ids_plot,
                               s['sample_id'], rmse_viz, plot_path)
                logger.log(f"         >>> Saved plot: {plot_path}")

                # Save best model
                if test_avg < best_rmse:
                    best_rmse = test_avg
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    logger.log(f"         >>> New best test RMSE! Saved.")

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log(f"Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.4f} A")
    logger.log("")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()
