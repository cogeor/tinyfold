#!/usr/bin/env python
"""
Stage 2 Continuous: Atom Refinement from Noisy Centroids.

Single-step atom prediction (no diffusion) from noisy centroid positions.
Uses the same data split as Stage 1 (train_10k_continuous).

Usage:
    python train_stage2_continuous.py \
        --load_split outputs/train_10k_continuous/split.json \
        --output_dir outputs/stage2_continuous \
        --n_steps 50000
"""

import sys
import math
import random
import numpy as np
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

# Model imports
from models.atomrefine_continuous import AtomRefinerContinuous
from models.dockq_utils import compute_dockq
from models.training_utils import random_rotation_matrix

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    GeometryLoss,
)
from data_split import load_split


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


# =============================================================================
# Data Loading
# =============================================================================

def load_sample_raw(table, i, normalize=True):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center coordinates
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    # Compute std
    original_std = coords.std()

    # Normalize to unit variance
    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids (mean of 4 backbone atoms)
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)  # [L, 3]

    # Residue-level features
    aa_seq = seq_res
    chain_ids = chain_res
    res_idx = torch.arange(n_res)

    return {
        'coords': coords,           # [N_atoms, 3]
        'coords_res': coords_res,   # [L, 4, 3]
        'centroids': centroids,     # [L, 3]
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'res_idx': res_idx,
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
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors
    coords = torch.zeros(B, max_atoms, 3)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)

    stds = []

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
        mask_atom[i, :N] = True

        stds.append(s['std'])

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }


# =============================================================================
# Loss Functions
# =============================================================================

def compute_atom_mse_loss(pred_atoms, gt_atoms, mask_res):
    """Compute MSE loss on atom coordinates with Kabsch alignment.

    Args:
        pred_atoms: [B, L, 4, 3] predicted atom positions
        gt_atoms: [B, L, 4, 3] ground truth atom positions
        mask_res: [B, L] valid residue mask

    Returns:
        loss: scalar
    """
    B, L, n_atoms, _ = pred_atoms.shape

    # Flatten to [B, L*4, 3]
    pred_flat = pred_atoms.view(B, L * n_atoms, 3)
    gt_flat = gt_atoms.view(B, L * n_atoms, 3)
    mask_flat = mask_res.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, L * n_atoms)

    # Kabsch align target to pred's frame (gradient flows through pred only)
    gt_aligned, pred_c = kabsch_align(gt_flat, pred_flat, mask_flat)

    # Compute per-sample MSE
    sq_diff = (pred_c - gt_aligned) ** 2
    sq_diff = sq_diff.sum(dim=-1)  # [B, L*4]

    # Average over valid positions
    n_valid = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
    per_sample_loss = (sq_diff * mask_flat.float()).sum(dim=1) / n_valid.squeeze()

    return per_sample_loss.mean()


def compute_atom_rmse(pred_atoms, gt_atoms, mask_res, std=1.0):
    """Compute RMSE in Angstroms."""
    B, L, n_atoms, _ = pred_atoms.shape

    # Flatten
    pred_flat = pred_atoms.view(B, L * n_atoms, 3)
    gt_flat = gt_atoms.view(B, L * n_atoms, 3)
    mask_flat = mask_res.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, L * n_atoms)

    # Kabsch align
    gt_aligned, pred_c = kabsch_align(gt_flat, pred_flat, mask_flat)

    # RMSE
    sq_diff = ((pred_c - gt_aligned) ** 2).sum(dim=-1)
    n_valid = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
    mse = (sq_diff * mask_flat.float()).sum(dim=1) / n_valid.squeeze()
    rmse = torch.sqrt(mse).mean()

    return rmse.item() * std


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Plot prediction vs ground truth."""
    fig = plt.figure(figsize=(12, 5))

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    chain_ids = chain_ids.detach().cpu().numpy()

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
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Continuous Training")

    # Data
    parser.add_argument("--load_split", type=str, required=True,
                        help="Path to split.json from Stage 1")

    # Model
    parser.add_argument("--c_token", type=int, default=256)
    parser.add_argument("--c_atom", type=int, default=128)
    parser.add_argument("--trunk_layers", type=int, default=3)
    parser.add_argument("--refine_layers", type=int, default=3)
    parser.add_argument("--local_atom_blocks", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)

    # Training
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--n_eval_train", type=int, default=50)

    # Noise and augmentation
    parser.add_argument("--centroid_noise", type=float, default=0.1,
                        help="Noise std for centroids (0.1 = ~1A in normalized space)")
    parser.add_argument("--augment_rotation", action="store_true",
                        help="Apply random rotation augmentation during training")

    # Loss weights
    parser.add_argument("--geom_weight", type=float, default=0.1)
    parser.add_argument("--bond_length_weight", type=float, default=1.0)
    parser.add_argument("--bond_angle_weight", type=float, default=0.1)
    parser.add_argument("--omega_weight", type=float, default=0.1)
    parser.add_argument("--o_chirality_weight", type=float, default=0.1)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_continuous")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

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
    logger.log("Stage 2 Continuous: Atom Refinement Training")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Log config
    logger.log("Configuration:")
    logger.log(f"  output_dir:     {args.output_dir}")
    logger.log(f"  load_split:     {args.load_split}")
    logger.log(f"  batch_size:     {args.batch_size}")
    logger.log(f"  grad_accum:     {args.grad_accum}")
    logger.log(f"  eff_batch:      {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:        {args.n_steps}")
    logger.log(f"  eval_every:     {args.eval_every}")
    logger.log(f"  lr:             {args.lr}")
    logger.log(f"  centroid_noise: {args.centroid_noise}")
    logger.log(f"  augment_rot:    {args.augment_rotation}")
    logger.log(f"  geom_weight:    {args.geom_weight}")
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

    # Load split
    logger.log(f"Loading split from: {args.load_split}")
    train_indices, test_indices, loaded_info = load_split(args.load_split)
    logger.log(f"Data split:")
    logger.log(f"  Training: {len(train_indices)} samples")
    logger.log(f"  Test: {len(test_indices)} samples")
    logger.log("")

    # Preload samples
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample_raw(table, idx, normalize=True) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx, normalize=True) for idx in test_indices}
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples")
    logger.log("")

    # Create model
    model = AtomRefinerContinuous(
        c_token=args.c_token,
        c_atom=args.c_atom,
        trunk_layers=args.trunk_layers,
        refine_layers=args.refine_layers,
        local_atom_blocks=args.local_atom_blocks,
        n_heads=args.n_heads,
        dropout=0.0,
    ).to(device)

    # Log model info
    param_counts = model.count_parameters()
    logger.log("Model: AtomRefinerContinuous")
    logger.log(f"  c_token:           {args.c_token}")
    logger.log(f"  c_atom:            {args.c_atom}")
    logger.log(f"  trunk_layers:      {args.trunk_layers}")
    logger.log(f"  refine_layers:     {args.refine_layers}")
    logger.log(f"  local_atom_blocks: {args.local_atom_blocks}")
    logger.log(f"  Parameters:")
    logger.log(f"    trunk:      {param_counts['trunk']:,}")
    logger.log(f"    centroid:   {param_counts['centroid']:,}")
    logger.log(f"    refine:     {param_counts['refine']:,}")
    logger.log(f"    local_atom: {param_counts['local_atom']:,}")
    logger.log(f"    output:     {param_counts['output']:,}")
    logger.log(f"    TOTAL:      {param_counts['total']:,}")
    logger.log("")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (args.n_steps - args.warmup_steps)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Geometry loss
    geometry_loss = None
    if args.geom_weight > 0:
        geometry_loss = GeometryLoss(
            bond_length_weight=args.bond_length_weight,
            bond_angle_weight=args.bond_angle_weight,
            omega_weight=args.omega_weight,
            o_chirality_weight=args.o_chirality_weight,
            cb_chirality_weight=0.0,
        )

    # Training loop
    logger.log("Starting training...")
    logger.log("")

    train_indices_list = list(train_samples.keys())
    test_indices_list = list(test_samples.keys())

    best_test_rmse = float('inf')
    start_time = time.time()

    for step in range(1, args.n_steps + 1):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_mse = 0.0
        total_geom = 0.0

        for _ in range(args.grad_accum):
            # Sample batch
            batch_indices = random.sample(train_indices_list, args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            # Get batch dimensions
            B = batch['centroids'].shape[0]

            # Apply rotation augmentation
            centroids = batch['centroids']
            gt_atoms = batch['coords_res']
            if args.augment_rotation:
                R = random_rotation_matrix(B, device)  # [B, 3, 3]
                centroids = torch.bmm(centroids, R)  # [B, L, 3]
                L = gt_atoms.shape[1]
                gt_atoms = torch.bmm(gt_atoms.view(B, L * 4, 3), R).view(B, L, 4, 3)

            # Add noise to centroids
            noisy_centroids = centroids + args.centroid_noise * torch.randn_like(centroids)

            # Forward pass
            pred_atoms = model(
                batch['aa_seq'],
                batch['chain_ids'],
                batch['res_idx'],
                noisy_centroids,
                batch['mask_res'],
            )  # [B, L, 4, 3]

            # MSE loss (gt_atoms already set above with rotation)
            loss_mse = compute_atom_mse_loss(pred_atoms, gt_atoms, batch['mask_res'])

            # Geometry loss (expects [B, L, 4, 3] and mask [B, L])
            loss_geom = torch.tensor(0.0, device=device)
            if geometry_loss is not None:
                geom_result = geometry_loss(pred_atoms, batch['mask_res'], gt_coords=gt_atoms)
                loss_geom = geom_result['total']

            # Total loss
            loss = loss_mse + args.geom_weight * loss_geom
            loss = loss / args.grad_accum

            loss.backward()

            total_loss += loss.item() * args.grad_accum
            total_mse += loss_mse.item()
            total_geom += loss_geom.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Average over accumulation steps
        total_mse /= args.grad_accum
        total_geom /= args.grad_accum

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            logger.log(f"Step {step:5d} | loss: {total_loss:.4f} | mse: {total_mse:.4f} | geom: {total_geom:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Evaluation
        if step % args.eval_every == 0:
            model.eval()

            with torch.no_grad():
                # Evaluate on test set
                test_rmses = []
                test_dockqs = []

                for idx in test_indices_list[:50]:  # Limit to 50 for speed
                    s = test_samples[idx]
                    sample_batch = collate_batch([s], device)

                    # Use GT centroids (no noise for evaluation)
                    pred_atoms = model(
                        sample_batch['aa_seq'],
                        sample_batch['chain_ids'],
                        sample_batch['res_idx'],
                        sample_batch['centroids'],
                        sample_batch['mask_res'],
                    )

                    gt_atoms = sample_batch['coords_res']
                    rmse = compute_atom_rmse(pred_atoms, gt_atoms, sample_batch['mask_res'], std=s['std'])
                    test_rmses.append(rmse)

                    # DockQ
                    L = s['n_res']
                    dockq_result = compute_dockq(
                        pred_atoms[0, :L],
                        gt_atoms[0, :L],
                        sample_batch['aa_seq'][0, :L],
                        sample_batch['chain_ids'][0, :L],
                        std=s['std'],
                    )
                    if dockq_result['dockq'] is not None:
                        test_dockqs.append(dockq_result['dockq'])

                # Evaluate on train subset
                train_rmses = []
                eval_train_indices = random.sample(train_indices_list, min(args.n_eval_train, len(train_indices_list)))

                for idx in eval_train_indices:
                    s = train_samples[idx]
                    sample_batch = collate_batch([s], device)

                    pred_atoms = model(
                        sample_batch['aa_seq'],
                        sample_batch['chain_ids'],
                        sample_batch['res_idx'],
                        sample_batch['centroids'],
                        sample_batch['mask_res'],
                    )

                    gt_atoms = sample_batch['coords_res']
                    rmse = compute_atom_rmse(pred_atoms, gt_atoms, sample_batch['mask_res'], std=s['std'])
                    train_rmses.append(rmse)

            avg_test_rmse = np.mean(test_rmses)
            avg_train_rmse = np.mean(train_rmses)
            avg_dockq = np.mean(test_dockqs) if test_dockqs else 0.0

            logger.log("")
            logger.log(f"  [Eval @ step {step}]")
            logger.log(f"    Train RMSE: {avg_train_rmse:.2f} A")
            logger.log(f"    Test RMSE:  {avg_test_rmse:.2f} A")
            logger.log(f"    Test DockQ: {avg_dockq:.3f}")
            logger.log("")

            # Save best model
            if avg_test_rmse < best_test_rmse:
                best_test_rmse = avg_test_rmse
                ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_rmse': avg_test_rmse,
                    'dockq': avg_dockq,
                    'args': vars(args),
                }, ckpt_path)
                logger.log(f"    Saved best model (RMSE: {avg_test_rmse:.2f} A)")

            # Plot first test sample
            s = test_samples[test_indices_list[0]]
            sample_batch = collate_batch([s], device)
            pred_atoms = model(
                sample_batch['aa_seq'],
                sample_batch['chain_ids'],
                sample_batch['res_idx'],
                sample_batch['centroids'],
                sample_batch['mask_res'],
            )
            L = s['n_res']
            pred_flat = pred_atoms[0, :L].view(-1, 3) * s['std']
            gt_flat = sample_batch['coords_res'][0, :L].view(-1, 3) * s['std']
            chain_ids_flat = sample_batch['chain_ids'][0, :L].unsqueeze(-1).expand(-1, 4).reshape(-1)

            # Kabsch align for visualization
            pred_aligned, gt_c = kabsch_align(
                pred_flat.unsqueeze(0),
                gt_flat.unsqueeze(0),
            )

            plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
            plot_prediction(
                pred_aligned[0], gt_c[0], chain_ids_flat,
                s['sample_id'], test_rmses[0], plot_path
            )

    # Final save
    ckpt_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'step': args.n_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, ckpt_path)

    logger.log("")
    logger.log("=" * 70)
    logger.log("Training complete!")
    logger.log(f"  Best test RMSE: {best_test_rmse:.2f} A")
    logger.log(f"  Models saved to: {args.output_dir}")
    logger.log("=" * 70)
    logger.close()


if __name__ == "__main__":
    main()
