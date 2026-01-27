#!/usr/bin/env python
"""
ResFold E2E Training with Multi-Sample Diffusion Conditioning.

Two-phase training:
- Phase 1: Train Stage 1 only (use train_resfold.py --mode stage1_only)
- Phase 2: Train Stage 2 E2E with K diffusion samples from Stage 1

The key innovation is that Stage 2 receives K diffusion samples during training,
learning to aggregate uncertain predictions into robust atom positions.

Usage:
    # Phase 1: Train Stage 1 (use existing script)
    python train_resfold.py --mode stage1_only --n_train 80 --n_steps 10000 \
        --continuous_sigma --output_dir outputs/resfold_stage1

    # Phase 2: Train Stage 2 E2E (this script)
    python train_resfold_e2e_v2.py \
        --checkpoint outputs/resfold_stage1/best_model.pt \
        --n_train 80 --n_steps 5000 \
        --output_dir outputs/resfold_e2e
"""

import sys
import os
import math
import random
import time
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model imports
from models import create_schedule, KarrasSchedule, VENoiser, kabsch_align_to_target
from models.resfold_e2e import ResFoldE2E, sample_e2e
from models.dockq_utils import compute_dockq

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
    compute_lddt_metrics,
)

from data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info, save_split, load_split,
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


# =============================================================================
# Data Loading (copied from train_resfold.py)
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

    # Compute std (always, for reference)
    original_std = coords.std()

    # Optionally normalize to unit variance
    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids (mean of 4 backbone atoms)
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)  # [L, 3]

    return {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
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
# Training Step
# =============================================================================

def train_step_e2e(
    model: ResFoldE2E,
    batch: dict,
    noiser: VENoiser,
    geom_loss_fn: GeometryLoss,
    args,
) -> dict:
    """E2E training step with multi-sample diffusion.

    1. Generate K diffusion samples from Stage 1
    2. Pass to Stage 2 for atom prediction
    3. Compute dual loss (Stage 1 + Stage 2)
    """
    model.train()

    B, L = batch['aa_seq'].shape
    device = batch['centroids'].device

    # Forward E2E
    result = model.forward_e2e(
        gt_centroids=batch['centroids'],
        aa_seq=batch['aa_seq'],
        chain_ids=batch['chain_ids'],
        res_idx=batch['res_idx'],
        mask=batch['mask_res'],
        noiser=noiser,
        self_cond_prob=args.self_cond_prob,
        stratified_sigma=args.stratified_sigma,
    )

    centroids_samples = result['centroids_samples']  # [B, K, L, 3]
    atoms_pred = result['atoms_pred']                # [B, L, 4, 3]
    K = centroids_samples.shape[1]

    # === Stage 1 Loss: Average MSE over K samples ===
    loss_s1 = 0.0
    for k in range(K):
        loss_s1 += compute_mse_loss(
            centroids_samples[:, k],
            batch['centroids'],
            batch['mask_res']
        )
    loss_s1 = loss_s1 / K

    # Distance consistency loss (on mean of samples)
    mean_centroids = centroids_samples.mean(dim=1)
    loss_dist = compute_distance_consistency_loss(
        mean_centroids, batch['centroids'], batch['mask_res']
    )

    # === Stage 2 Loss: Atom MSE + Geometry ===
    atoms_pred_flat = atoms_pred.view(B, -1, 3)
    gt_atoms_flat = batch['coords_res'].view(B, -1, 3)
    loss_atom_mse = compute_mse_loss(atoms_pred_flat, gt_atoms_flat, batch['mask_atom'])

    # Geometry losses
    geom_result = geom_loss_fn(atoms_pred, batch['mask_res'], gt_coords=batch['coords_res'])
    loss_geom = geom_result['total']

    loss_s2 = loss_atom_mse + args.geom_weight * loss_geom

    # === Combined Loss ===
    total_loss = args.s1_weight * (loss_s1 + args.dist_weight * loss_dist) + args.s2_weight * loss_s2

    return {
        'total': total_loss,
        'stage1': loss_s1.item(),
        'stage2': loss_s2.item(),
        'centroid_mse': loss_s1.item(),
        'dist': loss_dist.item(),
        'atom_mse': loss_atom_mse.item(),
        'geom': loss_geom.item(),
        'geom_bond': geom_result['bond_length'].item(),
        'geom_angle': geom_result['bond_angle'].item(),
        'geom_omega': geom_result['omega'].item(),
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_e2e(
    model: ResFoldE2E,
    samples: dict,
    indices: list,
    noiser: VENoiser,
    device: torch.device,
    args,
) -> dict:
    """Evaluate E2E model with full diffusion sampling."""
    model.eval()

    centroid_rmses = []
    atom_rmses = []
    lddt_scores = []
    ilddt_scores = []
    dockq_scores = []

    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)

        # Full E2E sampling
        result = sample_e2e(
            model, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            noiser, mask=batch['mask_res'],
            n_samples=args.n_samples,
            self_cond=args.self_cond_prob > 0,
            align_per_step=True,
            recenter=True,
        )

        centroids_pred = result['mean_centroids']  # [1, L, 3]
        atoms_pred = result['atoms_pred']          # [1, L, 4, 3]

        n_res = s['n_res']
        n_atoms = s['n_atoms']

        # Centroid RMSE
        gt_centroids = batch['centroids'][:, :n_res]
        pred_centroids = centroids_pred[:, :n_res]
        centroid_rmse = compute_rmse(pred_centroids, gt_centroids, batch['mask_res'][:, :n_res]).item() * s['std']
        centroid_rmses.append(centroid_rmse)

        # Atom RMSE
        atoms_pred_flat = atoms_pred.view(1, -1, 3)[:, :n_atoms]
        gt_atoms_flat = batch['coords'][:, :n_atoms]
        atom_rmse = compute_rmse(atoms_pred_flat, gt_atoms_flat, batch['mask_atom'][:, :n_atoms]).item() * s['std']
        atom_rmses.append(atom_rmse)

        # lDDT / ilDDT
        pred_coords_res = atoms_pred[:, :n_res]
        gt_coords_res = batch['coords_res'][:, :n_res]
        chain_ids = batch['chain_ids'][:, :n_res]

        lddt_result = compute_lddt_metrics(
            pred_coords_res, gt_coords_res, chain_ids,
            batch['mask_res'][:, :n_res],
            coord_scale=s['std']
        )
        lddt_scores.append(lddt_result['lddt'])
        if lddt_result['n_interface'] > 0:
            ilddt_scores.append(lddt_result['ilddt'])

        # DockQ
        dockq_result = compute_dockq(
            pred_coords_res[0], gt_coords_res[0],
            batch['aa_seq'][0, :n_res], chain_ids[0],
            std=s['std']
        )
        if dockq_result['dockq'] is not None:
            dockq_scores.append(dockq_result['dockq'])

    return {
        'centroid_rmse': np.mean(centroid_rmses),
        'atom_rmse': np.mean(atom_rmses),
        'lddt': np.mean(lddt_scores),
        'ilddt': np.mean(ilddt_scores) if ilddt_scores else 0.0,
        'dockq': np.mean(dockq_scores) if dockq_scores else 0.0,
        'n_samples': len(indices),
    }


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

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2], c='cyan', s=10, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2], c='orange', s=10, alpha=0.7)
    ax2.set_title(f'Prediction\nRMSE: {rmse:.2f} A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ResFold E2E Training with Multi-Sample Diffusion")

    # Data
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=14)
    parser.add_argument("--n_eval_train", type=int, default=50)
    parser.add_argument("--min_atoms", type=int, default=200)
    parser.add_argument("--max_atoms", type=int, default=400)
    parser.add_argument("--load_split", type=str, default=None,
                        help="Load train/test split from JSON")

    # Training
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Stage 1 checkpoint (required for E2E)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--freeze_stage1", action="store_true",
                        help="Freeze Stage 1 weights (default: fine-tune)")

    # Model - Stage 1 (must match checkpoint)
    parser.add_argument("--c_token", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=9)
    parser.add_argument("--denoiser_blocks", type=int, default=7)

    # Model - Stage 2
    parser.add_argument("--s2_layers", type=int, default=6)
    parser.add_argument("--s2_heads", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of diffusion samples (K)")
    parser.add_argument("--s2_aggregation", type=str, default="learned",
                        choices=["learned", "mean", "attention"])

    # Diffusion
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--sigma_data", type=float, default=1.0)

    # Self-conditioning
    parser.add_argument("--self_cond_prob", type=float, default=0.5)
    parser.add_argument("--no_stratified_sigma", dest="stratified_sigma", action="store_false")
    parser.set_defaults(stratified_sigma=True)

    # Loss weights
    parser.add_argument("--s1_weight", type=float, default=1.0,
                        help="Weight for Stage 1 loss")
    parser.add_argument("--s2_weight", type=float, default=1.0,
                        help="Weight for Stage 2 loss")
    parser.add_argument("--dist_weight", type=float, default=0.1,
                        help="Weight for distance consistency loss")
    parser.add_argument("--geom_weight", type=float, default=0.1,
                        help="Weight for geometry losses")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/resfold_e2e")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    # Header
    logger.log("=" * 70)
    logger.log("ResFold E2E Training with Multi-Sample Diffusion")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Config
    logger.log("Configuration:")
    logger.log(f"  checkpoint:     {args.checkpoint}")
    logger.log(f"  freeze_stage1:  {args.freeze_stage1}")
    logger.log(f"  n_samples (K):  {args.n_samples}")
    logger.log(f"  s2_layers:      {args.s2_layers}")
    logger.log(f"  s2_aggregation: {args.s2_aggregation}")
    logger.log(f"  batch_size:     {args.batch_size}")
    logger.log(f"  grad_accum:     {args.grad_accum}")
    logger.log(f"  eff_batch:      {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:        {args.n_steps}")
    logger.log(f"  lr:             {args.lr}")
    logger.log(f"  s1_weight:      {args.s1_weight}")
    logger.log(f"  s2_weight:      {args.s2_weight}")
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

    # Data split
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        train_indices, test_indices, split_info = load_split(args.load_split)
    else:
        split_config = DataSplitConfig(
            n_train=args.n_train,
            n_test=args.n_test,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            seed=42,
        )
        train_indices, test_indices = get_train_test_indices(table, split_config)
        split_info = get_split_info(table, split_config)
        save_split(split_info, os.path.join(args.output_dir, "split.json"))

    logger.log(f"Data: {len(train_indices)} train, {len(test_indices)} test")

    # Preload samples
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx) for idx in test_indices}
    all_samples = {**train_samples, **test_samples}
    logger.log(f"  Loaded {len(all_samples)} samples")
    logger.log("")

    # Create model
    model = ResFoldE2E(
        c_token=args.c_token,
        trunk_layers=args.trunk_layers,
        denoiser_blocks=args.denoiser_blocks,
        n_timesteps=args.T,
        s2_layers=args.s2_layers,
        s2_heads=args.s2_heads,
        n_samples=args.n_samples,
        s2_aggregation=args.s2_aggregation,
    ).to(device)

    # Load Stage 1 checkpoint
    logger.log(f"Loading Stage 1 checkpoint: {args.checkpoint}")
    load_result = model.load_stage1_checkpoint(args.checkpoint, device)
    logger.log(f"  Loaded from step {load_result['step']}")
    logger.log(f"  Missing keys: {load_result['missing_keys']}")

    # Set training mode
    if args.freeze_stage1:
        model.freeze_stage1()
        logger.log("  Stage 1: FROZEN")
    else:
        model.unfreeze_stage1()
        logger.log("  Stage 1: trainable (fine-tuning)")

    # Count parameters
    counts = model.count_parameters()
    logger.log("")
    logger.log("Model:")
    logger.log(f"  Stage 1: {counts['stage1']:,} ({counts['stage1_pct']:.1f}%)")
    logger.log(f"  Stage 2: {counts['stage2']:,} ({counts['stage2_pct']:.1f}%)")
    logger.log(f"  Total:   {counts['total']:,}")
    logger.log(f"  Trainable: {counts['total_trainable']:,}")
    logger.log("")

    # Diffusion
    schedule = KarrasSchedule(
        n_steps=args.T,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=7.0,
    )
    noiser = VENoiser(schedule, sigma_data=args.sigma_data).to(device)
    logger.log(f"Diffusion: VE, sigma=[{args.sigma_min}, {args.sigma_max}], T={args.T}")
    logger.log("")

    # Geometry loss
    geom_loss_fn = GeometryLoss(
        bond_length_weight=1.0,
        bond_angle_weight=0.1,
        omega_weight=0.1,
        o_chirality_weight=0.1,
    )
    logger.log(f"Geometry loss: {geom_loss_fn}")
    logger.log("")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.min_lr
    )

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    start_time = time.time()

    model.train()
    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_losses = {}

        for accum_step in range(args.grad_accum):
            # Sample batch
            batch_indices = random.choices(train_indices, k=args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            # Forward + loss
            losses = train_step_e2e(model, batch, noiser, geom_loss_fn, args)
            loss = losses['total'] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            # Accumulate for logging
            if accum_step == args.grad_accum - 1:
                accum_losses = losses

        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            lc = accum_losses
            logger.log(
                f"Step {step:5d} | loss: {accum_loss:.4f} | "
                f"s1: {lc['stage1']:.4f} | s2: {lc['stage2']:.4f} | "
                f"atm: {lc['atom_mse']:.4f} | geo: {lc['geom']:.3f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
            )

        # Evaluation
        if step % args.eval_every == 0:
            model.eval()

            # Evaluate on subset of train
            n_eval_train = min(args.n_eval_train, len(train_indices))
            eval_train_indices = random.sample(train_indices, n_eval_train)
            train_results = evaluate_e2e(
                model, all_samples, eval_train_indices, noiser, device, args
            )

            # Evaluate on test
            test_results = evaluate_e2e(
                model, all_samples, test_indices, noiser, device, args
            )

            logger.log(
                f"         >>> Train ({n_eval_train}): "
                f"CenRMSE={train_results['centroid_rmse']:.2f}A "
                f"AtmRMSE={train_results['atom_rmse']:.2f}A "
                f"lDDT={train_results['lddt']:.3f}"
            )
            logger.log(
                f"         >>> Test ({len(test_indices)}):  "
                f"CenRMSE={test_results['centroid_rmse']:.2f}A "
                f"AtmRMSE={test_results['atom_rmse']:.2f}A "
                f"lDDT={test_results['lddt']:.3f} "
                f"DockQ={test_results['dockq']:.3f}"
            )

            # Save best model
            if test_results['atom_rmse'] < best_rmse:
                best_rmse = test_results['atom_rmse']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'test_atom_rmse': test_results['atom_rmse'],
                    'test_lddt': test_results['lddt'],
                    'args': vars(args),
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.log(f"         >>> New best! Saved.")

            # Plot first test sample
            s = test_samples[test_indices[0]]
            batch = collate_batch([s], device)
            result = sample_e2e(
                model, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                noiser, mask=batch['mask_res'], n_samples=args.n_samples
            )
            n = s['n_atoms']
            pred = result['atoms_pred'][0].view(-1, 3)[:n] * s['std']
            target = batch['coords'][0, :n] * s['std']
            pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
            rmse = compute_rmse(pred_aligned, target_c).item()
            chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]

            plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
            plot_prediction(pred_aligned[0], target_c[0], chain_ids_plot, s['sample_id'], rmse, plot_path)
            logger.log(f"         >>> Saved plot: {plot_path}")

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log("Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.4f} A")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()
