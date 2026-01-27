#!/usr/bin/env python
"""
ResFold Iterative Atom Assembly Training (Phase 3).

Iterative construction: predict K atoms at a time, conditioned on already-placed atoms.
Uses hierarchical clustering to determine placement order.

Training:
1. Sample x ∈ [0, N-K] atoms as "known" (teacher forcing from GT)
2. Select next K atoms via proximity to known
3. Cross-attention: target queries attend to known atom coordinates
4. Loss: relative distance to known atoms + position MSE

Usage:
    python scripts/train_resfold_iterative.py \
        --checkpoint outputs/train_10k_continuous/best_model.pt \
        --n_train 80 --n_test 14 --n_steps 5000 \
        --k_atoms 4 --eval_every 500 \
        --output_dir outputs/resfold_iterative
"""

import sys
import os
import random
import time
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model imports
from models.resfold import ResidueDenoiser
from models.iterative_assembler import IterativeAtomAssembler
from models.clustering import (
    hierarchical_cluster_atoms,
    select_next_atoms_to_place,
    simulate_known_mask,
)

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_relative_distance_loss,
    compute_distance_consistency_loss,
    GeometryLoss,
)

# Diffusion imports (for residue denoising loss when not frozen)
from models.diffusion import VENoiser, create_schedule

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
# Data Loading
# =============================================================================

def load_sample_raw(table, i, normalize=True):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
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

    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'coords': coords,           # [N, 3] flat atom coords
        'coords_res': coords_res,   # [L, 4, 3]
        'centroids': centroids,     # [L, 3]
        'atom_types': atom_types,
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
    n_res_list = []
    n_atoms_list = []

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
        n_res_list.append(L)
        n_atoms_list.append(N)

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
        'n_res': n_res_list,
        'n_atoms': n_atoms_list,
    }


# =============================================================================
# Training Step
# =============================================================================

def train_step_iterative(
    trunk_model: ResidueDenoiser,
    assembler: IterativeAtomAssembler,
    batch: dict,
    noiser: VENoiser,
    geom_loss_fn: GeometryLoss,
    args,
) -> dict:
    """Iterative assembly training step with optional E2E training.

    Losses:
    1. Residue diffusion loss (when --freeze_stage1 is False)
    2. Atom placement loss (relative distance + MSE)
    3. Geometry losses (bond lengths, angles, omega)

    Training modes:
    - freeze_stage1=True: Only train assembler, trunk provides frozen embeddings
    - freeze_stage1=False: Joint training, gradients flow to trunk via atom loss
    """
    assembler.train()

    B = batch['aa_seq'].shape[0]
    device = batch['centroids'].device

    loss_components = {
        'relative_dist': 0.0, 'position_mse': 0.0, 'n_targets': 0,
        'residue_mse': 0.0, 'dist_consist': 0.0, 'geom': 0.0,
    }

    # === Stage 1: Residue diffusion (optional) ===
    loss_s1 = torch.tensor(0.0, device=device)

    if args.freeze_stage1:
        trunk_model.eval()
        with torch.no_grad():
            trunk_tokens = trunk_model.get_trunk_tokens(
                batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
            )
    else:
        trunk_model.train()
        # Get trunk tokens WITH gradients
        trunk_tokens = trunk_model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )

        # Residue denoising loss: sample noise level, predict denoised centroids
        sigma = noiser.sample_sigma(B, device)
        noisy_centroids = noiser.add_noise(batch['centroids'], sigma)

        # Denoise
        pred_centroids = trunk_model.denoise(
            noisy_centroids, sigma,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res'],
            trunk_tokens=trunk_tokens,
        )

        # Residue MSE loss
        loss_residue_mse = compute_mse_loss(
            pred_centroids, batch['centroids'], batch['mask_res']
        )

        # Distance consistency loss
        loss_dist = compute_distance_consistency_loss(
            pred_centroids, batch['centroids'], batch['mask_res']
        )

        loss_s1 = args.s1_weight * (loss_residue_mse + args.dist_weight * loss_dist)
        loss_components['residue_mse'] = loss_residue_mse.item()
        loss_components['dist_consist'] = loss_dist.item()

    # === Stage 2: Iterative atom assembly ===
    total_atom_loss = torch.tensor(0.0, device=device)
    all_pred_atoms = []  # For geometry loss

    # Process each sample in batch
    for b in range(B):
        n_res = batch['n_res'][b]
        n_atoms = batch['n_atoms'][b]
        chain_ids = batch['chain_ids'][b, :n_res]

        # Get GT atom coordinates [N, 3]
        gt_coords = batch['coords'][b, :n_atoms]

        # Sample number of known atoms: uniform in [0, N-K]
        k = args.k_atoms
        max_known = max(0, n_atoms - k)
        n_known = random.randint(0, max_known)

        # Simulate placement order to get known_mask
        known_mask = simulate_known_mask(
            gt_coords, chain_ids, n_known, k_per_step=k
        )

        # Select next K atoms to predict
        remaining = (~known_mask).sum().item()
        actual_k = min(k, remaining)

        if actual_k == 0:
            continue  # All atoms already placed

        target_idx = select_next_atoms_to_place(
            gt_coords, known_mask, actual_k, cluster_ids=None
        )

        # Get residue indices for target atoms (atom_idx // 4)
        target_res_idx = target_idx // 4

        # Prepare inputs for assembler (add batch dim)
        trunk_b = trunk_tokens[b:b+1]  # [1, L, C]

        # Known coordinates (teacher forcing from GT)
        n_known_actual = known_mask.sum().item()
        if n_known_actual > 0:
            known_coords = gt_coords[known_mask].unsqueeze(0)  # [1, n_known, 3]
            known_mask_b = torch.ones(1, n_known_actual, dtype=torch.bool, device=device)
        else:
            # Cold start: no known atoms
            known_coords = torch.zeros(1, 1, 3, device=device)
            known_mask_b = torch.zeros(1, 1, dtype=torch.bool, device=device)

        target_idx_b = target_idx.unsqueeze(0)      # [1, K]
        target_res_idx_b = target_res_idx.unsqueeze(0)  # [1, K]

        # Forward pass
        pred_coords = assembler(
            trunk_b, known_coords, known_mask_b, target_idx_b, target_res_idx_b
        )  # [1, K, 3]

        # Ground truth for target atoms
        gt_target = gt_coords[target_idx].unsqueeze(0)  # [1, K, 3]

        # Loss: relative distances to known atoms
        if n_known_actual > 0:
            loss_rel = compute_relative_distance_loss(
                pred_coords, gt_target, known_coords, known_mask_b,
                align_first=args.align_before_loss,
            )
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Direct MSE for stability
        loss_mse = F.mse_loss(pred_coords, gt_target)

        sample_loss = args.rel_dist_weight * loss_rel + args.mse_weight * loss_mse
        total_atom_loss = total_atom_loss + sample_loss

        loss_components['relative_dist'] += loss_rel.item()
        loss_components['position_mse'] += loss_mse.item()
        loss_components['n_targets'] += actual_k

        # Collect predictions for geometry loss (reconstruct full atom tensor)
        if args.geom_weight > 0:
            # Build full prediction: known from GT, target from pred
            full_pred = gt_coords.clone()
            full_pred[target_idx] = pred_coords[0]
            all_pred_atoms.append(full_pred.view(n_res, 4, 3))

    # Average atom loss over batch
    n_valid = max(1, sum(1 for b in range(B) if batch['n_atoms'][b] > args.k_atoms))
    total_atom_loss = total_atom_loss / n_valid

    for key in ['relative_dist', 'position_mse']:
        loss_components[key] /= n_valid

    # === Geometry Loss ===
    loss_geom = torch.tensor(0.0, device=device)
    if args.geom_weight > 0 and all_pred_atoms:
        # Pad to batch
        max_res = max(p.shape[0] for p in all_pred_atoms)
        pred_atoms_padded = torch.zeros(len(all_pred_atoms), max_res, 4, 3, device=device)
        mask_padded = torch.zeros(len(all_pred_atoms), max_res, dtype=torch.bool, device=device)

        for i, p in enumerate(all_pred_atoms):
            L = p.shape[0]
            pred_atoms_padded[i, :L] = p
            mask_padded[i, :L] = True

        geom_result = geom_loss_fn(pred_atoms_padded, mask_padded)
        loss_geom = geom_result['total']
        loss_components['geom'] = loss_geom.item()

    # === Combined Loss ===
    total_loss = loss_s1 + args.s2_weight * total_atom_loss + args.geom_weight * loss_geom

    return {
        'total': total_loss,
        **loss_components,
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_iterative(
    trunk_model: ResidueDenoiser,
    assembler: IterativeAtomAssembler,
    samples: dict,
    indices: list,
    device: torch.device,
    args,
) -> dict:
    """Evaluate iterative assembly with full construction."""
    trunk_model.eval()
    assembler.eval()

    atom_rmses = []

    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)

        n_res = s['n_res']
        n_atoms = s['n_atoms']
        k = args.k_atoms

        # Get trunk tokens
        trunk_tokens = trunk_model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )

        gt_coords = batch['coords'][0, :n_atoms]
        chain_ids = batch['chain_ids'][0, :n_res]

        # Iteratively construct
        constructed = torch.zeros(n_atoms, 3, device=device)
        known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

        while known_mask.sum() < n_atoms:
            remaining = n_atoms - known_mask.sum().item()
            actual_k = min(k, remaining)

            # Select next atoms (use GT for fair evaluation)
            target_idx = select_next_atoms_to_place(
                gt_coords, known_mask, actual_k, cluster_ids=None
            )

            if len(target_idx) == 0:
                break

            target_res_idx = target_idx // 4

            # Prepare inputs
            n_known = known_mask.sum().item()
            if n_known > 0:
                known_coords = constructed[known_mask].unsqueeze(0)
                known_mask_b = torch.ones(1, n_known, dtype=torch.bool, device=device)
            else:
                known_coords = torch.zeros(1, 1, 3, device=device)
                known_mask_b = torch.zeros(1, 1, dtype=torch.bool, device=device)

            pred_coords = assembler(
                trunk_tokens,
                known_coords,
                known_mask_b,
                target_idx.unsqueeze(0),
                target_res_idx.unsqueeze(0),
            )  # [1, K, 3]

            # Update constructed structure
            constructed[target_idx] = pred_coords[0]
            known_mask[target_idx] = True

        # Compute RMSE
        rmse = compute_rmse(
            constructed.unsqueeze(0),
            gt_coords.unsqueeze(0)
        ).item() * s['std']
        atom_rmses.append(rmse)

    return {
        'atom_rmse': np.mean(atom_rmses) if atom_rmses else 0.0,
        'atom_rmse_std': np.std(atom_rmses) if atom_rmses else 0.0,
        'n_samples': len(indices),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Plot prediction vs ground truth."""
    fig = plt.figure(figsize=(12, 5))

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    chain_np = chain_ids.cpu().numpy()

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    mask_a = chain_np == 0
    mask_b = chain_np == 1
    if mask_a.any():
        ax1.scatter(target_np[mask_a, 0], target_np[mask_a, 1], target_np[mask_a, 2], c='blue', s=10, alpha=0.7)
    if mask_b.any():
        ax1.scatter(target_np[mask_b, 0], target_np[mask_b, 1], target_np[mask_b, 2], c='red', s=10, alpha=0.7)
    ax1.set_title(f'{sample_id}\nGround Truth')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred_np[mask_a, 0], pred_np[mask_a, 1], pred_np[mask_a, 2], c='cyan', s=10, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred_np[mask_b, 0], pred_np[mask_b, 1], pred_np[mask_b, 2], c='orange', s=10, alpha=0.7)
    ax2.set_title(f'Prediction\nRMSE: {rmse:.2f} A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ResFold Iterative Assembly Training")

    # Data
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=14)
    parser.add_argument("--n_eval_train", type=int, default=50)
    parser.add_argument("--min_atoms", type=int, default=200)
    parser.add_argument("--max_atoms", type=int, default=600)
    parser.add_argument("--load_split", type=str, default=None)

    # Training
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Stage 1 checkpoint (trunk encoder)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint (for trunk encoder)")

    # Trunk model config (must match checkpoint)
    parser.add_argument("--c_token", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=14)
    parser.add_argument("--denoiser_blocks", type=int, default=10)

    # Iterative assembler config
    parser.add_argument("--assembler_layers", type=int, default=4)
    parser.add_argument("--assembler_heads", type=int, default=8)

    # Iterative assembly args
    parser.add_argument("--k_atoms", type=int, default=4,
                        help="Number of atoms to predict per step")
    parser.add_argument("--rel_dist_weight", type=float, default=1.0,
                        help="Weight for relative distance loss")
    parser.add_argument("--mse_weight", type=float, default=0.1,
                        help="Weight for direct MSE loss")
    parser.add_argument("--align_before_loss", action="store_true",
                        help="Kabsch align before computing loss")

    # Stage 1 training options
    parser.add_argument("--freeze_stage1", action="store_true",
                        help="Freeze Stage 1 (trunk) weights. If False, joint E2E training.")
    parser.add_argument("--s1_weight", type=float, default=1.0,
                        help="Weight for Stage 1 residue diffusion loss (only when not frozen)")
    parser.add_argument("--s2_weight", type=float, default=1.0,
                        help="Weight for Stage 2 atom placement loss")
    parser.add_argument("--dist_weight", type=float, default=0.1,
                        help="Weight for distance consistency loss (Stage 1)")
    parser.add_argument("--geom_weight", type=float, default=0.1,
                        help="Weight for geometry losses (bond lengths, angles)")

    # Diffusion config (for Stage 1 when not frozen)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=10.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/resfold_iterative")

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
    logger.log("ResFold Iterative Atom Assembly Training")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Config
    logger.log("Configuration:")
    logger.log(f"  checkpoint:       {args.checkpoint}")
    logger.log(f"  freeze_stage1:    {args.freeze_stage1}")
    logger.log(f"  k_atoms:          {args.k_atoms}")
    logger.log(f"  assembler_layers: {args.assembler_layers}")
    logger.log(f"  batch_size:       {args.batch_size}")
    logger.log(f"  grad_accum:       {args.grad_accum}")
    logger.log(f"  eff_batch:        {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:          {args.n_steps}")
    logger.log(f"  lr:               {args.lr}")
    logger.log("")
    logger.log("Loss weights:")
    logger.log(f"  s1_weight:        {args.s1_weight} {'(ignored - frozen)' if args.freeze_stage1 else ''}")
    logger.log(f"  s2_weight:        {args.s2_weight}")
    logger.log(f"  rel_dist_weight:  {args.rel_dist_weight}")
    logger.log(f"  mse_weight:       {args.mse_weight}")
    logger.log(f"  dist_weight:      {args.dist_weight}")
    logger.log(f"  geom_weight:      {args.geom_weight}")
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

    # Load trunk model (Stage 1 encoder)
    logger.log(f"Loading trunk model from: {args.checkpoint}")
    trunk_model = ResidueDenoiser(
        c_token=args.c_token,
        trunk_layers=args.trunk_layers,
        denoiser_blocks=args.denoiser_blocks,
        n_timesteps=args.T,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Handle pipeline checkpoint (stage1. prefix)
    if any(k.startswith('stage1.') for k in state_dict.keys()):
        state_dict = {k.replace('stage1.', ''): v for k, v in state_dict.items() if k.startswith('stage1.')}

    trunk_model.load_state_dict(state_dict, strict=False)

    trunk_params = sum(p.numel() for p in trunk_model.parameters())
    if args.freeze_stage1:
        trunk_model.eval()
        for p in trunk_model.parameters():
            p.requires_grad = False
        logger.log(f"  Trunk params: {trunk_params:,} (frozen)")
    else:
        trunk_model.train()
        logger.log(f"  Trunk params: {trunk_params:,} (trainable)")

    # Create iterative assembler
    assembler = IterativeAtomAssembler(
        c_token=args.c_token,
        n_layers=args.assembler_layers,
        n_heads=args.assembler_heads,
        dropout=0.0,
    ).to(device)

    assembler_params = assembler.count_parameters()
    logger.log(f"  Assembler params: {assembler_params['total']:,} (trainable)")
    logger.log("")

    # Create noiser (for residue diffusion loss when not frozen)
    schedule = create_schedule("karras", n_steps=args.T, sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    noiser = VENoiser(schedule)
    logger.log(f"Diffusion: n_steps={args.T}, sigma=[{args.sigma_min}, {args.sigma_max}]")

    # Create geometry loss
    geom_loss_fn = GeometryLoss(
        bond_length_weight=1.0,
        bond_angle_weight=0.5,
        omega_weight=0.5,
        o_chirality_weight=0.5,
    )
    logger.log("Geometry loss: bond_length=1.0, bond_angle=0.5, omega=0.5, o_chirality=0.5")
    logger.log("")

    # Optimizer - include trunk params if not frozen
    if args.freeze_stage1:
        trainable_params = list(assembler.parameters())
    else:
        trainable_params = list(trunk_model.parameters()) + list(assembler.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.min_lr
    )
    logger.log(f"Optimizer: AdamW, {sum(p.numel() for p in trainable_params):,} trainable params")

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    start_time = time.time()

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
            losses = train_step_iterative(
                trunk_model, assembler, batch, noiser, geom_loss_fn, args
            )
            loss = losses['total'] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

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
            if args.freeze_stage1:
                logger.log(
                    f"Step {step:5d} | loss: {accum_loss:.4f} | "
                    f"rel: {lc.get('relative_dist', 0):.4f} | "
                    f"mse: {lc.get('position_mse', 0):.4f} | "
                    f"geom: {lc.get('geom', 0):.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                )
            else:
                logger.log(
                    f"Step {step:5d} | loss: {accum_loss:.4f} | "
                    f"s1: {lc.get('residue_mse', 0):.4f} | "
                    f"rel: {lc.get('relative_dist', 0):.4f} | "
                    f"mse: {lc.get('position_mse', 0):.4f} | "
                    f"geom: {lc.get('geom', 0):.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                )

        # Evaluation
        if step % args.eval_every == 0:
            # Evaluate on subset of train
            n_eval_train = min(args.n_eval_train, len(train_indices))
            eval_train_indices = random.sample(train_indices, n_eval_train)
            train_results = evaluate_iterative(
                trunk_model, assembler, all_samples, eval_train_indices, device, args
            )

            # Evaluate on test
            test_results = evaluate_iterative(
                trunk_model, assembler, all_samples, test_indices, device, args
            )

            logger.log(
                f"         >>> Train ({n_eval_train}): "
                f"RMSE={train_results['atom_rmse']:.2f}A"
            )
            logger.log(
                f"         >>> Test ({len(test_indices)}):  "
                f"RMSE={test_results['atom_rmse']:.2f}A"
            )

            # Save best model
            if test_results['atom_rmse'] < best_rmse:
                best_rmse = test_results['atom_rmse']
                save_dict = {
                    'step': step,
                    'assembler_state_dict': assembler.state_dict(),
                    'test_atom_rmse': test_results['atom_rmse'],
                    'args': vars(args),
                }
                # Save trunk model if trained jointly
                if not args.freeze_stage1:
                    save_dict['trunk_state_dict'] = trunk_model.state_dict()
                torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
                logger.log(f"         >>> New best! Saved.")

            # Plot first test sample
            with torch.no_grad():
                s = test_samples[test_indices[0]]
                batch = collate_batch([s], device)
                n_atoms = s['n_atoms']
                gt_coords = batch['coords'][0, :n_atoms]
                chain_ids = batch['chain_ids'][0, :s['n_res']]

                # Run full iterative construction
                trunk_model.eval()
                trunk_tokens = trunk_model.get_trunk_tokens(
                    batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
                )
                assembler.eval()
                constructed = torch.zeros(n_atoms, 3, device=device)
                known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

                while known_mask.sum() < n_atoms:
                    remaining = n_atoms - known_mask.sum().item()
                    actual_k = min(args.k_atoms, remaining)
                    target_idx = select_next_atoms_to_place(gt_coords, known_mask, actual_k)
                    if len(target_idx) == 0:
                        break

                    n_known = known_mask.sum().item()
                    if n_known > 0:
                        known_coords = constructed[known_mask].unsqueeze(0)
                        known_mask_b = torch.ones(1, n_known, dtype=torch.bool, device=device)
                    else:
                        known_coords = torch.zeros(1, 1, 3, device=device)
                        known_mask_b = torch.zeros(1, 1, dtype=torch.bool, device=device)

                    pred = assembler(
                        trunk_tokens, known_coords, known_mask_b,
                        target_idx.unsqueeze(0), (target_idx // 4).unsqueeze(0)
                    )
                    constructed[target_idx] = pred[0]
                    known_mask[target_idx] = True

                # Align and plot
                pred_aligned, gt_centered = kabsch_align(
                    constructed.unsqueeze(0), gt_coords.unsqueeze(0)
                )
                rmse = compute_rmse(pred_aligned, gt_centered).item() * s['std']

                # Expand chain_ids to atom level
                chain_ids_atom = chain_ids.repeat_interleave(4)[:n_atoms]

                plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                plot_prediction(
                    pred_aligned[0] * s['std'], gt_centered[0] * s['std'],
                    chain_ids_atom, s['sample_id'], rmse, plot_path
                )
                logger.log(f"         >>> Saved plot: {plot_path}")

            # Restore train mode if needed
            if not args.freeze_stage1:
                trunk_model.train()
            assembler.train()

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
