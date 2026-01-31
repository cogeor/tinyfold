#!/usr/bin/env python
"""
ResFold Stage 2 Training: Atom prediction from residue tokens + centroid samples.

Architecture:
- Stage 1 (frozen or trainable): ResidueDenoiser -> trunk_tokens + centroid samples
- Stage 2 (trainable): ResFoldAssembler -> atom positions [B, L, 4, 3]

Inputs to Stage 2:
- trunk_tokens: [B, L, c_token] from Stage 1 trunk encoder
- centroid_samples: [B, K, L, 3] K=5 centroid diffusion samples

Training modes:
- freeze_stage1=True: Only train assembler, use GT centroids + noise as samples
- freeze_stage1=False: Joint E2E training with auxiliary residue loss

Usage:
    python scripts/train_resfold_stage2.py \\
        --checkpoint outputs/stage1_50k_small/best_model.pt \\
        --trunk_layers 9 --denoiser_blocks 7 \\
        --n_train 80 --n_test 14 --n_steps 5000 \\
        --freeze_stage1 \\
        --output_dir outputs/resfold_stage2
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

# Shared utilities
from script_utils import (
    Logger,
    load_sample_residue as load_sample_raw,
    collate_batch_residue as collate_batch,
    set_seed,
    save_config,
    get_data_path,
)

# Model imports
from models.resfold import ResidueDenoiser
from models.resfold_assembler import ResFoldAssembler
from models.diffusion import VENoiser, create_schedule
from models.samplers import create_sampler

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
)

from data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info, save_split, load_split,
)
from models.dockq_utils import compute_dockq


# Data loading functions: load_sample_raw and collate_batch imported from script_utils


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

    # Compute residue centroids and reshape
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
# Centroid Sampling
# =============================================================================

def generate_centroid_samples(
    trunk_model: ResidueDenoiser,
    noiser: VENoiser,
    aa_seq: torch.Tensor,
    chain_ids: torch.Tensor,
    res_idx: torch.Tensor,
    mask: torch.Tensor,
    n_samples: int = 5,
    gt_centroids: torch.Tensor = None,  # For training: denoise from noisy GT
    use_gt_shortcut: bool = False,  # If True, just return noisy GT (fast but wrong)
) -> torch.Tensor:
    """Generate K centroid samples via Stage 1 diffusion.

    Training mode (gt_centroids provided):
    - Add noise to GT at varying levels
    - Run Stage 1 denoising to get predictions
    - Returns K denoised samples (Stage 1 predictions)

    Inference mode (gt_centroids=None):
    - Full diffusion sampling from noise

    Args:
        trunk_model: Stage 1 model
        noiser: VE noiser
        aa_seq, chain_ids, res_idx, mask: Sequence info
        n_samples: Number of samples to generate
        gt_centroids: GT centroids for training
        use_gt_shortcut: Skip denoising, just return noisy GT (for debugging)

    Returns:
        centroid_samples: [B, K, L, 3]
    """
    B, L = aa_seq.shape
    device = aa_seq.device

    # Get trunk tokens (shared across samples)
    with torch.no_grad():
        trunk_tokens = trunk_model.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

    if gt_centroids is not None:
        # Training mode: denoise from noisy GT using Stage 1
        samples = []
        for k in range(n_samples):
            # Sample different noise levels for diversity
            sigma = noiser.sample_sigma(B, device) * (0.3 + k * 0.3)  # Varying noise
            noisy, _ = noiser.add_noise_continuous(gt_centroids, sigma)

            if use_gt_shortcut:
                # Fast shortcut: just use noisy GT (for debugging)
                samples.append(noisy.detach())
            else:
                # Proper: run Stage 1 denoising
                with torch.no_grad():
                    pred = trunk_model.forward_sigma_with_trunk(
                        noisy, trunk_tokens, sigma, mask
                    )
                samples.append(pred.detach())

        return torch.stack(samples, dim=1)  # [B, K, L, 3]
    else:
        # Inference mode: full diffusion sampling from noise
        samples = []
        sigmas = noiser.schedule.sigmas.to(device)

        for k in range(n_samples):
            # Sample from noise
            x_t = torch.randn(B, L, 3, device=device) * sigmas[0]

            # Denoise through schedule
            with torch.no_grad():
                for i in range(len(sigmas) - 1):
                    sigma_cur = sigmas[i]
                    sigma_next = sigmas[i + 1]

                    # Denoise step
                    pred = trunk_model.forward_sigma_with_trunk(
                        x_t, trunk_tokens, sigma_cur.expand(B), mask
                    )

                    # Euler step
                    d = (x_t - pred) / sigma_cur
                    x_t = x_t + d * (sigma_next - sigma_cur)

            samples.append(x_t)

        return torch.stack(samples, dim=1)  # [B, K, L, 3]


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    trunk_model: ResidueDenoiser,
    assembler: ResFoldAssembler,
    batch: dict,
    noiser: VENoiser,
    geom_loss_fn: GeometryLoss,
    args,
) -> dict:
    """Training step for Stage 2.

    1. Get trunk_tokens from Stage 1
    2. Generate K centroid samples (noisy GT for training)
    3. Predict atom positions with assembler
    4. Compute losses
    """
    assembler.train()

    B, L = batch['aa_seq'].shape
    device = batch['centroids'].device

    loss_components = {
        'atom_mse': 0.0, 'geom': 0.0, 'residue_mse': 0.0, 'dist': 0.0,
    }

    # === Stage 1: Get trunk tokens ===
    if args.freeze_stage1:
        trunk_model.eval()
        with torch.no_grad():
            trunk_tokens = trunk_model.get_trunk_tokens(
                batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
            )
            # Generate centroid samples: Stage 1 denoising predictions
            # If fast_centroid_train, skip Stage 1 denoising and use noisy GT directly
            centroid_samples = generate_centroid_samples(
                trunk_model, noiser,
                batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res'],
                n_samples=args.n_samples,
                gt_centroids=batch['centroids'],
                use_gt_shortcut=args.fast_centroid_train,
            )
        loss_s1 = torch.tensor(0.0, device=device)
    else:
        trunk_model.train()
        trunk_tokens = trunk_model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )

        # Residue denoising loss
        sigma = noiser.sample_sigma(B, device)
        noisy_centroids = noiser.add_noise(batch['centroids'], sigma)
        pred_centroids = trunk_model.denoise(
            noisy_centroids, sigma,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res'],
            trunk_tokens=trunk_tokens,
        )

        loss_residue_mse = compute_mse_loss(pred_centroids, batch['centroids'], batch['mask_res'])
        loss_dist = compute_distance_consistency_loss(pred_centroids, batch['centroids'], batch['mask_res'])
        loss_s1 = args.s1_weight * (loss_residue_mse + args.dist_weight * loss_dist)

        loss_components['residue_mse'] = loss_residue_mse.item()
        loss_components['dist'] = loss_dist.item()

        # Generate centroid samples (with gradients for E2E)
        centroid_samples = generate_centroid_samples(
            trunk_model, noiser,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res'],
            n_samples=args.n_samples,
            gt_centroids=batch['centroids'],
        )

    # === Stage 2: Predict atoms ===
    if args.use_masked_training:
        # Random atom masking
        atom_mask = torch.rand(B, L, 4, device=device) > args.mask_ratio
        pred_atoms = assembler.forward_masked(
            trunk_tokens, centroid_samples,
            batch['coords_res'], atom_mask, batch['mask_res']
        )
    else:
        pred_atoms = assembler(trunk_tokens, centroid_samples, batch['mask_res'])

    # === Losses ===
    # Atom MSE
    pred_flat = pred_atoms.view(B, -1, 3)
    gt_flat = batch['coords_res'].view(B, -1, 3)
    loss_atom_mse = compute_mse_loss(pred_flat, gt_flat, batch['mask_atom'])

    # Geometry losses (use mean std for batch)
    mean_std = sum(batch['stds']) / len(batch['stds'])
    geom_result = geom_loss_fn(pred_atoms, batch['mask_res'], coord_std=mean_std)
    loss_geom = geom_result['total']

    loss_s2 = loss_atom_mse + args.geom_weight * loss_geom

    loss_components['atom_mse'] = loss_atom_mse.item()
    loss_components['geom'] = loss_geom.item()

    # Combined loss
    total_loss = loss_s1 + args.s2_weight * loss_s2

    return {
        'total': total_loss,
        **loss_components,
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    trunk_model: ResidueDenoiser,
    assembler: ResFoldAssembler,
    samples: dict,
    indices: list,
    noiser: VENoiser,
    device: torch.device,
    args,
    use_iterative: bool = False,
    k_per_step: int = 4,
    compute_dockq_scores: bool = False,
) -> dict:
    """Evaluate model.

    Args:
        use_iterative: If True, use iterative inference (mask->predict->unmask)
        k_per_step: Atoms to fix per iteration (if iterative)
        compute_dockq_scores: If True, also compute DockQ (slower)
    """
    trunk_model.eval()
    assembler.eval()

    atom_rmses = []
    dockq_scores = []

    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)

        # Get trunk tokens
        trunk_tokens = trunk_model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )

        # Generate centroid samples
        centroid_samples = generate_centroid_samples(
            trunk_model, noiser,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res'],
            n_samples=args.n_samples,
            gt_centroids=batch['centroids'] if args.use_gt_centroids_eval else None,
        )

        # Predict atoms
        if use_iterative:
            pred_atoms = assembler.sample_iterative_residue(
                trunk_tokens, centroid_samples, batch['mask_res'],
                k_residues_per_step=k_per_step,
                update_centroids=True,
            )
        else:
            pred_atoms = assembler(trunk_tokens, centroid_samples, batch['mask_res'])

        # Compute RMSE
        n_res = s['n_res']
        pred_res = pred_atoms[0, :n_res]  # [L, 4, 3]
        gt_res = batch['coords_res'][0, :n_res]  # [L, 4, 3]

        pred_flat = pred_res.view(-1, 3)
        gt_flat = gt_res.view(-1, 3)
        rmse = compute_rmse(pred_flat.unsqueeze(0), gt_flat.unsqueeze(0)).item() * s['std']
        atom_rmses.append(rmse)

        # Compute DockQ if requested
        if compute_dockq_scores:
            dockq_result = compute_dockq(
                pred_res, gt_res,
                batch['aa_seq'][0, :n_res],
                batch['chain_ids'][0, :n_res],
                std=s['std']
            )
            if dockq_result['dockq'] is not None:
                dockq_scores.append(dockq_result['dockq'])

    result = {
        'atom_rmse': np.mean(atom_rmses) if atom_rmses else 0.0,
        'atom_rmse_std': np.std(atom_rmses) if atom_rmses else 0.0,
        'n_samples': len(indices),
    }

    if compute_dockq_scores and dockq_scores:
        result['dockq'] = np.mean(dockq_scores)
        result['dockq_std'] = np.std(dockq_scores)
        result['n_dockq_valid'] = len(dockq_scores)

    return result


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ResFold Stage 2 Training")

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

    # Stage 1 checkpoint
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--c_token", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=9)
    parser.add_argument("--denoiser_blocks", type=int, default=7)

    # Stage 2 (assembler) config
    parser.add_argument("--assembler_layers", type=int, default=19)
    parser.add_argument("--assembler_heads", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=5)

    # Training options
    parser.add_argument("--freeze_stage1", action="store_true")
    parser.add_argument("--use_masked_training", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--use_gt_centroids_eval", action="store_true",
                        help="Use GT centroids for evaluation (faster)")
    parser.add_argument("--fast_centroid_train", action="store_true",
                        help="Use noisy GT centroids directly (skip Stage 1 denoising during training)")

    # Loss weights
    parser.add_argument("--s1_weight", type=float, default=1.0)
    parser.add_argument("--s2_weight", type=float, default=1.0)
    parser.add_argument("--dist_weight", type=float, default=0.1)
    parser.add_argument("--geom_weight", type=float, default=0.1)

    # Diffusion
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=10.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/resfold_stage2")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility: set seed before anything else
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save config at startup
    save_config(args, args.output_dir)

    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    # Header
    logger.log("=" * 70)
    logger.log("ResFold Stage 2 Training (Atom Assembly)")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Config
    logger.log("Configuration:")
    logger.log(f"  checkpoint:       {args.checkpoint}")
    logger.log(f"  freeze_stage1:    {args.freeze_stage1}")
    logger.log(f"  assembler_layers: {args.assembler_layers}")
    logger.log(f"  n_samples:        {args.n_samples}")
    logger.log(f"  batch_size:       {args.batch_size}")
    logger.log(f"  grad_accum:       {args.grad_accum}")
    logger.log(f"  eff_batch:        {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:          {args.n_steps}")
    logger.log(f"  lr:               {args.lr}")
    logger.log("")
    logger.log("Loss weights:")
    logger.log(f"  s1_weight:        {args.s1_weight} {'(ignored - frozen)' if args.freeze_stage1 else ''}")
    logger.log(f"  s2_weight:        {args.s2_weight}")
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
    data_path = get_data_path()
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

    # Load trunk model (Stage 1)
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

    trunk_model.load_state_dict(state_dict, strict=True)

    trunk_params = sum(p.numel() for p in trunk_model.parameters())
    if args.freeze_stage1:
        trunk_model.eval()
        for p in trunk_model.parameters():
            p.requires_grad = False
        logger.log(f"  Trunk params: {trunk_params:,} (frozen)")
    else:
        trunk_model.train()
        logger.log(f"  Trunk params: {trunk_params:,} (trainable)")

    # Create assembler (Stage 2)
    assembler = ResFoldAssembler(
        c_token=args.c_token,
        n_samples=args.n_samples,
        n_layers=args.assembler_layers,
        n_heads=args.assembler_heads,
        dropout=0.0,
    ).to(device)

    assembler_params = assembler.count_parameters()
    logger.log(f"  Assembler params: {assembler_params['total']:,} (trainable)")
    logger.log("")

    # Create noiser
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
    logger.log("Geometry loss: bond=1.0, angle=0.5, omega=0.5, chirality=0.5")
    logger.log("")

    # Optimizer
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
            batch_indices = random.choices(train_indices, k=args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            losses = train_step(trunk_model, assembler, batch, noiser, geom_loss_fn, args)
            loss = losses['total'] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if accum_step == args.grad_accum - 1:
                accum_losses = losses

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
                    f"atom: {lc.get('atom_mse', 0):.4f} | "
                    f"geom: {lc.get('geom', 0):.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                )
            else:
                logger.log(
                    f"Step {step:5d} | loss: {accum_loss:.4f} | "
                    f"s1: {lc.get('residue_mse', 0):.4f} | "
                    f"atom: {lc.get('atom_mse', 0):.4f} | "
                    f"geom: {lc.get('geom', 0):.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                )

        # Evaluation
        if step % args.eval_every == 0:
            n_eval_train = min(args.n_eval_train, len(train_indices))
            eval_train_indices = random.sample(train_indices, n_eval_train)
            train_results = evaluate(
                trunk_model, assembler, all_samples, eval_train_indices, noiser, device, args,
                compute_dockq_scores=False  # Skip DockQ for train (too slow)
            )

            test_results = evaluate(
                trunk_model, assembler, all_samples, test_indices, noiser, device, args,
                compute_dockq_scores=False  # Skip DockQ during training (too slow)
            )

            train_log = f"         >>> Train ({n_eval_train}): RMSE={train_results['atom_rmse']:.2f}A"
            logger.log(train_log)

            test_log = f"         >>> Test ({len(test_indices)}):  RMSE={test_results['atom_rmse']:.2f}A"
            if 'dockq' in test_results:
                test_log += f" | DockQ={test_results['dockq']:.3f}"
            logger.log(test_log)

            # Save best model
            if test_results['atom_rmse'] < best_rmse:
                best_rmse = test_results['atom_rmse']
                save_dict = {
                    'step': step,
                    'assembler_state_dict': assembler.state_dict(),
                    'test_atom_rmse': test_results['atom_rmse'],
                    'test_dockq': test_results.get('dockq'),
                    'args': vars(args),
                }
                if not args.freeze_stage1:
                    save_dict['trunk_state_dict'] = trunk_model.state_dict()
                torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
                dockq_str = f", DockQ={test_results['dockq']:.3f}" if test_results.get('dockq') else ""
                logger.log(f"         >>> New best! Saved.{dockq_str}")

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
