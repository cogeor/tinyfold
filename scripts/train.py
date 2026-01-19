#!/usr/bin/env python
"""
TinyFold training script.

Supports multiple model architectures and noise types via factories.
Outputs training log to output_dir/train.log.

Usage:
    python train.py --noise_type linear_chain --output_dir outputs/train_linear_1M
    python train.py --noise_type gaussian --output_dir outputs/train_gaussian_1M
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from models import (
    create_model, list_models,
    create_schedule, create_noiser, list_noise_types,
    TimestepCurriculum,
)
from data_split import DataSplitConfig, get_train_test_indices, get_split_info


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Dual output to console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = open(log_path, 'w', buffering=1)  # Line buffered

    def log(self, msg: str = ""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# =============================================================================
# Sampling and alignment
# =============================================================================

@torch.no_grad()
def ddpm_sample(model, atom_types, atom_to_res, aa_seq, chain_ids, noiser, mask=None,
                clamp_val=3.0, x_linear=None):
    """Diffusion sampling loop.

    For Gaussian noise: starts from random noise, uses DDPM reverse.
    For linear_chain: starts from extended chain, uses interpolation reverse.

    Args:
        x_linear: Extended chain coordinates [B, N, 3] for linear_chain noise.
                  If None and noiser has reverse_step, will be generated.
    """
    device = atom_types.device
    B, N = atom_types.shape

    # Check if this is linear_chain noise (has reverse_step method)
    use_linear_chain = hasattr(noiser, 'reverse_step')

    if use_linear_chain:
        # Start from extended chain
        if x_linear is None:
            from models.diffusion import generate_extended_chain
            x_linear = torch.zeros(B, N, 3, device=device)
            for b in range(B):
                x_linear[b] = generate_extended_chain(
                    N, atom_to_res[b], atom_types[b], chain_ids[b], device
                )
            x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
            x_linear = x_linear / x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x = x_linear.clone()
    else:
        # Start from random noise
        x = torch.randn(B, N, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if use_linear_chain:
            # Linear chain reverse: interpolate between x0_pred and x_linear
            x = noiser.reverse_step(x, x0_pred, t, x_linear)
        else:
            # DDPM reverse
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

    return x


def kabsch_align(pred, target, mask=None):
    """Kabsch alignment for rotation-invariant comparison."""
    B, N, _ = pred.shape

    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        pred_mean = (pred * mask_exp).sum(dim=1, keepdim=True) / n_valid
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

    pred_c = pred - pred_mean
    target_c = target - target_mean

    if mask is not None:
        pred_c = pred_c * mask_exp
        target_c = target_c * mask_exp

    H = torch.bmm(pred_c.transpose(1, 2), target_c)
    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=pred.device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))
    pred_aligned = torch.bmm(pred_c, R.transpose(1, 2))

    return pred_aligned, target_c


def compute_loss(pred, target, mask=None):
    """MSE loss after Kabsch alignment."""
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

    if mask is not None:
        loss = (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        loss = sq_diff.mean()

    return loss


def compute_rmse(pred, target, mask=None):
    """RMSE after Kabsch alignment."""
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

    if mask is not None:
        rmse = torch.sqrt((sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1))
    else:
        rmse = torch.sqrt(sq_diff.mean())

    return rmse


# =============================================================================
# Data loading
# =============================================================================

def find_medium_samples(table, min_atoms=200, max_atoms=400):
    """Find samples with atom count in range."""
    indices = []
    for i in range(len(table)):
        n_atoms = len(table['atom_type'][i].as_py())
        if min_atoms <= n_atoms <= max_atoms:
            indices.append(i)
    return indices


def load_sample_raw(table, i):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    coords = coords.reshape(n_atoms, 3)
    aa_seq = seq_res[atom_to_res]
    chain_ids = chain_res[atom_to_res]

    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    return {
        'coords': coords,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'std': std.item(),
        'n_atoms': n_atoms,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    """Collate samples into a padded batch."""
    B = len(samples)
    max_atoms = max(s['n_atoms'] for s in samples)

    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    aa_seq = torch.zeros(B, max_atoms, dtype=torch.long)
    chain_ids = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    stds = []

    for i, s in enumerate(samples):
        n = s['n_atoms']
        coords[i, :n] = s['coords']
        atom_types[i, :n] = s['atom_types']
        atom_to_res[i, :n] = s['atom_to_res']
        aa_seq[i, :n] = s['aa_seq']
        chain_ids[i, :n] = s['chain_ids']
        mask[i, :n] = True
        stds.append(s['std'])

    return {
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'mask': mask.to(device),
        'stds': stds,
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Plot prediction vs ground truth for a single protein."""
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
    parser = argparse.ArgumentParser(description="TinyFold training")

    # Data
    parser.add_argument("--n_train", type=int, default=80, help="Number of training samples")
    parser.add_argument("--n_test", type=int, default=14, help="Number of test samples")
    parser.add_argument("--n_eval_train", type=int, default=200, help="Number of train samples to eval (0=all)")
    parser.add_argument("--min_atoms", type=int, default=200, help="Min atoms per sample")
    parser.add_argument("--max_atoms", type=int, default=400, help="Max atoms per sample")
    parser.add_argument("--batch_size", type=int, default=128)

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    # Model
    parser.add_argument("--model", type=str, default="attention_v2", choices=list_models())
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)

    # Diffusion
    parser.add_argument("--schedule", type=str, default="linear", help="Alpha-bar schedule")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=list_noise_types(),
                        help="Noise type: gaussian or linear_chain")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Gaussian noise scale for linear_chain")
    parser.add_argument("--T", type=int, default=50, help="Diffusion timesteps")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", help="Enable timestep curriculum")
    parser.add_argument("--curriculum_warmup", type=int, default=5000, help="Steps to reach full T")
    parser.add_argument("--curriculum_schedule", type=str, default="linear",
                        choices=["linear", "cosine"], help="Curriculum progression schedule")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/train")

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
    logger.log("TinyFold Training")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Script: {os.path.abspath(__file__)}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Log config
    logger.log("Configuration:")
    logger.log(f"  output_dir:  {args.output_dir}")
    logger.log(f"  n_train:     {args.n_train}")
    logger.log(f"  n_test:      {args.n_test}")
    logger.log(f"  batch_size:  {args.batch_size}")
    logger.log(f"  grad_accum:  {args.grad_accum}")
    logger.log(f"  eff_batch:   {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:     {args.n_steps}")
    logger.log(f"  eval_every:  {args.eval_every}")
    logger.log(f"  lr:          {args.lr}")
    logger.log(f"  model:       {args.model}")
    logger.log(f"  h_dim:       {args.h_dim}")
    logger.log(f"  n_layers:    {args.n_layers}")
    logger.log(f"  schedule:    {args.schedule}")
    logger.log(f"  noise_type:  {args.noise_type}")
    logger.log(f"  noise_scale: {args.noise_scale}")
    logger.log(f"  T:           {args.T}")
    if args.curriculum:
        logger.log(f"  curriculum:  enabled (warmup={args.curriculum_warmup}, schedule={args.curriculum_schedule})")
    logger.log("")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.log("")

    # Load data with deterministic train/test split
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    table = pq.read_table(data_path)

    # Deterministic split (same n_train always gives same samples)
    split_config = DataSplitConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        seed=42,
    )
    train_indices, test_indices = get_train_test_indices(table, split_config)
    split_info = get_split_info(table, split_config)

    logger.log(f"Data split (seed={split_config.seed}):")
    logger.log(f"  Eligible samples ({args.min_atoms}-{args.max_atoms} atoms): {split_info['eligible_samples']}")
    logger.log(f"  Training: {len(train_indices)} samples")
    logger.log(f"  Test: {len(test_indices)} samples (held out, never seen during training)")
    logger.log(f"  Train IDs: {split_info['train_ids'][:3]}...")
    logger.log(f"  Test IDs:  {split_info['test_ids'][:3]}...")
    logger.log("")

    # Preload train and test samples SEPARATELY
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx) for idx in test_indices}
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples")

    # Create model
    if args.model == "af3_style":
        # AF3-style uses different kwargs
        model = create_model(
            args.model,
            c_token=args.h_dim * 2,  # 256 for h_dim=128
            c_atom=args.h_dim,
            trunk_layers=args.n_layers + 3,  # 9 for n_layers=6
            denoiser_blocks=args.n_layers + 1,  # 7 for n_layers=6
            n_timesteps=args.T,
            dropout=0.0,
        ).to(device)
    else:
        model = create_model(
            args.model,
            h_dim=args.h_dim,
            n_heads=8,
            n_layers=args.n_layers,
            n_timesteps=args.T,
            dropout=0.0,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model: {args.model}")
    logger.log(f"  Parameters: {n_params:,}")
    logger.log("")

    # Create diffusion components
    schedule = create_schedule(args.schedule, T=args.T)
    if args.noise_type == "linear_chain":
        noiser = create_noiser(args.noise_type, schedule, noise_scale=args.noise_scale)
    else:
        noiser = create_noiser(args.noise_type, schedule)
    noiser = noiser.to(device)

    logger.log(f"Diffusion:")
    logger.log(f"  Schedule: {args.schedule}")
    logger.log(f"  Noise type: {args.noise_type}")
    if args.noise_type == "linear_chain":
        logger.log(f"  Noise scale: {args.noise_scale}")

    # Create curriculum if enabled
    curriculum = None
    if args.curriculum:
        curriculum = TimestepCurriculum(noiser.T, args.curriculum_warmup, args.curriculum_schedule)
        logger.log(f"  Curriculum: warmup={args.curriculum_warmup}, schedule={args.curriculum_schedule}")
    logger.log("")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=1e-5)

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    start_time = time.time()

    model.train()
    for step in range(1, args.n_steps + 1):
        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(args.grad_accum):
            # TRAINING ONLY uses train_samples (no data leakage)
            batch_indices = random.choices(train_indices, k=args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            if curriculum:
                t = curriculum.sample(args.batch_size, step, device)
            else:
                t = torch.randint(0, noiser.T, (args.batch_size,), device=device)

            # Add noise (unified API)
            # For linear_flow: target is velocity, for others: target is x_linear (unused)
            x_t, target = noiser.add_noise(
                batch['coords'], t,
                atom_to_res=batch['atom_to_res'],
                atom_type=batch['atom_types'],
                chain_ids=batch['chain_ids'],
            )

            pred = model(x_t, batch['atom_types'], batch['atom_to_res'],
                         batch['aa_seq'], batch['chain_ids'], t, batch['mask'])

            # Different loss for linear_flow (velocity prediction) vs others (x0 prediction)
            if args.noise_type == "linear_flow":
                # Direct MSE on velocity (no Kabsch - velocity is in same frame)
                sq_diff = ((pred - target) ** 2).sum(dim=-1)
                loss = (sq_diff * batch['mask'].float()).sum() / batch['mask'].float().sum().clamp(min=1)
            else:
                loss = compute_loss(pred, batch['coords'], batch['mask'])

            # Scale loss for accumulation
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss = accum_loss  # For logging

        if step % 100 == 0:
            elapsed = time.time() - start_time
            if curriculum:
                t_max = curriculum.get_max_t(step)
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | t_max: {t_max:2d} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            else:
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on TRAIN set (random subset for speed)
                n_eval = args.n_eval_train if args.n_eval_train > 0 else len(train_indices)
                n_eval = min(n_eval, len(train_indices))
                eval_train_indices = random.sample(train_indices, n_eval)
                train_rmses = []
                for idx in eval_train_indices:
                    s = train_samples[idx]
                    batch = collate_batch([s], device)
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                    rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                    train_rmses.append(rmse)
                train_avg = sum(train_rmses) / len(train_rmses)

                # Evaluate on TEST set (full set - never seen during training)
                test_rmses = []
                for idx in test_indices:
                    s = test_samples[idx]
                    batch = collate_batch([s], device)
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                    rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                    test_rmses.append(rmse)
                test_avg = sum(test_rmses) / len(test_rmses)

                logger.log(f"         >>> Train RMSE ({n_eval}): {train_avg:.4f} A | Test RMSE ({len(test_indices)}): {test_avg:.4f} A")

                # Plot first train sample
                s = train_samples[train_indices[0]]
                batch = collate_batch([s], device)
                x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                     batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])

                n = s['n_atoms']
                pred = x_pred[0, :n] * s['std']
                target = batch['coords'][0, :n] * s['std']
                pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                rmse_viz = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item()  # Already in Angstroms

                plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                plot_prediction(pred_aligned[0], target_c[0], s['chain_ids'],
                               s['sample_id'], rmse_viz, plot_path)
                logger.log(f"         >>> Saved plot: {plot_path}")

                # Save best model based on TEST RMSE (generalization)
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

    # Final eval with best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.log("")
    logger.log("=" * 70)
    logger.log("Final evaluation (3 samples each)")
    logger.log("=" * 70)
    model.eval()
    with torch.no_grad():
        # Evaluate TRAIN set
        logger.log(f"\nTRAIN SET ({len(train_indices)} samples):")
        train_final_rmses = []
        for idx in train_indices:
            s = train_samples[idx]
            batch = collate_batch([s], device)
            rmses = []
            for _ in range(3):
                x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                     batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                rmses.append(rmse)
            mean_rmse = sum(rmses) / len(rmses)
            train_final_rmses.append(mean_rmse)
            logger.log(f"  {s['sample_id']}: {mean_rmse:.2f} A")
        train_overall = sum(train_final_rmses) / len(train_final_rmses)
        logger.log(f"  --- Train mean: {train_overall:.2f} A")

        # Evaluate TEST set (never seen during training)
        logger.log(f"\nTEST SET ({len(test_indices)} samples) - NEVER SEEN DURING TRAINING:")
        test_final_rmses = []
        for idx in test_indices:
            s = test_samples[idx]
            batch = collate_batch([s], device)
            rmses = []
            for _ in range(3):
                x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                     batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                rmses.append(rmse)
            mean_rmse = sum(rmses) / len(rmses)
            test_final_rmses.append(mean_rmse)
            logger.log(f"  {s['sample_id']}: {mean_rmse:.2f} A")
        test_overall = sum(test_final_rmses) / len(test_final_rmses)
        logger.log(f"  --- Test mean: {test_overall:.2f} A")

        logger.log("")
        logger.log("=" * 70)
        logger.log(f"FINAL RESULTS:")
        logger.log(f"  Train RMSE: {train_overall:.2f} A")
        logger.log(f"  Test RMSE:  {test_overall:.2f} A")
        logger.log(f"  Gap:        {test_overall - train_overall:.2f} A")
        logger.log("=" * 70)

    logger.log("")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()
