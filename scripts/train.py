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
)


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
def ddpm_sample(model, atom_types, atom_to_res, aa_seq, chain_ids, noiser, mask=None, clamp_val=3.0):
    """DDPM sampling loop."""
    device = atom_types.device
    B, N = atom_types.shape
    x = torch.randn(B, N, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
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
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model
    parser.add_argument("--model", type=str, default="attention_v2", choices=list_models())
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)

    # Diffusion
    parser.add_argument("--schedule", type=str, default="cosine", help="Alpha-bar schedule")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=list_noise_types(),
                        help="Noise type: gaussian or linear_chain")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Gaussian noise scale for linear_chain")
    parser.add_argument("--T", type=int, default=50, help="Diffusion timesteps")

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
    logger.log(f"  n_samples:   {args.n_samples}")
    logger.log(f"  batch_size:  {args.batch_size}")
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

    medium_indices = find_medium_samples(table, 200, 400)
    logger.log(f"Found {len(medium_indices)} samples with 200-400 atoms")

    train_indices = medium_indices[:args.n_samples]
    logger.log(f"Training on {len(train_indices)} samples")
    logger.log("")

    # Preload
    logger.log("Preloading samples...")
    all_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}

    # Create model
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
        batch_indices = random.choices(train_indices, k=args.batch_size)
        batch_samples = [all_samples[idx] for idx in batch_indices]
        batch = collate_batch(batch_samples, device)

        t = torch.randint(0, noiser.T, (args.batch_size,), device=device)

        # Add noise (unified API)
        x_t, _ = noiser.add_noise(
            batch['coords'], t,
            atom_to_res=batch['atom_to_res'],
            atom_type=batch['atom_types'],
            chain_ids=batch['chain_ids'],
        )

        x0_pred = model(x_t, batch['atom_types'], batch['atom_to_res'],
                        batch['aa_seq'], batch['chain_ids'], t, batch['mask'])
        loss = compute_loss(x0_pred, batch['coords'], batch['mask'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            logger.log(f"Step {step:5d} | loss: {loss.item():.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on subset
                eval_indices = train_indices[:min(20, len(train_indices))]
                train_rmses = []
                for idx in eval_indices:
                    s = all_samples[idx]
                    batch = collate_batch([s], device)
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                    rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                    train_rmses.append(rmse)

                train_avg = sum(train_rmses) / len(train_rmses)
                logger.log(f"         >>> Eval RMSE (first 20): {train_avg:.4f} A")

                # Plot first sample
                s = all_samples[train_indices[0]]
                batch = collate_batch([s], device)
                x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                     batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])

                n = s['n_atoms']
                pred = x_pred[0, :n] * s['std']
                target = batch['coords'][0, :n] * s['std']
                pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                rmse_viz = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item() * s['std']

                plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                plot_prediction(pred_aligned[0], target_c[0], s['chain_ids'],
                               s['sample_id'], rmse_viz, plot_path)
                logger.log(f"         >>> Saved plot: {plot_path}")

                if train_avg < best_rmse:
                    best_rmse = train_avg
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'rmse': train_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    logger.log(f"         >>> New best! Saved.")

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log(f"Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best RMSE: {best_rmse:.4f} A")

    # Final eval
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.log("")
    logger.log("Final evaluation (3 samples each, first 30):")
    model.eval()
    with torch.no_grad():
        all_rmses = []
        for idx in train_indices[:30]:
            s = all_samples[idx]
            batch = collate_batch([s], device)
            rmses = []
            for _ in range(3):
                x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                     batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])
                rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                rmses.append(rmse)
            mean_rmse = sum(rmses) / len(rmses)
            all_rmses.append(mean_rmse)
            logger.log(f"  {s['sample_id']}: {mean_rmse:.2f} A")

        overall = sum(all_rmses) / len(all_rmses)
        logger.log(f"\nOverall mean: {overall:.2f} A")

    logger.log("")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()
