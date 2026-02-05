#!/usr/bin/env python
"""
Clean Stage 1 Training Script - x0 Prediction with DDPM

Key design choices:
1. x0 prediction: Network predicts clean coordinates directly
2. DDPM noise: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
3. Rigid-aligned loss: Kabsch align prediction to GT before MSE
4. Distance consistency: Additional loss on pairwise distances

Usage:
    python scripts/train_stage1_clean.py --n_train 10000 --n_steps 50000 --output_dir outputs/stage1_10k
"""

import os
import sys
import random
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resfold import ResidueDenoiser
from models import create_schedule, create_noiser


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

def load_sample(table, i):
    """Load and normalize a single sample."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    seq = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(coords) // 3
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Normalize: center and scale to unit variance
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    # Compute residue centroids
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'centroids': centroids,
        'aa_seq': seq,
        'chain_ids': chain,
        'res_idx': torch.arange(n_res),
        'n_res': n_res,
        'std': std.item(),
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    """Collate into padded batch."""
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)

    centroids = torch.zeros(B, max_res, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask = torch.zeros(B, max_res, dtype=torch.bool)

    for i, s in enumerate(samples):
        L = s['n_res']
        centroids[i, :L] = s['centroids']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask[i, :L] = True

    return {
        'centroids': centroids.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask': mask.to(device),
    }


# =============================================================================
# Dynamic Batching (binning by size)
# =============================================================================

class DynamicBatchSampler:
    """Sample batches with similar-sized proteins to minimize padding."""

    def __init__(self, samples: dict, max_tokens: int = 8000, n_buckets: int = 8):
        self.samples = samples
        self.max_tokens = max_tokens

        # Sort samples by size
        sizes = [(idx, s['n_res']) for idx, s in samples.items()]
        sizes.sort(key=lambda x: x[1])

        # Create buckets
        n_per_bucket = len(sizes) // n_buckets
        self.buckets = []
        for i in range(n_buckets):
            start = i * n_per_bucket
            end = (i + 1) * n_per_bucket if i < n_buckets - 1 else len(sizes)
            bucket_samples = [idx for idx, _ in sizes[start:end]]
            if bucket_samples:
                max_res = max(samples[idx]['n_res'] for idx in bucket_samples)
                batch_size = max(1, max_tokens // max_res)
                self.buckets.append({
                    'indices': bucket_samples,
                    'max_res': max_res,
                    'batch_size': batch_size,
                })

    def sample_batch(self):
        """Sample a batch from a random bucket."""
        bucket = random.choice(self.buckets)
        indices = random.choices(bucket['indices'], k=bucket['batch_size'])
        return indices, bucket['batch_size']

    def get_info(self):
        """Get bucket info for logging."""
        return [{'bucket': i, 'max_res': b['max_res'], 'batch_size': b['batch_size'],
                 'tokens': b['max_res'] * b['batch_size']}
                for i, b in enumerate(self.buckets)]


# =============================================================================
# Loss Functions
# =============================================================================

def kabsch_align(pred, target, mask=None):
    """Kabsch alignment: find optimal rotation to align pred to target."""
    B = pred.shape[0]

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


def mse_loss(pred, target, mask=None):
    """MSE loss after Kabsch alignment."""
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

    if mask is not None:
        loss = (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        loss = sq_diff.mean()

    return loss


def dist_loss(pred, target, mask=None):
    """Distance consistency loss (preserves pairwise distances)."""
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        loss = (dist_diff * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1)
    else:
        loss = dist_diff.mean()

    return loss


def rmse(pred, target, mask=None):
    """RMSE after Kabsch alignment."""
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

    if mask is not None:
        return torch.sqrt((sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1))
    else:
        return torch.sqrt(sq_diff.mean())


# =============================================================================
# Training
# =============================================================================

def train_step(model, batch, noiser, device, dist_weight=0.1):
    """Single training step with x0 prediction."""
    B = batch['centroids'].shape[0]
    gt = batch['centroids']
    mask = batch['mask']

    # Sample timestep
    t = torch.randint(0, noiser.T, (B,), device=device)

    # Add noise (DDPM forward process)
    noise = torch.randn_like(gt)
    sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
    sqrt_1_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
    x_t = sqrt_ab * gt + sqrt_1_ab * noise

    # Model predicts clean x0
    x_pred = model(x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t, mask)

    # Loss
    loss_mse = mse_loss(x_pred, gt, mask)
    loss_dist = dist_loss(x_pred, gt, mask)
    total = loss_mse + dist_weight * loss_dist

    return {
        'total': total,
        'mse': loss_mse.item(),
        'dist': loss_dist.item(),
    }


@torch.no_grad()
def sample(model, batch, noiser, device):
    """DDPM sampling with x0 prediction."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask']

    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        x0_pred = model(x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

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


@torch.no_grad()
def evaluate(model, samples, noiser, device, n_eval=None):
    """Evaluate on samples, return mean RMSE."""
    model.eval()
    rmses = []

    eval_samples = samples[:n_eval] if n_eval else samples

    for s in eval_samples:
        batch = collate_batch([s], device)
        pred = sample(model, batch, noiser, device)
        r = rmse(pred, batch['centroids'], batch['mask']).item() * s['std']
        rmses.append(r)

    model.train()
    return sum(rmses) / len(rmses)


# =============================================================================
# Plotting
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse_val, output_path):
    """Plot predicted vs target structure."""
    fig = plt.figure(figsize=(14, 6))

    # Target
    ax1 = fig.add_subplot(121, projection='3d')
    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    if mask_a.any():
        ax1.scatter(target[mask_a, 0], target[mask_a, 1], target[mask_a, 2],
                   c='blue', s=10, alpha=0.7, label='Chain A')
    if mask_b.any():
        ax1.scatter(target[mask_b, 0], target[mask_b, 1], target[mask_b, 2],
                   c='red', s=10, alpha=0.7, label='Chain B')
    ax1.set_title(f'Ground Truth: {sample_id}')
    ax1.legend()

    # Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2],
                   c='cyan', s=10, alpha=0.7, label='Chain A')
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2],
                   c='orange', s=10, alpha=0.7, label='Chain B')
    ax2.set_title(f'Prediction (RMSE: {rmse_val:.2f}Å)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_loss_curves(steps, losses, output_path):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, [l['total'] for l in losses])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, [l['mse'] for l in losses])
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('MSE Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, [l['dist'] for l in losses])
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Distance Loss')
    axes[2].set_title('Distance Consistency Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='data/processed/samples.parquet')
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--min_atoms', type=int, default=160)
    parser.add_argument('--max_atoms', type=int, default=1816)

    # Model
    parser.add_argument('--c_token', type=int, default=256)
    parser.add_argument('--trunk_layers', type=int, default=6)
    parser.add_argument('--denoiser_blocks', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--T', type=int, default=50)

    # Training
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dist_weight', type=float, default=0.1)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--n_eval', type=int, default=20)

    # Dynamic batching
    parser.add_argument('--dynamic_batch', action='store_true', default=True)
    parser.add_argument('--max_tokens', type=int, default=8000)

    # Output
    parser.add_argument('--output_dir', default='outputs/stage1_10k')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--load_split', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Header
    logger.log("=" * 70)
    logger.log("Stage 1 Training - x0 Prediction (DDPM)")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log()
    logger.log(f"Device: {device}")
    if device.type == 'cuda':
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.log()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, args.data_path)

    logger.log(f"Loading data from: {data_path}")
    table = pq.read_table(data_path)
    logger.log(f"  Total samples: {len(table)}")

    # Data split
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        with open(args.load_split, 'r') as f:
            split = json.load(f)
        train_indices = split['train_indices']
        test_indices = split['test_indices']
    else:
        # Filter by atom count
        eligible = []
        for i in range(len(table)):
            n_atoms = len(table['atom_coords'][i].as_py()) // 3
            if args.min_atoms <= n_atoms <= args.max_atoms:
                eligible.append(i)

        logger.log(f"  Eligible samples ({args.min_atoms}-{args.max_atoms} atoms): {len(eligible)}")

        random.seed(42)
        random.shuffle(eligible)
        train_indices = eligible[:args.n_train]
        test_indices = eligible[args.n_train:args.n_train + args.n_test]

        # Save split
        split_path = os.path.join(args.output_dir, 'split.json')
        with open(split_path, 'w') as f:
            json.dump({'train_indices': train_indices, 'test_indices': test_indices}, f)
        logger.log(f"  Saved split to: {split_path}")

    logger.log(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Load samples
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample(table, idx) for idx in train_indices}
    test_samples = [load_sample(table, idx) for idx in test_indices]
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test")
    logger.log()

    # Model
    logger.log("Model configuration:")
    logger.log(f"  c_token: {args.c_token}")
    logger.log(f"  trunk_layers: {args.trunk_layers}")
    logger.log(f"  denoiser_blocks: {args.denoiser_blocks}")
    logger.log(f"  n_heads: {args.n_heads}")
    logger.log(f"  T: {args.T}")

    model = ResidueDenoiser(
        c_token=args.c_token,
        trunk_layers=args.trunk_layers,
        trunk_heads=args.n_heads,
        denoiser_blocks=args.denoiser_blocks,
        denoiser_heads=args.n_heads,
        n_timesteps=args.T,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log()
    logger.log(f"Model: ResidueDenoiser")
    logger.log(f"  Parameters: {n_params:,}")
    logger.log()

    # Load checkpoint
    start_step = 0
    if args.checkpoint:
        logger.log(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        logger.log(f"  Resumed from step {start_step}")
        logger.log()

    # Diffusion
    schedule = create_schedule('linear', T=args.T)
    schedule.to(device)
    noiser = create_noiser('gaussian', schedule)

    logger.log(f"Diffusion: T={args.T}, schedule=linear")
    logger.log()

    # Dynamic batching
    if args.dynamic_batch:
        sampler = DynamicBatchSampler(train_samples, max_tokens=args.max_tokens)
        logger.log(f"Dynamic batching: max_tokens={args.max_tokens}")
        for info in sampler.get_info():
            logger.log(f"  Bucket {info['bucket']}: max_res={info['max_res']}, "
                      f"batch_size={info['batch_size']}, tokens={info['tokens']}")
    else:
        sampler = None
        logger.log(f"Fixed batching: batch_size={args.batch_size}")

    logger.log()
    logger.log(f"Gradient accumulation: {args.grad_accum}")
    logger.log(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.log()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=1e-5
    )

    # Advance scheduler if resuming
    for _ in range(start_step):
        scheduler.step()

    logger.log(f"Training for {args.n_steps} steps...")
    logger.log(f"  lr: {args.lr}")
    logger.log(f"  dist_weight: {args.dist_weight}")
    logger.log(f"  eval_every: {args.eval_every}")
    logger.log("=" * 70)

    # Training
    start_time = time.time()
    running_loss = 0.0
    running_mse = 0.0
    running_dist = 0.0
    running_cnt = 0
    best_rmse = float('inf')

    loss_history = []
    loss_steps = []

    step = start_step

    while step < args.n_steps:
        # Gradient accumulation cycle
        accum_idx = (step - start_step) % args.grad_accum
        if accum_idx == 0:
            optimizer.zero_grad()

        # Get batch
        if args.dynamic_batch:
            batch_indices, _ = sampler.sample_batch()
            batch_samples = [train_samples[idx] for idx in batch_indices]
        else:
            batch_samples = random.choices(list(train_samples.values()), k=args.batch_size)

        batch = collate_batch(batch_samples, device)

        # Forward pass
        losses = train_step(model, batch, noiser, device, args.dist_weight)
        (losses['total'] / args.grad_accum).backward()

        # Step optimizer at end of accumulation cycle
        if accum_idx == args.grad_accum - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        step += 1

        # Track losses
        running_loss += losses['total'].item()
        running_mse += losses['mse']
        running_dist += losses['dist']
        running_cnt += 1

        # Log every 100 steps
        if step % 100 == 0:
            avg_loss = running_loss / running_cnt
            avg_mse = running_mse / running_cnt
            avg_dist = running_dist / running_cnt
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time

            logger.log(f"Step {step:5d} | loss: {avg_loss:.4f} | "
                      f"mse: {avg_mse:.4f} | dist: {avg_dist:.4f} | "
                      f"lr: {lr:.2e} | {elapsed:.0f}s")

            loss_history.append({'total': avg_loss, 'mse': avg_mse, 'dist': avg_dist})
            loss_steps.append(step)

            running_loss = 0.0
            running_mse = 0.0
            running_dist = 0.0
            running_cnt = 0

        # Evaluate
        if step % args.eval_every == 0:
            train_rmse = evaluate(model, list(train_samples.values()), noiser, device, n_eval=args.n_eval)
            test_rmse = evaluate(model, test_samples, noiser, device, n_eval=args.n_eval)

            logger.log(f"         >>> Train RMSE: {train_rmse:.2f}A | Test RMSE: {test_rmse:.2f}A")

            # Save best model
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'args': vars(args),
                }, os.path.join(args.output_dir, 'best.pt'))
                logger.log(f"         >>> New best model saved! RMSE: {test_rmse:.2f}A")

            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'checkpoint.pt'))

            # Plot loss curves
            if loss_history:
                plot_loss_curves(loss_steps, loss_history,
                               os.path.join(args.output_dir, 'plots', 'loss_curves.png'))

            # Plot sample predictions
            for i, s in enumerate(test_samples[:3]):
                batch = collate_batch([s], device)
                pred = sample(model, batch, noiser, device)
                pred_np = pred[0, :s['n_res']].cpu().numpy() * s['std']
                target_np = batch['centroids'][0, :s['n_res']].cpu().numpy() * s['std']
                chain_np = batch['chain_ids'][0, :s['n_res']].cpu().numpy()

                # Align for visualization
                pred_aligned, target_c = kabsch_align(
                    pred[:, :s['n_res']] * s['std'],
                    batch['centroids'][:, :s['n_res']] * s['std'],
                    batch['mask'][:, :s['n_res']]
                )
                r = rmse(pred[:, :s['n_res']], batch['centroids'][:, :s['n_res']],
                        batch['mask'][:, :s['n_res']]).item() * s['std']

                plot_prediction(
                    pred_aligned[0].cpu().numpy(),
                    target_c[0].cpu().numpy(),
                    chain_np,
                    s['sample_id'],
                    r,
                    os.path.join(args.output_dir, 'plots', f'pred_step{step}_sample{i}.png')
                )

    # Final
    logger.log("=" * 70)
    logger.log("Training complete!")
    logger.log(f"  Total time: {time.time() - start_time:.0f}s ({(time.time() - start_time)/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.2f}A")
    logger.log("=" * 70)

    logger.close()


if __name__ == '__main__':
    main()
