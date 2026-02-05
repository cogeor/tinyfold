#!/usr/bin/env python
"""
Stage 1 Training: Diffusion on residue centroids (x0 prediction).

Simple, clean training script - no rollout, no Stage 2.

Usage:
    python scripts/train_stage1.py --output_dir outputs/stage1_full
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

from models import create_schedule, create_noiser
from models.resfold_pipeline import ResFoldPipeline
from models.geometry_losses import ContactLoss, compute_lddt, compute_ilddt
from data_split import DataSplitConfig, get_train_test_indices, save_split, DynamicBatchSampler


# =============================================================================
# Config - 7.5M Stage 1 parameters
# =============================================================================

MODEL_CONFIG = {
    'c_token_s1': 192,
    'trunk_layers': 5,
    'trunk_heads': 4,
    'denoiser_blocks': 10,
    'denoiser_heads': 4,
    'c_token_s2': 192,
    's2_layers': 6,
    's2_heads': 4,
    'n_timesteps': 50,
}


# =============================================================================
# Logging
# =============================================================================

class Logger:
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
# Loss Functions
# =============================================================================

def kabsch_align(pred, target, mask=None):
    """Kabsch alignment for rotation-invariant comparison."""
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


def compute_mse_loss(pred, target, mask=None):
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


def compute_distance_loss(pred, target, mask=None):
    """Distance consistency loss."""
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        loss = (dist_diff * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1)
    else:
        loss = dist_diff.mean()

    return loss


# =============================================================================
# Data Loading
# =============================================================================

def load_sample(table, i):
    """Load and normalize a single sample."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
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

    # Residue centroids (mean of 4 backbone atoms)
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'centroids': centroids,
        'coords_res': coords_res,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    """Collate samples into padded batch."""
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)

    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask = torch.zeros(B, max_res, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s['n_res']
        centroids[i, :n] = s['centroids']
        coords_res[i, :n] = s['coords_res']
        aa_seq[i, :n] = s['aa_seq']
        chain_ids[i, :n] = s['chain_ids']
        res_idx[i, :n] = s['res_idx']
        mask[i, :n] = True

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask': mask.to(device),
    }


# =============================================================================
# Training Step (x0 prediction only)
# =============================================================================

def train_step(model, batch, noiser, device, contact_loss_fn, dist_weight=0.1, contact_weight=0.1):
    """Single training step with x0 prediction."""
    model.train()

    B = batch['aa_seq'].shape[0]
    gt = batch['centroids']
    mask = batch['mask']

    # Sample timestep [1, T)
    t = torch.randint(1, noiser.T, (B,), device=device)

    # Add noise: x_t = sqrt(ab) * x0 + sqrt(1-ab) * noise
    noise = torch.randn_like(gt)
    sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
    sqrt_1_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
    x_t = sqrt_ab * gt + sqrt_1_ab * noise

    # Predict x0 directly
    x0_pred = model.forward_stage1(
        x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t, mask
    )

    # Losses
    mse = compute_mse_loss(x0_pred, gt, mask)
    dist = compute_distance_loss(x0_pred, gt, mask)

    contact_result = contact_loss_fn(
        pred_centroids=x0_pred, gt_centroids=gt,
        chain_ids=batch['chain_ids'], mask=mask
    )
    contact = contact_result['stage1']

    total = mse + dist_weight * dist + contact_weight * contact

    return {
        'total': total,
        'mse': mse.item(),
        'dist': dist.item(),
        'contact': contact.item(),
    }


# =============================================================================
# Sampling (DDPM with x0 prediction)
# =============================================================================

@torch.no_grad()
def sample_centroids(model, batch, noiser, device):
    """Sample centroids using DDPM with x0 prediction."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask']

    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(1, noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict x0
        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t_batch, mask
        )
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        if t > 1:
            ab = noiser.alpha_bar[t].item()
            ab_prev = noiser.alpha_bar[t - 1].item()
            beta = noiser.betas[t].item()
            alpha = noiser.alphas[t].item()

            coef1 = math.sqrt(ab_prev) * beta / (1 - ab)
            coef2 = math.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
            mean = coef1 * x0_pred + coef2 * x

            var = beta * (1 - ab_prev) / (1 - ab)
            x = mean + math.sqrt(var) * torch.randn_like(x)
        else:
            x = x0_pred

    return x


# =============================================================================
# Evaluation (batched for speed)
# =============================================================================

@torch.no_grad()
def evaluate(model, samples, indices, noiser, device, n_eval=100, batch_size=16):
    """Evaluate centroid RMSE on subset using batched sampling."""
    model.eval()

    eval_indices = random.sample(list(indices), min(n_eval, len(indices)))
    rmses = []
    lddts = []

    # Process in batches
    for i in range(0, len(eval_indices), batch_size):
        batch_indices = eval_indices[i:i+batch_size]
        batch_samples = [samples[idx] for idx in batch_indices]
        batch = collate_batch(batch_samples, device)

        # Batched sampling
        pred = sample_centroids(model, batch, noiser, device)

        # Compute metrics per sample
        for j, idx in enumerate(batch_indices):
            s = samples[idx]
            n = s['n_res']
            rmse = compute_rmse(pred[j:j+1, :n], batch['centroids'][j:j+1, :n]).item() * s['std']
            rmses.append(rmse)

            lddt = compute_lddt(pred[j:j+1, :n], batch['centroids'][j:j+1, :n], coord_scale=s['std'])
            lddts.append(lddt.item())

    return {
        'rmse': sum(rmses) / len(rmses),
        'lddt': sum(lddts) / len(lddts),
        'n_eval': len(eval_indices),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Single plot: GT vs prediction side by side."""
    fig = plt.figure(figsize=(12, 5))

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    # Ground truth
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    if mask_a.any():
        ax1.scatter(target[mask_a, 0], target[mask_a, 1], target[mask_a, 2], c='blue', s=20, alpha=0.7)
    if mask_b.any():
        ax1.scatter(target[mask_b, 0], target[mask_b, 1], target[mask_b, 2], c='red', s=20, alpha=0.7)
    ax1.set_title(f'{sample_id}\nGround Truth')

    # Prediction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2], c='cyan', s=20, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2], c='orange', s=20, alpha=0.7)
    ax2.set_title(f'Prediction\nRMSE: {rmse:.2f} A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/samples.parquet')
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--n_eval_train', type=int, default=100, help='Train samples for eval')
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--dist_weight', type=float, default=0.1)
    parser.add_argument('--contact_weight', type=float, default=0.1)
    parser.add_argument('--dynamic_batch', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=8000)
    parser.add_argument('--output_dir', default='outputs/stage1_full')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    # Header
    logger.log("=" * 70)
    logger.log("Stage 1 Training: Centroid Diffusion (x0 prediction)")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Device: {device}")
    if device.type == 'cuda':
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.log()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, args.data_path)
    logger.log(f"Loading data: {data_path}")
    table = pq.read_table(data_path)
    n_total = len(table)
    logger.log(f"  Total samples: {n_total}")

    # Split: use all eligible samples (filter by atom count)
    config = DataSplitConfig(n_train=0, n_test=args.n_test, min_atoms=160, max_atoms=2000, seed=42)
    # Count eligible samples
    n_atoms_list = [len(table['atom_type'][i].as_py()) for i in range(n_total)]
    n_eligible = sum(1 for n in n_atoms_list if config.min_atoms <= n <= config.max_atoms)
    n_train = n_eligible - args.n_test
    config = DataSplitConfig(n_train=n_train, n_test=args.n_test, min_atoms=160, max_atoms=2000, seed=42)
    train_indices, test_indices = get_train_test_indices(table, config)
    logger.log(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Save split
    split_path = os.path.join(args.output_dir, 'split.json')
    from data_split import get_split_info
    save_split(get_split_info(table, config), split_path)
    logger.log(f"  Saved split: {split_path}")

    # Preload samples
    logger.log("Preloading samples...")
    samples = {}
    for idx in train_indices + test_indices:
        samples[idx] = load_sample(table, idx)
    logger.log(f"  Loaded {len(samples)} samples")

    # Model (Stage 1 only trainable)
    logger.log()
    logger.log("Model config:")
    for k, v in MODEL_CONFIG.items():
        logger.log(f"  {k}: {v}")

    model = ResFoldPipeline(**MODEL_CONFIG).to(device)
    model.set_training_mode('stage1_only')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  Trainable params: {n_params:,}")

    # Checkpoint
    start_step = 0
    if args.checkpoint:
        logger.log(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        logger.log(f"  Resuming from step {start_step}")

    # Diffusion
    schedule = create_schedule('linear', T=args.T)
    schedule.to(device)
    noiser = create_noiser('gaussian', schedule)
    logger.log(f"Diffusion: T={args.T}, schedule=linear")

    # Contact loss
    contact_loss_fn = ContactLoss(threshold=1.0, min_seq_sep=5, inter_chain_weight=2.0, stage="stage1")
    logger.log(f"Contact loss: {contact_loss_fn}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=1e-5)

    # Dynamic batching
    if args.dynamic_batch:
        train_samples = {idx: samples[idx] for idx in train_indices}
        sampler = DynamicBatchSampler(train_samples, base_batch_size=args.batch_size, max_tokens=args.max_tokens)
        logger.log(f"Dynamic batching: max_tokens={args.max_tokens}")
        for info in sampler.get_batch_sizes():
            logger.log(f"  Bucket {info['bucket']}: max_res={info['max_res']}, bs={info['batch_size']}")
    else:
        sampler = None
        logger.log(f"Fixed batching: bs={args.batch_size}")

    logger.log(f"Gradient accumulation: {args.grad_accum}")
    logger.log(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.log()
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    # Training loop
    step = start_step
    best_rmse = float('inf')
    start_time = time.time()

    running_loss = 0.0
    running_mse = 0.0
    running_dist = 0.0
    running_con = 0.0
    running_cnt = 0

    while step < args.n_steps:
        accum_idx = (step - start_step) % args.grad_accum
        if accum_idx == 0:
            optimizer.zero_grad()

        # Get batch
        if args.dynamic_batch:
            batch_indices, _ = sampler.sample_batch()
        else:
            batch_indices = random.sample(train_indices, args.batch_size)

        batch = collate_batch([samples[idx] for idx in batch_indices], device)

        # Train step
        losses = train_step(model, batch, noiser, device, contact_loss_fn,
                           dist_weight=args.dist_weight, contact_weight=args.contact_weight)

        (losses['total'] / args.grad_accum).backward()

        if accum_idx == args.grad_accum - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        step += 1

        # Running averages
        running_loss += losses['total'].item()
        running_mse += losses['mse']
        running_dist += losses['dist']
        running_con += losses['contact']
        running_cnt += 1

        # Log
        if step % args.log_every == 0:
            avg_loss = running_loss / running_cnt
            avg_mse = running_mse / running_cnt
            avg_dist = running_dist / running_cnt
            avg_con = running_con / running_cnt
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time

            logger.log(
                f"Step {step:6d} | loss: {avg_loss:.4f} | "
                f"mse: {avg_mse:.3f} dst: {avg_dist:.3f} con: {avg_con:.3f} | "
                f"lr: {lr:.2e} | {elapsed:.0f}s"
            )

            running_loss = 0.0
            running_mse = 0.0
            running_dist = 0.0
            running_con = 0.0
            running_cnt = 0

        # Evaluate
        if step % args.eval_every == 0:
            train_results = evaluate(model, samples, train_indices, noiser, device, n_eval=args.n_eval_train)
            test_results = evaluate(model, samples, test_indices, noiser, device, n_eval=len(test_indices))

            logger.log(
                f"  >>> Train: RMSE={train_results['rmse']:.2f}A lDDT={train_results['lddt']:.3f} (n={train_results['n_eval']})"
            )
            logger.log(
                f"  >>> Test:  RMSE={test_results['rmse']:.2f}A lDDT={test_results['lddt']:.3f} (n={test_results['n_eval']})"
            )

            # Save best
            if test_results['rmse'] < best_rmse:
                best_rmse = test_results['rmse']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_rmse': test_results['rmse'],
                    'test_lddt': test_results['lddt'],
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.log(f"  >>> New best! RMSE: {best_rmse:.2f}A")

            # Checkpoint
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, 'checkpoint.pt'))

            # Single plot (random test sample)
            model.eval()
            plot_idx = random.choice(test_indices)
            s = samples[plot_idx]
            batch = collate_batch([s], device)
            pred = sample_centroids(model, batch, noiser, device)

            n = s['n_res']
            pred_aligned, gt_aligned = kabsch_align(pred[:, :n], batch['centroids'][:, :n])
            rmse = compute_rmse(pred[:, :n], batch['centroids'][:, :n]).item() * s['std']

            plot_prediction(
                pred_aligned[0] * s['std'], gt_aligned[0] * s['std'],
                batch['chain_ids'][0, :n], s['sample_id'], rmse,
                os.path.join(args.output_dir, 'plots', f'step{step}.png')
            )
            model.train()

    # Final
    logger.log()
    logger.log("=" * 70)
    logger.log("Training complete!")
    logger.log(f"Best test RMSE: {best_rmse:.2f}A")
    logger.log("=" * 70)
    logger.close()


if __name__ == '__main__':
    main()
