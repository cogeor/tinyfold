#!/usr/bin/env python
"""
Training script for coiled-coil experiment.

Compares Gaussian vs LinearChain noise for coiled-coil structures.

Usage:
    # Gaussian noise (baseline)
    python train_coil_experiment.py --noise_type gaussian --output_dir outputs/coil_experiment/gaussian

    # Linear chain noise
    python train_coil_experiment.py --noise_type linear_chain --output_dir outputs/coil_experiment/linear_chain
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_schedule, create_noiser
from models.resfold_pipeline import ResFoldPipeline
from tinyfold.model.losses import kabsch_align, compute_mse_loss, compute_rmse


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


def load_sample_raw(table, i, normalize=True):
    """Load sample from parquet table."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    original_std = coords.std()
    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

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
    """Collate samples into padded batch."""
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)

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
        atom_types[i, :N] = s['atom_types']
        atom_to_res[i, :N] = s['atom_to_res']

        stds.append(s['std'])

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
    }


@torch.no_grad()
def sample_centroids(model, batch, noiser, device, noise_type='gaussian'):
    """DDPM sampling for centroids."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    if noise_type == 'linear_chain':
        # Start from extended chain
        from models.diffusion import generate_extended_chain
        x_linear = torch.zeros(B, L, 3, device=device)
        for b in range(B):
            # Create per-residue atom info (4 atoms per residue, use CA as representative)
            n_res = batch['n_res'][b]
            # For centroids, treat each residue as one "atom"
            atom_to_res = torch.arange(n_res, device=device)
            atom_type = torch.ones(n_res, dtype=torch.long, device=device)  # All CA
            chain_ids_b = batch['chain_ids'][b, :n_res]

            x_linear[b, :n_res] = generate_extended_chain(
                n_res, atom_to_res, atom_type, chain_ids_b, device
            )
        # Normalize
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_std = x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x_linear = x_linear / x_std
        # Start from extended chain - model was trained in this frame
        x = x_linear.clone()
    else:
        x = torch.randn(B, L, 3, device=device)
        x_linear = None

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t_batch, mask
        )
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        if t > 0:
            if noise_type == 'linear_chain' and x_linear is not None:
                # No alignment needed - both x0_pred and x_linear are in same frame
                # Model was trained to predict in x_linear's frame
                sqrt_ab_prev = noiser.schedule.sqrt_alpha_bar[t - 1]
                sqrt_one_minus_ab_prev = noiser.schedule.sqrt_one_minus_alpha_bar[t - 1]
                x = sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev * x_linear
            else:
                # Standard DDPM
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

        # Recenter
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
        else:
            centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid

    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Coiled-coil experiment")

    parser.add_argument("--noise_type", type=str, default="gaussian",
                        choices=["gaussian", "linear_chain"])
    parser.add_argument("--output_dir", type=str, required=True)

    # Model (~1M params)
    parser.add_argument("--c_token_s1", type=int, default=128)
    parser.add_argument("--trunk_layers", type=int, default=4)
    parser.add_argument("--denoiser_blocks", type=int, default=4)

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=50)

    # Data
    parser.add_argument("--coil_samples", type=str,
                        default="outputs/coil_experiment/coil_samples.json")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    logger.log("=" * 70)
    logger.log("Coiled-Coil Experiment")
    logger.log("=" * 70)
    logger.log(f"Noise type: {args.noise_type}")
    logger.log(f"Output: {args.output_dir}")
    logger.log(f"Model: c_token={args.c_token_s1}, trunk={args.trunk_layers}, denoiser={args.denoiser_blocks}")
    logger.log("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # Load coil samples
    with open(args.coil_samples) as f:
        coil_data = json.load(f)

    train_indices = coil_data['train_indices']
    test_indices = coil_data['test_indices']
    logger.log(f"Train samples: {len(train_indices)}")
    logger.log(f"Test samples: {len(test_indices)}")

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    train_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx) for idx in test_indices}
    logger.log(f"Loaded {len(train_samples)} train, {len(test_samples)} test")

    # Create model
    model = ResFoldPipeline(
        c_token_s1=args.c_token_s1,
        trunk_layers=args.trunk_layers,
        denoiser_blocks=args.denoiser_blocks,
        n_timesteps=args.T,
        stage1_only=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # Create noiser
    schedule = create_schedule("linear", T=args.T)
    noiser = create_noiser(args.noise_type, schedule)
    noiser = noiser.to(device)
    logger.log(f"Noiser: {args.noise_type}")
    logger.log("")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=1e-5)

    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    start_time = time.time()

    model.train()
    for step in range(1, args.n_steps + 1):
        # Sample batch
        batch_indices = random.choices(train_indices, k=args.batch_size)
        batch_samples = [train_samples[idx] for idx in batch_indices]
        batch = collate_batch(batch_samples, device)

        # Sample timesteps and add noise
        B = len(batch_samples)
        t = torch.randint(0, noiser.T, (B,), device=device)

        if args.noise_type == 'linear_chain':
            # Generate extended chain for each sample
            from models.diffusion import generate_extended_chain
            x_linear = torch.zeros_like(batch['centroids'])
            for b in range(B):
                n_res = batch['n_res'][b]
                atom_to_res = torch.arange(n_res, device=device)
                atom_type = torch.ones(n_res, dtype=torch.long, device=device)
                chain_ids_b = batch['chain_ids'][b, :n_res]

                x_linear[b, :n_res] = generate_extended_chain(
                    n_res, atom_to_res, atom_type, chain_ids_b, device
                )

            # Normalize
            x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
            x_linear_std = x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            x_linear = x_linear / x_linear_std

            # NO alignment during training - match inference distribution
            # Model must learn to handle extended chain in arbitrary orientation
            # Align GT to extended chain frame instead (so interpolation makes sense)
            gt_aligned, _ = kabsch_align(batch['centroids'], x_linear, batch['mask_res'])

            # Interpolate: both in extended chain's frame
            sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            x_t = sqrt_ab * gt_aligned + sqrt_one_minus_ab * x_linear
        else:
            # Standard Gaussian noise
            noise = torch.randn_like(batch['centroids'])
            sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            x_t = sqrt_ab * batch['centroids'] + sqrt_one_minus_ab * noise

        # Forward
        centroids_pred = model.forward_stage1(
            x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t, batch['mask_res']
        )

        # Loss - use appropriate target based on noise type
        if args.noise_type == 'linear_chain':
            # Target is GT aligned to extended chain's frame
            loss = compute_mse_loss(centroids_pred, gt_aligned, batch['mask_res'])
        else:
            loss = compute_mse_loss(centroids_pred, batch['centroids'], batch['mask_res'])

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
                # Evaluate on train
                train_rmses = []
                for idx in train_indices[:20]:
                    s = train_samples[idx]
                    batch = collate_batch([s], device)
                    pred = sample_centroids(model, batch, noiser, device, args.noise_type)
                    rmse = compute_rmse(pred, batch['centroids'], batch['mask_res']).item() * s['std']
                    train_rmses.append(rmse)
                train_avg = np.mean(train_rmses)

                # Evaluate on test
                test_rmses = []
                for idx in test_indices:
                    s = test_samples[idx]
                    batch = collate_batch([s], device)
                    pred = sample_centroids(model, batch, noiser, device, args.noise_type)
                    rmse = compute_rmse(pred, batch['centroids'], batch['mask_res']).item() * s['std']
                    test_rmses.append(rmse)
                test_avg = np.mean(test_rmses)

                logger.log(f"         >>> Train RMSE (20): {train_avg:.2f} A | Test RMSE ({len(test_indices)}): {test_avg:.2f} A")

                if test_avg < best_rmse:
                    best_rmse = test_avg
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    logger.log(f"         >>> New best! Saved.")

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log(f"Training complete")
    logger.log(f"  Total time: {total_time:.0f}s")
    logger.log(f"  Best test RMSE: {best_rmse:.2f} A")
    logger.close()


if __name__ == "__main__":
    main()
