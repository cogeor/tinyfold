#!/usr/bin/env python
"""
ResFold End-to-End Training Script (Simplified).

Smaller model (15M params total: 10M Stage1, 5M Stage2) trained end-to-end.

Losses:
- Stage 1: centroid MSE + contact loss
- Stage 2: atom MSE + geometry losses
- Balanced so both stages contribute equally

Usage:
    python scripts/train_resfold_e2e.py --n_train 80 --n_steps 10000
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
from models.geometry_losses import GeometryLoss, ContactLoss, compute_lddt_metrics
from models.dockq_utils import compute_dockq
from data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info, save_split, load_split,
    DynamicBatchSampler
)


# =============================================================================
# Config
# =============================================================================

# 10M model config: 7.5M Stage1 (75%) + 2.5M Stage2 (25%)
# 4 heads instead of 8, c_token must match between stages
MODEL_CONFIG = {
    'c_token_s1': 192,
    'trunk_layers': 5,
    'trunk_heads': 4,
    'denoiser_blocks': 10,
    'denoiser_heads': 4,
    'c_token_s2': 192,  # Must match c_token_s1
    's2_layers': 6,
    's2_heads': 4,
    'n_timesteps': 50,
}


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


def compute_distance_consistency_loss(pred_centroids, target_centroids, mask=None):
    """Loss for preserving inter-residue distances.

    Encourages predicted centroids to have similar pairwise distances
    as the ground truth centroids.
    """
    # Compute pairwise distances [B, L, L]
    pred_dist = torch.cdist(pred_centroids, pred_centroids)
    target_dist = torch.cdist(target_centroids, target_centroids)

    # MSE on distances
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        # Create pairwise mask [B, L, L]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        loss = (dist_diff * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1)
    else:
        loss = dist_diff.mean()

    return loss


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
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors
    coords = torch.zeros(B, max_atoms, 3)
    mask_atoms = torch.zeros(B, max_atoms, dtype=torch.bool)

    for i, s in enumerate(samples):
        n_res = s['n_res']
        n_atoms = s['n_atoms']

        centroids[i, :n_res] = s['centroids']
        coords_res[i, :n_res] = s['coords_res']
        aa_seq[i, :n_res] = s['aa_seq']
        chain_ids[i, :n_res] = s['chain_ids']
        res_idx[i, :n_res] = s['res_idx']
        mask_res[i, :n_res] = True

        coords[i, :n_atoms] = s['coords']
        mask_atoms[i, :n_atoms] = True

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'coords': coords.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'mask_atoms': mask_atoms.to(device),
    }


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    model, batch, noiser, device,
    contact_loss_fn, geom_loss_fn,
    stage1_weight=1.0, stage2_weight=1.0,
    contact_weight=0.1, geom_weight=0.1,
    dist_weight=0.1, grad_accum=1, mode='e2e',
):
    """Single training step with x0 prediction (matches train_resfold.py).

    Args:
        mode: 'stage1' (only train diffusion), 'stage2' (only train atom refiner), 'e2e' (both)

    Returns dict with all loss components. Does NOT call optimizer.step().
    """
    model.train()

    B, L = batch['aa_seq'].shape

    # Get ground truth
    gt_centroids = batch['centroids']  # [B, L, 3]
    gt_atoms = batch['coords_res']     # [B, L, 4, 3]
    mask = batch['mask_res']
    mask_atoms = batch['mask_atoms']

    # Initialize loss components
    centroid_mse = torch.tensor(0.0, device=device)
    dist_loss = torch.tensor(0.0, device=device)
    contact_loss = torch.tensor(0.0, device=device)
    stage1_loss = torch.tensor(0.0, device=device)
    atom_mse = torch.tensor(0.0, device=device)
    geom_loss = torch.tensor(0.0, device=device)
    stage2_loss = torch.tensor(0.0, device=device)
    geom_result = {'bond_length': torch.tensor(0.0), 'bond_angle': torch.tensor(0.0), 'omega': torch.tensor(0.0)}

    # === Stage 1: Diffusion on centroids (x0 prediction) ===
    if mode in ['stage1', 'e2e']:
        # Sample timestep from [1, T) - skip t=0 (trivial, no noise)
        t = torch.randint(1, noiser.T, (B,), device=device)

        # Add noise to centroids
        noise = torch.randn_like(gt_centroids)
        sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * gt_centroids + sqrt_one_minus_ab * noise

        # Predict x0 (clean centroids) directly
        centroids_pred = model.forward_stage1(
            x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t, mask
        )

        # Stage 1 loss: MSE on centroids + distance consistency
        centroid_mse = compute_mse_loss(centroids_pred, gt_centroids, mask)
        dist_loss = compute_distance_consistency_loss(centroids_pred, gt_centroids, mask)

        # Contact loss
        contact_result = contact_loss_fn(
            pred_centroids=centroids_pred,
            gt_centroids=gt_centroids,
            chain_ids=batch['chain_ids'],
            mask=mask,
        )
        contact_loss = contact_result['stage1']

        stage1_loss = centroid_mse + dist_weight * dist_loss + contact_weight * contact_loss

    # === Stage 2: Atoms from centroids (teacher forcing) ===
    if mode in ['stage2', 'e2e']:
        # Use GT centroids for training (teacher forcing)
        atoms_pred = model.forward_stage2(
            gt_centroids,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            mask
        )

        # Flatten to atoms for MSE
        atoms_pred_flat = atoms_pred.view(B, -1, 3)  # [B, L*4, 3]
        gt_atoms_flat = gt_atoms.view(B, -1, 3)

        atom_mse = compute_mse_loss(atoms_pred_flat, gt_atoms_flat, mask_atoms)

        # Geometry losses
        geom_result = geom_loss_fn(
            coords=atoms_pred,
            mask=mask,
            gt_coords=gt_atoms,
        )
        geom_loss = geom_result['total']

        stage2_loss = atom_mse + geom_weight * geom_loss

    # === Combined loss ===
    if mode == 'stage1':
        total_loss = stage1_loss
    elif mode == 'stage2':
        total_loss = stage2_loss
    else:  # e2e
        total_loss = stage1_weight * stage1_loss + stage2_weight * stage2_loss

    # Scale loss for gradient accumulation
    (total_loss / grad_accum).backward()

    return {
        'total': total_loss.item(),
        'stage1': stage1_loss.item() if isinstance(stage1_loss, torch.Tensor) else stage1_loss,
        'stage2': stage2_loss.item() if isinstance(stage2_loss, torch.Tensor) else stage2_loss,
        'centroid_mse': centroid_mse.item() if isinstance(centroid_mse, torch.Tensor) else centroid_mse,
        'dist': dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss,
        'contact': contact_loss.item() if isinstance(contact_loss, torch.Tensor) else contact_loss,
        'atom_mse': atom_mse.item() if isinstance(atom_mse, torch.Tensor) else atom_mse,
        'geom': geom_loss.item() if isinstance(geom_loss, torch.Tensor) else geom_loss,
        'geom_bnd': geom_result['bond_length'].item() if isinstance(geom_result['bond_length'], torch.Tensor) else geom_result['bond_length'],
        'geom_ang': geom_result['bond_angle'].item() if isinstance(geom_result['bond_angle'], torch.Tensor) else geom_result['bond_angle'],
        'geom_omg': geom_result['omega'].item() if isinstance(geom_result['omega'], torch.Tensor) else geom_result['omega'],
    }


def rollout_step(
    model, batch, noiser, device,
    contact_loss_fn, geom_loss_fn,
    stage1_weight=1.0, stage2_weight=1.0,
    contact_weight=0.1, geom_weight=0.1,
    dist_weight=0.1, K=5, clamp_val=3.0, grad_accum=1, mode='e2e',
):
    """K-step rollout training for Stage 1 diffusion.

    Only used when mode includes stage1 training. For stage2-only mode, this is skipped.

    Args:
        K: Number of reverse diffusion steps to run
        mode: 'stage1' (only train diffusion), 'stage2' (skip rollout), 'e2e' (both)
    """
    # Rollout only makes sense for Stage 1 training
    if mode == 'stage2':
        # For stage2-only, just do a regular train_step
        return train_step(model, batch, noiser, device, contact_loss_fn, geom_loss_fn,
                         stage1_weight, stage2_weight, contact_weight, geom_weight, dist_weight, grad_accum, mode)

    model.train()

    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Get ground truth
    gt_centroids = batch['centroids']  # [B, L, 3]
    gt_atoms = batch['coords_res']     # [B, L, 4, 3]
    mask_atoms = batch['mask_atoms']

    # Start from pure noise at t=T-1
    x = torch.randn(B, L, 3, device=device)

    # Run K reverse diffusion steps (x0 prediction)
    for step in range(K):
        t_val = noiser.T - 1 - step
        if t_val < 1:
            break
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Predict x0 directly (not epsilon)
        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t, mask
        )
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # DDPM reverse step to x_{t-1}
        if t_val > 1:
            ab = noiser.alpha_bar[t_val].item()
            ab_prev = noiser.alpha_bar[t_val - 1].item()
            beta = noiser.betas[t_val].item()
            alpha = noiser.alphas[t_val].item()

            coef1 = math.sqrt(ab_prev) * beta / (1 - ab)
            coef2 = math.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
            mean = coef1 * x0_pred + coef2 * x

            var = beta * (1 - ab_prev) / (1 - ab)
            x = mean + math.sqrt(var) * torch.randn_like(x)
        else:
            x = x0_pred

    # === Stage 1 loss: MSE + distance consistency ===
    centroids_pred = x0_pred
    centroid_mse = compute_mse_loss(centroids_pred, gt_centroids, mask)
    dist_loss = compute_distance_consistency_loss(centroids_pred, gt_centroids, mask)

    # Contact loss on final centroids
    contact_result = contact_loss_fn(
        pred_centroids=centroids_pred,
        gt_centroids=gt_centroids,
        chain_ids=batch['chain_ids'],
        mask=mask,
    )
    contact_loss = contact_result['stage1']

    stage1_loss = centroid_mse + dist_weight * dist_loss + contact_weight * contact_loss

    # === Stage 2: Atoms from centroids (teacher forcing) ===
    atom_mse = torch.tensor(0.0, device=device)
    geom_loss = torch.tensor(0.0, device=device)
    stage2_loss = torch.tensor(0.0, device=device)

    if mode == 'e2e':
        # Use GT centroids for training (teacher forcing)
        atoms_pred = model.forward_stage2(
            gt_centroids,
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            mask
        )

        # Atom MSE (simple, no Kabsch)
        atoms_pred_flat = atoms_pred.view(B, -1, 3)
        gt_atoms_flat = gt_atoms.view(B, -1, 3)
        sq_diff_atoms = ((atoms_pred_flat - gt_atoms_flat) ** 2).sum(dim=-1)
        atom_mse = (sq_diff_atoms * mask_atoms.float()).sum() / mask_atoms.float().sum().clamp(min=1)

        # Geometry losses
        geom_result = geom_loss_fn(
            coords=atoms_pred,
            mask=mask,
            gt_coords=gt_atoms,
        )
        geom_loss = geom_result['total']

        stage2_loss = atom_mse + geom_weight * geom_loss

    # === Combined loss ===
    if mode == 'stage1':
        total_loss = stage1_loss
    else:  # e2e
        total_loss = stage1_weight * stage1_loss + stage2_weight * stage2_loss

    # Scale loss for gradient accumulation
    (total_loss / grad_accum).backward()

    return {
        'total': total_loss.item(),
        'stage1': stage1_loss.item(),
        'stage2': stage2_loss.item() if isinstance(stage2_loss, torch.Tensor) else stage2_loss,
        'centroid_mse': centroid_mse.item(),
        'dist': dist_loss.item(),
        'contact': contact_loss.item(),
        'atom_mse': atom_mse.item() if isinstance(atom_mse, torch.Tensor) else atom_mse,
        'geom': geom_loss.item() if isinstance(geom_loss, torch.Tensor) else geom_loss,
        'K': K,
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def sample_full(model, batch, noiser, device, clamp_val=3.0):
    """Full DDPM sampling from noise to atoms using x0 prediction."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Start from noise
    x = torch.randn(B, L, 3, device=device)

    # Sample from T-1 down to 1 (skip t=0)
    for t in reversed(range(1, noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict x0 directly (not epsilon)
        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t_batch, mask
        )
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # DDPM reverse step to x_{t-1}
        if t > 1:
            ab_t = noiser.alpha_bar[t].item()
            ab_prev = noiser.alpha_bar[t - 1].item()
            beta = noiser.betas[t].item()
            alpha = noiser.alphas[t].item()

            coef1 = math.sqrt(ab_prev) * beta / (1 - ab_t)
            coef2 = math.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
            mean = coef1 * x0_pred + coef2 * x

            var = beta * (1 - ab_prev) / (1 - ab_t)
            x = mean + math.sqrt(var) * torch.randn_like(x)
        else:
            # t=1: final step, use x0_pred directly
            x = x0_pred

    # Stage 2: atoms from sampled centroids
    centroids = x
    atoms = model.forward_stage2(
        centroids, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], mask
    )

    return centroids, atoms


@torch.no_grad()
def evaluate(model, samples, indices, noiser, device, logger, n_eval=50):
    """Evaluate model on subset of samples."""
    model.eval()

    # Randomly sample indices
    eval_indices = random.sample(list(indices), min(n_eval, len(indices)))

    centroid_rmses = []
    atom_rmses = []
    lddt_scores = []
    ilddt_scores = []

    for idx in eval_indices:
        s = samples[idx]
        batch = collate_batch([s], device)

        # Sample from noise
        centroids_pred, atoms_pred = sample_full(model, batch, noiser, device)

        # Centroid RMSE
        n_res = s['n_res']
        gt_centroids = batch['centroids'][:, :n_res]
        pred_centroids = centroids_pred[:, :n_res]
        centroid_rmse = compute_rmse(pred_centroids, gt_centroids).item() * s['std']
        centroid_rmses.append(centroid_rmse)

        # Atom RMSE
        n_atoms = s['n_atoms']
        atoms_pred_flat = atoms_pred.view(1, -1, 3)[:, :n_atoms]
        gt_atoms_flat = batch['coords'][:, :n_atoms]
        atom_rmse = compute_rmse(atoms_pred_flat, gt_atoms_flat).item() * s['std']
        atom_rmses.append(atom_rmse)

        # lDDT / ilDDT
        pred_coords_res = atoms_pred[:, :n_res]
        gt_coords_res = batch['coords_res'][:, :n_res]
        chain_ids = batch['chain_ids'][:, :n_res]

        lddt_result = compute_lddt_metrics(
            pred_coords_res, gt_coords_res, chain_ids, coord_scale=s['std']
        )
        lddt_scores.append(lddt_result['lddt'])
        if lddt_result['n_interface'] > 0:
            ilddt_scores.append(lddt_result['ilddt'])

    results = {
        'centroid_rmse': sum(centroid_rmses) / len(centroid_rmses),
        'atom_rmse': sum(atom_rmses) / len(atom_rmses),
        'lddt': sum(lddt_scores) / len(lddt_scores),
        'ilddt': sum(ilddt_scores) / len(ilddt_scores) if ilddt_scores else 0.0,
        'n_eval': len(eval_indices),
    }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path, lddt=None):
    """Plot prediction vs ground truth (matching train_resfold.py style)."""
    fig = plt.figure(figsize=(12, 5))

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    # Ground truth
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    if mask_a.any():
        ax1.scatter(target[mask_a, 0], target[mask_a, 1], target[mask_a, 2], c='blue', s=10, alpha=0.7)
    if mask_b.any():
        ax1.scatter(target[mask_b, 0], target[mask_b, 1], target[mask_b, 2], c='red', s=10, alpha=0.7)
    ax1.set_title(f'{sample_id}\nGround Truth')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # Prediction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2], c='cyan', s=10, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2], c='orange', s=10, alpha=0.7)
    title_str = f'Prediction\nRMSE: {rmse:.2f} A'
    if lddt is not None:
        title_str += f' | lDDT: {lddt:.3f}'
    ax2.set_title(title_str)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='data/processed/samples.parquet')
    parser.add_argument('--n_train', type=int, default=80)
    parser.add_argument('--n_test', type=int, default=14)
    parser.add_argument('--load_split', type=str, default=None)

    # Training
    parser.add_argument('--mode', type=str, default='e2e', choices=['stage1', 'stage2', 'e2e'],
                        help='Training mode: stage1 (diffusion only), stage2 (atom refiner only), e2e (both)')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--n_eval', type=int, default=50)

    # Diffusion
    parser.add_argument('--T', type=int, default=50)

    # Loss weights
    parser.add_argument('--stage1_weight', type=float, default=1.0)
    parser.add_argument('--stage2_weight', type=float, default=1.0)
    parser.add_argument('--contact_weight', type=float, default=0.1)
    parser.add_argument('--geom_weight', type=float, default=0.1)
    parser.add_argument('--dist_weight', type=float, default=0.1,
                        help='Weight for distance consistency loss (Stage 1)')

    # Dynamic batching
    parser.add_argument('--dynamic_batch', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=20000)

    # Rollout training
    parser.add_argument('--rollout_prob', type=float, default=0.2,
                        help='Probability of using rollout training vs single-step')
    parser.add_argument('--rollout_K', type=int, default=5,
                        help='Number of diffusion steps in rollout')

    # Output
    parser.add_argument('--output_dir', default='outputs/resfold_e2e')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    # Header
    logger.log("=" * 70)
    logger.log("ResFold End-to-End Training (Smaller Model)")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Script: {os.path.abspath(__file__)}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    n_samples = len(table)
    logger.log(f"  Total samples: {n_samples}")

    # Data split
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        train_indices, test_indices, split_info = load_split(args.load_split)
        logger.log(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
    else:
        config = DataSplitConfig(
            n_train=args.n_train,
            n_test=args.n_test,
            min_atoms=320,
            max_atoms=1816,
            seed=42,
        )
        train_indices, test_indices = get_train_test_indices(table, config)
        logger.log(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

        # Save split
        split_path = os.path.join(args.output_dir, 'split.json')
        info = get_split_info(table, config)
        save_split(info, split_path)
        logger.log(f"  Saved split to: {split_path}")

    # Preload samples
    logger.log("Preloading samples...")
    samples = {}
    for idx in train_indices + test_indices:
        samples[idx] = load_sample_raw(table, idx)
    logger.log(f"  Loaded {len(samples)} samples")

    # Create model
    logger.log()
    logger.log("Model configuration:")
    for k, v in MODEL_CONFIG.items():
        logger.log(f"  {k}: {v}")

    model = ResFoldPipeline(
        c_token_s1=MODEL_CONFIG['c_token_s1'],
        trunk_layers=MODEL_CONFIG['trunk_layers'],
        trunk_heads=MODEL_CONFIG['trunk_heads'],
        denoiser_blocks=MODEL_CONFIG['denoiser_blocks'],
        denoiser_heads=MODEL_CONFIG['denoiser_heads'],
        c_token_s2=MODEL_CONFIG['c_token_s2'],
        s2_layers=MODEL_CONFIG['s2_layers'],
        s2_heads=MODEL_CONFIG['s2_heads'],
        n_timesteps=MODEL_CONFIG['n_timesteps'],
    ).to(device)

    # Count parameters
    counts = model.count_parameters()
    logger.log()
    logger.log(f"Model: ResFold (end-to-end, smaller)")
    logger.log(f"  Stage 1 params: {counts['stage1']:,} ({counts['stage1_pct']:.1f}%)")
    logger.log(f"  Stage 2 params: {counts['stage2']:,} ({counts['stage2_pct']:.1f}%)")
    logger.log(f"  Total params:   {counts['total']:,}")

    # Set training mode (freeze appropriate parameters)
    logger.log()
    logger.log(f"Training mode: {args.mode}")
    if args.mode == 'stage1':
        model.set_training_mode('stage1_only')
        logger.log("  Stage 1: trainable, Stage 2: frozen")
    elif args.mode == 'stage2':
        model.set_training_mode('stage2_only')
        logger.log("  Stage 1: frozen, Stage 2: trainable")
    else:  # e2e
        model.set_training_mode('end_to_end')
        logger.log("  Stage 1: trainable, Stage 2: trainable")

    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint:
        logger.log()
        logger.log(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        logger.log(f"  Loaded from step {start_step}")

    # Diffusion
    schedule = create_schedule('linear', T=args.T)
    schedule.to(device)  # Move schedule tensors to device
    noiser = create_noiser('gaussian', schedule)
    logger.log()
    logger.log(f"Diffusion: T={args.T}, schedule=linear")

    # Loss functions
    contact_loss_fn = ContactLoss(
        threshold=1.0,
        min_seq_sep=5,
        include_intra=True,
        include_inter=True,
        inter_chain_weight=2.0,
        stage="stage1",
    )

    geom_loss_fn = GeometryLoss(
        bond_length_weight=1.0,
        bond_angle_weight=0.1,
        omega_weight=0.1,
        o_chirality_weight=0.1,
        cb_chirality_weight=0.0,
    )

    logger.log()
    logger.log(f"Contact loss: {contact_loss_fn}")
    logger.log(f"Geometry loss: {geom_loss_fn}")

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.log(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=1e-5
    )

    # Loss weights
    logger.log()
    logger.log("Loss weights:")
    logger.log(f"  stage1_weight: {args.stage1_weight}")
    logger.log(f"  stage2_weight: {args.stage2_weight}")
    logger.log(f"  contact_weight: {args.contact_weight}")
    logger.log(f"  geom_weight: {args.geom_weight}")

    # Rollout training
    logger.log()
    logger.log("Rollout training:")
    logger.log(f"  rollout_prob: {args.rollout_prob}")
    logger.log(f"  rollout_K: {args.rollout_K}")

    # Batching
    if args.dynamic_batch:
        # Filter samples to only include training indices
        train_samples = {idx: samples[idx] for idx in train_indices}
        sampler = DynamicBatchSampler(
            train_samples,
            base_batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
        logger.log()
        logger.log(f"Dynamic batching: max_tokens={args.max_tokens}")
        # Log batch sizes per bucket
        for info in sampler.get_batch_sizes():
            logger.log(f"  Bucket {info['bucket']}: max_res={info['max_res']}, batch_size={info['batch_size']}, tokens={info['tokens']}")
    else:
        sampler = None
        logger.log()
        logger.log(f"Fixed batching: batch_size={args.batch_size}")

    # Effective batch size
    eff_batch = args.batch_size * args.grad_accum
    logger.log()
    logger.log(f"Gradient accumulation: {args.grad_accum}")
    logger.log(f"Effective batch size: {eff_batch}")

    # Training
    logger.log()
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    step = start_step
    best_atom_rmse = float('inf')
    start_time = time.time()

    # Running averages
    running_loss = 0.0
    running_s1 = 0.0
    running_s2 = 0.0
    running_mse = 0.0  # centroid MSE
    running_dist = 0.0  # distance consistency loss
    running_con = 0.0
    running_atm = 0.0
    running_geo = 0.0
    running_cnt = 0

    rollout_count = 0

    while step < args.n_steps:
        # Gradient accumulation: zero_grad at start of cycle
        accum_idx = (step - start_step) % args.grad_accum
        if accum_idx == 0:
            optimizer.zero_grad()

        # Get batch
        if args.dynamic_batch:
            batch_indices, _ = sampler.sample_batch()
        else:
            batch_indices = random.sample(train_indices, args.batch_size)

        batch_samples = [samples[idx] for idx in batch_indices]
        batch = collate_batch(batch_samples, device)

        # Decide whether to use rollout or single-step training
        # Rollout only applies to Stage 1 training
        use_rollout = args.mode != 'stage2' and random.random() < args.rollout_prob

        if use_rollout:
            losses = rollout_step(
                model, batch, noiser, device,
                contact_loss_fn, geom_loss_fn,
                stage1_weight=args.stage1_weight,
                stage2_weight=args.stage2_weight,
                contact_weight=args.contact_weight,
                geom_weight=args.geom_weight,
                dist_weight=args.dist_weight,
                K=args.rollout_K,
                grad_accum=args.grad_accum,
                mode=args.mode,
            )
            rollout_count += 1
        else:
            losses = train_step(
                model, batch, noiser, device,
                contact_loss_fn, geom_loss_fn,
                stage1_weight=args.stage1_weight,
                stage2_weight=args.stage2_weight,
                contact_weight=args.contact_weight,
                geom_weight=args.geom_weight,
                dist_weight=args.dist_weight,
                grad_accum=args.grad_accum,
                mode=args.mode,
            )

        # Gradient accumulation: step optimizer at end of cycle
        if accum_idx == args.grad_accum - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        step += 1

        # Update running averages
        running_loss += losses['total']
        running_s1 += losses['stage1']
        running_s2 += losses['stage2']
        running_mse += losses.get('centroid_mse', 0.0)
        running_dist += losses.get('dist', 0.0)
        running_con += losses['contact']
        running_atm += losses['atom_mse']
        running_geo += losses['geom']
        running_cnt += 1

        # Log every 100 steps
        if step % 100 == 0:
            avg_loss = running_loss / running_cnt
            avg_s1 = running_s1 / running_cnt
            avg_s2 = running_s2 / running_cnt
            avg_mse = running_mse / running_cnt
            avg_dist = running_dist / running_cnt
            avg_con = running_con / running_cnt
            avg_atm = running_atm / running_cnt
            avg_geo = running_geo / running_cnt
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time

            # Mode-specific logging
            if args.mode == 'stage1':
                logger.log(
                    f"Step {step:5d} | loss: {avg_loss:.4f} | "
                    f"s1: {avg_s1:.4f} (mse:{avg_mse:.3f} dst:{avg_dist:.3f} con:{avg_con:.3f}) | "
                    f"lr: {lr:.2e} | {elapsed:.0f}s"
                )
            elif args.mode == 'stage2':
                logger.log(
                    f"Step {step:5d} | loss: {avg_loss:.4f} | "
                    f"s2: {avg_s2:.4f} (atm:{avg_atm:.3f} geo:{avg_geo:.3f}) | "
                    f"lr: {lr:.2e} | {elapsed:.0f}s"
                )
            else:  # e2e
                logger.log(
                    f"Step {step:5d} | loss: {avg_loss:.4f} | "
                    f"s1: {avg_s1:.4f} (mse:{avg_mse:.3f} dst:{avg_dist:.3f} con:{avg_con:.3f}) | "
                    f"s2: {avg_s2:.4f} (atm:{avg_atm:.3f} geo:{avg_geo:.3f}) | "
                    f"lr: {lr:.2e} | {elapsed:.0f}s"
                )

            running_loss = 0.0
            running_s1 = 0.0
            running_s2 = 0.0
            running_mse = 0.0
            running_dist = 0.0
            running_con = 0.0
            running_atm = 0.0
            running_geo = 0.0
            running_cnt = 0

        # Evaluate
        if step % args.eval_every == 0:
            train_results = evaluate(
                model, samples, train_indices, noiser, device, logger, n_eval=args.n_eval
            )
            test_results = evaluate(
                model, samples, test_indices, noiser, device, logger, n_eval=len(test_indices)
            )

            logger.log(
                f"         >>> Train: CenRMSE={train_results['centroid_rmse']:.2f}A "
                f"AtmRMSE={train_results['atom_rmse']:.2f}A lDDT={train_results['lddt']:.3f} "
                f"ilDDT={train_results['ilddt']:.3f}"
            )
            logger.log(
                f"         >>> Test:  CenRMSE={test_results['centroid_rmse']:.2f}A "
                f"AtmRMSE={test_results['atom_rmse']:.2f}A lDDT={test_results['lddt']:.3f} "
                f"ilDDT={test_results['ilddt']:.3f}"
            )

            # Save best model
            if test_results['atom_rmse'] < best_atom_rmse:
                best_atom_rmse = test_results['atom_rmse']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_atom_rmse': test_results['atom_rmse'],
                    'test_lddt': test_results['lddt'],
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.log(f"         >>> New best model saved! RMSE: {best_atom_rmse:.2f}A")

            # Save checkpoint
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, 'checkpoint.pt'))

            # Plot train and test samples
            model.eval()
            plot_dir = os.path.join(args.output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)

            # Plot first few train samples + first few test samples
            plot_indices = train_indices[:2] + test_indices[:2]
            for plot_idx in plot_indices:
                s = samples[plot_idx]
                batch = collate_batch([s], device)

                centroids_pred, atoms_pred = sample_full(model, batch, noiser, device)

                n = s['n_atoms']
                pred = atoms_pred[0].view(-1, 3)[:n] * s['std']
                target = batch['coords'][0, :n] * s['std']

                pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                # RMSE is already in Angstroms since pred/target are scaled
                rmse = compute_rmse(pred.unsqueeze(0), target.unsqueeze(0)).item()

                chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]

                # lDDT
                n_res = s['n_res']
                pred_res = atoms_pred[:, :n_res]
                gt_res = batch['coords_res'][:, :n_res]
                lddt_result = compute_lddt_metrics(pred_res, gt_res, batch['chain_ids'][:, :n_res], coord_scale=s['std'])

                # Mark train vs test in filename
                prefix = "train" if plot_idx in train_indices else "test"
                plot_prediction(
                    pred_aligned[0], target_c[0], chain_ids_plot,
                    f"{prefix}_{s['sample_id']}",
                    rmse,
                    os.path.join(plot_dir, f"step{step}_{prefix}_{s['sample_id']}.png"),
                    lddt=lddt_result['lddt'],
                )

            model.train()

    # Final summary
    logger.log()
    logger.log("=" * 70)
    logger.log("Training complete!")
    logger.log(f"Best test atom RMSE: {best_atom_rmse:.2f}A")
    logger.log(f"Rollout steps: {rollout_count} / {step} ({100*rollout_count/step:.1f}%)")
    logger.log("=" * 70)

    logger.close()


if __name__ == '__main__':
    main()
