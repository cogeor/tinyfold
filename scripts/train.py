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
    kabsch_align_to_target,
    # Continuous sigma (VE noise)
    KarrasSchedule,
    VENoiser,
    # Samplers
    create_sampler,
)
from models.training_utils import (
    random_rigid_augment,
    random_rotation_matrix,
    af3_loss_weight,
)
from models.self_conditioning import (
    self_conditioning_training_step,
    sample_step_with_self_cond,
)
from models.dockq_utils import compute_dockq
from data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info,
    save_split, load_split,
)

# Loss imports - use consolidated losses from tinyfold.model.losses
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
    ContactLoss,
    compute_lddt_metrics,
)


# =============================================================================
# Logging (simple logger for backward compatibility)
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
                clamp_val=3.0, x_linear=None, noise_type="gaussian",
                align_per_step=False, recenter=False):
    """Diffusion sampling loop.

    For Gaussian noise: starts from random noise, uses DDPM reverse.
    For linear_chain: starts from extended chain, uses interpolation reverse.
    For linear_flow: iterative refinement from extended chain.

    Args:
        x_linear: Extended chain coordinates [B, N, 3] for linear_chain/linear_flow.
                  If None, will be generated.
        noise_type: "gaussian", "linear_chain", or "linear_flow"
        align_per_step: If True, Kabsch-align x0_pred to current x before update.
                        This fixes frame drift (Boltz-1 style). Default False for
                        backward compatibility.
        recenter: If True, re-center coordinates after each step to avoid
                  translation drift. Default False for backward compatibility.
    """
    device = atom_types.device
    B, N = atom_types.shape

    # Generate extended chain if needed (deterministic - no rotation)
    def get_x_linear():
        from models.diffusion import generate_extended_chain
        xl = torch.zeros(B, N, 3, device=device)
        for b in range(B):
            xl[b] = generate_extended_chain(
                N, atom_to_res[b], atom_types[b], chain_ids[b], device,
                apply_rotation=False  # Deterministic for inference
            )
        xl = xl - xl.mean(dim=1, keepdim=True)
        xl = xl / xl.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return xl

    # === LINEAR_FLOW: Iterative refinement with x0 prediction ===
    if noise_type == "linear_flow":
        if x_linear is None:
            x_linear = get_x_linear()
        x = x_linear.clone()

        # Iteratively predict x0 and interpolate towards it
        # Each step: x_t -> model -> x0_pred -> interpolate to x_{t+1}
        for t in range(noiser.T):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            # Predict x0 from current state
            x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
            x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

            # NEW: Kabsch-align x0_pred to current x's frame (fixes drift)
            if align_per_step:
                x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            if t < noiser.T - 1:
                # Interpolate: move towards x0_pred
                # At t+1, alpha increases, so we're closer to x0
                alpha_next = noiser.schedule.sqrt_alpha_bar[t + 1]
                one_minus_alpha_next = noiser.schedule.sqrt_one_minus_alpha_bar[t + 1]
                x = alpha_next * x0_pred + one_minus_alpha_next * x_linear
            else:
                # Final step: return x0_pred directly
                x = x0_pred

            # NEW: Re-center to avoid translation drift
            if recenter:
                if mask is not None:
                    mask_exp = mask.unsqueeze(-1).float()
                    n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                    centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
                else:
                    centroid = x.mean(dim=1, keepdim=True)
                x = x - centroid

        return x

    # === LINEAR_CHAIN: DDPM with extended chain start ===
    use_linear_chain = hasattr(noiser, 'reverse_step')
    if use_linear_chain:
        if x_linear is None:
            x_linear = get_x_linear()
        x = x_linear.clone()
        # Start from t=T (pure x_linear) and go down to t=0
        t_range = reversed(range(noiser.T + 1))
    else:
        # === GAUSSIAN: Start from random noise ===
        x = torch.randn(B, N, 3, device=device)
        # Standard DDPM: t goes from T-1 to 0
        t_range = reversed(range(noiser.T))

    for t in t_range:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        # Use forward_direct for linear_chain with AF3-style (no residual scaling)
        if use_linear_chain and hasattr(model, 'forward_direct'):
            x0_pred = model.forward_direct(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        else:
            x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # NEW: Kabsch-align x0_pred to current x's frame (fixes drift)
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

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

        # NEW: Re-center to avoid translation drift
        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    return x


@torch.no_grad()
def ddpm_sample_ve(model, atom_types, atom_to_res, aa_seq, chain_ids, noiser, mask=None,
                   clamp_val=3.0, align_per_step=True, recenter=True):
    """VE (Euler) sampling for af3_style with continuous sigma.

    Directly mirrors sample_centroids_ve from train_resfold.py.
    Uses Karras/EDM-style Euler sampling with the model's forward_sigma method.

    Args:
        model: Model with forward_sigma(x, atom_types, atom_to_res, aa_seq, chain_ids, sigma, mask)
        atom_types: [B, N] atom type indices
        atom_to_res: [B, N] residue index per atom
        aa_seq: [B, N] amino acid type per atom
        chain_ids: [B, N] chain ID per atom
        noiser: VENoiser with .sigmas attribute (Karras schedule)
        mask: [B, N] valid atom mask
        clamp_val: Clamp x0 predictions to [-clamp_val, clamp_val]
        align_per_step: Kabsch-align x0_pred to current x (fixes drift)
        recenter: Re-center coordinates each step (avoids translation drift)

    Returns:
        x: [B, N, 3] sampled coordinates
    """
    device = atom_types.device
    B, N = atom_types.shape

    # Get sigma schedule from noiser (decreasing: sigma_max -> sigma_min)
    sigmas = noiser.sigmas.to(device)

    # Initialize at highest noise level (VE: x = sigma * noise)
    x = sigmas[0] * torch.randn(B, N, 3, device=device)

    # Euler sampling loop (same as sample_centroids_ve in train_resfold.py)
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Create sigma tensor for batch
        sigma_batch = sigma.expand(B)

        # Predict x0 using continuous sigma conditioning
        x0_pred = model.forward_sigma(x, atom_types, atom_to_res, aa_seq, chain_ids,
                                       sigma_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # Kabsch-align x0_pred to current x
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        # Euler step: x_next = x + (sigma_next - sigma) * (x - x0_pred) / sigma
        d = (x - x0_pred) / sigma  # Direction toward x0
        dt = sigma_next - sigma    # Negative (decreasing sigma)
        x = x + d * dt

        # Re-center
        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    return x


# Alias for backward compatibility (compute_loss -> compute_mse_loss)
compute_loss = compute_mse_loss


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
    parser.add_argument("--load_split", type=str, default=None,
                        help="Load train/test split from JSON file")

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum LR for cosine annealing")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    # Model
    parser.add_argument("--model", type=str, default="attention_v2", choices=list_models())
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--trunk_layers", type=int, default=None,
                        help="Override trunk layer count for af3_style (default: n_layers+3)")
    parser.add_argument("--denoiser_blocks", type=int, default=None,
                        help="Override denoiser block count for af3_style (default: n_layers+1)")

    # Diffusion
    parser.add_argument("--schedule", type=str, default="linear", help="Alpha-bar schedule")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=list_noise_types(),
                        help="Noise type: gaussian or linear_chain")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Gaussian noise scale for linear_chain")
    parser.add_argument("--T", type=int, default=50, help="Diffusion timesteps")

    # Continuous sigma (AF3-style VE noise)
    parser.add_argument("--continuous_sigma", action="store_true",
                        help="Use AF3-style continuous sigma (VE) instead of discrete timesteps")
    parser.add_argument("--sigma_data", type=float, default=1.0,
                        help="Data std for VE noise (1.0 for normalized coords)")
    parser.add_argument("--sigma_min", type=float, default=0.002,
                        help="Minimum sigma for VE noise")
    parser.add_argument("--sigma_max", type=float, default=10.0,
                        help="Maximum sigma for VE noise")
    parser.add_argument("--sigma_sampling", type=str, default="log_uniform",
                        choices=["log_uniform", "stratified"],
                        help="Sigma sampling method: log_uniform (default) or stratified (better high-sigma coverage)")

    # Self-conditioning
    parser.add_argument("--self_cond_prob", type=float, default=0.0,
                        help="Self-conditioning probability (0 to disable, 0.5 recommended)")

    # Augmentation
    parser.add_argument("--augment_rotation", action="store_true",
                        help="Apply random rotation augmentation during training")
    parser.add_argument("--translate_aug", type=float, default=0.0,
                        help="Random translation augmentation scale (0 to disable)")

    # Loss functions
    parser.add_argument("--dist_weight", type=float, default=0.0,
                        help="Weight for distance consistency loss")
    parser.add_argument("--geom_weight", type=float, default=0.0,
                        help="Weight for geometry loss (requires af3_style model)")
    parser.add_argument("--loss_weighting", action="store_true",
                        help="Apply AF3-style loss weighting by noise level")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", help="Enable timestep curriculum")
    parser.add_argument("--curriculum_warmup", type=int, default=5000, help="Steps to reach full T")
    parser.add_argument("--curriculum_schedule", type=str, default="linear",
                        choices=["linear", "cosine"], help="Curriculum progression schedule")

    # Sampling (evaluation)
    parser.add_argument("--align_per_step", action="store_true",
                        help="Kabsch-align x0_pred to x_t each step (fixes drift, Boltz-1 style)")
    parser.add_argument("--recenter", action="store_true",
                        help="Re-center coordinates each step (avoids translation drift)")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/train")

    # Resume from checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")

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

    # Load or create split
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        train_indices, test_indices, split_info = load_split(args.load_split)
        logger.log(f"  Training: {len(train_indices)} samples")
        logger.log(f"  Test: {len(test_indices)} samples")
    else:
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

        # Save split for reproducibility
        split_path = os.path.join(args.output_dir, 'split.json')
        save_split(split_info, split_path)
        logger.log(f"  Split saved to: {split_path}")
    logger.log("")

    # Preload train and test samples SEPARATELY
    logger.log("Preloading samples...")
    train_samples = {idx: load_sample_raw(table, idx) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx) for idx in test_indices}
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples")

    # Create model
    if args.model == "af3_style":
        # AF3-style uses different kwargs
        # Allow explicit override of layer counts
        trunk_layers = args.trunk_layers if args.trunk_layers is not None else args.n_layers + 3
        denoiser_blocks = args.denoiser_blocks if args.denoiser_blocks is not None else args.n_layers + 1
        model = create_model(
            args.model,
            c_token=args.h_dim * 2,  # 256 for h_dim=128
            c_atom=args.h_dim,
            trunk_layers=trunk_layers,
            denoiser_blocks=denoiser_blocks,
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
    if args.model == "af3_style":
        logger.log(f"  trunk_layers: {trunk_layers}")
        logger.log(f"  denoiser_blocks: {denoiser_blocks}")
    logger.log("")

    # Create diffusion components
    if args.continuous_sigma:
        # AF3-style VE noise with Karras schedule
        karras_schedule = KarrasSchedule(
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            n_steps=args.T,
        )
        noiser = VENoiser(karras_schedule, sigma_data=args.sigma_data)
        noiser = noiser.to(device)
        sigmas = karras_schedule.sigmas.to(device)
        logger.log(f"Diffusion (continuous sigma / VE):")
        logger.log(f"  sigma_min: {args.sigma_min}")
        logger.log(f"  sigma_max: {args.sigma_max}")
        logger.log(f"  sigma_data: {args.sigma_data}")
        logger.log(f"  T (steps): {args.T}")
    else:
        schedule = create_schedule(args.schedule, T=args.T)
        if args.noise_type in ["linear_chain", "linear_flow"]:
            noiser = create_noiser(args.noise_type, schedule, noise_scale=args.noise_scale)
        else:
            noiser = create_noiser(args.noise_type, schedule)
        noiser = noiser.to(device)
        sigmas = None
        logger.log(f"Diffusion:")
        logger.log(f"  Schedule: {args.schedule}")
        logger.log(f"  Noise type: {args.noise_type}")
        if args.noise_type == "linear_chain":
            logger.log(f"  Noise scale: {args.noise_scale}")

    # Create curriculum if enabled (only for discrete timesteps)
    curriculum = None
    if args.curriculum and not args.continuous_sigma:
        curriculum = TimestepCurriculum(noiser.T, args.curriculum_warmup, args.curriculum_schedule)
        logger.log(f"  Curriculum: warmup={args.curriculum_warmup}, schedule={args.curriculum_schedule}")

    # Log augmentation and loss settings
    logger.log("")
    logger.log("Augmentation:")
    logger.log(f"  rotation: {args.augment_rotation}")
    logger.log(f"  translation: {args.translate_aug}")
    logger.log("")
    logger.log("Loss weights:")
    logger.log(f"  dist_weight: {args.dist_weight}")
    logger.log(f"  geom_weight: {args.geom_weight}")
    logger.log(f"  loss_weighting (AF3): {args.loss_weighting}")
    logger.log(f"  self_cond_prob: {args.self_cond_prob}")
    logger.log("")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=args.min_lr)

    # Create geometry loss if needed
    geom_loss_fn = None
    if args.geom_weight > 0:
        geom_loss_fn = GeometryLoss(
            bond_length_weight=1.0,
            bond_angle_weight=0.1,
            omega_weight=0.1,
            o_chirality_weight=0.1,
            cb_chirality_weight=0.0,
        )

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
        accum_mse_loss = 0.0
        accum_dist_loss = 0.0
        accum_geom_loss = 0.0

        for accum_step in range(args.grad_accum):
            # TRAINING ONLY uses train_samples (no data leakage)
            batch_indices = random.choices(train_indices, k=args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)
            B = args.batch_size

            # Get clean coordinates (target)
            x0 = batch['coords']

            # Apply rotation augmentation BEFORE noising
            if args.augment_rotation:
                R = random_rotation_matrix(B, device)  # [B, 3, 3]
                x0 = torch.bmm(x0, R.transpose(1, 2))  # [B, N, 3]

            # Apply translation augmentation
            if args.translate_aug > 0:
                T_aug = torch.randn(B, 1, 3, device=device) * args.translate_aug
                x0 = x0 + T_aug

            # Sample noise level / timestep
            if args.continuous_sigma:
                # VE noise: sample sigma based on selected method
                if args.sigma_sampling == "stratified":
                    sigma = noiser.sample_sigma_stratified(B, device)
                else:
                    # Default log-uniform sampling
                    log_sigma = torch.rand(B, device=device) * (math.log(args.sigma_max) - math.log(args.sigma_min)) + math.log(args.sigma_min)
                    sigma = log_sigma.exp()
                # Add noise: x_t = x_0 + sigma * epsilon
                noise = torch.randn_like(x0)
                x_input = x0 + sigma.view(B, 1, 1) * noise
            else:
                if curriculum:
                    t = curriculum.sample(B, step, device)
                else:
                    # For linear_chain: sample t from 0 to T (inclusive) so model sees pure x_linear
                    # For gaussian: sample t from 0 to T-1 (standard DDPM)
                    if args.noise_type == "linear_chain":
                        t = torch.randint(0, noiser.T + 1, (B,), device=device)
                    else:
                        t = torch.randint(0, noiser.T, (B,), device=device)

                # Add noise (unified API)
                x_input, target = noiser.add_noise(
                    x0, t,
                    atom_to_res=batch['atom_to_res'],
                    atom_type=batch['atom_types'],
                    chain_ids=batch['chain_ids'],
                )
                sigma = None  # Not used for discrete timesteps

            # Forward pass - predict x0
            if args.continuous_sigma:
                # VE mode: use forward_sigma if available
                if hasattr(model, 'forward_sigma'):
                    pred = model.forward_sigma(
                        x_input, batch['atom_types'], batch['atom_to_res'],
                        batch['aa_seq'], batch['chain_ids'], sigma, batch['mask']
                    )
                else:
                    # Fallback: convert sigma to discrete timestep (approximate)
                    t_approx = torch.zeros(B, dtype=torch.long, device=device)
                    pred = model(x_input, batch['atom_types'], batch['atom_to_res'],
                                 batch['aa_seq'], batch['chain_ids'], t_approx, batch['mask'])
            else:
                # For AF3-style with linear_chain: use forward_direct (no scaling)
                if args.noise_type == "linear_chain" and hasattr(model, 'forward_direct'):
                    pred = model.forward_direct(x_input, batch['atom_types'], batch['atom_to_res'],
                                                batch['aa_seq'], batch['chain_ids'], t, batch['mask'])
                else:
                    pred = model(x_input, batch['atom_types'], batch['atom_to_res'],
                                 batch['aa_seq'], batch['chain_ids'], t, batch['mask'])

            # Compute MSE loss
            use_kabsch = (args.noise_type != "linear_chain")
            if args.noise_type == "linear_flow" and not args.continuous_sigma:
                mse_loss = compute_loss(pred, target, batch['mask'], use_kabsch=use_kabsch)
            else:
                mse_loss = compute_loss(pred, x0, batch['mask'], use_kabsch=use_kabsch)

            # Apply AF3-style loss weighting if enabled
            if args.loss_weighting and args.continuous_sigma:
                weight = af3_loss_weight(sigma, args.sigma_data)
                mse_loss = mse_loss * weight.mean()

            loss = mse_loss
            accum_mse_loss += mse_loss.item() / args.grad_accum

            # Distance consistency loss
            if args.dist_weight > 0:
                dist_loss = compute_distance_consistency_loss(pred, x0, batch['mask'])
                loss = loss + args.dist_weight * dist_loss
                accum_dist_loss += dist_loss.item() / args.grad_accum

            # Geometry loss (only for af3_style which predicts atoms)
            if args.geom_weight > 0 and geom_loss_fn is not None and args.model == "af3_style":
                # Reshape pred from [B, N_atoms, 3] to [B, L, 4, 3] for geometry loss
                # Determine L (number of residues) from atom_to_res
                L = batch['atom_to_res'].max().item() + 1
                pred_res = pred.view(B, L, 4, 3)
                x0_res = x0.view(B, L, 4, 3)
                mask_res = batch['mask'].view(B, L, 4)[:, :, 0]  # [B, L]
                geom_losses = geom_loss_fn(pred_res, mask_res, gt_coords=x0_res)
                geom_loss = geom_losses['total']
                loss = loss + args.geom_weight * geom_loss
                accum_geom_loss += geom_loss.item() / args.grad_accum

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
            loss_str = f"loss: {loss:.6f}"
            if args.dist_weight > 0:
                loss_str += f" (mse: {accum_mse_loss:.4f}, dist: {accum_dist_loss:.4f})"
            if args.geom_weight > 0:
                loss_str += f" (geom: {accum_geom_loss:.4f})"
            if curriculum:
                t_max = curriculum.get_max_t(step)
                logger.log(f"Step {step:5d} | {loss_str} | t_max: {t_max:2d} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            else:
                logger.log(f"Step {step:5d} | {loss_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

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
                    if args.continuous_sigma:
                        x_pred = ddpm_sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                                                batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                                align_per_step=args.align_per_step,
                                                recenter=args.recenter)
                    else:
                        x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                             batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                             noise_type=args.noise_type,
                                             align_per_step=args.align_per_step,
                                             recenter=args.recenter)
                    rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                    train_rmses.append(rmse)
                train_avg = sum(train_rmses) / len(train_rmses)

                # Evaluate on TEST set (full set - never seen during training)
                test_rmses = []
                for idx in test_indices:
                    s = test_samples[idx]
                    batch = collate_batch([s], device)
                    if args.continuous_sigma:
                        x_pred = ddpm_sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                                                batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                                align_per_step=args.align_per_step,
                                                recenter=args.recenter)
                    else:
                        x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                             batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                             noise_type=args.noise_type,
                                             align_per_step=args.align_per_step,
                                             recenter=args.recenter)
                    rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                    test_rmses.append(rmse)
                test_avg = sum(test_rmses) / len(test_rmses)

                logger.log(f"         >>> Train RMSE ({n_eval}): {train_avg:.4f} A | Test RMSE ({len(test_indices)}): {test_avg:.4f} A")

                # Plot first train sample
                s = train_samples[train_indices[0]]
                batch = collate_batch([s], device)
                if args.continuous_sigma:
                    x_pred = ddpm_sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                                            batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                            align_per_step=args.align_per_step,
                                            recenter=args.recenter)
                else:
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                         noise_type=args.noise_type,
                                         align_per_step=args.align_per_step,
                                         recenter=args.recenter)

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
                if args.continuous_sigma:
                    x_pred = ddpm_sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                                            batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                            align_per_step=args.align_per_step,
                                            recenter=args.recenter)
                else:
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                         noise_type=args.noise_type,
                                         align_per_step=args.align_per_step,
                                         recenter=args.recenter)
                rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                rmses.append(rmse)
            mean_rmse = sum(rmses) / len(rmses)
            train_final_rmses.append(mean_rmse)
        train_overall = sum(train_final_rmses) / len(train_final_rmses)
        logger.log(f"  Mean: {train_overall:.2f} A")

        # Evaluate TEST set (never seen during training)
        logger.log(f"\nTEST SET ({len(test_indices)} samples):")
        test_final_rmses = []
        for idx in test_indices:
            s = test_samples[idx]
            batch = collate_batch([s], device)
            rmses = []
            for _ in range(3):
                if args.continuous_sigma:
                    x_pred = ddpm_sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                                            batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                            align_per_step=args.align_per_step,
                                            recenter=args.recenter)
                else:
                    x_pred = ddpm_sample(model, batch['atom_types'], batch['atom_to_res'],
                                         batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'],
                                         noise_type=args.noise_type,
                                         align_per_step=args.align_per_step,
                                         recenter=args.recenter)
                rmse = compute_rmse(x_pred, batch['coords'], batch['mask']).item() * s['std']
                rmses.append(rmse)
            mean_rmse = sum(rmses) / len(rmses)
            test_final_rmses.append(mean_rmse)
        test_overall = sum(test_final_rmses) / len(test_final_rmses)
        logger.log(f"  Mean: {test_overall:.2f} A")

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
