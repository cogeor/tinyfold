#!/usr/bin/env python
"""
IterFold Training: Anchor-conditioned protein structure prediction.

Alternative to diffusion-based ResFold. Instead of denoising, we condition
on known residue positions (anchors) and predict atom coordinates.

Key features:
- SO(3) rotation augmentation (default: enabled)
- Random anchor masking (10-30% of residues anchored)
- DockQ evaluation metric
- Simple MSE + geometry loss

Usage:
    python scripts/train_iterfold.py \\
        --n_train 80 --n_test 14 --n_steps 5000 \\
        --output_dir outputs/iterfold
"""

import sys
import os
import random
import time
from datetime import datetime
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq

# Shared utilities from script_utils
from script_utils import (
    Logger,
    set_seed,
    save_config,
    get_data_path,
    plot_prediction,
)

# Training utilities from tinyfold.training
from tinyfold.training import (
    load_sample_raw,
    collate_batch,
    random_rotation_matrix,
    apply_rigid_augment,
    apply_rotation_augment,
)

# Model imports
from tinyfold.model.iterfold import IterFold
from tinyfold.model.metrics import compute_dockq
from tinyfold.model.resfold.clustering import select_next_residues_to_place

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
)
from tinyfold.model.losses.geometry import GeometryLoss, pairwise_distance_loss

from tinyfold.training.data_split import (
    DataSplitConfig, get_train_test_indices, get_split_info, save_split, load_split,
)


# =============================================================================
# Masking Strategy
# =============================================================================

def sample_training_masks(
    B: int,
    L: int,
    chain_ids: torch.Tensor,
    mask_res: torch.Tensor,
    device: torch.device,
    mask_ratio_min: float = 0.5,
    mask_ratio_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Sample masks for training.

    Strategy:
    - Sample mask_ratio uniformly from [mask_ratio_min, mask_ratio_max]
    - Randomly mask that fraction of residues
    - Unmasked (visible) residues get GT centroids as input
    - Returns visible_mask and neighbor_mask (masked neighbors of visible)

    Args:
        B: Batch size
        L: Sequence length
        chain_ids: Chain IDs [B, L]
        mask_res: Valid residue mask [B, L]
        device: torch device
        mask_ratio_min: Min fraction of residues to mask (default 0.5)
        mask_ratio_max: Max fraction of residues to mask (default 1.0)

    Returns:
        visible_mask: [B, L] bool - residues with GT centroid input
        neighbor_mask: [B, L] bool - masked residues that neighbor visible ones
        mask_ratio: float - the sampled mask ratio for this batch
    """
    # Sample mask ratio uniformly
    mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)

    # Randomly select which residues are visible (unmasked)
    visible_mask = torch.rand(B, L, device=device) > mask_ratio
    visible_mask = visible_mask & mask_res  # Only valid residues

    # Ensure at least 1 residue is visible per sample (prevents MSE=0 edge case)
    for b in range(B):
        if not visible_mask[b].any():
            valid_indices = mask_res[b].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                # Pick a random valid residue to be visible
                idx = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                visible_mask[b, idx] = True

    # Find masked neighbors of visible residues (in same chain)
    visible_pad = F.pad(visible_mask.float(), (1, 1), value=0)  # [B, L+2]
    chain_pad = F.pad(chain_ids, (1, 1), value=-1)  # [B, L+2]

    # Left neighbor is visible: visible[i-1] -> neighbor at i
    left_visible = visible_pad[:, :-2].bool()  # [B, L]
    same_chain_left = (chain_ids == chain_pad[:, :-2])

    # Right neighbor is visible: visible[i+1] -> neighbor at i
    right_visible = visible_pad[:, 2:].bool()  # [B, L]
    same_chain_right = (chain_ids == chain_pad[:, 2:])

    # Neighbor mask: neighbors of visible, in same chain, NOT visible itself, and valid
    neighbor_of_visible = (left_visible & same_chain_left) | (right_visible & same_chain_right)
    neighbor_mask = neighbor_of_visible & ~visible_mask & mask_res

    return visible_mask, neighbor_mask, mask_ratio


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    model: IterFold,
    batch: dict,
    args,
    geometry_loss_fn: GeometryLoss = None,
) -> dict:
    """Training step for IterFold.

    Masking strategy:
    - Mask `mask_ratio` fraction of residues randomly
    - Visible (unmasked) residues get GT centroids (or atoms) as input
    - Loss computed on BOTH visible and neighbor residues

    Steps:
    1. Apply rotation augmentation (if enabled)
    2. Sample visible/neighbor masks
    3. Build input positions (GT for visible, zeros for masked)
    4. Predict atoms
    5. Kabsch align for rotation-invariant loss
    6. Compute separate losses for visible and neighbor residues
    7. Compute geometry loss (bond lengths, angles)
    """
    model.train()

    B, L = batch['aa_seq'].shape
    device = batch['centroids'].device

    # Get ground truth
    gt_atoms = batch['coords_res']      # [B, L, 4, 3]
    gt_centroids = batch['centroids']   # [B, L, 3]

    # 1. Rigid augmentation: rotation + translation (if enabled)
    if args.rotation_augment:
        gt_atoms, gt_centroids = apply_rigid_augment(gt_atoms, gt_centroids, translation_scale=2.0)

    # 2. Sample masks (mask_ratio varies uniformly from min to max)
    visible_mask, neighbor_mask, mask_ratio = sample_training_masks(
        B, L, batch['chain_ids'], batch['mask_res'], device,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
    )

    # 3. Build input: GT for visible, zeros for masked
    if args.use_atom_anchor_decoder:
        # Atom-based: pass atom positions [B, L, 4, 3]
        visible_expanded = visible_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, L, 1, 1]
        anchor_input = gt_atoms * visible_expanded  # [B, L, 4, 3]
    else:
        # Centroid-based: pass centroids [B, L, 3]
        anchor_input = gt_centroids * visible_mask.unsqueeze(-1).float()  # [B, L, 3]

    # 4. Predict atoms
    pred_atoms = model(
        batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
        anchor_input, batch['mask_res']
    )

    # 5. Direct loss in rotated frame (no Kabsch during training!)
    # 
    # The model receives anchor positions in the rotated frame (after rotation augmentation).
    # It should predict atoms in that same frame. Kabsch alignment during training is WRONG
    # because it allows degenerate solutions (e.g., Z-axis collapse) to achieve low loss.
    # Kabsch is only used during evaluation for fair comparison when no anchors are given.

    # 6. Compute per-atom squared errors
    diff_sq = (pred_atoms - gt_atoms).pow(2).sum(dim=-1)  # [B, L, 4]
    valid_atom_mask = batch['mask_res'].unsqueeze(-1).expand(-1, -1, 4)

    # Visible loss: teaches model to correctly use anchor positions
    # (anchor_pos + small_offset -> prediction should match GT)
    visible_atom_mask = visible_mask.unsqueeze(-1).expand(-1, -1, 4) & valid_atom_mask
    visible_count = visible_atom_mask.sum().clamp(min=1)
    loss_visible = (diff_sq * visible_atom_mask).sum() / visible_count

    # Neighbor loss: prediction of masked neighbors
    neighbor_atom_mask = neighbor_mask.unsqueeze(-1).expand(-1, -1, 4) & valid_atom_mask
    neighbor_count = neighbor_atom_mask.sum().clamp(min=1)
    loss_neighbor = (diff_sq * neighbor_atom_mask).sum() / neighbor_count

    # MSE loss: visible (anchor usage) + neighbor (propagation)
    # Both are needed for proper learning
    mse_loss = loss_visible + loss_neighbor

    # 7. Geometry loss (bond lengths, angles) to enforce proper backbone structure
    geom_loss = 0.0
    geom_losses = {}
    if geometry_loss_fn is not None and args.geom_weight > 0:
        # Use mean std for the batch (normalized coords)
        mean_std = sum(batch['stds']) / len(batch['stds'])
        geom_losses = geometry_loss_fn(
            pred_atoms,
            mask=batch['mask_res'],
            gt_coords=gt_atoms,
            coord_std=mean_std,
        )
        geom_loss = geom_losses['total'] * args.geom_weight

    # 8. Pairwise distance loss to enforce global structure
    dist_loss = 0.0
    if args.dist_weight > 0:
        dist_loss = pairwise_distance_loss(
            pred_atoms, gt_atoms,
            mask=batch['mask_res'],
            n_sample=64,
        ) * args.dist_weight

    # 9. Centroid loss (helps frame decoder learn positions)
    centroid_loss = 0.0
    if args.centroid_weight > 0:
        # GT and pred centroids
        gt_centroid = gt_atoms.mean(dim=2)  # [B, L, 3]
        pred_centroid = pred_atoms.mean(dim=2)  # [B, L, 3]
        
        # Compute centroid MSE (only on valid residues)
        centroid_diff_sq = (pred_centroid - gt_centroid).pow(2).sum(dim=-1)  # [B, L]
        valid_count = batch['mask_res'].sum().clamp(min=1)
        centroid_loss = (centroid_diff_sq * batch['mask_res']).sum() / valid_count
        centroid_loss = centroid_loss * args.centroid_weight

    total_loss = mse_loss + geom_loss + dist_loss + centroid_loss

    # Stats
    n_visible = visible_mask.sum().item()
    n_neighbor = neighbor_mask.sum().item()
    n_valid = batch['mask_res'].sum().item()

    result = {
        'total': total_loss,
        'loss_mse': mse_loss.item(),
        'loss_centroid': centroid_loss.item() if isinstance(centroid_loss, torch.Tensor) else centroid_loss,
        'loss_vis': loss_visible.item(),
        'loss_nbr': loss_neighbor.item(),
        'loss_dist': dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss,
        'visible_ratio': n_visible / n_valid if n_valid > 0 else 0,
        'neighbor_ratio': n_neighbor / n_valid if n_valid > 0 else 0,
        'mask_ratio': mask_ratio,
    }

    # Add geometry loss components
    if geom_losses:
        result['loss_geom'] = geom_losses['total'].item()
        result['loss_bond'] = geom_losses.get('bond_length', torch.tensor(0.0)).item()
        result['loss_angle'] = geom_losses.get('bond_angle', torch.tensor(0.0)).item()

    return result


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def iterative_inference(
    model: IterFold,
    batch: dict,
    device: torch.device,
    max_iterations: int = 100,
    use_atom_anchor: bool = False,
) -> torch.Tensor:
    """Iterative structure prediction starting from a random seed residue.

    Algorithm:
    1. Pick a random residue as seed, give it a random position
    2. Predict all atoms using current anchors
    3. Find neighbors of anchored residues that are not yet anchored
    4. Add them as new anchors using predicted centroids/atoms
    5. Repeat until all residues are anchored

    Args:
        model: IterFold model
        batch: Batch dict with aa_seq, chain_ids, res_idx, mask_res
        device: torch device
        max_iterations: Safety limit on iterations
        use_atom_anchor: If True, use atom positions for anchoring (for AtomAnchorDecoder)

    Returns:
        pred_atoms: [1, L, 4, 3] predicted atom coordinates
    """
    L = batch['aa_seq'].shape[1]
    mask = batch['mask_res'][0]  # [L]
    chain_ids = batch['chain_ids'][0]  # [L]
    valid_indices = mask.nonzero(as_tuple=True)[0]

    # Initialize anchors based on decoder type
    if use_atom_anchor:
        # Atom-based: [1, L, 4, 3]
        anchor_input = torch.zeros(1, L, 4, 3, device=device)
    else:
        # Centroid-based: [1, L, 3]
        anchor_input = torch.zeros(1, L, 3, device=device)
    anchored = torch.zeros(L, dtype=torch.bool, device=device)

    # Pick random seed residue and give it a random position (normalized coords)
    seed_idx = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
    if use_atom_anchor:
        # Initialize seed with small random atom positions
        anchor_input[0, seed_idx] = torch.randn(4, 3, device=device) * 0.1
    else:
        anchor_input[0, seed_idx] = torch.randn(3, device=device) * 0.1
    anchored[seed_idx] = True

    for iteration in range(max_iterations):
        # Predict atoms with current anchors
        pred_atoms = model(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            anchor_input, batch['mask_res']
        )  # [1, L, 4, 3]

        # Find neighbors of anchored residues (in same chain) that are not yet anchored
        new_anchors = torch.zeros(L, dtype=torch.bool, device=device)
        for i in range(L):
            if anchored[i] and mask[i]:
                # Check left neighbor
                if i > 0 and mask[i-1] and chain_ids[i-1] == chain_ids[i] and not anchored[i-1]:
                    new_anchors[i-1] = True
                # Check right neighbor
                if i < L-1 and mask[i+1] and chain_ids[i+1] == chain_ids[i] and not anchored[i+1]:
                    new_anchors[i+1] = True

        if not new_anchors.any():
            # No new neighbors to add - we're done (or disconnected chains)
            break

        # Add new anchors using predicted positions
        anchored = anchored | new_anchors
        if use_atom_anchor:
            # Use predicted atoms directly
            anchor_input[0, new_anchors] = pred_atoms[0, new_anchors]
        else:
            # Use predicted centroids
            pred_centroids = pred_atoms[0].mean(dim=1)  # [L, 3]
            anchor_input[0, new_anchors] = pred_centroids[new_anchors]

    # Final prediction with all anchors
    pred_atoms = model(
        batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
        anchor_input, batch['mask_res']
    )

    return pred_atoms


@torch.no_grad()
def evaluate(
    model: IterFold,
    samples: dict,
    indices: list,
    device: torch.device,
    args,
    compute_dockq_scores: bool = False,
) -> dict:
    """Evaluate model using one-shot inference (ALL masked, no anchors).

    Predicts structure blind without any GT information.
    Uses Kabsch alignment for rotation-invariant RMSE comparison.

    Args:
        compute_dockq_scores: If True, also compute DockQ (slower)
    """
    model.eval()

    atom_rmses = []
    dockq_scores = []

    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)
        L = batch['aa_seq'].shape[1]
        n_res = s['n_res']

        gt_res = batch['coords_res'][0, :n_res]  # [L, 4, 3]
        gt_flat = gt_res.view(-1, 3)
        mask_flat = torch.ones(1, gt_flat.shape[0], dtype=torch.bool, device=device)

        # === One-shot inference (ALL masked, no anchors) ===
        if args.use_atom_anchor_decoder:
            # Atom-based: zeros [1, L, 4, 3]
            anchor_input = torch.zeros(1, L, 4, 3, device=device)
        else:
            # Centroid-based: zeros [1, L, 3]
            anchor_input = torch.zeros(1, L, 3, device=device)
        pred_atoms = model(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            anchor_input, batch['mask_res']
        )
        pred_flat = pred_atoms[0, :n_res].view(-1, 3)
        gt_aligned, pred_aligned = kabsch_align(
            gt_flat.unsqueeze(0), pred_flat.unsqueeze(0), mask_flat
        )
        rmse = compute_rmse(pred_aligned, gt_aligned).item() * s['std']
        atom_rmses.append(rmse)

        # Compute DockQ if requested (on one-shot prediction)
        if compute_dockq_scores:
            dockq_result = compute_dockq(
                pred_atoms[0, :n_res], gt_res,
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


@torch.no_grad()
def evaluate_final(
    model: IterFold,
    samples: dict,
    indices: list,
    device: torch.device,
    args,
) -> dict:
    """Final evaluation with one-shot prediction and full DockQ metrics.

    Uses one-shot prediction (no anchors) - same as training eval.
    """
    model.eval()

    atom_rmses = []
    dockq_scores = []
    fnat_scores = []
    irms_scores = []
    lrms_scores = []

    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)

        # One-shot prediction: NO anchors
        L = batch['aa_seq'].shape[1]
        if args.use_atom_anchor_decoder:
            # Atom-based: [1, L, 4, 3]
            anchor_input = torch.zeros(1, L, 4, 3, device=device)
        else:
            # Centroid-based: [1, L, 3]
            anchor_input = torch.zeros(1, L, 3, device=device)

        pred_atoms = model(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            anchor_input, batch['mask_res']
        )

        n_res = s['n_res']
        pred_res = pred_atoms[0, :n_res]
        gt_res = batch['coords_res'][0, :n_res]

        # RMSE with Kabsch alignment
        pred_flat = pred_res.view(-1, 3)
        gt_flat = gt_res.view(-1, 3)
        mask_flat = torch.ones(1, pred_flat.shape[0], dtype=torch.bool, device=device)
        gt_aligned, pred_aligned = kabsch_align(
            gt_flat.unsqueeze(0), pred_flat.unsqueeze(0), mask_flat
        )
        rmse = compute_rmse(pred_aligned, gt_aligned).item() * s['std']
        atom_rmses.append(rmse)

        # DockQ
        dockq_result = compute_dockq(
            pred_res, gt_res,
            batch['aa_seq'][0, :n_res],
            batch['chain_ids'][0, :n_res],
            std=s['std']
        )
        if dockq_result['dockq'] is not None:
            dockq_scores.append(dockq_result['dockq'])
            fnat_scores.append(dockq_result['fnat'])
            irms_scores.append(dockq_result['irms'])
            lrms_scores.append(dockq_result['lrms'])

    result = {
        'atom_rmse': np.mean(atom_rmses),
        'atom_rmse_std': np.std(atom_rmses),
    }

    if dockq_scores:
        result['dockq'] = np.mean(dockq_scores)
        result['dockq_std'] = np.std(dockq_scores)
        result['fnat'] = np.mean(fnat_scores)
        result['irms'] = np.mean(irms_scores)
        result['lrms'] = np.mean(lrms_scores)
        result['n_dockq_valid'] = len(dockq_scores)

        # DockQ quality bins
        dockq_arr = np.array(dockq_scores)
        result['pct_acceptable'] = 100 * (dockq_arr >= 0.23).mean()
        result['pct_medium'] = 100 * (dockq_arr >= 0.49).mean()
        result['pct_high'] = 100 * (dockq_arr >= 0.80).mean()

    return result


# Visualization: plot_prediction imported from script_utils


# =============================================================================
# Main
# =============================================================================

def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="YAML config profile for default arguments")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="IterFold Training")

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

    # Model config
    parser.add_argument("--c_token", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=9)
    parser.add_argument("--decoder_layers", type=int, default=12)
    parser.add_argument("--n_atom_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--use_geometric_decoder", action="store_true",
                        help="Use new geometric atom decoder with backbone priors")
    parser.add_argument("--use_frame_decoder", action="store_true",
                        help="Use frame-based decoder (centroid + rotation)")
    parser.add_argument("--use_atom_anchor_decoder", action="store_true",
                        help="Use atom-based anchor decoder (conditions on atom positions, not centroids)")
    parser.add_argument("--c_atom", type=int, default=128,
                        help="Atom feature dimension (for geometric/atom anchor decoder)")

    # Masking
    parser.add_argument("--mask_ratio_min", type=float, default=0.5,
                        help="Min fraction of residues to mask (default 0.5)")
    parser.add_argument("--mask_ratio_max", type=float, default=1.0,
                        help="Max fraction of residues to mask (default 1.0)")

    # Geometry loss
    parser.add_argument("--geom_weight", type=float, default=1.0,
                        help="Weight for geometry loss (bond lengths, angles). Default 1.0")
    parser.add_argument("--dist_weight", type=float, default=1.0,
                        help="Weight for pairwise distance loss (global structure). Default 1.0")
    parser.add_argument("--centroid_weight", type=float, default=0.0,
                        help="Weight for centroid loss (frame decoder). Default 0.0")

    # Augmentation
    parser.add_argument("--rotation_augment", action="store_true", default=True,
                        help="Apply SO(3) rotation augmentation with Kabsch-aligned loss (default: True)")
    parser.add_argument("--no_rotation_augment", action="store_false", dest="rotation_augment",
                        help="Disable rotation augmentation")

    # Inference
    parser.add_argument("--n_inference_iters", type=int, default=10)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/iterfold")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config profile for default arguments")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            config_defaults = yaml.safe_load(f) or {}
        valid_dests = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in config_defaults.items() if k in valid_dests}
        parser.set_defaults(**filtered)

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility: set seed before anything else
    set_seed(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save config at startup (before training, in case of crash)
    save_config(args, args.output_dir)

    logger = Logger(os.path.join(args.output_dir, 'train.log'))

    # Header
    logger.log("=" * 70)
    logger.log("IterFold Training (Anchor-Conditioned Structure Prediction)")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Config
    decoder_type = "atom_anchor" if args.use_atom_anchor_decoder else \
                   "frame" if args.use_frame_decoder else \
                   "geometric" if args.use_geometric_decoder else "anchor"
    logger.log("Configuration:")
    logger.log(f"  decoder_type:     {decoder_type}")
    logger.log(f"  c_token:          {args.c_token}")
    logger.log(f"  trunk_layers:     {args.trunk_layers}")
    logger.log(f"  decoder_layers:   {args.decoder_layers}")
    logger.log(f"  n_atom_layers:    {args.n_atom_layers}")
    logger.log(f"  batch_size:       {args.batch_size}")
    logger.log(f"  grad_accum:       {args.grad_accum}")
    logger.log(f"  eff_batch:        {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:          {args.n_steps}")
    logger.log(f"  lr:               {args.lr}")
    logger.log("")
    logger.log("Masking:")
    logger.log(f"  mask_ratio:       [{args.mask_ratio_min}, {args.mask_ratio_max}] (uniform sampling)")
    logger.log("")
    logger.log("Auxiliary losses:")
    logger.log(f"  geom_weight:      {args.geom_weight}")
    logger.log(f"  dist_weight:      {args.dist_weight}")
    logger.log("")
    logger.log("Augmentation:")
    logger.log(f"  rotation_augment: {args.rotation_augment}")
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
            seed=args.seed,
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
    model = IterFold(
        c_token=args.c_token,
        trunk_layers=args.trunk_layers,
        decoder_layers=args.decoder_layers,
        n_atom_layers=args.n_atom_layers,
        trunk_heads=args.n_heads,
        decoder_heads=args.n_heads,
        use_geometric_decoder=args.use_geometric_decoder,
        use_frame_decoder=args.use_frame_decoder,
        use_atom_anchor_decoder=args.use_atom_anchor_decoder,
        c_atom=args.c_atom,
    ).to(device)

    params = model.count_parameters()
    logger.log(f"Model: IterFold")
    logger.log(f"  Total params:   {params['total']:,}")
    logger.log(f"  Trunk params:   {params['trunk']:,} ({params['trunk_pct']:.1f}%)")
    logger.log(f"  Decoder params: {params['decoder']:,} ({params['decoder_pct']:.1f}%)")
    logger.log("")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.min_lr
    )
    logger.log(f"Optimizer: AdamW, {params['total']:,} trainable params")

    # Geometry loss (enforces bond lengths, angles for proper backbone structure)
    geometry_loss_fn = GeometryLoss(
        bond_length_weight=1.0,
        bond_angle_weight=1.0,
        omega_weight=0.5,
        o_chirality_weight=0.5,
        cb_chirality_weight=0.0,
        bound_losses=True,
    )
    logger.log(f"Geometry loss: {geometry_loss_fn}")

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    best_dockq = 0.0
    start_time = time.time()

    history = {
        'step': [], 'loss': [], 'visible_ratio': [], 'loss_ratio': [],
        'train_rmse': [], 'test_rmse': [], 'test_dockq': [],
    }

    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_losses = {}

        for accum_step in range(args.grad_accum):
            batch_indices = random.choices(train_indices, k=args.batch_size)
            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(batch_samples, device)

            losses = train_step(model, batch, args, geometry_loss_fn)
            loss = losses['total'] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if accum_step == args.grad_accum - 1:
                accum_losses = losses

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            lc = accum_losses
            log_str = (
                f"Step {step:5d} | loss: {accum_loss:.4f} | "
                f"mse: {lc.get('loss_mse', 0):.4f} | "
                f"dist: {lc.get('loss_dist', 0):.4f} | "
                f"geom: {lc.get('loss_geom', 0):.4f} | "
                f"m%: {lc.get('mask_ratio', 0):.2f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
            )
            logger.log(log_str)

        # Evaluation (genuine iterative inference from scratch - no GT info)
        if step % args.eval_every == 0:
            n_eval_train = min(args.n_eval_train, len(train_indices))
            eval_train_indices = random.sample(train_indices, n_eval_train)

            train_results = evaluate(
                model, all_samples, eval_train_indices, device, args,
                compute_dockq_scores=False
            )

            test_results = evaluate(
                model, all_samples, test_indices, device, args,
                compute_dockq_scores=True
            )

            train_log = f"         >>> Train ({n_eval_train}): RMSE={train_results['atom_rmse']:.2f}Å"
            logger.log(train_log)

            test_log = f"         >>> Test ({len(test_indices)}):  RMSE={test_results['atom_rmse']:.2f}Å"
            if 'dockq' in test_results:
                test_log += f" | DockQ={test_results['dockq']:.3f}"
            logger.log(test_log)

            # Plot single example prediction (one-shot, ALL masked)
            plot_idx = test_indices[0]
            plot_sample = all_samples[plot_idx]
            plot_batch = collate_batch([plot_sample], device)

            with torch.no_grad():
                L_plot = plot_batch['aa_seq'].shape[1]
                if args.use_atom_anchor_decoder:
                    # Atom-based: zeros [1, L, 4, 3]
                    anchor_input_plot = torch.zeros(1, L_plot, 4, 3, device=device)
                else:
                    # Centroid-based: zeros [1, L, 3]
                    anchor_input_plot = torch.zeros(1, L_plot, 3, device=device)
                plot_pred = model(
                    plot_batch['aa_seq'], plot_batch['chain_ids'], plot_batch['res_idx'],
                    anchor_input_plot, plot_batch['mask_res']
                )

            n_res = plot_sample['n_res']
            pred_flat = plot_pred[0, :n_res].view(-1, 3)
            gt_flat = plot_batch['coords_res'][0, :n_res].view(-1, 3)

            # Kabsch align for visualization
            mask_flat = torch.ones(1, pred_flat.shape[0], dtype=torch.bool, device=device)
            gt_aligned, pred_aligned = kabsch_align(
                gt_flat.unsqueeze(0), pred_flat.unsqueeze(0), mask_flat
            )
            plot_rmse = compute_rmse(pred_aligned, gt_aligned).item() * plot_sample['std']

            # Expand chain_ids to atom level for plotting
            chain_ids_plot = plot_batch['chain_ids'][0, :n_res].unsqueeze(-1).expand(-1, 4).reshape(-1)

            plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
            plot_prediction(
                pred_aligned[0], gt_aligned[0], chain_ids_plot,
                plot_sample['sample_id'], plot_rmse, plot_path
            )
            logger.log(f"         >>> Saved plot: {plot_path}")

            # Track history
            history['step'].append(step)
            history['loss'].append(accum_loss)
            history['visible_ratio'].append(accum_losses.get('visible_ratio', 0))
            history['loss_ratio'].append(accum_losses.get('loss_ratio', 0))
            history['train_rmse'].append(train_results['atom_rmse'])
            history['test_rmse'].append(test_results['atom_rmse'])
            history['test_dockq'].append(test_results.get('dockq', 0))

            # Save best model
            if test_results['atom_rmse'] < best_rmse:
                best_rmse = test_results['atom_rmse']
                best_dockq = test_results.get('dockq', 0)
                save_dict = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'test_atom_rmse': test_results['atom_rmse'],
                    'test_dockq': test_results.get('dockq'),
                    'args': vars(args),
                }
                torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
                dockq_str = f", DockQ={test_results['dockq']:.3f}" if test_results.get('dockq') else ""
                logger.log(f"         >>> New best! Saved.{dockq_str}")

    # Final evaluation with full DockQ
    logger.log("")
    logger.log("=" * 70)
    logger.log("Final Evaluation (one-shot, no anchors)")
    logger.log("=" * 70)

    final_results = evaluate_final(model, all_samples, test_indices, device, args)

    logger.log(f"Test set ({len(test_indices)} samples):")
    logger.log(f"  RMSE:  {final_results['atom_rmse']:.2f} +/- {final_results['atom_rmse_std']:.2f} Å")
    if 'dockq' in final_results:
        logger.log(f"  DockQ: {final_results['dockq']:.3f} +/- {final_results['dockq_std']:.3f}")
        logger.log(f"  Fnat:  {final_results['fnat']:.3f}")
        logger.log(f"  iRMS:  {final_results['irms']:.2f} A")
        logger.log(f"  LRMS:  {final_results['lrms']:.2f} A")
        logger.log("")
        logger.log("  Quality distribution:")
        logger.log(f"    Acceptable (>=0.23): {final_results['pct_acceptable']:.1f}%")
        logger.log(f"    Medium (>=0.49):     {final_results['pct_medium']:.1f}%")
        logger.log(f"    High (>=0.80):       {final_results['pct_high']:.1f}%")

    # Save final checkpoint
    torch.save({
        'step': args.n_steps,
        'model_state_dict': model.state_dict(),
        'final_results': final_results,
        'args': vars(args),
    }, os.path.join(args.output_dir, 'final_model.pt'))

    # Plot training curves
    if history['step']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(history['step'], history['loss'], label='Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training Loss')

        # RMSE
        axes[0, 1].plot(history['step'], history['train_rmse'], label='Train')
        axes[0, 1].plot(history['step'], history['test_rmse'], label='Test')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('RMSE (A)')
        axes[0, 1].legend()
        axes[0, 1].set_title('Atom RMSE')

        # DockQ
        axes[1, 0].plot(history['step'], history['test_dockq'])
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('DockQ')
        axes[1, 0].set_title('Test DockQ')
        axes[1, 0].axhline(y=0.23, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        axes[1, 0].axhline(y=0.49, color='green', linestyle='--', alpha=0.5, label='Medium')
        axes[1, 0].axhline(y=0.80, color='blue', linestyle='--', alpha=0.5, label='High')
        axes[1, 0].legend()

        # Empty plot for summary
        axes[1, 1].axis('off')
        summary_text = (
            f"Final Results\n"
            f"─────────────\n"
            f"Best RMSE: {best_rmse:.2f} A\n"
            f"Best DockQ: {best_dockq:.3f}\n\n"
            f"Iterative Eval\n"
            f"─────────────\n"
            f"RMSE: {final_results['atom_rmse']:.2f} A\n"
        )
        if 'dockq' in final_results:
            summary_text += (
                f"DockQ: {final_results['dockq']:.3f}\n"
                f"Fnat: {final_results['fnat']:.3f}\n"
            )
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=150)
        plt.close()

    # Final summary
    total_time = time.time() - start_time
    logger.log("")
    logger.log("=" * 70)
    logger.log("Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.4f} A")
    logger.log(f"  Best test DockQ: {best_dockq:.4f}")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()
