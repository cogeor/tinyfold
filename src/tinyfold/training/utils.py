"""Training utilities for AF3-style diffusion training.

This module provides:
1. Random rigid augmentation (rotation + translation)
2. Multi-copy training with trunk reuse (48-copy optimization)
3. Loss weighting schemes

Usage:
    from models.training_utils import (
        random_rigid_augment,
        MultiCopyTrainer,
        af3_loss_weight,
    )

    # Simple augmentation
    x_aug = random_rigid_augment(x, mask)

    # Multi-copy training (AF3-style)
    trainer = MultiCopyTrainer(n_copies=48)
    loss = trainer.train_step(model, batch, noiser, compute_loss_fn)
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, Tuple



from .augmentation import random_rotation_matrix, apply_rigid_augment as random_rigid_augment


def af3_loss_weight(sigma: Tensor, sigma_data: float = 1.0) -> Tensor:
    """AF3/EDM-style loss weighting based on noise level.

    From EDM paper (Karras et al. 2022): λ(σ) = (σ² + σ_data²) / (σ + σ_data)²
    This gives weights in range [0.5, 1.0] - higher for extreme noise levels.

    Args:
        sigma: Noise level [B]
        sigma_data: Data standard deviation (1.0 for normalized coords)

    Returns:
        weight: [B] loss weights
    """
    return (sigma**2 + sigma_data**2) / (sigma + sigma_data + 1e-8)**2


def timestep_to_sigma(t: Tensor, noiser) -> Tensor:
    """Convert discrete timestep to equivalent sigma.

    For VP diffusion: sigma^2 = (1 - alpha_bar) / alpha_bar

    Args:
        t: Timestep indices [B]
        noiser: Diffusion noiser with alpha_bar schedule

    Returns:
        sigma: Equivalent noise level [B]
    """
    alpha_bar = noiser.alpha_bar[t]
    sigma = torch.sqrt((1 - alpha_bar) / alpha_bar.clamp(min=1e-8))
    return sigma


class MultiCopyTrainer:
    """AF3-style multi-copy training with trunk reuse.

    The key insight: trunk (encoder) only needs to run ONCE per sample,
    but we can run the denoiser on MANY augmented copies with different:
    - Random rotations/translations
    - Different noise levels (timesteps)
    - Different noise realizations

    This is much more efficient than running the full model N times.
    """

    def __init__(
        self,
        n_copies: int = 48,
        augment_rotation: bool = True,
        augment_translation: float = 0.0,
    ):
        """
        Args:
            n_copies: Number of augmented copies per sample
            augment_rotation: Apply random rotation to each copy
            augment_translation: Scale of random translation (0 = none)
        """
        self.n_copies = n_copies
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        noiser,
        compute_loss_fn: Callable,
        use_loss_weighting: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Single training step with multi-copy optimization.

        Args:
            model: ResidueDenoiser with forward_with_trunk method
            batch: Dict with 'centroids', 'aa_seq', 'chain_ids', 'res_idx', 'mask_res'
            noiser: Diffusion noiser (GaussianNoise or similar)
            compute_loss_fn: Loss function (pred, target, mask) -> loss
            use_loss_weighting: Apply AF3-style loss weighting

        Returns:
            loss: Scalar loss value
            metrics: Dict with intermediate metrics
        """
        device = batch['centroids'].device
        B, L, _ = batch['centroids'].shape
        mask = batch.get('mask_res')

        # === STEP 1: Run trunk ONCE on sequence features (no coordinates!) ===
        with torch.no_grad():
            trunk_tokens = model.get_trunk_tokens(
                batch['aa_seq'],
                batch['chain_ids'],
                batch['res_idx'],
                mask,
            )  # [B, L, c_token]

        # === STEP 2: Create N augmented copies ===
        all_losses = []
        all_timesteps = []

        for _ in range(self.n_copies):
            # Augment clean coordinates
            x0_aug = random_rigid_augment(
                batch['centroids'],
                mask,
                rotation=self.augment_rotation,
                translation_scale=self.augment_translation,
            )

            # Sample timestep
            t = torch.randint(0, noiser.T, (B,), device=device)
            all_timesteps.append(t.float().mean().item())

            # Add noise
            x_t, _ = noiser.add_noise(x0_aug, t)

            # Apply same augmentation to x_t for consistency
            # (Actually, since noise is isotropic, we can add noise THEN augment)
            # But for correctness, we augment x0 first, then add noise

            # === STEP 3: Run denoiser with pre-computed trunk ===
            x0_pred = model.forward_with_trunk(x_t, trunk_tokens, t, mask)

            # === STEP 4: Compute loss ===
            # Target is the augmented clean coordinates
            loss = compute_loss_fn(x0_pred, x0_aug, mask)

            # Optional: AF3-style loss weighting
            if use_loss_weighting:
                sigma = timestep_to_sigma(t, noiser)
                weight = af3_loss_weight(sigma)
                loss = (loss * weight.view(-1, 1, 1)).mean() if loss.dim() > 0 else loss * weight.mean()

            all_losses.append(loss)

        # Average over copies
        total_loss = torch.stack(all_losses).mean()

        metrics = {
            'n_copies': self.n_copies,
            'avg_timestep': sum(all_timesteps) / len(all_timesteps),
        }

        return total_loss, metrics


def expand_for_multi_copy(batch: Dict[str, Tensor], n_copies: int) -> Dict[str, Tensor]:
    """Expand batch tensors for vectorized multi-copy training.

    Instead of looping, expand batch dimension: [B, ...] -> [B * n_copies, ...]
    Then run everything in one forward pass.

    Args:
        batch: Original batch dict
        n_copies: Number of copies per sample

    Returns:
        expanded_batch: Batch with expanded tensors
    """
    expanded = {}
    for key, val in batch.items():
        if isinstance(val, Tensor):
            # Repeat along batch dimension
            # [B, ...] -> [B, n_copies, ...] -> [B * n_copies, ...]
            shape = val.shape
            expanded_val = val.unsqueeze(1).expand(-1, n_copies, *[-1] * (len(shape) - 1))
            expanded[key] = expanded_val.reshape(-1, *shape[1:])
        else:
            expanded[key] = val
    return expanded


class VectorizedMultiCopyTrainer:
    """Vectorized version of multi-copy training (more GPU efficient).

    Instead of looping over copies, we expand the batch and run one forward pass.
    This is faster on GPU but uses more memory.

    Memory usage: O(B * n_copies * L * c_token)
    Speed: ~n_copies times faster than looping
    """

    def __init__(
        self,
        n_copies: int = 48,
        augment_rotation: bool = True,
        augment_translation: float = 0.0,
    ):
        self.n_copies = n_copies
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        noiser,
        compute_loss_fn: Callable,
        use_loss_weighting: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Vectorized multi-copy training step.

        More GPU-efficient than looping, but uses more memory.
        """
        device = batch['centroids'].device
        B, L, _ = batch['centroids'].shape
        mask = batch.get('mask_res')

        # === STEP 1: Run trunk ONCE on sequence features (no coordinates!) ===
        # Don't use no_grad here since we might want to train trunk too
        trunk_tokens = model.get_trunk_tokens(
            batch['aa_seq'],
            batch['chain_ids'],
            batch['res_idx'],
            mask,
        )  # [B, L, c_token]

        # === STEP 2: Expand for n_copies ===
        # trunk_tokens: [B, L, c] -> [B * n_copies, L, c]
        trunk_tokens_exp = trunk_tokens.unsqueeze(1).expand(-1, self.n_copies, -1, -1)
        trunk_tokens_exp = trunk_tokens_exp.reshape(B * self.n_copies, L, -1)

        # Expand other tensors
        centroids_exp = batch['centroids'].unsqueeze(1).expand(-1, self.n_copies, -1, -1)
        centroids_exp = centroids_exp.reshape(B * self.n_copies, L, 3)

        if mask is not None:
            mask_exp = mask.unsqueeze(1).expand(-1, self.n_copies, -1)
            mask_exp = mask_exp.reshape(B * self.n_copies, L)
        else:
            mask_exp = None

        # === STEP 3: Apply different augmentations to each copy ===
        x0_aug = random_rigid_augment(
            centroids_exp,
            mask_exp,
            rotation=self.augment_rotation,
            translation_scale=self.augment_translation,
        )

        # Sample different timesteps for each copy
        t = torch.randint(0, noiser.T, (B * self.n_copies,), device=device)

        # Add noise
        x_t, _ = noiser.add_noise(x0_aug, t)

        # === STEP 4: Run denoiser (vectorized) ===
        x0_pred = model.forward_with_trunk(x_t, trunk_tokens_exp, t, mask_exp)

        # === STEP 5: Compute loss ===
        loss = compute_loss_fn(x0_pred, x0_aug, mask_exp)

        # Optional: AF3-style loss weighting
        if use_loss_weighting:
            sigma = timestep_to_sigma(t, noiser)
            weight = af3_loss_weight(sigma)
            # Weight per sample
            if loss.dim() == 0:
                loss = loss * weight.mean()
            else:
                loss = (loss.view(B * self.n_copies, -1).mean(dim=1) * weight).mean()

        metrics = {
            'n_copies': self.n_copies,
            'effective_batch_size': B * self.n_copies,
            'avg_timestep': t.float().mean().item(),
        }

        return loss, metrics
