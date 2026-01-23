"""MSE-based losses with Kabsch alignment for structure prediction.

Provides rotation-invariant loss functions for comparing predicted
and ground truth protein structures.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


def kabsch_align(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Kabsch alignment for rotation-invariant comparison.

    Computes optimal rotation to align pred to target using SVD.
    Both tensors are centered before alignment.

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        pred_aligned: Aligned predicted coordinates [B, N, 3]
        target_centered: Centered target coordinates [B, N, 3]
    """
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

    # SVD for optimal rotation
    H = torch.bmm(pred_c.transpose(1, 2), target_c)
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=pred.device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))
    pred_aligned = torch.bmm(pred_c, R.transpose(1, 2))

    return pred_aligned, target_c


def compute_mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    use_kabsch: bool = True,
) -> Tensor:
    """MSE loss with optional Kabsch alignment.

    Uses per-sample averaging for correct gradient accumulation behavior.
    Each sample contributes equally regardless of sequence length.

    IMPORTANT: Kabsch aligns TARGET to PRED (not pred to target).
    This is correct because:
    - The model outputs predictions in its own coordinate frame
    - With rotation augmentation, target is in a random rotated frame
    - We align target to pred's frame so the model learns the correct SHAPE
    - If we aligned pred to target, the model couldn't learn which frame to use

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]
        use_kabsch: Whether to apply Kabsch alignment (default True)

    Returns:
        loss: Scalar MSE loss (averaged per-sample, then across batch)
    """
    if use_kabsch:
        # Align TARGET to PRED's frame (not pred to target!)
        target_aligned, pred_c = kabsch_align(target, pred, mask)
        # CRITICAL: Detach aligned target so gradients don't flow through Kabsch SVD
        # This follows Boltz implementation - gradients should only flow through pred
        target_aligned = target_aligned.detach()
    else:
        # Direct MSE without alignment - just center both
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_c = pred - pred_mean
        target_aligned = target - target_mean

    sq_diff = ((pred_c - target_aligned) ** 2).sum(dim=-1)  # [B, N]

    if mask is not None:
        # Per-sample loss: average over valid positions within each sample
        n_valid_per_sample = mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample_loss = (sq_diff * mask.float()).sum(dim=1) / n_valid_per_sample  # [B]
        # Average across samples
        loss = per_sample_loss.mean()
    else:
        loss = sq_diff.mean()

    return loss


def compute_rmse(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """RMSE after Kabsch alignment.

    Aligns TARGET to PRED's frame (consistent with compute_mse_loss).

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        rmse: Scalar RMSE value
    """
    # Align target to pred's frame (not pred to target!)
    target_aligned, pred_c = kabsch_align(target, pred, mask)
    target_aligned = target_aligned.detach()  # Consistent with compute_mse_loss
    sq_diff = ((pred_c - target_aligned) ** 2).sum(dim=-1)

    if mask is not None:
        rmse = torch.sqrt((sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1))
    else:
        rmse = torch.sqrt(sq_diff.mean())

    return rmse


def compute_distance_consistency_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Loss for preserving pairwise distances.

    Encourages predicted coordinates to have similar pairwise distances
    as the ground truth. This is rotation/translation invariant without
    explicit alignment.

    Uses per-sample averaging for correct gradient accumulation behavior.

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., centroids)
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        loss: Scalar distance consistency loss (averaged per-sample, then across batch)
    """
    # Compute pairwise distances [B, N, N]
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    # MSE on distances
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        # Create pairwise mask [B, N, N]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        # Per-sample loss: average over valid pairs within each sample
        n_valid_pairs_per_sample = pair_mask.sum(dim=(1, 2)).clamp(min=1)  # [B]
        per_sample_loss = (dist_diff * pair_mask.float()).sum(dim=(1, 2)) / n_valid_pairs_per_sample  # [B]
        # Average across samples
        loss = per_sample_loss.mean()
    else:
        loss = dist_diff.mean()

    return loss
