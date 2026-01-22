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

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]
        use_kabsch: Whether to apply Kabsch alignment (default True)

    Returns:
        loss: Scalar MSE loss
    """
    if use_kabsch:
        pred_aligned, target_c = kabsch_align(pred, target, mask)
    else:
        # Direct MSE without alignment - just center both
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_aligned = pred - pred_mean
        target_c = target - target_mean

    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

    if mask is not None:
        loss = (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        loss = sq_diff.mean()

    return loss


def compute_rmse(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """RMSE after Kabsch alignment.

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        rmse: Scalar RMSE value
    """
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)

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

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., centroids)
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        loss: Scalar distance consistency loss
    """
    # Compute pairwise distances [B, N, N]
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    # MSE on distances
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        # Create pairwise mask [B, N, N]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        loss = (dist_diff * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1)
    else:
        loss = dist_diff.mean()

    return loss
