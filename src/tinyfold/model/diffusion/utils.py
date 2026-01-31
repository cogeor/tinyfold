"""Diffusion utilities."""

import torch
from torch import Tensor
from typing import Optional

def kabsch_align_to_target(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Kabsch-align pred INTO target's coordinate frame.

    This is the key fix for diffusion sampling drift (Boltz-1 style).
    Unlike standard kabsch_align() which returns both tensors centered,
    this returns pred transformed to match target's frame exactly.

    The problem: during diffusion sampling, the model predicts x0 in a
    potentially different rigid frame than the current x_t. If we naively
    interpolate (coef1 * x0_pred + coef2 * x_t), the result is warped garbage.

    The fix: before interpolation, Kabsch-align x0_pred to x_t's frame.

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., x0_pred from denoiser)
        target: Target frame coordinates [B, N, 3] (e.g., current x_t)
        mask: Optional mask for valid positions [B, N]

    Returns:
        pred_aligned: Prediction aligned to target's frame [B, N, 3]
    """
    B = pred.shape[0]
    device = pred.device

    # Compute centroids
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()  # [B, N, 1]
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        pred_mean = (pred * mask_exp).sum(dim=1, keepdim=True) / n_valid
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

    # Center both
    pred_c = pred - pred_mean
    target_c = target - target_mean

    if mask is not None:
        pred_c = pred_c * mask_exp
        target_c = target_c * mask_exp

    # SVD for optimal rotation: find R that minimizes ||target_c - pred_c @ R^T||
    H = torch.bmm(pred_c.transpose(1, 2), target_c)  # [B, 3, 3]
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case (ensure proper rotation, not reflection)
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    # Optimal rotation matrix
    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # [B, 3, 3]

    # Apply rotation to centered pred, then translate to target's frame
    pred_rotated = torch.bmm(pred_c, R.transpose(1, 2))  # [B, N, 3]
    pred_aligned = pred_rotated + target_mean  # Translate to target's centroid

    return pred_aligned
