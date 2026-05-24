"""Diffusion utilities."""

import torch
from torch import Tensor
from typing import Optional

from tinyfold.model.geometry import kabsch_rigid


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

    Adapter over :func:`tinyfold.model.geometry.kabsch_rigid`. The shared
    helper's ``aligned`` is already in target's translated frame, which is
    exactly what this contract returns.

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., x0_pred from denoiser)
        target: Target frame coordinates [B, N, 3] (e.g., current x_t)
        mask: Optional mask for valid positions [B, N]

    Returns:
        pred_aligned: Prediction aligned to target's frame [B, N, 3]
    """
    _, _, pred_aligned = kabsch_rigid(pred, target, mask)
    return pred_aligned
