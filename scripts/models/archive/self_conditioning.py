"""Self-Conditioning for Diffusion Models.

Self-conditioning helps diffusion models handle distribution shift during inference
by training them to refine their own predictions.

The Problem:
- Training: x_t = ground_truth + sigma * noise (fresh Gaussian noise)
- Inference: x_t = previous_prediction + accumulated_error (not fresh noise)
- At low sigma, the model expects x_t to be very close to ground truth
- But during inference, x_t has structural error from previous steps
- Model fails because it was never trained on this distribution

The Solution (Self-Conditioning):
- During training, with probability p_self_cond, first run the model to get x0_prev
- Then run the model again, conditioning on x0_prev (detached)
- Model learns to refine imperfect predictions
- During inference, always condition on the previous step's prediction

Reference: "Improving Diffusion Model Efficiency Through Patching" and AF3 supplementary.
"""

from typing import Optional, Callable, Tuple
import torch
from torch import Tensor


class SelfConditioningMixin:
    """Mixin class providing self-conditioning functionality for diffusion models.

    To use, inherit from this mixin and implement:
    - _forward_impl(x_t, ..., x0_prev=None) -> x0_pred

    The mixin provides:
    - forward_with_self_cond(): Training forward with self-conditioning
    - sample_with_self_cond(): Inference with self-conditioning
    """

    def _forward_impl(
        self,
        x_t: Tensor,
        x0_prev: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Override this in subclass. Returns x0_pred."""
        raise NotImplementedError


def self_conditioning_training_step(
    model_forward: Callable,
    x_t: Tensor,
    p_self_cond: float = 0.5,
    **forward_kwargs,
) -> Tuple[Tensor, bool]:
    """Execute one training step with optional self-conditioning.

    With probability p_self_cond:
    1. Run model to get x0_prev (detached, no grad)
    2. Run model again with x0_prev as conditioning

    Otherwise:
    1. Run model without self-conditioning (x0_prev = zeros)

    Args:
        model_forward: Function that takes (x_t, x0_prev, **kwargs) -> x0_pred
        x_t: Noisy input [B, L, 3]
        p_self_cond: Probability of using self-conditioning (default 0.5)
        **forward_kwargs: Additional kwargs passed to model_forward

    Returns:
        x0_pred: Model prediction [B, L, 3]
        used_self_cond: Whether self-conditioning was used
    """
    use_self_cond = torch.rand(1).item() < p_self_cond

    if use_self_cond:
        # First pass: get initial prediction (no gradient)
        with torch.no_grad():
            x0_prev = model_forward(x_t, x0_prev=None, **forward_kwargs)
            x0_prev = x0_prev.detach()

        # Second pass: refine with self-conditioning
        x0_pred = model_forward(x_t, x0_prev=x0_prev, **forward_kwargs)
    else:
        # No self-conditioning - pass zeros
        x0_pred = model_forward(x_t, x0_prev=None, **forward_kwargs)

    return x0_pred, use_self_cond


def sample_step_with_self_cond(
    model_forward: Callable,
    x_t: Tensor,
    x0_prev: Optional[Tensor],
    sigma: Tensor,
    sigma_next: Tensor,
    **forward_kwargs,
) -> Tuple[Tensor, Tensor]:
    """Execute one sampling step with self-conditioning.

    Uses the previous x0_pred as conditioning, then applies Euler update.

    Args:
        model_forward: Function that takes (x_t, x0_prev, sigma, **kwargs) -> x0_pred
        x_t: Current noisy state [B, L, 3]
        x0_prev: Previous x0 prediction (or None for first step)
        sigma: Current noise level [B] or scalar
        sigma_next: Next noise level [B] or scalar
        **forward_kwargs: Additional kwargs passed to model_forward

    Returns:
        x_next: Updated state after Euler step [B, L, 3]
        x0_pred: Current x0 prediction (for use in next step)
    """
    # Model prediction with self-conditioning
    x0_pred = model_forward(x_t, x0_prev=x0_prev, sigma=sigma, **forward_kwargs)

    # Euler step: x_next = (sigma_next/sigma) * x_t + (1 - sigma_next/sigma) * x0_pred
    if sigma_next.item() > 0:
        ratio = sigma_next / sigma
        x_next = ratio * x_t + (1 - ratio) * x0_pred
    else:
        x_next = x0_pred

    return x_next, x0_pred


def create_self_cond_embedding(c_token: int) -> torch.nn.Module:
    """Create embedding layer for x0_prev conditioning.

    Returns a module that embeds [B, L, 3] -> [B, L, c_token].
    Uses the same architecture as coord_embed for consistency.

    Args:
        c_token: Token dimension

    Returns:
        nn.Module that embeds 3D coordinates to token space
    """
    return torch.nn.Linear(3, c_token)
