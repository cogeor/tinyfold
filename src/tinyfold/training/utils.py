"""Training loss-weighting utilities."""

import warnings

import torch
from torch import Tensor


def edm_loss_weight(sigma: Tensor, sigma_data: float = 1.0) -> Tensor:
    """EDM/Karras 2022 per-sample loss weighting.

    Reference: Karras, Aittala, Aila, Laine (2022),
      "Elucidating the Design Space of Diffusion-Based Generative Models",
      NeurIPS 2022, Eq. 7 (the "effective weight" lambda(sigma)).

    Closed form:
        lambda(sigma) = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2

    Derivation: in EDM preconditioning, the model's output is scaled by
    c_out(sigma) = sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)
    (see ResFoldOneStep._edm_coefficients). The training MSE on the raw
    network output F is therefore implicitly multiplied by c_out**2 when
    measured in data space. To make every noise level contribute equally
    to gradient signal we multiply the data-space MSE by 1 / c_out**2,
    which is exactly the lambda above.

    Asymptotics (sigma_data = 1):
        sigma -> 0   : lambda -> 1 / sigma**2   (blows up; rescues small-sigma
                                                 samples that c_out squashes)
        sigma -> inf : lambda -> 1               (high-sigma samples already
                                                 have full gradient magnitude)
        sigma = 1    : lambda = 2

    Args:
        sigma:      [B] per-sample noise level (sigma, not log-sigma).
        sigma_data: float, must match the model's sigma_data (1.0 throughout
                    tinyfold; see onestep.py:134, denoiser.py:295).

    Returns:
        weight: [B] per-sample loss weights. MUST be multiplied at the
                per-sample MSE level, NOT after reducing across the batch.
    """
    # Tiny epsilon on sigma only (sigma_data is a known positive constant);
    # protects against the schedule occasionally returning sigma == 0.
    sigma_safe = sigma.clamp(min=1e-8)
    return (sigma_safe ** 2 + sigma_data ** 2) / (sigma_safe * sigma_data) ** 2


def af3_loss_weight(sigma: Tensor, sigma_data: float = 1.0) -> Tensor:
    """Deprecated alias for edm_loss_weight (kept for old call sites)."""
    warnings.warn(
        "af3_loss_weight is deprecated; use edm_loss_weight (same formula, "
        "corrected from the buggy (sigma+sigma_data)**2 denominator).",
        DeprecationWarning,
        stacklevel=2,
    )
    return edm_loss_weight(sigma, sigma_data)
