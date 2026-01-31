"""Diffusion samplers for TinyFold.

Samplers handle the reverse diffusion process (inference/generation).
Each sampler implements the same interface and can be swapped easily.

Usage:
    from models.samplers import create_sampler, list_samplers

    # Create a sampler
    sampler = create_sampler("ddpm_kabsch")

    # Sample from model
    x_pred = sampler.sample(model, shape, model_kwargs, noiser, device)

Available samplers:
    - ddpm: Standard DDPM reverse (baseline, original behavior)
    - ddpm_kabsch: DDPM with per-step Kabsch alignment (fixes drift)
    - ddpm_kabsch_recenter: DDPM + Kabsch + recentering (recommended)
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Optional, Callable

from .utils import kabsch_align_to_target


class BaseSampler(ABC):
    """Abstract base class for diffusion samplers."""

    def __init__(
        self,
        clamp_val: float = 3.0,
        align_per_step: bool = False,
        recenter: bool = False,
    ):
        """
        Args:
            clamp_val: Value to clamp predictions to [-clamp_val, clamp_val]
            align_per_step: If True, Kabsch-align x0_pred to x_t each step
            recenter: If True, re-center coordinates after each step
        """
        self.clamp_val = clamp_val
        self.align_per_step = align_per_step
        self.recenter = recenter

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: Dict[str, Any],
        noiser,
        device: torch.device,
        forward_fn: Optional[Callable] = None,
    ) -> Tensor:
        """Run reverse diffusion sampling.

        Args:
            model: The denoising model
            shape: Output shape (B, N, 3) for atoms or (B, L, 3) for centroids
            model_kwargs: Dict of model inputs (atom_types, aa_seq, chain_ids, etc.)
            noiser: Diffusion noiser with schedule (alpha_bar, betas, etc.)
            device: torch device
            forward_fn: Optional custom forward function. If None, uses model().
                        Signature: forward_fn(model, x, t, **model_kwargs) -> x0_pred

        Returns:
            x0_pred: Final predicted coordinates [B, N, 3] or [B, L, 3]
        """
        pass

    def _apply_alignment(self, x0_pred: Tensor, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Apply Kabsch alignment if enabled."""
        if self.align_per_step:
            return kabsch_align_to_target(x0_pred, x, mask)
        return x0_pred

    def _apply_recenter(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Apply recentering if enabled."""
        if not self.recenter:
            return x

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
        else:
            centroid = x.mean(dim=1, keepdim=True)
        return x - centroid


class DDPMSampler(BaseSampler):
    """Standard DDPM reverse sampler.

    This is the baseline sampler that matches the original ddpm_sample() behavior
    when align_per_step=False and recenter=False.
    """

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: Dict[str, Any],
        noiser,
        device: torch.device,
        forward_fn: Optional[Callable] = None,
    ) -> Tensor:
        B, N, _ = shape
        mask = model_kwargs.get('mask') or model_kwargs.get('mask_res')

        # Default forward function
        if forward_fn is None:
            def forward_fn(model, x, t, **kwargs):
                return model(x, **kwargs, t=t)

        # Start from random noise
        x = torch.randn(B, N, 3, device=device)

        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict x0
            x0_pred = forward_fn(model, x, t_batch, **model_kwargs)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            # Optional: Kabsch-align x0_pred to current x
            x0_pred = self._apply_alignment(x0_pred, x, mask)

            # DDPM reverse step
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

            # Optional: Re-center
            x = self._apply_recenter(x, mask)

        return x


class HeunSampler(BaseSampler):
    """Heun (2nd order) sampler for improved accuracy.

    Uses predictor-corrector approach:
    1. Predict next state with Euler
    2. Evaluate slope at predicted point
    3. Average slopes for final update

    This reduces integration error compared to standard Euler/DDPM,
    at the cost of 2x model evaluations per step.
    """

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: Dict[str, Any],
        noiser,
        device: torch.device,
        forward_fn: Optional[Callable] = None,
    ) -> Tensor:
        B, N, _ = shape
        mask = model_kwargs.get('mask') or model_kwargs.get('mask_res')

        if forward_fn is None:
            def forward_fn(model, x, t, **kwargs):
                return model(x, **kwargs, t=t)

        # Start from random noise
        x = torch.randn(B, N, 3, device=device)

        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Get schedule values
            ab_t = noiser.alpha_bar[t]
            if t > 0:
                ab_prev = noiser.alpha_bar[t - 1]
                beta = noiser.betas[t]
                alpha = noiser.alphas[t]

            # === Predictor (Euler step) ===
            x0_pred = forward_fn(model, x, t_batch, **model_kwargs)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)
            x0_pred = self._apply_alignment(x0_pred, x, mask)

            if t > 0:
                # Compute slope d1 = (x - x0_pred) / sqrt(1 - ab_t)
                # For DDPM, we use the posterior mean formula
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x

                # Euler prediction (no noise for predictor)
                x_pred = mean

                # === Corrector (evaluate at predicted point) ===
                t_prev_batch = torch.full((B,), t - 1, device=device, dtype=torch.long)
                x0_pred_corr = forward_fn(model, x_pred, t_prev_batch, **model_kwargs)
                x0_pred_corr = torch.clamp(x0_pred_corr, -self.clamp_val, self.clamp_val)
                x0_pred_corr = self._apply_alignment(x0_pred_corr, x_pred, mask)

                # Average the two x0 predictions
                x0_avg = 0.5 * (x0_pred + x0_pred_corr)

                # Final update with averaged prediction
                mean_corr = coef1 * x0_avg + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean_corr + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

            x = self._apply_recenter(x, mask)

        return x


class DeterministicDDIMSampler(BaseSampler):
    """Deterministic DDIM sampler (no noise injection).

    Useful for:
    - Reproducible sampling (same seed = same output)
    - Faster sampling with fewer steps
    - Debugging (removes stochasticity)
    """

    def __init__(self, eta: float = 0.0, **kwargs):
        """
        Args:
            eta: Noise scale (0 = deterministic, 1 = DDPM-like)
        """
        super().__init__(**kwargs)
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: Dict[str, Any],
        noiser,
        device: torch.device,
        forward_fn: Optional[Callable] = None,
    ) -> Tensor:
        B, N, _ = shape
        mask = model_kwargs.get('mask') or model_kwargs.get('mask_res')

        if forward_fn is None:
            def forward_fn(model, x, t, **kwargs):
                return model(x, **kwargs, t=t)

        x = torch.randn(B, N, 3, device=device)

        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            x0_pred = forward_fn(model, x, t_batch, **model_kwargs)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)
            x0_pred = self._apply_alignment(x0_pred, x, mask)

            if t > 0:
                ab_t = noiser.alpha_bar[t]
                ab_prev = noiser.alpha_bar[t - 1]

                # DDIM update (deterministic when eta=0)
                # x_{t-1} = sqrt(ab_prev) * x0_pred + sqrt(1 - ab_prev) * direction
                # where direction points from x0_pred toward noise

                # Predict noise from x0_pred
                sqrt_ab_t = torch.sqrt(ab_t)
                sqrt_one_minus_ab_t = torch.sqrt(1 - ab_t)
                eps_pred = (x - sqrt_ab_t * x0_pred) / sqrt_one_minus_ab_t

                # Compute sigma for optional stochasticity
                sigma = self.eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)

                # DDIM update
                sqrt_ab_prev = torch.sqrt(ab_prev)
                sqrt_one_minus_ab_prev_minus_sigma = torch.sqrt(1 - ab_prev - sigma ** 2)

                x = sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev_minus_sigma * eps_pred
                if self.eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

            x = self._apply_recenter(x, mask)

        return x


class EDMSampler(BaseSampler):
    """EDM-style sampler for VE (variance-exploding) diffusion.

    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., 2022).

    Uses Euler or Heun integration on the probability flow ODE.
    Works with VENoiser and KarrasSchedule.
    """

    def __init__(
        self,
        use_heun: bool = True,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float('inf'),
        s_noise: float = 1.0,
        **kwargs
    ):
        """
        Args:
            use_heun: Use Heun's method (2nd order) instead of Euler (1st order)
            s_churn: Amount of stochasticity (0 = deterministic)
            s_tmin: Minimum sigma for stochasticity
            s_tmax: Maximum sigma for stochasticity
            s_noise: Noise inflation factor
        """
        super().__init__(**kwargs)
        self.use_heun = use_heun
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: Dict[str, Any],
        noiser,
        device: torch.device,
        forward_fn: Optional[Callable] = None,
    ) -> Tensor:
        """EDM sampling with optional Heun's method.

        Args:
            noiser: Should be VENoiser with KarrasSchedule (has .sigmas attribute)
        """
        B, N, _ = shape
        mask = model_kwargs.get('mask') or model_kwargs.get('mask_res')

        # Get sigma schedule (requires VENoiser/KarrasSchedule)
        if hasattr(noiser, 'sigmas'):
            sigmas = noiser.sigmas.to(device)
        else:
            # Fallback: convert alpha_bar to sigma
            sigmas = torch.sqrt((1 - noiser.alpha_bar) / noiser.alpha_bar.clamp(min=1e-8))
            sigmas = sigmas.to(device)

        # Model forward function
        if forward_fn is None:
            def forward_fn(model, x, t, **kwargs):
                return model(x, **kwargs, t=t)

        # Initialize at highest noise level
        x = sigmas[0] * torch.randn(B, N, 3, device=device)

        # Main sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Optional: add stochasticity (s_churn)
            gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1) if self.s_tmin <= sigma <= self.s_tmax else 0
            sigma_hat = sigma * (1 + gamma)
            if gamma > 0:
                eps = torch.randn_like(x) * self.s_noise
                x = x + (sigma_hat**2 - sigma**2).sqrt() * eps

            # Convert sigma to timestep for model
            t_batch = noiser.schedule.sigma_to_t(sigma_hat.expand(B)) if hasattr(noiser, 'schedule') and hasattr(noiser.schedule, 'sigma_to_t') else torch.full((B,), i, device=device, dtype=torch.long)

            # Predict x0 (denoised)
            x0_pred = forward_fn(model, x, t_batch, **model_kwargs)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            # Kabsch align
            x0_pred = self._apply_alignment(x0_pred, x, mask)

            # Compute derivative: d = (x - x0_pred) / sigma_hat
            d = (x - x0_pred) / sigma_hat

            # Euler step
            dt = sigma_next - sigma_hat  # Negative (decreasing sigma)
            x_next = x + d * dt

            # Optional: Heun's method (2nd order correction)
            if self.use_heun and sigma_next > 0:
                # Evaluate at predicted point
                t_next = noiser.schedule.sigma_to_t(sigma_next.expand(B)) if hasattr(noiser, 'schedule') and hasattr(noiser.schedule, 'sigma_to_t') else torch.full((B,), i + 1, device=device, dtype=torch.long)

                x0_pred_next = forward_fn(model, x_next, t_next, **model_kwargs)
                x0_pred_next = torch.clamp(x0_pred_next, -self.clamp_val, self.clamp_val)
                x0_pred_next = self._apply_alignment(x0_pred_next, x_next, mask)

                d_next = (x_next - x0_pred_next) / sigma_next

                # Average derivatives (trapezoidal rule)
                d_avg = 0.5 * (d + d_next)
                x_next = x + d_avg * dt

            x = x_next

            # Recenter
            x = self._apply_recenter(x, mask)

        return x


# =============================================================================
# Registry
# =============================================================================

_SAMPLERS = {
    "ddpm": lambda **kw: DDPMSampler(align_per_step=False, recenter=False, **kw),
    "ddpm_kabsch": lambda **kw: DDPMSampler(align_per_step=True, recenter=False, **kw),
    "ddpm_kabsch_recenter": lambda **kw: DDPMSampler(align_per_step=True, recenter=True, **kw),
    "heun": lambda **kw: HeunSampler(align_per_step=True, recenter=True, **kw),
    "ddim": lambda **kw: DeterministicDDIMSampler(eta=0.0, align_per_step=True, recenter=True, **kw),
    "ddim_stochastic": lambda **kw: DeterministicDDIMSampler(eta=1.0, align_per_step=True, recenter=True, **kw),
    "edm": lambda **kw: EDMSampler(use_heun=True, align_per_step=True, recenter=True, **kw),
    "edm_euler": lambda **kw: EDMSampler(use_heun=False, align_per_step=True, recenter=True, **kw),
}


def list_samplers() -> list:
    """Return list of available sampler names."""
    return list(_SAMPLERS.keys())


def create_sampler(name: str, **kwargs) -> BaseSampler:
    """Create a sampler by name.

    Args:
        name: Sampler name (see list_samplers())
        **kwargs: Override default sampler options (clamp_val, align_per_step, recenter)

    Returns:
        Instantiated sampler
    """
    if name not in _SAMPLERS:
        available = ", ".join(_SAMPLERS.keys())
        raise ValueError(f"Unknown sampler: {name}. Available: {available}")
    return _SAMPLERS[name](**kwargs)
