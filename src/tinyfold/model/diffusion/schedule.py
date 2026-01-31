"""Diffusion noise schedules.

Provides cosine and linear schedules for alpha_bar, the cumulative
product of (1 - beta) that controls noise levels at each timestep.
"""

import math

import torch
import torch.nn as nn


class CosineSchedule:
    """Cosine schedule for alpha_bar (from Nichol & Dhariwal 2021).

    Non-nn.Module version for use with noise types.
    """

    def __init__(self, T: int = 50, s: float = 0.008):
        self.T = T
        self.s = s

        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1]])
        self.betas = 1 - self.alphas

    def to(self, device):
        for attr in ["alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar", "alphas", "betas"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


class LinearSchedule:
    """Linear alpha_bar schedule. Directly interpolates alpha_bar from 1 to 0."""

    def __init__(self, T: int = 50):
        self.T = T

        t = torch.arange(T + 1, dtype=torch.float32)
        alpha_bar = 1 - t / T
        alpha_bar = alpha_bar.clamp(min=1e-6)  # Avoid exactly 0

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-6)])
        self.betas = 1 - self.alphas

    def to(self, device):
        for attr in ["alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar", "alphas", "betas"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


class DiffusionSchedule(nn.Module):
    """Cosine noise schedule for diffusion.

    Implements the cosine schedule from "Improved Denoising Diffusion
    Probabilistic Models" (Nichol & Dhariwal, 2021).

    Stores:
    - betas[t]: noise variance at step t
    - alphas[t]: 1 - betas[t]
    - alpha_bar[t]: cumulative product of alphas up to t
    - sqrt_alpha_bar[t]: for forward diffusion
    - sqrt_one_minus_alpha_bar[t]: for forward diffusion
    """

    def __init__(self, T: int = 16, s: float = 0.008):
        """
        Args:
            T: Number of diffusion timesteps
            s: Small offset for numerical stability
        """
        super().__init__()
        self.T = T

        # Compute alpha_bar using cosine schedule
        steps = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]

        # Clip for numerical stability
        alpha_bar = torch.clamp(alpha_bar, 1e-5, 1.0)

        # Compute betas from alpha_bar
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, 0, 0.999)

        alphas = 1 - betas

        # Register buffers (moved to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar[1:])  # [T]
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar[1:]))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar[1:])
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x0.

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: [N, 3] clean coordinates
            t: int or [1] timestep in [0, T-1]
            noise: optional pre-sampled noise
        Returns:
            x_t: [N, 3] noised coordinates
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Handle int or tensor t
        if isinstance(t, int):
            sqrt_ab = self.sqrt_alpha_bar[t]
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]
        else:
            sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1)
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)

        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: int | torch.Tensor,
        eps_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from noisy x_t and predicted noise.

        x0_hat = (x_t - sqrt(1-alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)

        Args:
            x_t: [N, 3] noisy coordinates
            t: timestep
            eps_hat: [N, 3] predicted noise
        Returns:
            x0_hat: [N, 3] predicted clean coordinates
        """
        if isinstance(t, int):
            sqrt_ab = self.sqrt_alpha_bar[t]
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]
        else:
            sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1)
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)

        # Add epsilon for numerical stability when sqrt_ab is very small (near t=0)

# =============================================================================
# Karras Schedule (EDM-style continuous sigma)
# =============================================================================

class KarrasSchedule:
    """Karras/EDM-style continuous sigma schedule.

    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., 2022). Uses continuous sigma values instead of discrete timesteps.

    Schedule: sigma[i] = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    For normalized coordinates (std=1), use sigma_min=0.002, sigma_max=10.0.
    """

    def __init__(
        self,
        n_steps: int = 200,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
    ):
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        # Build schedule: decreasing from sigma_max to sigma_min
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        inv_rho = 1.0 / rho
        sigmas = (sigma_max ** inv_rho + steps * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
        self.sigmas = sigmas  # [n_steps+1], from high to low

        # For compatibility with VP-style code, create fake alpha_bar
        # VP: x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
        # VE: x_t = x0 + sigma * eps
        # Mapping: sigma^2 = (1 - ab) / ab  =>  ab = 1 / (1 + sigma^2)
        self.alpha_bar = 1.0 / (1.0 + sigmas ** 2)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        # Fake betas/alphas for interface compatibility
        self.betas = torch.zeros(n_steps + 1)
        self.alphas = torch.ones(n_steps + 1)

    @property
    def T(self):
        return self.n_steps

    def to(self, device):
        for attr in ['sigmas', 'alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'betas', 'alphas']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to equivalent discrete timestep index."""
        # Find nearest sigma in schedule
        dists = (self.sigmas.unsqueeze(0) - sigma.unsqueeze(1)).abs()
        return dists.argmin(dim=1)

