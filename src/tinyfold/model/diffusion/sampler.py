"""Diffusion samplers for inference.

Provides DDIM (deterministic) and DDPM (stochastic) sampling loops.
"""

from __future__ import annotations

import torch
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tinyfold.model.diffusion.schedule import DiffusionSchedule


class DDIMSampler:
    """Deterministic DDIM sampling loop.

    Implements DDIM (Denoising Diffusion Implicit Models) for fast,
    deterministic sampling from the diffusion model.
    """

    def __init__(self, schedule: "DiffusionSchedule", eta: float = 0.0):
        """
        Args:
            schedule: Diffusion schedule with alpha_bar buffers
            eta: Noise scale (0 = deterministic, 1 = DDPM-like)
        """
        self.schedule = schedule
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Run full sampling from T to 0.

        Args:
            denoise_fn: Function(x_t, t) -> eps_hat
                Should handle edge building and conditioning internally
            shape: (N_atom, 3) output shape
            device: torch device
        Returns:
            x0: [N_atom, 3] sampled coordinates
        """
        T = self.schedule.T

        # Start from noise
        x = torch.randn(shape, device=device)

        for t in reversed(range(T)):
            # Predict noise
            eps_hat = denoise_fn(x, t)

            # DDIM update step
            x = self._ddim_step(x, eps_hat, t)

        return x

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        eps_hat: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Single DDIM reverse step.

        DDIM update rule:
        x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
                  + sqrt(1 - alpha_bar_{t-1} - sigma^2) * eps_hat
                  + sigma * noise

        With eta=0 (deterministic), sigma=0.

        Args:
            x_t: [N, 3] current noisy coordinates
            eps_hat: [N, 3] predicted noise
            t: current timestep
        Returns:
            x_prev: [N, 3] coordinates at t-1
        """
        device = x_t.device

        alpha_bar_t = self.schedule.alpha_bar[t]

        if t > 0:
            alpha_bar_prev = self.schedule.alpha_bar[t - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # Predict x0
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1_ab = torch.sqrt(1 - alpha_bar_t)
        x0_pred = (x_t - sqrt_1_ab * eps_hat) / sqrt_ab

        # Compute sigma for stochasticity
        if self.eta > 0 and t > 0:
            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=device)

        # DDIM update
        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
        sqrt_1_ab_prev = torch.sqrt(1 - alpha_bar_prev - sigma**2)

        x_prev = sqrt_ab_prev * x0_pred + sqrt_1_ab_prev * eps_hat

        # Add noise if eta > 0
        if self.eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    def sample_with_trajectory(
        self,
        denoise_fn,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample and return intermediate states for visualization.

        Args:
            denoise_fn: Function(x_t, t) -> eps_hat
            shape: (N_atom, 3) output shape
            device: torch device
        Returns:
            x0: [N_atom, 3] final sampled coordinates
            trajectory: list of [N_atom, 3] intermediate states
        """
        T = self.schedule.T
        trajectory = []

        x = torch.randn(shape, device=device)
        trajectory.append(x.clone())

        for t in reversed(range(T)):
            eps_hat = denoise_fn(x, t)
            x = self._ddim_step(x, eps_hat, t)
            trajectory.append(x.clone())

        return x, trajectory


class DDPMSampler:
    """Stochastic DDPM sampling loop.

    Implements standard DDPM reverse process with noise injection at each step.
    Supports Gaussian and linear chain noise types.
    """

    def __init__(
        self,
        noiser,
        clamp_val: float = 3.0,
    ):
        """
        Args:
            noiser: Noiser object (GaussianNoise, LinearChainNoise, etc.)
            clamp_val: Value to clamp predictions to
        """
        self.noiser = noiser
        self.clamp_val = clamp_val

    @property
    def T(self):
        return self.noiser.T

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        noise_type: str = "gaussian",
    ) -> torch.Tensor:
        """Run full DDPM sampling from T to 0.

        Args:
            model: Model that takes (x_t, atom_types, atom_to_res, aa_seq, chain_ids, t, mask)
                   and returns x0_pred
            batch: Batch dict with keys: atom_types, atom_to_res, aa_seq, chain_ids, mask
            device: torch device
            noise_type: "gaussian", "linear_chain", or "linear_flow"

        Returns:
            x0: [B, N, 3] sampled coordinates
        """
        B, N = batch["atom_types"].shape

        # Get x_linear for linear chain/flow methods
        x_linear = None
        if noise_type in ["linear_chain", "linear_flow"]:
            from tinyfold.model.diffusion.noise import generate_extended_chain
            x_linear = torch.zeros(B, N, 3, device=device)
            for b in range(B):
                x_linear[b] = generate_extended_chain(
                    N,
                    batch["atom_to_res"][b],
                    batch["atom_types"][b],
                    batch["chain_ids"][b],
                    device,
                    apply_rotation=False,
                )
            x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
            x_linear = x_linear / x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)

        # Initialize starting point
        if noise_type == "linear_flow":
            x = x_linear.clone()
            t_range = range(self.T)  # Forward in time
        elif noise_type == "linear_chain":
            x = x_linear.clone()
            t_range = reversed(range(self.T + 1))  # From T down to 0
        else:
            x = torch.randn(B, N, 3, device=device)
            t_range = reversed(range(self.T))

        # Sampling loop
        for t in t_range:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Get model prediction
            if noise_type == "linear_chain" and hasattr(model, "forward_direct"):
                x0_pred = model.forward_direct(
                    x, batch["atom_types"], batch["atom_to_res"],
                    batch["aa_seq"], batch["chain_ids"], t_batch, batch["mask"]
                )
            else:
                x0_pred = model(
                    x, batch["atom_types"], batch["atom_to_res"],
                    batch["aa_seq"], batch["chain_ids"], t_batch, batch["mask"]
                )

            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            # Reverse step
            if noise_type == "linear_flow":
                if t < self.T - 1:
                    alpha_next = self.noiser.schedule.sqrt_alpha_bar[t + 1]
                    one_minus_alpha_next = self.noiser.schedule.sqrt_one_minus_alpha_bar[t + 1]
                    x = alpha_next * x0_pred + one_minus_alpha_next * x_linear
                else:
                    x = x0_pred
            elif noise_type == "linear_chain":
                x = self.noiser.reverse_step(x, x0_pred, t, x_linear)
            else:
                # Standard DDPM
                x = self._ddpm_step(x, x0_pred, t)

        return x

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Single DDPM reverse step.

        Args:
            x_t: [B, N, 3] current noisy coordinates
            x0_pred: [B, N, 3] predicted clean coordinates
            t: current timestep

        Returns:
            x_prev: [B, N, 3] coordinates at t-1
        """
        if t == 0:
            return x0_pred

        ab_t = self.noiser.alpha_bar[t]
        ab_prev = self.noiser.alpha_bar[t - 1]
        beta = self.noiser.betas[t]
        alpha = self.noiser.alphas[t]

        coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
        coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
        mean = coef1 * x0_pred + coef2 * x_t

        var = beta * (1 - ab_prev) / (1 - ab_t)
        x_prev = mean + torch.sqrt(var) * torch.randn_like(x_t)

        return x_prev
