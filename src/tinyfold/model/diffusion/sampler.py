"""DDIM sampler for inference."""

import torch

from tinyfold.model.diffusion.schedule import DiffusionSchedule


class DDIMSampler:
    """Deterministic DDIM sampling loop.

    Implements DDIM (Denoising Diffusion Implicit Models) for fast,
    deterministic sampling from the diffusion model.
    """

    def __init__(self, schedule: DiffusionSchedule, eta: float = 0.0):
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
