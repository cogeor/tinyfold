"""Timestep curriculum for diffusion training.

Implements curriculum learning for diffusion models - start with easy
denoising tasks (low timesteps) and gradually increase difficulty.
"""

import math
import torch


class TimestepCurriculum:
    """Curriculum for diffusion timesteps - start easy, increase difficulty.

    During early training, only sample from low timesteps (easy denoising).
    Gradually increase max timestep as training progresses.
    """

    def __init__(
        self,
        T: int,
        warmup_steps: int,
        schedule: str = "linear",
    ):
        """
        Args:
            T: Maximum timestep (from noiser)
            warmup_steps: Training steps to reach full T
            schedule: "linear" or "cosine" progression
        """
        self.T = T
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def get_max_t(self, step: int) -> int:
        """Get maximum timestep for current training step.

        Args:
            step: Current training step

        Returns:
            Maximum timestep to sample from
        """
        if step >= self.warmup_steps:
            return self.T

        progress = step / self.warmup_steps

        if self.schedule == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        # At least t_max=1 to avoid empty range
        return max(1, int(progress * self.T))

    def sample(
        self,
        batch_size: int,
        step: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample timesteps with curriculum constraint.

        Args:
            batch_size: Number of timesteps to sample
            step: Current training step
            device: Device to create tensor on

        Returns:
            [batch_size] tensor of timesteps
        """
        t_max = self.get_max_t(step)
        return torch.randint(0, t_max, (batch_size,), device=device)

    def __repr__(self):
        return f"TimestepCurriculum(T={self.T}, warmup={self.warmup_steps}, schedule={self.schedule})"
