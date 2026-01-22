"""Diffusion components for coordinate generation.

Provides:
- Schedules: CosineSchedule, LinearSchedule, DiffusionSchedule
- Noise types: GaussianNoise, LinearChainNoise, LinearChainFlow
- Samplers: DDIMSampler, DDPMSampler
- Curriculum: TimestepCurriculum
- Factory functions: create_schedule, create_noiser
"""

from tinyfold.model.diffusion.sampler import DDIMSampler, DDPMSampler
from tinyfold.model.diffusion.schedule import (
    DiffusionSchedule,
    CosineSchedule,
    LinearSchedule,
)
from tinyfold.model.diffusion.noise import (
    GaussianNoise,
    LinearChainNoise,
    LinearChainFlow,
    generate_extended_chain,
)
from tinyfold.model.diffusion.curriculum import TimestepCurriculum


# Registry for factory functions
_SCHEDULES = {
    "cosine": CosineSchedule,
    "linear": LinearSchedule,
}

_NOISE_TYPES = {
    "gaussian": GaussianNoise,
    "linear_chain": LinearChainNoise,
    "linear_flow": LinearChainFlow,
}


def list_schedules() -> list[str]:
    """Return list of available schedule names."""
    return list(_SCHEDULES.keys())


def list_noise_types() -> list[str]:
    """Return list of available noise type names."""
    return list(_NOISE_TYPES.keys())


def create_schedule(name: str, **kwargs):
    """Create a schedule by name.

    Args:
        name: "cosine" or "linear"
        **kwargs: Schedule-specific args (e.g., T=50)

    Returns:
        Schedule object
    """
    if name not in _SCHEDULES:
        raise ValueError(f"Unknown schedule: {name}. Available: {list(_SCHEDULES.keys())}")
    return _SCHEDULES[name](**kwargs)


def create_noiser(noise_type: str, schedule, **kwargs):
    """Create a noiser by name.

    Args:
        noise_type: "gaussian", "linear_chain", or "linear_flow"
        schedule: The schedule to use
        **kwargs: Noise-type specific args (e.g., noise_scale for linear_chain)

    Returns:
        Noiser object
    """
    if noise_type not in _NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(_NOISE_TYPES.keys())}")
    return _NOISE_TYPES[noise_type](schedule, **kwargs)


__all__ = [
    # Schedules
    "DiffusionSchedule",
    "CosineSchedule",
    "LinearSchedule",
    # Noise types
    "GaussianNoise",
    "LinearChainNoise",
    "LinearChainFlow",
    "generate_extended_chain",
    # Samplers
    "DDIMSampler",
    "DDPMSampler",
    # Curriculum
    "TimestepCurriculum",
    # Factory functions
    "create_schedule",
    "create_noiser",
    "list_schedules",
    "list_noise_types",
]
