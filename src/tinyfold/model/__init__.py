"""TinyFold model components.

Provides:
- Diffusion components (schedules, noise types, samplers, factory functions)
- Loss functions (MSE, geometry, contact, lDDT)
"""

from tinyfold.model.config import ModelConfig

# Diffusion components
from tinyfold.model.diffusion import (
    CosineSchedule,
    DDPMSampler,
    DeterministicDDIMSampler,
    DiffusionSchedule,
    GaussianNoise,
    LinearChainFlow,
    LinearChainNoise,
    LinearSchedule,
    create_noiser,
    create_schedule,
    generate_extended_chain,
    list_noise_types,
    list_schedules,
)

# Loss functions
from tinyfold.model.losses import (
    ContactLoss,
    GeometryLoss,
    compute_distance_consistency_loss,
    compute_ilddt,
    compute_lddt,
    compute_lddt_metrics,
    compute_mse_loss,
    compute_rmse,
    kabsch_align,
)

__all__ = [
    # Core
    "ModelConfig",
    # Diffusion - factory functions
    "create_schedule",
    "create_noiser",
    "list_schedules",
    "list_noise_types",
    # Diffusion - schedules
    "DiffusionSchedule",
    "CosineSchedule",
    "LinearSchedule",
    # Diffusion - noise
    "GaussianNoise",
    "LinearChainNoise",
    "LinearChainFlow",
    "generate_extended_chain",
    # Diffusion - samplers
    "DeterministicDDIMSampler",
    "DDPMSampler",
    # Losses
    "kabsch_align",
    "compute_mse_loss",
    "compute_rmse",
    "compute_distance_consistency_loss",
    "GeometryLoss",
    "ContactLoss",
    "compute_lddt",
    "compute_ilddt",
    "compute_lddt_metrics",
]
