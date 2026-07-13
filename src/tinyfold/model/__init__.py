"""TinyFold model components.

Provides:
- Model registry for creating models by name
- Diffusion components (schedules, noise types, samplers)
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
    TimestepCurriculum,
    generate_extended_chain,
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

# Registry (factory functions)
from tinyfold.model.registry import (
    create_model,
    create_noiser,
    create_schedule,
    get_model_class,
    list_models,
    list_noise_types,
    list_schedules,
    register_model,
)

__all__ = [
    # Core
    "ModelConfig",
    # Registry
    "create_model",
    "create_schedule",
    "create_noiser",
    "list_models",
    "list_schedules",
    "list_noise_types",
    "get_model_class",
    "register_model",
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
    # Diffusion - curriculum
    "TimestepCurriculum",
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
