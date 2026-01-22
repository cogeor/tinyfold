"""TinyFold model components.

Provides:
- Model registry for creating models by name
- Diffusion components (schedules, noise types, samplers)
- Loss functions (MSE, geometry, contact, lDDT)
- Pairformer trunk
- EGNN denoiser
"""

from tinyfold.model.config import ModelConfig
from tinyfold.model.ppi_model import PPIModel

# Registry (factory functions)
from tinyfold.model.registry import (
    create_model,
    create_schedule,
    create_noiser,
    list_models,
    list_schedules,
    list_noise_types,
    get_model_class,
    register_model,
)

# Diffusion components
from tinyfold.model.diffusion import (
    DiffusionSchedule,
    CosineSchedule,
    LinearSchedule,
    GaussianNoise,
    LinearChainNoise,
    LinearChainFlow,
    DDIMSampler,
    DDPMSampler,
    TimestepCurriculum,
    generate_extended_chain,
)

# Loss functions
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
    ContactLoss,
    compute_lddt,
    compute_ilddt,
    compute_lddt_metrics,
)

__all__ = [
    # Core
    "ModelConfig",
    "PPIModel",
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
    "DDIMSampler",
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
