"""Training infrastructure for TinyFold.

Provides unified logging, metrics tracking, checkpointing, data loading,
and augmentation utilities across all training scripts.
"""

from .logger import TrainingLogger
from .run_naming import generate_run_name
from .metrics import MetricTracker, LossComponents
from .objective import LossRegistry, LossComposer, LossTerm
from .checkpointing import save_checkpoint, load_checkpoint, CheckpointManager
from .data import load_sample, collate_batch, load_sample_raw
from .augmentation import (
    random_rotation_matrix,
    apply_rigid_augment,
    apply_rotation_augment,
)
from .setup import (
    get_or_create_split,
    create_diffusion_components,
    load_model_checkpoint,
    create_train_sampler,
)

__all__ = [
    # Logging & Metrics
    "TrainingLogger",
    "generate_run_name",
    "MetricTracker",
    "LossComponents",
    "LossRegistry",
    "LossComposer",
    "LossTerm",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    # Data
    "load_sample",
    "load_sample_raw",
    "collate_batch",
    # Augmentation
    "random_rotation_matrix",
    "apply_rigid_augment",
    "apply_rotation_augment",
    # Setup utilities
    "get_or_create_split",
    "create_diffusion_components",
    "load_model_checkpoint",
    "create_train_sampler",
]


