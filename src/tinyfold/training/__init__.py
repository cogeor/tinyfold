"""Training infrastructure for TinyFold.

Provides unified logging, metrics tracking, checkpointing, data loading,
and augmentation utilities across all training scripts.
"""

from .augmentation import (
    apply_rigid_augment,
    apply_rotation_augment,
    random_rotation_matrix,
)
from .checkpointing import CheckpointManager, load_checkpoint, save_checkpoint
from .data import collate_batch, load_sample, load_sample_raw
from .logger import TrainingLogger
from .metrics import LossComponents, MetricTracker
from .objective import LossComposer, LossRegistry, LossTerm
from .registry_append import append_registry_row
from .run_naming import generate_run_name
from .setup import (
    create_diffusion_components,
    create_train_sampler,
    get_or_create_split,
    load_model_checkpoint,
)

__all__ = [
    # Logging & Metrics
    "TrainingLogger",
    "generate_run_name",
    "append_registry_row",
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


