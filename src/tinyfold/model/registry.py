"""Model registry for TinyFold.

Provides factory functions for creating models, schedules, and noisers.
This module centralizes model creation and provides type-safe interfaces.

Usage:
    from tinyfold.model.registry import create_model, list_models
    from tinyfold.model.registry import create_schedule, create_noiser

    model = create_model("resfold_onestep")
    schedule = create_schedule("cosine", T=50)
    noiser = create_noiser("gaussian", schedule)
"""

import importlib
from typing import Any

# Schedule/noiser factories live in one place (tinyfold.model.diffusion); re-export
# them here so ``registry`` stays the single import surface without duplicating the
# tables (the local copies used to drift — they lacked karras/ve).
from tinyfold.model.diffusion import (
    create_noiser as create_noiser,
)
from tinyfold.model.diffusion import (
    create_schedule as create_schedule,
)
from tinyfold.model.diffusion import (
    list_noise_types as list_noise_types,
)
from tinyfold.model.diffusion import (
    list_schedules as list_schedules,
)

# Lazy imports to avoid circular dependencies
_MODEL_CLASSES: dict[str, str] = {
    # resfold line; resfold_onestep is the supported headline model.
    "resfold_stage1": "tinyfold.model.resfold.denoiser.ResidueDenoiser",
    "resfold_stage2": "tinyfold.model.resfold.refiner.AtomRefinerV2",
    "resfold_stage2_multi": "tinyfold.model.resfold.atomrefine_multi_sample.AtomRefinerV2MultiSample",
    "resfold": "tinyfold.model.resfold.pipeline.ResFoldPipeline",
    "resfold_e2e": "tinyfold.model.resfold.e2e.ResFoldE2E",
    "resfold_assembler": "tinyfold.model.resfold.assembler.ResFoldAssembler",
    "resfold_onestep": "tinyfold.model.resfold.onestep.ResFoldOneStep",
}

# Cache for loaded classes
_loaded_classes: dict[str, type] = {}


def _load_class(full_path: str) -> type:
    """Dynamically load a class from module path."""
    if full_path in _loaded_classes:
        return _loaded_classes[full_path]

    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _loaded_classes[full_path] = cls
    return cls


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODEL_CLASSES.keys())


def create_model(name: str, **kwargs) -> Any:
    """Create a model by name.

    Args:
        name: Model name (see list_models())
        **kwargs: Model-specific arguments (h_dim, n_layers, etc.)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is unknown
    """
    if name not in _MODEL_CLASSES:
        available = ", ".join(_MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    cls = _load_class(_MODEL_CLASSES[name])
    return cls(**kwargs)


def get_model_class(name: str) -> type:
    """Get model class by name (for inspection without instantiation)."""
    if name not in _MODEL_CLASSES:
        available = ", ".join(_MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return _load_class(_MODEL_CLASSES[name])


def register_model(name: str, module_path: str):
    """Register a new model class.

    Args:
        name: Name to register under
        module_path: Full module path to class (e.g., "mymodule.MyModel")
    """
    _MODEL_CLASSES[name] = module_path
