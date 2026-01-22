"""Model registry for TinyFold.

Provides factory functions for creating models, schedules, and noisers.
This module centralizes model creation and provides type-safe interfaces.

Usage:
    from tinyfold.model.registry import create_model, list_models
    from tinyfold.model.registry import create_schedule, create_noiser

    model = create_model("attention_v2", h_dim=128, n_layers=6)
    schedule = create_schedule("cosine", T=50)
    noiser = create_noiser("gaussian", schedule)
"""

from typing import Dict, Any, Type, List, Optional
import importlib


# Lazy imports to avoid circular dependencies
_MODEL_CLASSES: Dict[str, str] = {
    "attention_v2": "models.attention_v2.AttentionDiffusionV2",
    "hierarchical": "models.hierarchical.HierarchicalDecoder",
    "pairformer": "models.pairformer_decoder.PairformerDecoder",
    "af3_style": "models.af3_style.AF3StyleDecoder",
    "resfold_stage1": "models.resfold.ResidueDenoiser",
    "resfold_stage2": "models.atomrefine.AtomRefiner",
    "resfold": "models.resfold_pipeline.ResFoldPipeline",
}

_SCHEDULE_CLASSES: Dict[str, str] = {
    "cosine": "tinyfold.model.diffusion.CosineSchedule",
    "linear": "tinyfold.model.diffusion.LinearSchedule",
}

_NOISER_CLASSES: Dict[str, str] = {
    "gaussian": "tinyfold.model.diffusion.GaussianNoise",
    "linear_chain": "tinyfold.model.diffusion.LinearChainNoise",
    "linear_flow": "tinyfold.model.diffusion.LinearChainFlow",
}

# Cache for loaded classes
_loaded_classes: Dict[str, Type] = {}


def _load_class(full_path: str) -> Type:
    """Dynamically load a class from module path."""
    if full_path in _loaded_classes:
        return _loaded_classes[full_path]

    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _loaded_classes[full_path] = cls
    return cls


def list_models() -> List[str]:
    """Return list of available model names."""
    return list(_MODEL_CLASSES.keys())


def list_schedules() -> List[str]:
    """Return list of available schedule names."""
    return list(_SCHEDULE_CLASSES.keys())


def list_noise_types() -> List[str]:
    """Return list of available noise type names."""
    return list(_NOISER_CLASSES.keys())


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


def create_schedule(name: str, **kwargs) -> Any:
    """Create a schedule by name.

    Args:
        name: Schedule name ("cosine" or "linear")
        **kwargs: Schedule-specific args (e.g., T=50)

    Returns:
        Schedule object
    """
    if name not in _SCHEDULE_CLASSES:
        available = ", ".join(_SCHEDULE_CLASSES.keys())
        raise ValueError(f"Unknown schedule: {name}. Available: {available}")

    cls = _load_class(_SCHEDULE_CLASSES[name])
    return cls(**kwargs)


def create_noiser(noise_type: str, schedule: Any, **kwargs) -> Any:
    """Create a noiser by name.

    Args:
        noise_type: Noise type name ("gaussian", "linear_chain", "linear_flow")
        schedule: Schedule object to use
        **kwargs: Noiser-specific args (e.g., noise_scale)

    Returns:
        Noiser object
    """
    if noise_type not in _NOISER_CLASSES:
        available = ", ".join(_NOISER_CLASSES.keys())
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {available}")

    cls = _load_class(_NOISER_CLASSES[noise_type])
    return cls(schedule, **kwargs)


def get_model_class(name: str) -> Type:
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


def register_schedule(name: str, module_path: str):
    """Register a new schedule class."""
    _SCHEDULE_CLASSES[name] = module_path


def register_noiser(name: str, module_path: str):
    """Register a new noiser class."""
    _NOISER_CLASSES[name] = module_path
