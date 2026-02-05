"""Model factory for TinyFold decoders and diffusion components.
(Backward compatibility wrapper around tinyfold.model)

Usage:
    from models import create_model, list_models
    from models import create_schedule, create_noiser, list_schedules, list_noise_types
"""

import warnings

# Only import archive models that don't have problematic import chains
# AF3StyleDecoder and AttentionDiffusionV2 only depend on .base (safe)
# PairformerDecoder, HierarchicalDecoder import from tinyfold.model which can fail
from .archive.af3_style import AF3StyleDecoder
from .archive.attention_v2 import AttentionDiffusionV2

# These have problematic imports - load lazily only if needed
AtomRefiner = None
HierarchicalDecoder = None
PairformerDecoder = None

# These imports may fail if the refactor left broken internal references.
# Wrap in try/except so the server can still start with af3_style.
try:
    from tinyfold.model.resfold import (
        ResidueDenoiser,
        ResFoldPipeline,
        ResFoldE2E,
        sample_e2e,
        ResFoldAssembler,
        AtomRefinerV2,
        AtomRefinerV2MultiSample,
    )
    from tinyfold.model.resfold import clustering
except ImportError as e:
    warnings.warn(f"Could not import resfold models: {e}")
    ResidueDenoiser = None
    ResFoldPipeline = None
    ResFoldE2E = None
    sample_e2e = None
    ResFoldAssembler = None
    AtomRefinerV2 = None
    AtomRefinerV2MultiSample = None
    clustering = None

try:
    from tinyfold.model.iterfold import IterFold, AnchorDecoder
except ImportError as e:
    warnings.warn(f"Could not import iterfold models: {e}")
    IterFold = None
    AnchorDecoder = None

from tinyfold.model.losses import (
    GeometryLoss,
    bond_length_loss,
    bond_angle_loss,
    omega_loss,
    o_chirality_loss,
    virtual_cb_loss,
    dihedral_angle,
    BOND_LENGTHS_ANGSTROM,
    get_normalized_bond_lengths,
    BOND_ANGLES,
)

from tinyfold.model.diffusion import (
    CosineSchedule,
    LinearSchedule,
    KarrasSchedule,
    GaussianNoise,
    LinearChainNoise,
    LinearChainFlow,
    VENoiser,
    TimestepCurriculum,
    create_schedule,
    create_noiser,
    list_schedules,
    list_noise_types,
    generate_extended_chain,
    kabsch_align_to_target,
    BaseSampler,
    DDPMSampler,
    HeunSampler,
    DeterministicDDIMSampler,
    EDMSampler,
    create_sampler,
    list_samplers,
)

from tinyfold.training.utils import (
    random_rigid_augment,
    random_rotation_matrix,
    af3_loss_weight,
    MultiCopyTrainer,
    VectorizedMultiCopyTrainer,
)
from .archive.self_conditioning import (
    self_conditioning_training_step,
    sample_step_with_self_cond,
    create_self_cond_embedding,
)


# Model registry - only include models that imported successfully
_MODELS = {
    "attention_v2": AttentionDiffusionV2,
    "af3_style": AF3StyleDecoder,
}

if HierarchicalDecoder is not None:
    _MODELS["hierarchical"] = HierarchicalDecoder
if PairformerDecoder is not None:
    _MODELS["pairformer"] = PairformerDecoder

if ResidueDenoiser is not None:
    _MODELS["resfold_stage1"] = ResidueDenoiser
if AtomRefiner is not None:
    _MODELS["resfold_stage2"] = AtomRefiner
if AtomRefinerV2MultiSample is not None:
    _MODELS["resfold_stage2_multi"] = AtomRefinerV2MultiSample
if ResFoldPipeline is not None:
    _MODELS["resfold"] = ResFoldPipeline
if ResFoldE2E is not None:
    _MODELS["resfold_e2e"] = ResFoldE2E
if ResFoldAssembler is not None:
    _MODELS["resfold_assembler"] = ResFoldAssembler
if IterFold is not None:
    _MODELS["iterfold"] = IterFold


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODELS.keys())


def create_model(name: str, **kwargs):
    """Create a model by name."""
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return _MODELS[name](**kwargs)


def get_model_class(name: str) -> type:
    """Get model class by name."""
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return _MODELS[name]


__all__ = [
    # Models
    "AttentionDiffusionV2",
    "HierarchicalDecoder",
    "PairformerDecoder",
    "AF3StyleDecoder",
    "ResidueDenoiser",
    "AtomRefiner",
    "AtomRefinerV2",
    "AtomRefinerV2MultiSample",
    "ResFoldPipeline",
    "ResFoldE2E",
    "sample_e2e",
    "ResFoldAssembler",
    "IterFold",
    "AnchorDecoder",
    # Clustering utilities
    "clustering",
    "GeometryLoss",
    "create_model",
    "list_models",
    "get_model_class",
    # Diffusion
    "CosineSchedule",
    "create_schedule",
    "list_schedules",
    "GaussianNoise",
    "LinearChainNoise",
    "LinearChainFlow",
    "create_noiser",
    "list_noise_types",
    "generate_extended_chain",
    "kabsch_align_to_target",
    "LinearSchedule",
    "KarrasSchedule",
    "VENoiser",
    "TimestepCurriculum",
    "BaseSampler",
    "DDPMSampler",
    "HeunSampler",
    "DeterministicDDIMSampler",
    "EDMSampler",
    "create_sampler",
    "list_samplers",
    # Training utilities
    "random_rigid_augment",
    "random_rotation_matrix",
    "af3_loss_weight",
    "MultiCopyTrainer",
    "VectorizedMultiCopyTrainer",
    # Self-conditioning
    "self_conditioning_training_step",
    "sample_step_with_self_cond",
    "create_self_cond_embedding",
]
