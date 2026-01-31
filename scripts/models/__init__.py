"""Model factory for TinyFold decoders and diffusion components.
(Backward compatibility wrapper around tinyfold.model)

Usage:
    from models import create_model, list_models
    from models import create_schedule, create_noiser, list_schedules, list_noise_types
"""

# Import from new locations in src/tinyfold/model/
from tinyfold.model.attention_v2 import AttentionDiffusionV2
from tinyfold.model.af3_style import AF3StyleDecoder
from tinyfold.model.resfold import (
    ResidueDenoiser,
    ResFoldPipeline,
    ResFoldE2E,
    sample_e2e,
    ResFoldAssembler,
    AtomRefinerV2,
    AtomRefinerV2MultiSample,
)
from tinyfold.model.iterfold import IterFold, AnchorDecoder
from tinyfold.model.resfold import clustering

# Archived/Deprecated models
# (If they were moved to archive/, import from there, or stub them out)
from .archive import AtomRefiner, HierarchicalDecoder, PairformerDecoder

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
from tinyfold.model.self_conditioning import (
    self_conditioning_training_step,
    sample_step_with_self_cond,
    create_self_cond_embedding,
)


# Model registry
_MODELS = {
    "attention_v2": AttentionDiffusionV2,
    "hierarchical": HierarchicalDecoder,
    "pairformer": PairformerDecoder,
    "af3_style": AF3StyleDecoder,
    "resfold_stage1": ResidueDenoiser,
    "resfold_stage2": AtomRefiner,
    "resfold_stage2_multi": AtomRefinerV2MultiSample,
    "resfold": ResFoldPipeline,
    "resfold_e2e": ResFoldE2E,
    "resfold_assembler": ResFoldAssembler,
    "iterfold": IterFold,
}


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
