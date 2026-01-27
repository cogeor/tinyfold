"""Model factory for TinyFold decoders and diffusion components.

Usage:
    from models import create_model, list_models
    from models import create_schedule, create_noiser, list_schedules, list_noise_types

    # Models
    model = create_model("attention_v2", h_dim=128, n_layers=6)

    # Diffusion (separate schedule from noise type)
    schedule = create_schedule("cosine", T=50)
    noiser = create_noiser("gaussian", schedule)  # or "linear_chain"
"""

from .base import BaseDecoder, sinusoidal_pos_enc
from .attention_v2 import AttentionDiffusionV2
from .af3_style import AF3StyleDecoder
from .resfold import ResidueDenoiser
from .atomrefine_v2 import AtomRefinerV2
from .atomrefine_continuous import AtomRefinerContinuous
from .atomrefine_multi_sample import AtomRefinerV2MultiSample
from .resfold_pipeline import ResFoldPipeline
from .resfold_e2e import ResFoldE2E, sample_e2e
from .iterative_assembler import IterativeAtomAssembler
from .resfold_assembler import ResFoldAssembler
from .clustering import (
    compute_bond_connectivity,
    hierarchical_cluster_atoms,
    select_next_atoms_to_place,
    get_placement_order,
    simulate_known_mask,
)
# Archived models (kept for backward compatibility with old checkpoints)
from .archive import AtomRefiner, HierarchicalDecoder, PairformerDecoder
# GeometryLoss and related functions now come from tinyfold.model.losses
# Import them here for backward compatibility with scripts that import from models
from tinyfold.model.losses import (
    GeometryLoss,
    bond_length_loss,
    bond_angle_loss,
    omega_loss,
    o_chirality_loss,
    virtual_cb_loss,
    dihedral_angle,
    BOND_LENGTHS,
    BOND_ANGLES,
)
from .diffusion import (
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
)
from .samplers import (
    BaseSampler,
    DDPMSampler,
    HeunSampler,
    DeterministicDDIMSampler,
    EDMSampler,
    create_sampler,
    list_samplers,
)
from .training_utils import (
    random_rigid_augment,
    random_rotation_matrix,
    af3_loss_weight,
    MultiCopyTrainer,
    VectorizedMultiCopyTrainer,
)
from .self_conditioning import (
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
    "iterative_assembler": IterativeAtomAssembler,
    "resfold_assembler": ResFoldAssembler,
}


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODELS.keys())


def create_model(name: str, **kwargs) -> BaseDecoder:
    """Create a model by name.

    Args:
        name: Model name (see list_models())
        **kwargs: Model-specific arguments (h_dim, n_layers, etc.)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is unknown
    """
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return _MODELS[name](**kwargs)


def get_model_class(name: str) -> type:
    """Get model class by name (for inspection without instantiation)."""
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return _MODELS[name]


__all__ = [
    # Models
    "BaseDecoder",
    "AttentionDiffusionV2",
    "HierarchicalDecoder",
    "PairformerDecoder",
    "AF3StyleDecoder",
    "ResidueDenoiser",
    "AtomRefiner",
    "AtomRefinerV2",
    "AtomRefinerV2MultiSample",
    "AtomRefinerContinuous",
    "ResFoldPipeline",
    "ResFoldE2E",
    "sample_e2e",
    "IterativeAtomAssembler",
    "ResFoldAssembler",
    # Clustering utilities
    "compute_bond_connectivity",
    "hierarchical_cluster_atoms",
    "select_next_atoms_to_place",
    "get_placement_order",
    "simulate_known_mask",
    "GeometryLoss",
    "create_model",
    "list_models",
    "get_model_class",
    "sinusoidal_pos_enc",
    # Diffusion - schedules
    "CosineSchedule",
    "create_schedule",
    "list_schedules",
    # Diffusion - noise types
    "GaussianNoise",
    "LinearChainNoise",
    "LinearChainFlow",
    "create_noiser",
    "list_noise_types",
    "generate_extended_chain",
    "kabsch_align_to_target",
    # Diffusion - schedules
    "LinearSchedule",
    "KarrasSchedule",
    # Diffusion - noise types
    "VENoiser",
    # Diffusion - curriculum
    "TimestepCurriculum",
    # Diffusion - samplers
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
