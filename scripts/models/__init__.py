"""Model factory for TinyFold decoders and diffusion components.
(Backward compatibility wrapper around tinyfold.model)

DEPRECATED: This module is kept for backward compatibility only.
New code should import directly from tinyfold.model instead:
    from tinyfold.model import create_model, list_models
    from tinyfold.model.archive import AF3StyleDecoder, AttentionDiffusionV2

Usage (legacy):
    from models import create_model, list_models
    from models import create_schedule, create_noiser, list_schedules, list_noise_types
"""

import warnings

# Import archive models from tinyfold.model.archive (new location)
from tinyfold.model.archive import (
    AF3StyleDecoder,
    AttentionDiffusionV2,
    HierarchicalDecoder,
    PairformerDecoder,
    AtomRefiner,
    AtomRefinerContinuous,
    MultiSampler,
    aggregate_samples,
    sample_centroids_multi,
    IterativeAtomAssembler,
    GeometricAtomDecoder,
    GeometricAtomDecoderV2,
    BaseDecoder,
    sinusoidal_pos_enc,
    self_conditioning_training_step,
    sample_step_with_self_cond,
    create_self_cond_embedding,
)

# Model registry - start with archive models
_MODELS = {
    "attention_v2": AttentionDiffusionV2,
    "af3_style": AF3StyleDecoder,
    "hierarchical": HierarchicalDecoder,
    "pairformer": PairformerDecoder,
}

# Import tinyfold.model components
try:
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
except ImportError as e:
    warnings.warn(f"Could not import tinyfold.model.losses: {e}")
    GeometryLoss = None
    bond_length_loss = None
    bond_angle_loss = None
    omega_loss = None
    o_chirality_loss = None
    virtual_cb_loss = None
    dihedral_angle = None
    BOND_LENGTHS_ANGSTROM = None
    get_normalized_bond_lengths = None
    BOND_ANGLES = None

try:
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
except ImportError as e:
    warnings.warn(f"Could not import tinyfold.model.diffusion: {e}")
    CosineSchedule = None
    LinearSchedule = None
    KarrasSchedule = None
    GaussianNoise = None
    LinearChainNoise = None
    LinearChainFlow = None
    VENoiser = None
    TimestepCurriculum = None
    create_schedule = None
    create_noiser = None
    list_schedules = None
    list_noise_types = None
    generate_extended_chain = None
    kabsch_align_to_target = None
    BaseSampler = None
    DDPMSampler = None
    HeunSampler = None
    DeterministicDDIMSampler = None
    EDMSampler = None
    create_sampler = None
    list_samplers = None

try:
    from tinyfold.training.utils import (
        random_rigid_augment,
        random_rotation_matrix,
        af3_loss_weight,
        MultiCopyTrainer,
        VectorizedMultiCopyTrainer,
    )
except ImportError as e:
    warnings.warn(f"Could not import tinyfold.training.utils: {e}")
    random_rigid_augment = None
    random_rotation_matrix = None
    af3_loss_weight = None
    MultiCopyTrainer = None
    VectorizedMultiCopyTrainer = None

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

    _MODELS["resfold_stage1"] = ResidueDenoiser
    _MODELS["resfold_stage2_multi"] = AtomRefinerV2MultiSample
    _MODELS["resfold"] = ResFoldPipeline
    _MODELS["resfold_e2e"] = ResFoldE2E
    _MODELS["resfold_assembler"] = ResFoldAssembler
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
    _MODELS["iterfold"] = IterFold
except ImportError as e:
    warnings.warn(f"Could not import iterfold models: {e}")
    IterFold = None
    AnchorDecoder = None


def list_models():
    """Return list of available model names."""
    return list(_MODELS.keys())


def create_model(name, **kwargs):
    """Create a model by name."""
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _MODELS[name](**kwargs)


def get_model_class(name):
    """Get model class by name."""
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _MODELS[name]
