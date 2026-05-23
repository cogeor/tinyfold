"""Archived models - kept for backward compatibility with old checkpoints.

These models are deprecated but maintained for loading old checkpoints
and for reference. New code should use the models in tinyfold.model.resfold
or tinyfold.model.iterfold instead.

Models:
- BaseDecoder: Base class for diffusion decoders
- AttentionDiffusionV2: Original attention-based diffusion decoder
- AF3StyleDecoder: AlphaFold3-style decoder with trunk and denoiser
- HierarchicalDecoder: Residue-level decoder with atom offset prediction
- PairformerDecoder: Pairformer-based diffusion decoder
- AtomRefiner: Stage 2 atom refinement from centroids
- AtomRefinerContinuous: Stage 2 atom refinement (single-shot)
- MultiSampler: Multi-sampling for improved inference
- IterativeAtomAssembler: Iterative atom position prediction
- GeometricAtomDecoder: Atom decoder with geometric priors
- GeometricAtomDecoderV2: Version 2 with local frame computation

Utilities:
- sinusoidal_pos_enc: Sinusoidal positional encoding
- self_conditioning_training_step: Training with self-conditioning
- sample_step_with_self_cond: Sampling with self-conditioning
- create_self_cond_embedding: Create embedding layer for self-conditioning
"""

# Base class and utilities
from .base import BaseDecoder, sinusoidal_pos_enc

# Main models
from .attention_v2 import AttentionDiffusionV2
from .af3_style import AF3StyleDecoder
from .hierarchical import HierarchicalDecoder
from .pairformer_decoder import PairformerDecoder
from .atomrefine import AtomRefiner
from .atomrefine_continuous import AtomRefinerContinuous
from .iterative_assembler import IterativeAtomAssembler
from .atom_decoder import (
    GeometricAtomDecoder,
    GeometricAtomDecoderV2,
    AtomAttentionBlock,
)

# Self-conditioning utilities
from .self_conditioning import (
    SelfConditioningMixin,
    self_conditioning_training_step,
    sample_step_with_self_cond,
    create_self_cond_embedding,
)

# Multi-sampling utilities
from .multi_sample import (
    MultiSampler,
    aggregate_samples,
    sample_centroids_multi,
    AggregationMethod,
)

__all__ = [
    # Base
    "BaseDecoder",
    "sinusoidal_pos_enc",
    # Models
    "AttentionDiffusionV2",
    "AF3StyleDecoder",
    "HierarchicalDecoder",
    "PairformerDecoder",
    "AtomRefiner",
    "AtomRefinerContinuous",
    "IterativeAtomAssembler",
    "GeometricAtomDecoder",
    "GeometricAtomDecoderV2",
    "AtomAttentionBlock",
    # Self-conditioning
    "SelfConditioningMixin",
    "self_conditioning_training_step",
    "sample_step_with_self_cond",
    "create_self_cond_embedding",
    # Multi-sampling
    "MultiSampler",
    "aggregate_samples",
    "sample_centroids_multi",
    "AggregationMethod",
]
