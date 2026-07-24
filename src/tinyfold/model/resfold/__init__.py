"""ResFold model components.

The supported model is **ResFoldOneStep**: a single network with parallel
centroid + atom heads, trained end-to-end (~11.8M params). The legacy two-stage
pipeline (ResFoldPipeline / ResFoldE2E / ResFoldAssembler and the Stage-2
refiners) has been removed — every recorded run and every config is onestep.
"""

from tinyfold.model.resfold.confidence_head import ConfidenceHead
from tinyfold.model.resfold.denoiser import ResidueDenoiser
from tinyfold.model.resfold.onestep import ResFoldOneStep

__all__ = [
    "ConfidenceHead",
    "ResFoldOneStep",
    "ResidueDenoiser",
]
