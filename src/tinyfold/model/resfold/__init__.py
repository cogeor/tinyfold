"""ResFold model components.

Models in this repo:
- **ResFoldOneStep** — the SUPPORTED headline model: a single network with
  parallel centroid + atom heads, trained end-to-end (11.8M params). New work
  should use this.
- ResFoldPipeline — the legacy two-stage variant (Stage 1 ResidueDenoiser
  centroid diffusion -> Stage 2 AtomRefinerV2 atom refinement). Kept for
  reference / older checkpoints.
"""

from tinyfold.model.resfold.denoiser import ResidueDenoiser
from tinyfold.model.resfold.pipeline import ResFoldPipeline
from tinyfold.model.resfold.refiner import AtomRefinerV2
from tinyfold.model.resfold.assembler import ResFoldAssembler
from tinyfold.model.resfold.e2e import ResFoldE2E, sample_e2e
from tinyfold.model.resfold.atomrefine_multi_sample import AtomRefinerV2MultiSample
from tinyfold.model.resfold.confidence_head import ConfidenceHead
from tinyfold.model.resfold.onestep import ResFoldOneStep

__all__ = [
    "ResFoldOneStep",
    "ResidueDenoiser",
    "ResFoldPipeline",
    "AtomRefinerV2",
    "ResFoldAssembler",
    "ResFoldE2E",
    "sample_e2e",
    "AtomRefinerV2MultiSample",
    "ConfidenceHead",
]
