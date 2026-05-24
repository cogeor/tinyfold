"""Geometry primitives shared across the model.

Currently exports the Kabsch rigid-alignment helper used by the loss code,
the diffusion sampler's frame-alignment path, and the Boltz-style Kabsch
interpolation sampler (PLAN
``.delegate/work/20260524-051951-prio01-retrain/03/PLAN.md``).
"""

from tinyfold.model.geometry.kabsch import kabsch_rigid

__all__ = ["kabsch_rigid"]
