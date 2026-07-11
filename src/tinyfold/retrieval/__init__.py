"""Retrieval / template library for TinyFold.

Moves geometry OUT of the model weights and into an external structural library
the small model retrieves from and refines. See
``notes/2026-07-11-retrieval-library-SPEC.md`` (local).

Public surface:
    * ``build_template_pair_features`` — AF3-style RELATIVE pair features
      (CA-CA distogram + local-frame unit vectors) computed from per-residue
      backbone template coordinates. Rotation/translation invariant.
    * ``TEMPLATE_FEAT_DIM`` — channel count for a given RBF bin count.
"""

from .template_features import (
    TEMPLATE_FEAT_DIM,
    build_template_pair_features,
    local_frames,
    template_feat_dim,
)

__all__ = [
    "TEMPLATE_FEAT_DIM",
    "template_feat_dim",
    "build_template_pair_features",
    "local_frames",
]
