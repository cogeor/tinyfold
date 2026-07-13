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
from .template_source import (
    ORACLE_SOURCES,
    VALID_SOURCES,
    make_template_inputs,
)

__all__ = [
    "ORACLE_SOURCES",
    "TEMPLATE_FEAT_DIM",
    "VALID_SOURCES",
    "build_template_pair_features",
    "local_frames",
    "make_template_inputs",
    "template_feat_dim",
]
