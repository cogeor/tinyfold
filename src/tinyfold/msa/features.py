"""Paired MSA -> APC-corrected coevolution pair features ``[L, L, F]``.

What this computes
------------------
Mutual information between alignment columns, with the **average product
correction** (APC) -- the classic, robust coevolution contact score. It is
feature source (a) from spec §5 ("APC-corrected covariance / DCA couplings").

Why MI+APC and not full mfDCA: mean-field DCA needs the inverse of a
``(L*21) x (L*21)`` covariance (L=600 -> a 12600^2 matrix), which is both heavy
and numerically delicate at the MSA depths DIPS actually has. MI+APC captures the
same coevolution signal at a fraction of the cost and has no failure mode when
the MSA is shallow. If Step 0 shows deep pairing, upgrading this one function to
mfDCA (or MSA-Transformer row attention) is the natural follow-up -- the model
side does not change.

Why APC matters: raw MI has a strong background from per-column entropy (highly
variable columns look coupled to everything). APC subtracts the outer product of
row/column means, which is what turns MI into a usable contact prior.

Channels (F = 4):
    0  mi_apc    APC-corrected MI -- the contact signal
    1  mi_raw    uncorrected MI
    2  coverage  fraction of (weighted) rows where BOTH i and j are non-gap
    3  log_neff  per-complex depth, broadcast; lets a linear proj discount
                 shallow-MSA targets (graceful degradation, spec §11)

Everything is in numpy and runs offline -- the result is cached per complex
(~2.9 MB at L=600, fp16), mirroring the ESM-cache pattern.
"""

from __future__ import annotations

import numpy as np

from tinyfold.msa.a3m import (
    MSA_GAP_IDX,
    MSA_NUM_STATES,
    MsaRecord,
    encode_msa,
    sequence_weights,
)

# 4 channels: mi_apc, mi_raw, coverage, log_neff.
MSA_FEAT_DIM = 4

__all__ = [
    "MSA_FEAT_DIM",
    "build_msa_pair_features",
    "coevolution_features",
    "encode_msa",
    "msa_feat_dim",
    "sequence_weights",
]


def msa_feat_dim() -> int:
    """Channel count of :func:`coevolution_features` (mirrors template_feat_dim)."""
    return MSA_FEAT_DIM


def _one_hot(codes: np.ndarray) -> np.ndarray:
    """``[N, L, 21]`` float32 one-hot."""
    n, length = codes.shape
    oh = np.zeros((n, length, MSA_NUM_STATES), dtype=np.float32)
    oh[np.arange(n)[:, None], np.arange(length)[None, :], codes] = 1.0
    return oh


def coevolution_features(
    seqs: list[str],
    *,
    reweight: bool = True,
    identity_threshold: float = 0.8,
    chunk_size: int = 64,
    eps: float = 1e-9,
) -> np.ndarray:
    """Coevolution pair features for a paired alignment.

    Args:
        seqs: paired alignment rows, all of length L. Row n of chain A must
            correspond to row n of chain B (see :mod:`tinyfold.msa.pairing`) --
            that correspondence IS the signal.
        reweight: apply redundancy reweighting (disable only in tests).
        chunk_size: rows of the joint computed at once. Bounds memory; does not
            change the result.

    Returns:
        ``[L, L, 4]`` float32: (mi_apc, mi_raw, coverage, log_neff).
    """
    codes = encode_msa(seqs)
    n, length = codes.shape

    w = sequence_weights(codes, identity_threshold) if reweight else np.ones(n)
    meff = float(w.sum())
    p = (w / meff).astype(np.float32)  # normalized row weights, sum = 1

    oh = _one_hot(codes)                                  # [N, L, 21]
    f_i = np.einsum("n,nia->ia", p, oh)                   # [L, 21] single freqs

    flat = oh.reshape(n, length * MSA_NUM_STATES)         # [N, L*21]
    flat_w = flat * p[:, None]                            # weighted rows

    # Non-gap indicator for the coverage channel.
    is_res = (codes != MSA_GAP_IDX).astype(np.float32)    # [N, L]
    coverage = np.einsum("n,ni,nj->ij", p, is_res, is_res).astype(np.float32)

    mi = np.zeros((length, length), dtype=np.float32)
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        c = stop - start
        # Joint freqs for this row-chunk: f_ij[i, j, a, b].
        # One BLAS matmul, then reshape -- an explicit einsum over n would
        # materialise an [N, c, L, 21, 21] intermediate.
        block = flat_w[:, start * MSA_NUM_STATES : stop * MSA_NUM_STATES].T @ flat
        f_ij = block.reshape(c, MSA_NUM_STATES, length, MSA_NUM_STATES)
        f_ij = f_ij.transpose(0, 2, 1, 3)                  # [c, L, 21, 21]

        # Independent expectation f_i(a) * f_j(b).
        outer = f_i[start:stop, None, :, None] * f_i[None, :, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = f_ij * np.log(f_ij / (outer + eps))
        # 0 * log 0 == 0 by convention.
        mi[start:stop] = np.where(f_ij > eps, term, 0.0).sum(axis=(2, 3))

    # A column is trivially informative about itself; that is not a contact and
    # would dominate the APC background.
    np.fill_diagonal(mi, 0.0)
    mi = 0.5 * (mi + mi.T)                                 # symmetrise fp noise

    mi_apc = _apc(mi)

    log_neff = np.float32(np.log1p(meff))
    feats = np.empty((length, length, MSA_FEAT_DIM), dtype=np.float32)
    feats[..., 0] = mi_apc
    feats[..., 1] = mi
    feats[..., 2] = coverage
    feats[..., 3] = log_neff
    return feats


def _apc(mi: np.ndarray) -> np.ndarray:
    """Average product correction: ``mi - (mi_i. * mi_.j) / mi_..``."""
    total = mi.mean()
    if total <= 0:
        return mi.copy()
    row = mi.mean(axis=1, keepdims=True)
    col = mi.mean(axis=0, keepdims=True)
    out = mi - (row @ col) / total
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def build_msa_pair_features(
    paired_a: list[MsaRecord],
    paired_b: list[MsaRecord],
    **kwargs,
) -> np.ndarray:
    """Concatenate a paired A/B alignment into one complex and featurise it.

    ``paired_a[n]`` and ``paired_b[n]`` must be the same organism -- that is what
    :func:`tinyfold.msa.pairing.pair_msas` guarantees. The cross-chain block of
    the returned matrix is the docking prior we are actually after.

    Returns:
        ``[L, L, 4]`` with ``L = LA + LB``, matching the concatenated
        ``seq`` layout used throughout the dataset.
    """
    if len(paired_a) != len(paired_b):
        raise ValueError(
            "paired MSAs must have the same number of rows "
            f"(got {len(paired_a)} vs {len(paired_b)}); pair them with pair_msas()"
        )
    rows = [ra.seq + rb.seq for ra, rb in zip(paired_a, paired_b)]
    return coevolution_features(rows, **kwargs)
