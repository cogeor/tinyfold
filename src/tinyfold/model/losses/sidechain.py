"""Symmetry-corrected sidechain losses (stage-3).

Design: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §9.

    "chi-symmetry correction (MANDATORY). ASP/GLU/PHE/TYR/ARG symmetric terminal
     groups; naive loss double-penalizes 180-degree-equivalent rotamers."

The problem, concretely: ASP's two carboxyl oxygens OD1/OD2 are chemically
indistinguishable. A prediction that places them perfectly but with the labels
exchanged is CORRECT, yet a naive per-atom MSE charges it the full squared
distance between the two oxygens -- roughly 2.2 A apart. Left uncorrected the
model is trained to guess an arbitrary labelling, which is unlearnable noise on
~20% of residues.

The fix is AF2's: score against both the ground truth and the RENAMED ground
truth (:func:`tinyfold.atom14.alt_atom14_permutation`) and keep the better one,
independently PER RESIDUE.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tinyfold.atom14 import restype_alt_permutation

_ALT_PERM_CACHE: dict[torch.device, Tensor] = {}


def alt_permutation_table(device: torch.device) -> Tensor:
    """``[21, 14]`` long alt-permutation table, cached per device."""
    if device not in _ALT_PERM_CACHE:
        _ALT_PERM_CACHE[device] = torch.as_tensor(
            restype_alt_permutation(), dtype=torch.long, device=device
        )
    return _ALT_PERM_CACHE[device]


def make_alt_gt(gt_atom14: Tensor, aatype: Tensor) -> Tensor:
    """Renamed ("alt") ground truth: ``alt[..., i, :] = gt[..., perm[i], :]``.

    Args:
        gt_atom14: ``[B, L, 14, 3]``.
        aatype:    ``[B, L]`` long residue-type indices (project AA order).

    Returns:
        ``[B, L, 14, 3]``, identical to ``gt_atom14`` for unambiguous residues.
    """
    perm = alt_permutation_table(gt_atom14.device)[aatype]      # [B, L, 14]
    idx = perm.unsqueeze(-1).expand(-1, -1, -1, 3)              # [B, L, 14, 3]
    return torch.gather(gt_atom14, dim=2, index=idx)


def symmetric_sidechain_loss(
    pred_atom14: Tensor,   # [B, L, 14, 3]
    gt_atom14: Tensor,     # [B, L, 14, 3]
    atom_mask: Tensor,     # [B, L, 14] bool/float, present heavy atoms
    aatype: Tensor,        # [B, L] long
    reduction: str = "mean",
) -> Tensor:
    """Per-atom MSE, minimised over the two equivalent atom labellings.

    The min is taken PER RESIDUE: each residue independently picks the labelling
    that fits, which is what makes the correction sound -- a global choice would
    force every ambiguous residue in a structure to flip together.

    Args:
        reduction: ``"mean"`` (over present atoms) or ``"per_residue"``
            (``[B, L]``, useful for diagnostics and for the overfit gate).

    Returns:
        Scalar, or ``[B, L]`` when ``reduction="per_residue"``.
    """
    if reduction not in ("mean", "per_residue"):
        raise ValueError(f"unknown reduction {reduction!r}")

    alt_gt = make_alt_gt(gt_atom14, aatype)
    m = atom_mask.to(pred_atom14.dtype)                          # [B, L, 14]

    # Squared error per atom, masked.
    se_gt = ((pred_atom14 - gt_atom14) ** 2).sum(-1) * m         # [B, L, 14]
    se_alt = ((pred_atom14 - alt_gt) ** 2).sum(-1) * m

    per_res_gt = se_gt.sum(-1)                                   # [B, L]
    per_res_alt = se_alt.sum(-1)
    # Per-residue choice of labelling.
    best = torch.minimum(per_res_gt, per_res_alt)                # [B, L]

    if reduction == "per_residue":
        return best

    denom = m.sum().clamp(min=1.0)
    return best.sum() / denom


def sidechain_rmsd(
    pred_atom14: Tensor,
    gt_atom14: Tensor,
    atom_mask: Tensor,
    aatype: Tensor,
) -> Tensor:
    """Symmetry-corrected sidechain RMSD in coordinate units (the S5 gate metric).

    Only SIDECHAIN slots (4..13) count: the backbone is frozen in stage 3, so
    including it would flatter the number with atoms the stage never predicts.

    The overfit gate (§11) requires this below ~0.5 A on a single complex before
    any data or scaling work is justified.
    """
    sc = slice(4, None)
    m = atom_mask[..., sc].to(pred_atom14.dtype)
    alt_gt = make_alt_gt(gt_atom14, aatype)

    se_gt = ((pred_atom14[..., sc, :] - gt_atom14[..., sc, :]) ** 2).sum(-1) * m
    se_alt = ((pred_atom14[..., sc, :] - alt_gt[..., sc, :]) ** 2).sum(-1) * m

    use_alt = (se_alt.sum(-1) < se_gt.sum(-1)).unsqueeze(-1)     # [B, L, 1]
    se = torch.where(use_alt, se_alt, se_gt)

    denom = m.sum().clamp(min=1.0)
    return torch.sqrt(se.sum() / denom)
