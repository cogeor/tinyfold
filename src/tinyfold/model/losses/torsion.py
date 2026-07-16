"""Symmetry-corrected TORSION-space losses (stage-3, variant A / B4).

Design: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §9.

The torsion analogue of :mod:`tinyfold.model.losses.sidechain`. Where the offset
loss corrects the ATOM-RENAMING ambiguity in Cartesian space, a torsion loss must
correct the CHI-PI-PERIODIC ambiguity: chi2 of ASP/PHE/TYR and chi3 of GLU are
symmetric under a 180-degree flip (the terminal group is indistinguishable), so a
prediction off by exactly pi on such a chi is CORRECT and must not be penalised.

chi is compared as (cos, sin) unit vectors (no +-pi wraparound cliff). For a
pi-periodic chi the flipped ground truth is simply the negated vector, so the loss
is ``min(||v - u||^2, ||v + u||^2)``, chosen per (residue, chi).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tinyfold.atom14 import restype_chi_pi_periodic

_PI_PERIODIC_CACHE: dict[torch.device, Tensor] = {}


def chi_pi_periodic_table(device: torch.device) -> Tensor:
    """``[21, 4]`` bool pi-periodic table, cached per device."""
    if device not in _PI_PERIODIC_CACHE:
        _PI_PERIODIC_CACHE[device] = torch.as_tensor(
            restype_chi_pi_periodic(), dtype=torch.bool, device=device
        )
    return _PI_PERIODIC_CACHE[device]


def torsion_symmetry_loss(
    pred_vec: Tensor,   # [B, L, 4, 2] unit (cos, sin) prediction from the head
    gt_chi: Tensor,     # [B, L, 4] ground-truth chi (radians)
    chi_mask: Tensor,   # [B, L, 4] bool, valid chis
    aatype: Tensor,     # [B, L] long
    reduction: str = "mean",
) -> Tensor:
    """(cos,sin) L2 loss, minimised over the pi-flip for pi-periodic chis.

    The min is per (residue, chi): each ambiguous chi independently picks the
    labelling that fits. Returns a scalar (``"mean"`` over valid chis) or the
    per-chi loss ``[B, L, 4]`` (``"none"``).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")

    gt_vec = torch.stack([torch.cos(gt_chi), torch.sin(gt_chi)], dim=-1)   # [B,L,4,2]
    base = ((pred_vec - gt_vec) ** 2).sum(-1)                             # [B,L,4]
    flip = ((pred_vec + gt_vec) ** 2).sum(-1)                             # pi-flip = -gt_vec
    periodic = chi_pi_periodic_table(pred_vec.device)[aatype]             # [B,L,4]
    per_chi = torch.where(periodic, torch.minimum(base, flip), base)

    m = chi_mask.to(pred_vec.dtype)
    per_chi = per_chi * m
    if reduction == "none":
        return per_chi
    return per_chi.sum() / m.sum().clamp(min=1.0)


def chi_mae_deg(
    pred_chi: Tensor,   # [B, L, 4] radians
    gt_chi: Tensor,     # [B, L, 4] radians
    chi_mask: Tensor,   # [B, L, 4] bool
    aatype: Tensor,     # [B, L] long
) -> Tensor:
    """Mean absolute chi error in DEGREES, wrapped and pi-symmetry-corrected.

    Reporting metric (not a training loss). For pi-periodic chis the error is
    taken mod pi, so a correct-but-flipped rotamer reads ~0.
    """
    d = pred_chi - gt_chi
    periodic = chi_pi_periodic_table(pred_chi.device)[aatype]
    # Wrap to (-pi, pi]; for periodic chis wrap to (-pi/2, pi/2].
    period = torch.where(periodic, torch.full_like(d, math.pi), torch.full_like(d, 2 * math.pi))
    err = torch.remainder(d + period / 2, period) - period / 2
    err = err.abs() * (180.0 / math.pi)
    m = chi_mask.to(pred_chi.dtype)
    return (err * m).sum() / m.sum().clamp(min=1.0)
