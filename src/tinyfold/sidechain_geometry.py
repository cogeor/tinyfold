"""Sidechain torsion (chi) geometry: extraction and rigid placement (variant A).

Design authority: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §4A. This is
the geometry half of the torsion sidechain variant -- the map between atom14
Cartesian coords and the chi1..chi4 torsion representation the head diffuses:

    extract_chi : atom14 coords -> chi angles [.,L,4] + mask
    place_chi   : chi angles + backbone -> atom14 coords   (see sidechain_frames)

Everything is torch and batched. Angles are radians in (-pi, pi].

Why torsions (vs the free-offset variant B): chi has <=4 DOF/residue and valid
bond lengths/angles BY CONSTRUCTION, which is exactly the property the Cartesian
free-offset head could not hold (it regressed toward a per-residue mean at mid
sigma; see the S5 finding). The placement geometry (bond lengths/angles/fixed
dihedrals) is data-derived, not lifted from AF2 -- see sidechain_frames.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tinyfold.atom14 import restype_chi_atom14_indices, restype_chi_mask


def dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-8) -> Tensor:
    """Signed dihedral angle of the four points p0-p1-p2-p3 (radians, atan2).

    Each ``p*`` is ``[..., 3]``; returns ``[...]``. The angle is the rotation
    about the p1->p2 axis taking the p0 side onto the p3 side, sign by the
    right-hand rule -- the standard IUPAC torsion convention.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    # Components of b0, b2 perpendicular to the central axis.
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


# Device/dtype-cached lookup tables (indexed by aatype).
_CHI_IDX_CACHE: dict[torch.device, Tensor] = {}
_CHI_MASK_CACHE: dict[torch.device, Tensor] = {}


def _chi_tables(device: torch.device) -> tuple[Tensor, Tensor]:
    if device not in _CHI_IDX_CACHE:
        idx = torch.from_numpy(restype_chi_atom14_indices()).to(device)      # [21,4,4]
        msk = torch.from_numpy(restype_chi_mask()).to(device)                # [21,4]
        _CHI_IDX_CACHE[device] = idx
        _CHI_MASK_CACHE[device] = msk
    return _CHI_IDX_CACHE[device], _CHI_MASK_CACHE[device]


def extract_chi(
    coords_atom14: Tensor,   # [..., L, 14, 3]
    aatype: Tensor,          # [..., L] long
    atom_mask: Tensor | None = None,  # [..., L, 14] bool (resolved atoms)
) -> tuple[Tensor, Tensor]:
    """atom14 coords -> ``(chi [...,L,4], chi_mask [...,L,4])``.

    ``chi_mask`` is True where the residue type defines that chi AND all four
    atoms are resolved (when ``atom_mask`` is given). Absent chis read 0.
    """
    device = coords_atom14.device
    chi_idx, chi_present = _chi_tables(device)               # [21,4,4], [21,4]
    idx = chi_idx[aatype]                                    # [...,L,4,4]
    present = chi_present[aatype]                            # [...,L,4]

    # Gather the four atoms of each chi: expand idx to index the atom axis.
    lead = coords_atom14.shape[:-2]                          # (..., L)
    gather_idx = idx.reshape(*lead, 16, 1).expand(*lead, 16, 3)
    atoms = torch.gather(coords_atom14, -2, gather_idx).reshape(*lead, 4, 4, 3)
    p0, p1, p2, p3 = atoms[..., 0, :], atoms[..., 1, :], atoms[..., 2, :], atoms[..., 3, :]
    chi = dihedral(p0, p1, p2, p3)                           # [...,L,4]

    mask = present
    if atom_mask is not None:
        # Every one of the 4 atoms must be resolved for the chi to be valid.
        four_present = torch.gather(atom_mask, -1, idx.reshape(*lead, 16)).reshape(*lead, 4, 4)
        mask = mask & four_present.all(dim=-1)
    chi = chi * mask                                         # zero absent chis
    return chi, mask
