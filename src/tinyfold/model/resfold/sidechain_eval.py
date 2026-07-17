"""All-atom sidechain evaluation for the third (torsion) diffusion stage (C3).

Two pieces the sidechain A/B needs, kept out of the hot training path:

1. ``build_atom14_geometry`` -- the dataset-averaged idealized geometry that
   ``place_chi`` needs at inference. Per-complex geometry (as the overfit gate uses)
   is unavailable at inference, so we average once over a fixed atom14 sample; this
   is the ~0.25 A round-trip floor from the B2 gate. Cached per (dir, n, device).

2. ``sidechain_rmsd_binned`` -- the headline number, symmetry-corrected sidechain
   RMSD, reported OVERALL and BINNED by backbone quality. Sidechain accuracy is
   bounded by backbone error, so a single mean would wrongly blame chi for a bad
   backbone; the binning separates "the packer is wrong" from "the backbone under
   it is wrong".
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tinyfold.model.losses.sidechain import make_alt_gt
from tinyfold.sidechain_frames import derive_ideal_geometry

_GEOM_CACHE: dict = {}


def build_atom14_geometry(
    atom14_dir: str | Path,
    n_complexes: int = 200,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Dataset-averaged idealized geometry for ``place_chi`` (cached).

    Loads up to ``n_complexes`` atom14 npz (deterministic sorted order) and calls
    ``derive_ideal_geometry`` over the pooled residues. Padded per-residue so a
    ragged set of lengths stacks; padding is masked out via ``mask_atom14``.
    """
    key = (str(atom14_dir), int(n_complexes), str(device))
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]

    paths = sorted(Path(atom14_dir).glob("*.npz"))[:n_complexes]
    if not paths:
        raise ValueError(f"no atom14 npz under {atom14_dir}")
    coords_list, mask_list, seq_list = [], [], []
    for p in paths:
        with np.load(p) as z:
            coords_list.append(torch.from_numpy(np.asarray(z["coords_atom14"])).float())
            mask_list.append(torch.from_numpy(np.asarray(z["mask_atom14"])).bool())
            seq_list.append(torch.from_numpy(np.asarray(z["seq_indices"])).long())
    # Concatenate along the residue axis (geometry is per-residue-instance, so a
    # flat [sum_L, 14, *] stack is exactly what derive_ideal_geometry averages).
    coords = torch.cat(coords_list, dim=0).to(device)      # [sumL, 14, 3]
    amask = torch.cat(mask_list, dim=0).to(device)         # [sumL, 14]
    aseq = torch.cat(seq_list, dim=0).to(device)           # [sumL]
    geom = derive_ideal_geometry(coords, aseq, amask)
    _GEOM_CACHE[key] = geom
    return geom


def sidechain_rmsd_binned(
    pred_atom14: Tensor,   # [B, L, 14, 3]
    gt_atom14: Tensor,     # [B, L, 14, 3]
    atom_mask: Tensor,     # [B, L, 14] bool
    aatype: Tensor,        # [B, L] long
    res_mask: Tensor,      # [B, L] bool (valid residues)
    bb_quality: Tensor | None = None,   # [B, L] higher = better backbone (e.g. lDDT)
    bins: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.01),
) -> dict:
    """Symmetry-corrected per-residue sidechain RMSD, overall + binned by backbone.

    Only sidechain slots (4..13) count (backbone is a separate stage). Returns
    ``{"overall": float, "n": int, "bins": [{"lo","hi","rmsd","n"}...]}``. When
    ``bb_quality`` is None only ``overall`` is meaningful.
    """
    sc = slice(4, None)
    m = atom_mask[..., sc].to(pred_atom14.dtype)                      # [B,L,10]
    alt_gt = make_alt_gt(gt_atom14, aatype)

    se_gt = ((pred_atom14[..., sc, :] - gt_atom14[..., sc, :]) ** 2).sum(-1) * m
    se_alt = ((pred_atom14[..., sc, :] - alt_gt[..., sc, :]) ** 2).sum(-1) * m
    use_alt = (se_alt.sum(-1) < se_gt.sum(-1)).unsqueeze(-1)          # [B,L,1]
    se = torch.where(use_alt, se_alt, se_gt)                         # [B,L,10]

    per_res_se = se.sum(-1)                                          # [B,L]
    per_res_n = m.sum(-1)                                            # [B,L] atoms/res
    valid = res_mask & (per_res_n > 0)

    def _rmsd(sel: Tensor) -> tuple[float, int]:
        n_atoms = (per_res_n * sel).sum().clamp(min=1.0)
        return (torch.sqrt((per_res_se * sel).sum() / n_atoms).item(),
                int(sel.sum().item()))

    overall_rmsd, overall_n = _rmsd(valid)
    out: dict = {"overall": overall_rmsd, "n": overall_n, "bins": []}
    if bb_quality is not None:
        for lo, hi in pairwise(bins):
            sel = valid & (bb_quality >= lo) & (bb_quality < hi)
            r, n = _rmsd(sel)
            out["bins"].append({"lo": lo, "hi": hi, "rmsd": r, "n": n})
    return out
