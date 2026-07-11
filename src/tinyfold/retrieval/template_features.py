"""AF3-style RELATIVE template pair features.

Design (see notes/2026-07-11-retrieval-library-BUILD-REPORT.md, D3)
-------------------------------------------------------------------
A template is stored PER-RESIDUE as backbone coords aligned (by sequence) to our
residues: ``coords_res [B, L, 4, 3]`` (N, CA, C, O), a coverage mask
``mask [B, L]``, and a ``frame_id [B, L]`` saying which rigid group each
residue's coordinates live in. The O(L^2) pair features are built HERE, once per
sample, inside the trunk's pair track — so nothing O(L^2) is ever cached or
cropped.

The features are RELATIVE and therefore invariant to how the template is posed:

* **CA-CA distogram** — radial-basis encoding of ``||CA_i - CA_j||``. Invariant
  to global rotation + translation.
* **Local-frame unit vector** — direction of ``CA_j - CA_i`` expressed in
  residue i's backbone frame (Gram-Schmidt on N,CA,C), then unit-normalized.
  Invariant to global rotation (the frame rotates with the structure), and
  carries orientation the scalar distance cannot.
* **Coverage channel** — 1.0 where the pair is valid, else 0.0, so the embedder
  can tell "covered but far" from "no template here".

A pair (i, j) is VALID iff both residues are covered AND share a frame:
``mask_i & mask_j & (frame_id_i == frame_id_j)``. For an oracle (whole-complex)
template every residue is in frame 0 → all covered pairs valid, including the
cross-chain docking. For a real monomer template each chain sits in its own
frame → only intra-chain pairs are valid and the model must still solve docking.

Distances are in the MODEL'S normalized coordinate units (coords divided by the
run's ``global_scale``). ``d_max`` is therefore in those units; the default
``4.0`` spans ~40 A at ``global_scale ~= 11``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def template_feat_dim(n_rbf: int) -> int:
    """Channel count: RBF distogram + 3 (unit vector) + 1 (coverage)."""
    return int(n_rbf) + 3 + 1


# Convenience constant for the default bin count used throughout the codebase.
TEMPLATE_FEAT_DIM = template_feat_dim(32)


def local_frames(coords_res: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-residue backbone frames via Gram-Schmidt on (N, CA, C).

    Args:
        coords_res: ``[..., L, 4, 3]`` backbone atoms ordered N, CA, C, O.
        eps: numerical floor for normalization.

    Returns:
        ``R [..., L, 3, 3]`` rotation whose COLUMNS are the frame axes
        (e1, e2, e3) in the global frame. To express a global vector ``v`` in
        residue i's local frame use ``R_i^T @ v``.
    """
    n = coords_res[..., 0, :]
    ca = coords_res[..., 1, :]
    c = coords_res[..., 2, :]

    v1 = c - ca                                   # CA -> C
    v2 = n - ca                                   # CA -> N
    e1 = F.normalize(v1, dim=-1, eps=eps)
    # Remove the e1 component from v2, then normalize -> e2.
    u2 = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    # Stack as columns: R[..., :, k] = e_{k+1}.
    R = torch.stack([e1, e2, e3], dim=-1)         # [..., L, 3, 3]
    return R


def build_template_pair_features(
    coords_res: Tensor,      # [B, L, 4, 3] template backbone (normalized units)
    mask: Tensor,            # [B, L] bool coverage
    frame_id: Tensor,        # [B, L] long rigid-group id (oracle: all 0)
    n_rbf: int = 32,
    d_min: float = 0.0,
    d_max: float = 4.0,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Build relative template pair features.

    Returns:
        feats:      ``[B, L, L, F]`` with ``F = template_feat_dim(n_rbf)``.
                    Zeroed on invalid pairs.
        pair_valid: ``[B, L, L]`` bool.
    """
    B, L = mask.shape
    device, dtype = coords_res.device, coords_res.dtype
    ca = coords_res[..., 1, :]                                   # [B, L, 3]

    # --- pairwise CA-CA displacement & distance -----------------------------
    # disp[b, i, j] = CA_j - CA_i  (vector rooted at residue i).
    disp = ca.unsqueeze(1) - ca.unsqueeze(2)                     # [B, L, L, 3]
    dist = disp.norm(dim=-1)                                     # [B, L, L]

    # --- RBF distogram ------------------------------------------------------
    centers = torch.linspace(d_min, d_max, n_rbf, device=device, dtype=dtype)
    sigma = (d_max - d_min) / max(n_rbf - 1, 1)
    rbf = torch.exp(-((dist.unsqueeze(-1) - centers) / (sigma + eps)) ** 2)  # [B, L, L, n_rbf]

    # --- local-frame unit vector (dir of CA_j in frame of i) ----------------
    R = local_frames(coords_res, eps=eps)                       # [B, L, 3, 3]
    # Express disp (rooted at i) in i's local frame: R_i^T @ disp_ij.
    # R^T: [B, L, 3, 3]; einsum over the axis dim.
    Rt = R.transpose(-1, -2)                                    # [B, L, 3, 3] = R_i^T
    # local_vec[b,i,j,m] = sum_k Rt[b,i,m,k] * disp[b,i,j,k]
    local_vec = torch.einsum("bimk,bijk->bijm", Rt, disp)      # [B, L, L, 3]
    unit_vec = F.normalize(local_vec, dim=-1, eps=eps)          # [B, L, L, 3]

    # --- validity mask ------------------------------------------------------
    same_frame = frame_id.unsqueeze(2) == frame_id.unsqueeze(1)  # [B, L, L]
    cov = mask.unsqueeze(2) & mask.unsqueeze(1)                   # [B, L, L]
    pair_valid = cov & same_frame                                # [B, L, L]
    valid_f = pair_valid.to(dtype).unsqueeze(-1)                 # [B, L, L, 1]

    feats = torch.cat([rbf, unit_vec, valid_f], dim=-1)          # [B, L, L, F]
    feats = feats * valid_f                                       # zero invalid pairs
    return feats, pair_valid
