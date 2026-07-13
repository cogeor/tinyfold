"""Shared Kabsch rigid-alignment primitive.

This module collapses three near-identical SVD recipes from across the codebase
into a single helper. See PLAN
``.delegate/work/20260524-051951-prio01-retrain/03/PLAN.md`` D1 for the design
discussion. Callers historically lived at:

- ``tinyfold.model.losses.mse.kabsch_align`` (training-time RMSE / MSE loss)
- ``tinyfold.model.diffusion.utils.kabsch_align_to_target`` (``align_per_step``
  sampler path; aligns x0_pred onto x_t)
- ``tinyfold.model.losses.mse.compute_c_rmsd`` (chain-A-then-apply-to-both
  complex RMSD; kept inline because it needs ``(R, t)`` separately)

The Boltz "Kabsch interpolation" sampler (Loop 03) is the new caller that
needs the freshly-stepped ``x`` rigid-aligned onto the previous step's ``x`` —
returning ``(R, t, aligned)`` lets every caller pick whichever slice it wants.
"""


import torch
from torch import Tensor


def kabsch_rigid(
    src: Tensor,
    tgt: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Find ``(R, t)`` minimising ``||R @ src + t - tgt||^2`` per batch element.

    Implements the standard Kabsch SVD recipe with proper-rotation enforcement
    via the sign-flip-D trick. Masked positions are zeroed out before SVD and
    their per-batch centroid divisors clamped to >=1.

    Args:
        src:  ``[B, N, 3]`` source coordinates (the ones that get moved).
        tgt:  ``[B, N, 3]`` target coordinates (the frame to align onto).
        mask: optional ``[B, N]`` boolean mask of valid positions.

    Returns:
        R:       ``[B, 3, 3]`` proper rotation matrices (det = +1).
        t:       ``[B, 3]``    translations such that ``R @ src_mean + t = tgt_mean``.
        aligned: ``[B, N, 3]`` equal to
                 ``torch.bmm(src, R.transpose(1, 2)) + t[:, None, :]``.
                 Note this is in ``tgt``'s translated frame (NOT centred at the
                 origin); callers wanting the centred convention should
                 subtract ``tgt_mean`` themselves.
    """
    assert src.shape == tgt.shape, \
        f"kabsch_rigid: src/tgt shape mismatch {src.shape} vs {tgt.shape}"
    assert src.dim() == 3 and src.shape[-1] == 3, \
        f"kabsch_rigid: expected [B, N, 3], got {src.shape}"

    B = src.shape[0]
    device = src.device

    # --- Centroids (mask-aware) ---
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).to(src.dtype)  # [B, N, 1]
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1).to(src.dtype)
        src_mean = (src * mask_exp).sum(dim=1, keepdim=True) / n_valid  # [B, 1, 3]
        tgt_mean = (tgt * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        src_mean = src.mean(dim=1, keepdim=True)
        tgt_mean = tgt.mean(dim=1, keepdim=True)

    # --- Centre ---
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    if mask is not None:
        src_c = src_c * mask_exp
        tgt_c = tgt_c * mask_exp

    # --- SVD: H = src_c^T @ tgt_c ---
    H = torch.bmm(src_c.transpose(1, 2), tgt_c)  # [B, 3, 3]
    U, _S, Vt = torch.linalg.svd(H)

    # Reflection guard: ensure det(R) = +1 by flipping the sign of the last
    # singular component if needed.
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=device, dtype=src.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # [B, 3, 3]

    # Translation so that R @ src_mean + t = tgt_mean.
    # src_mean is [B, 1, 3]; bmm needs a [B, 3, 1] column vec — easiest path
    # is matmul on the [B, 1, 3] @ R^T which gives [B, 1, 3].
    src_mean_rot = torch.bmm(src_mean, R.transpose(1, 2))  # [B, 1, 3]
    t = (tgt_mean - src_mean_rot).squeeze(1)  # [B, 3]

    # Apply R, t to the full src tensor.
    aligned = torch.bmm(src, R.transpose(1, 2)) + t.unsqueeze(1)  # [B, N, 3]

    return R, t, aligned
