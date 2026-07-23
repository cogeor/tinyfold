"""MSE-based losses with Kabsch alignment for structure prediction.

Provides rotation-invariant loss functions for comparing predicted
and ground truth protein structures.
"""


import torch
from torch import Tensor

from tinyfold.model.geometry import kabsch_rigid


def kabsch_align(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Kabsch alignment for rotation-invariant comparison.

    Computes optimal rotation to align pred to target using SVD.
    Both tensors are centered before alignment.

    Adapter over :func:`tinyfold.model.geometry.kabsch_rigid`: that helper
    returns ``aligned`` in target's TRANSLATED frame (i.e. ``+ target_mean``),
    while the historical ``kabsch_align`` contract returns BOTH ``pred_aligned``
    and ``target_centered`` in the ORIGIN-CENTRED frame. We subtract
    ``target_mean`` from the helper's output to restore that convention.

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        pred_aligned: Aligned predicted coordinates [B, N, 3] (origin-centred)
        target_centered: Centered target coordinates [B, N, 3] (origin-centred)
    """
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).to(target.dtype)
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1).to(target.dtype)
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        target_mean = target.mean(dim=1, keepdim=True)

    _, _, aligned = kabsch_rigid(pred, target, mask)
    # Helper returned aligned in target's translated frame; subtract target_mean
    # to put it back in the origin-centred frame the old contract used.
    pred_aligned = aligned - target_mean
    target_c = target - target_mean
    if mask is not None:
        pred_aligned = pred_aligned * mask_exp
        target_c = target_c * mask_exp

    return pred_aligned, target_c


def compute_mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    use_kabsch: bool = True,
    reduction: str = 'mean',
) -> Tensor:
    """MSE loss with optional Kabsch alignment.

    Uses per-sample averaging for correct gradient accumulation behavior.
    Each sample contributes equally regardless of sequence length.

    IMPORTANT: Kabsch aligns TARGET to PRED (not pred to target).
    This is correct because:
    - The model outputs predictions in its own coordinate frame
    - With rotation augmentation, target is in a random rotated frame
    - We align target to pred's frame so the model learns the correct SHAPE
    - If we aligned pred to target, the model couldn't learn which frame to use

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]
        use_kabsch: Whether to apply Kabsch alignment (default True)
        reduction: 'mean' returns a scalar (default; existing behavior);
                   'per_sample' returns a [B] tensor of per-sample MSE
                   (used by EDM/Karras 2022 per-sample loss weighting; see
                   tinyfold.training.utils.edm_loss_weight).

    Returns:
        loss: Scalar MSE loss when reduction='mean', else [B] per-sample MSE.
    """
    if reduction not in ('mean', 'per_sample'):
        raise ValueError(
            f"compute_mse_loss: reduction must be 'mean' or 'per_sample', got {reduction!r}"
        )

    if use_kabsch:
        # Align TARGET to PRED's frame (not pred to target!)
        target_aligned, pred_c = kabsch_align(target, pred, mask)
        # CRITICAL: Detach aligned target so gradients don't flow through Kabsch SVD
        # This follows Boltz implementation - gradients should only flow through pred
        target_aligned = target_aligned.detach()
    else:
        # Direct MSE without alignment - just center both
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_c = pred - pred_mean
        target_aligned = target - target_mean

    sq_diff = ((pred_c - target_aligned) ** 2).sum(dim=-1)  # [B, N]

    if mask is not None:
        # Per-sample loss: average over valid positions within each sample
        n_valid_per_sample = mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample_loss = (sq_diff * mask.float()).sum(dim=1) / n_valid_per_sample  # [B]
        if reduction == 'per_sample':
            return per_sample_loss
        # Average across samples
        loss = per_sample_loss.mean()
    else:
        if reduction == 'per_sample':
            # Average over (N, 3) per sample first; sq_diff is [B, N] -> [B]
            return sq_diff.mean(dim=1)
        loss = sq_diff.mean()

    return loss


def compute_rmse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """RMSE after Kabsch alignment.

    Aligns TARGET to PRED's frame (consistent with compute_mse_loss).

    Args:
        pred: Predicted coordinates [B, N, 3]
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]

    Returns:
        rmse: Scalar RMSE value
    """
    # Align target to pred's frame (not pred to target!)
    target_aligned, pred_c = kabsch_align(target, pred, mask)
    target_aligned = target_aligned.detach()  # Consistent with compute_mse_loss
    sq_diff = ((pred_c - target_aligned) ** 2).sum(dim=-1)

    if mask is not None:
        rmse = torch.sqrt((sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1))
    else:
        rmse = torch.sqrt(sq_diff.mean())

    return rmse


def compute_c_rmsd(
    pred_ca: Tensor,        # [B, L, 3] predicted CA (or centroid) coords
    gt_ca: Tensor,          # [B, L, 3] ground-truth CA coords
    chain_ids: Tensor,      # [B, L] long, values in {0, 1}; 0 == chain A
    mask: Tensor | None = None,  # [B, L] bool, valid residues
) -> Tensor:
    """Complex-RMSD: Kabsch-align pred chain A onto GT chain A, apply that
    SINGLE rigid transform to the full prediction (both chains), then return
    RMSD over all valid CA.

    This measures inter-chain placement: identical to per-chain RMSD if
    chain B sits where GT says it does after chain A is aligned, and large
    when the predicted complex has the right chains but the wrong relative
    pose.

    Implementation notes:
        - Reuses the same SVD recipe as ``kabsch_align`` but keeps ``R`` and
          ``t`` explicit so we can apply the chain-A transform to the WHOLE
          complex (chain B included).
        - Requires at least 3 valid chain-A CA per batch element (asserted).
    """
    assert pred_ca.shape == gt_ca.shape, \
        f"shape mismatch {pred_ca.shape} vs {gt_ca.shape}"
    assert chain_ids.shape == pred_ca.shape[:2], \
        f"chain_ids shape {chain_ids.shape} != {pred_ca.shape[:2]}"

    B, L, _ = pred_ca.shape
    device = pred_ca.device

    # Chain A mask (chain id == 0), AND'd with validity mask.
    chain_a = (chain_ids == 0)
    if mask is not None:
        chain_a = chain_a & mask.bool()
        full_mask = mask.bool()
    else:
        full_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Guard: need >=3 chain-A residues per batch element for a stable Kabsch.
    n_a = chain_a.sum(dim=1)
    assert (n_a >= 3).all(), \
        f"compute_c_rmsd needs >=3 chain-A residues per sample, got {n_a.tolist()}"

    # --- Per-batch chain-A Kabsch fit (R, t) ---
    # Centroids over chain-A only.
    chain_a_f = chain_a.unsqueeze(-1).float()
    n_a_exp = chain_a_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1,1]
    pred_a_mean = (pred_ca * chain_a_f).sum(dim=1, keepdim=True) / n_a_exp
    gt_a_mean   = (gt_ca   * chain_a_f).sum(dim=1, keepdim=True) / n_a_exp

    pred_a_c = (pred_ca - pred_a_mean) * chain_a_f
    gt_a_c   = (gt_ca   - gt_a_mean)   * chain_a_f

    # H = pred_a_c^T @ gt_a_c  -> SVD -> R that maps pred onto gt.
    H = torch.bmm(pred_a_c.transpose(1, 2), gt_a_c)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d
    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # [B,3,3]
    # Translation so that R @ pred_a_mean + t = gt_a_mean.
    t = gt_a_mean.squeeze(1) - torch.bmm(pred_a_mean, R.transpose(1, 2)).squeeze(1)  # [B,3]

    # --- Apply (R, t) to FULL prediction (both chains) ---
    pred_aligned = torch.bmm(pred_ca, R.transpose(1, 2)) + t.unsqueeze(1)  # [B,L,3]

    # --- RMSD over all valid CA (both chains) ---
    sq_diff = ((pred_aligned - gt_ca) ** 2).sum(dim=-1)  # [B,L]
    full_mask_f = full_mask.float()
    n_valid = full_mask_f.sum().clamp(min=1.0)
    rmsd = torch.sqrt((sq_diff * full_mask_f).sum() / n_valid)
    return rmsd


def compute_relative_distance_loss(
    pred_coords: Tensor,       # [B, K, 3] predicted coordinates
    gt_coords: Tensor,         # [B, K, 3] ground truth for target atoms
    known_coords: Tensor,      # [B, M, 3] coordinates of known atoms
    known_mask: Tensor | None = None,  # [B, M] mask for valid known atoms
    align_first: bool = True,
) -> Tensor:
    """Compute loss on distances from predicted atoms to known atoms.

    Instead of penalizing absolute positions, this penalizes the distance
    from each predicted atom to each known atom. This makes the loss
    invariant to global translation/rotation.

    Optionally performs Kabsch alignment of predicted to ground truth
    first (considering only the predicted atoms).

    Args:
        pred_coords: Predicted coordinates for K target atoms [B, K, 3]
        gt_coords: Ground truth coordinates for K target atoms [B, K, 3]
        known_coords: Coordinates of M already-placed atoms [B, M, 3]
        known_mask: Boolean mask for valid known atoms [B, M]
        align_first: Whether to Kabsch-align pred to gt before computing loss

    Returns:
        loss: Scalar loss value
    """
    B, K, _ = pred_coords.shape
    M = known_coords.shape[1]

    # Handle empty known case
    if M == 0 or (known_mask is not None and not known_mask.any()):
        # Fall back to direct MSE if no known atoms
        return ((pred_coords - gt_coords) ** 2).sum(dim=-1).mean()

    if align_first and K >= 3:
        # Kabsch align predicted to ground truth
        pred_aligned, _ = kabsch_align(pred_coords, gt_coords)
    else:
        pred_aligned = pred_coords

    # Compute distances from predicted to known atoms [B, K, M]
    pred_dists = torch.cdist(pred_aligned, known_coords)  # [B, K, M]
    gt_dists = torch.cdist(gt_coords, known_coords)       # [B, K, M]

    # MSE on distances, masked
    sq_diff = (pred_dists - gt_dists) ** 2

    if known_mask is not None:
        mask_exp = known_mask.unsqueeze(1).float()  # [B, 1, M]
        # Per-target loss: average over valid known atoms
        n_valid = mask_exp.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        per_target_loss = (sq_diff * mask_exp).sum(dim=-1) / n_valid.squeeze(-1)  # [B, K]
        loss = per_target_loss.mean()
    else:
        loss = sq_diff.mean()

    return loss


def _backbone_frames(coords: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Per-residue rigid frames from backbone N/CA/C via Gram-Schmidt.

    ``coords`` is ``[B, L, 4, 3]`` (N, CA, C, O). Returns ``(R, t)`` where
    ``R`` is ``[B, L, 3, 3]`` with the local basis as COLUMNS and ``t`` is the
    CA position ``[B, L, 3]`` -- so ``x_global = R @ x_local + t`` and the
    inverse (global -> local) is ``R^T (x_global - t)``.
    """
    n = coords[..., 0, :]
    ca = coords[..., 1, :]
    c = coords[..., 2, :]
    e1 = c - ca
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)
    v2 = n - ca
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # columns are basis vectors
    return R, ca


def compute_fape_loss(
    pred_coords: Tensor,       # [B, L, 4, 3] predicted backbone
    target_coords: Tensor,     # [B, L, 4, 3] ground-truth backbone
    chain_ids: Tensor,         # [B, L]
    mask: Tensor | None = None,  # [B, L]
    clamp: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
) -> Tensor:
    """Frame-aligned point error with an inter-chain-unclamped tail (C6).

    Builds a local frame per residue from its backbone, expresses every other
    residue's CA in each frame (in pred and target separately), and takes the
    per-pair distance error. Intra-chain pairs are clamped at ``clamp`` (AF2's
    10 A); INTER-chain pairs are left UNCLAMPED, so a badly-docked interface
    keeps producing gradient no matter how far off it is -- AF-Multimer's recipe
    "to provide a better gradient signal for incorrect interfaces". Errors are
    divided by ``length_scale`` (AF2 = 10 A). FAPE is invariant to a global rigid
    transform of the inputs (everything is frame-relative).
    """
    B, L = chain_ids.shape
    device = pred_coords.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

    Rp, tp = _backbone_frames(pred_coords)
    Rt, tt = _backbone_frames(target_coords)
    xp = pred_coords[..., 1, :]      # CA points [B, L, 3]
    xt = target_coords[..., 1, :]

    # Local coords of point j in frame i: R_i^T (x_j - t_i)  -> [B, i, j, 3]
    diff_p = xp[:, None, :, :] - tp[:, :, None, :]
    diff_t = xt[:, None, :, :] - tt[:, :, None, :]
    loc_p = torch.einsum('bick,bijc->bijk', Rp, diff_p)
    loc_t = torch.einsum('bick,bijc->bijk', Rt, diff_t)

    d = torch.sqrt(((loc_p - loc_t) ** 2).sum(-1) + eps)  # [B, i, j]

    inter = chain_ids.unsqueeze(-1) != chain_ids.unsqueeze(-2)  # [B, i, j]
    # Intra-chain clamped; inter-chain unclamped.
    d_used = torch.where(inter, d, d.clamp(max=clamp))

    pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()
    denom = pair_mask.sum(dim=(1, 2)).clamp(min=1)
    per_sample = (d_used * pair_mask).sum(dim=(1, 2)) / denom  # [B]
    return (per_sample / length_scale).mean()


def compute_distance_consistency_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    chain_ids: Tensor | None = None,
    interchain_weight: float = 1.0,
) -> Tensor:
    """Loss for preserving pairwise distances.

    Encourages predicted coordinates to have similar pairwise distances
    as the ground truth. This is rotation/translation invariant without
    explicit alignment.

    Uses per-sample averaging for correct gradient accumulation behavior.

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., centroids)
        target: Target coordinates [B, N, 3]
        mask: Optional mask for valid positions [B, N]
        chain_ids: Optional [B, N] chain labels. Required to up-weight
            inter-chain residue pairs (C6).
        interchain_weight: Multiplier for pairs spanning the two chains
            (default 1.0 = off; every pair weighted equally, bitwise-identical
            to the pre-C6 loss). AF-Multimer gives interface pairs a stronger
            gradient signal for wrong docking; this is the distance-loss analogue.

    Returns:
        loss: Scalar distance consistency loss (averaged per-sample, then across batch)
    """
    # Compute pairwise distances [B, N, N]
    pred_dist = torch.cdist(pred, pred)
    target_dist = torch.cdist(target, target)

    # MSE on distances
    dist_diff = (pred_dist - target_dist) ** 2

    if mask is not None:
        # Create pairwise mask [B, N, N]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        w = pair_mask.float()
        if chain_ids is not None and interchain_weight != 1.0:
            # Up-weight inter-chain pairs. interchain_weight==1.0 short-circuits
            # above so the default path is byte-identical.
            inter = chain_ids.unsqueeze(-1) != chain_ids.unsqueeze(-2)  # [B,N,N]
            w = w * torch.where(
                inter,
                torch.as_tensor(interchain_weight, dtype=w.dtype, device=w.device),
                torch.ones((), dtype=w.dtype, device=w.device),
            )
        # Per-sample loss: weighted average over valid pairs within each sample
        denom = w.sum(dim=(1, 2)).clamp(min=1)  # [B]
        per_sample_loss = (dist_diff * w).sum(dim=(1, 2)) / denom  # [B]
        # Average across samples
        loss = per_sample_loss.mean()
    else:
        loss = dist_diff.mean()

    return loss
