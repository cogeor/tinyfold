"""lDDT metrics (Local Distance Difference Test) for structure evaluation.

lDDT is a superposition-free metric that compares local pairwise distances.
For each residue, it checks what fraction of distances to nearby residues
are preserved within given thresholds.
"""

import torch
from torch import Tensor
from typing import Optional, List, Dict


DEFAULT_COORD_SCALE = 10.0  # Default normalization scale


def compute_lddt(
    pred_ca: Tensor,
    gt_ca: Tensor,
    mask: Optional[Tensor] = None,
    inclusion_radius: float = 15.0,
    thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> Tensor:
    """Compute lDDT (Local Distance Difference Test) for CA atoms.

    lDDT is a superposition-free metric that compares local pairwise distances.
    For each residue, it checks what fraction of distances to nearby residues
    (within inclusion_radius) are preserved within the given thresholds.

    Args:
        pred_ca: [B, L, 3] or [L, 3] predicted CA coordinates
        gt_ca: [B, L, 3] or [L, 3] ground truth CA coordinates
        mask: [B, L] or [L] residue mask (True = valid)
        inclusion_radius: Only consider pairs within this distance in GT (Angstroms)
        thresholds: Distance error thresholds in Angstroms (default: [0.5, 1, 2, 4])
        coord_scale: Scale factor if coords are normalized (default: 10.0)

    Returns:
        lDDT score (0-1), averaged over all valid residues
    """
    # Handle unbatched input
    if pred_ca.dim() == 2:
        pred_ca = pred_ca.unsqueeze(0)
        gt_ca = gt_ca.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, L, _ = pred_ca.shape
    device = pred_ca.device

    # Scale to Angstroms if normalized
    pred_ca = pred_ca * coord_scale
    gt_ca = gt_ca * coord_scale

    # Compute pairwise distances [B, L, L]
    pred_dists = torch.cdist(pred_ca, pred_ca)
    gt_dists = torch.cdist(gt_ca, gt_ca)

    # Mask for valid pairs within inclusion radius (in GT)
    # Exclude self-distances (diagonal)
    valid_pairs = gt_dists < inclusion_radius
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pairs = valid_pairs & diag_mask

    # Apply residue mask if provided
    if mask is not None:
        mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # [B, L, L]
        valid_pairs = valid_pairs & mask_2d

    # Compute distance errors
    dist_errors = torch.abs(pred_dists - gt_dists)

    # Count how many thresholds each pair satisfies
    scores = torch.zeros_like(dist_errors)
    for thresh in thresholds:
        scores = scores + (dist_errors < thresh).float()
    scores = scores / len(thresholds)  # Normalize to 0-1

    # Average over valid pairs for each residue, then over residues
    n_valid_per_res = valid_pairs.float().sum(dim=-1).clamp(min=1)  # [B, L]
    lddt_per_res = (scores * valid_pairs.float()).sum(dim=-1) / n_valid_per_res  # [B, L]

    # Average over valid residues
    if mask is not None:
        n_valid_res = mask.float().sum(dim=-1).clamp(min=1)  # [B]
        lddt = (lddt_per_res * mask.float()).sum(dim=-1) / n_valid_res  # [B]
    else:
        lddt = lddt_per_res.mean(dim=-1)  # [B]

    return lddt.mean()  # Scalar


def compute_interface_mask(
    ca_coords: Tensor,
    chain_ids: Tensor,
    interface_threshold: float = 8.0,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> Tensor:
    """Compute mask for interface residues.

    Interface residues are those with CA within interface_threshold of any
    CA from the other chain.

    Args:
        ca_coords: [B, L, 3] or [L, 3] CA coordinates
        chain_ids: [B, L] or [L] chain IDs (0 or 1)
        interface_threshold: Distance cutoff in Angstroms
        coord_scale: Scale factor if coords are normalized

    Returns:
        [B, L] or [L] boolean mask (True = interface residue)
    """
    # Handle unbatched input
    squeezed = False
    if ca_coords.dim() == 2:
        ca_coords = ca_coords.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)
        squeezed = True

    B, L, _ = ca_coords.shape
    device = ca_coords.device

    # Scale to Angstroms
    ca_coords = ca_coords * coord_scale

    # Compute pairwise distances
    dists = torch.cdist(ca_coords, ca_coords)  # [B, L, L]

    # Mask for inter-chain pairs
    chain_a = chain_ids == 0  # [B, L]
    chain_b = chain_ids == 1  # [B, L]
    inter_chain = (
        (chain_a.unsqueeze(-1) & chain_b.unsqueeze(-2)) |
        (chain_b.unsqueeze(-1) & chain_a.unsqueeze(-2))
    )  # [B, L, L]

    # Find residues with any inter-chain contact
    has_contact = (dists < interface_threshold) & inter_chain  # [B, L, L]
    interface_mask = has_contact.any(dim=-1)  # [B, L]

    if squeezed:
        interface_mask = interface_mask.squeeze(0)

    return interface_mask


def compute_ilddt(
    pred_ca: Tensor,
    gt_ca: Tensor,
    chain_ids: Tensor,
    mask: Optional[Tensor] = None,
    interface_threshold: float = 8.0,
    inclusion_radius: float = 15.0,
    thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> Tensor:
    """Compute ilDDT (interface lDDT) for CA atoms.

    Same as lDDT but only computed for interface residues (residues within
    interface_threshold of the other chain in the ground truth).

    Args:
        pred_ca: [B, L, 3] or [L, 3] predicted CA coordinates
        gt_ca: [B, L, 3] or [L, 3] ground truth CA coordinates
        chain_ids: [B, L] or [L] chain IDs (0 or 1)
        mask: [B, L] or [L] residue mask (True = valid)
        interface_threshold: Distance cutoff for interface definition (Angstroms)
        inclusion_radius: Only consider pairs within this distance in GT (Angstroms)
        thresholds: Distance error thresholds in Angstroms
        coord_scale: Scale factor if coords are normalized

    Returns:
        ilDDT score (0-1), averaged over interface residues
    """
    # Handle unbatched input
    if pred_ca.dim() == 2:
        pred_ca = pred_ca.unsqueeze(0)
        gt_ca = gt_ca.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, L, _ = pred_ca.shape
    device = pred_ca.device

    # Compute interface mask from ground truth
    interface_mask = compute_interface_mask(gt_ca, chain_ids, interface_threshold, coord_scale)

    # Combine with residue mask if provided
    if mask is not None:
        combined_mask = mask & interface_mask
    else:
        combined_mask = interface_mask

    # Check if there are any interface residues
    if not combined_mask.any():
        return torch.tensor(0.0, device=device)

    # Scale to Angstroms
    pred_ca_scaled = pred_ca * coord_scale
    gt_ca_scaled = gt_ca * coord_scale

    # Compute pairwise distances [B, L, L]
    pred_dists = torch.cdist(pred_ca_scaled, pred_ca_scaled)
    gt_dists = torch.cdist(gt_ca_scaled, gt_ca_scaled)

    # Mask for valid pairs within inclusion radius (in GT)
    valid_pairs = gt_dists < inclusion_radius
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pairs = valid_pairs & diag_mask

    # Apply combined mask (both residues must be valid, at least one must be interface)
    if mask is not None:
        mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        valid_pairs = valid_pairs & mask_2d

    # For ilDDT, only count pairs where at least one residue is at interface
    interface_2d = interface_mask.unsqueeze(-1) | interface_mask.unsqueeze(-2)
    valid_pairs = valid_pairs & interface_2d

    # Compute distance errors
    dist_errors = torch.abs(pred_dists - gt_dists)

    # Count how many thresholds each pair satisfies
    scores = torch.zeros_like(dist_errors)
    for thresh in thresholds:
        scores = scores + (dist_errors < thresh).float()
    scores = scores / len(thresholds)

    # Average over valid pairs for interface residues
    n_valid_per_res = valid_pairs.float().sum(dim=-1).clamp(min=1)
    lddt_per_res = (scores * valid_pairs.float()).sum(dim=-1) / n_valid_per_res

    # Average only over interface residues
    n_interface = combined_mask.float().sum(dim=-1).clamp(min=1)
    ilddt = (lddt_per_res * combined_mask.float()).sum(dim=-1) / n_interface

    return ilddt.mean()


def compute_lddt_metrics(
    pred_coords: Tensor,
    gt_coords: Tensor,
    chain_ids: Tensor,
    mask: Optional[Tensor] = None,
    interface_threshold: float = 8.0,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> Dict[str, float]:
    """Compute both lDDT and ilDDT metrics.

    Convenience function that computes both metrics and returns them as a dict.
    Expects atom coordinates [B, L, 4, 3] and extracts CA (index 1).

    Args:
        pred_coords: [B, L, 4, 3] predicted atom coordinates (N, CA, C, O)
        gt_coords: [B, L, 4, 3] ground truth atom coordinates
        chain_ids: [B, L] chain IDs
        mask: [B, L] residue mask
        interface_threshold: Distance cutoff for interface definition
        coord_scale: Scale factor if coords are normalized

    Returns:
        dict with 'lddt', 'ilddt', 'n_interface' keys
    """
    # Extract CA atoms (index 1)
    pred_ca = pred_coords[..., 1, :]  # [B, L, 3]
    gt_ca = gt_coords[..., 1, :]  # [B, L, 3]

    lddt = compute_lddt(pred_ca, gt_ca, mask, coord_scale=coord_scale)
    ilddt = compute_ilddt(pred_ca, gt_ca, chain_ids, mask, interface_threshold, coord_scale=coord_scale)

    # Count interface residues
    interface_mask = compute_interface_mask(gt_ca, chain_ids, interface_threshold, coord_scale)
    if mask is not None:
        n_interface = (interface_mask & mask).float().sum().item()
    else:
        n_interface = interface_mask.float().sum().item()

    return {
        "lddt": lddt.item(),
        "ilddt": ilddt.item(),
        "n_interface": int(n_interface),
    }
