"""Contact-based losses for protein structure prediction.

Penalizes deviation in pairwise distances for residue pairs that are
close in 3D space but not sequential neighbors.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict


# Default contact parameters
DEFAULT_CONTACT_THRESHOLD = 10.0  # Angstroms
DEFAULT_MIN_SEQ_SEP = 5  # Minimum sequence separation for non-local


def compute_contact_mask(
    centroids: Tensor,
    chain_ids: Tensor,
    threshold: float = 1.0,  # Normalized (10A / 10 = 1.0)
    min_seq_sep: int = 5,
    include_intra: bool = True,
    include_inter: bool = True,
) -> Tensor:
    """Compute contact mask from ground truth centroids.

    Args:
        centroids: [L, 3] centroid coordinates (normalized)
        chain_ids: [L] chain IDs (0 or 1)
        threshold: Distance threshold for contacts (normalized)
        min_seq_sep: Minimum sequence separation for intra-chain contacts
        include_intra: Include intra-chain non-local contacts
        include_inter: Include inter-chain contacts

    Returns:
        [L, L] boolean mask (upper triangular, True = contact)
    """
    L = centroids.shape[0]
    device = centroids.device

    # Pairwise distances
    dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # [L, L]

    # Sequence separation
    seq_idx = torch.arange(L, device=device)
    seq_sep = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1))  # [L, L]

    # Chain masks
    same_chain = chain_ids.unsqueeze(0) == chain_ids.unsqueeze(1)  # [L, L]
    cross_chain = ~same_chain

    # Contact criteria
    close = dist < threshold

    # Upper triangular to avoid double counting
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    # Build contact mask
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)

    if include_intra:
        # Intra-chain: same chain, non-local (seq_sep > min_seq_sep)
        intra = close & same_chain & (seq_sep > min_seq_sep) & upper
        mask = mask | intra

    if include_inter:
        # Inter-chain: different chains (always non-local)
        inter = close & cross_chain & upper
        mask = mask | inter

    return mask


def contact_loss_centroids(
    pred_centroids: Tensor,
    gt_centroids: Tensor,
    contact_mask: Tensor,
    chain_ids: Optional[Tensor] = None,
    inter_chain_weight: float = 1.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Contact-based loss for Stage 1 (centroids).

    Penalizes deviation in pairwise distances for contact pairs.

    Args:
        pred_centroids: [B, L, 3] predicted centroids
        gt_centroids: [B, L, 3] ground truth centroids
        contact_mask: [B, L, L] precomputed contact mask from GT
        chain_ids: [B, L] chain IDs for inter-chain weighting (optional)
        inter_chain_weight: Weight multiplier for inter-chain contacts
        mask: [B, L] residue mask (optional)

    Returns:
        Scalar loss
    """
    # Pairwise distances [B, L, L]
    pred_dist = torch.cdist(pred_centroids, pred_centroids)
    gt_dist = torch.cdist(gt_centroids, gt_centroids)

    # Squared difference
    sq_diff = (pred_dist - gt_dist) ** 2

    # Apply inter-chain weighting if requested
    if chain_ids is not None and inter_chain_weight != 1.0:
        cross_chain = chain_ids.unsqueeze(-1) != chain_ids.unsqueeze(-2)  # [B, L, L]
        weights = torch.where(cross_chain, inter_chain_weight, 1.0)
        sq_diff = sq_diff * weights

    # Apply residue mask if provided
    if mask is not None:
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # [B, L, L]
        contact_mask = contact_mask & pair_mask

    # Compute masked mean
    n_contacts = contact_mask.float().sum().clamp(min=1)
    loss = (sq_diff * contact_mask.float()).sum() / n_contacts

    return loss


def contact_loss_atoms(
    pred_atoms: Tensor,
    gt_atoms: Tensor,
    contact_mask: Tensor,
    chain_ids: Optional[Tensor] = None,
    inter_chain_weight: float = 1.0,
    mask: Optional[Tensor] = None,
    distance_type: str = "min",
) -> Tensor:
    """Contact-based loss for Stage 2 (atoms).

    Penalizes deviation in pairwise distances for contact pairs.
    Distance can be computed as min atom-atom or CA-CA.

    Args:
        pred_atoms: [B, L, 4, 3] predicted atom positions (N, CA, C, O)
        gt_atoms: [B, L, 4, 3] ground truth atom positions
        contact_mask: [B, L, L] precomputed contact mask from GT centroids
        chain_ids: [B, L] chain IDs for inter-chain weighting (optional)
        inter_chain_weight: Weight multiplier for inter-chain contacts
        mask: [B, L] residue mask (optional)
        distance_type: "min" for min atom-atom, "ca" for CA-CA distance

    Returns:
        Scalar loss
    """
    B, L = pred_atoms.shape[:2]

    if distance_type == "ca":
        # CA-CA distance (atom index 1)
        pred_ca = pred_atoms[:, :, 1, :]  # [B, L, 3]
        gt_ca = gt_atoms[:, :, 1, :]
        pred_dist = torch.cdist(pred_ca, pred_ca)  # [B, L, L]
        gt_dist = torch.cdist(gt_ca, gt_ca)
    elif distance_type == "min":
        # Minimum atom-atom distance between residues
        pred_flat = pred_atoms.view(B, L * 4, 3)
        gt_flat = gt_atoms.view(B, L * 4, 3)

        # Full pairwise [B, L*4, L*4]
        pred_dist_full = torch.cdist(pred_flat, pred_flat)
        gt_dist_full = torch.cdist(gt_flat, gt_flat)

        # Reshape to [B, L, 4, L, 4] and take min over atom pairs
        pred_dist_full = pred_dist_full.view(B, L, 4, L, 4)
        gt_dist_full = gt_dist_full.view(B, L, 4, L, 4)

        pred_dist = pred_dist_full.min(dim=2)[0].min(dim=-1)[0]  # [B, L, L]
        gt_dist = gt_dist_full.min(dim=2)[0].min(dim=-1)[0]
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")

    # Squared difference
    sq_diff = (pred_dist - gt_dist) ** 2

    # Apply inter-chain weighting if requested
    if chain_ids is not None and inter_chain_weight != 1.0:
        cross_chain = chain_ids.unsqueeze(-1) != chain_ids.unsqueeze(-2)
        weights = torch.where(cross_chain, inter_chain_weight, 1.0)
        sq_diff = sq_diff * weights

    # Apply residue mask if provided
    if mask is not None:
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        contact_mask = contact_mask & pair_mask

    # Compute masked mean
    n_contacts = contact_mask.float().sum().clamp(min=1)
    loss = (sq_diff * contact_mask.float()).sum() / n_contacts

    return loss


class ContactLoss(nn.Module):
    """Contact-based auxiliary loss for protein structure prediction.

    Penalizes deviation in pairwise distances for residue pairs that are
    close in 3D space but not sequential neighbors.

    Args:
        threshold: Contact distance threshold in normalized units (default 1.0 = 10A)
        min_seq_sep: Minimum sequence separation for intra-chain contacts
        include_intra: Include intra-chain non-local contacts
        include_inter: Include inter-chain contacts
        inter_chain_weight: Weight multiplier for inter-chain contacts
        stage: "stage1" for centroids, "stage2" for atoms, "both" for both
        atom_distance_type: "ca" or "min" for Stage 2
    """

    def __init__(
        self,
        threshold: float = 1.0,
        min_seq_sep: int = 5,
        include_intra: bool = True,
        include_inter: bool = True,
        inter_chain_weight: float = 2.0,
        stage: str = "stage1",
        atom_distance_type: str = "ca",
    ):
        super().__init__()
        self.threshold = threshold
        self.min_seq_sep = min_seq_sep
        self.include_intra = include_intra
        self.include_inter = include_inter
        self.inter_chain_weight = inter_chain_weight
        self.stage = stage
        self.atom_distance_type = atom_distance_type

    def compute_contact_masks(
        self,
        gt_centroids: Tensor,
        chain_ids: Tensor,
    ) -> Tensor:
        """Compute contact masks for a batch.

        Args:
            gt_centroids: [B, L, 3] ground truth centroids
            chain_ids: [B, L] chain IDs

        Returns:
            [B, L, L] contact masks
        """
        B, L = gt_centroids.shape[:2]
        device = gt_centroids.device

        masks = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        for b in range(B):
            masks[b] = compute_contact_mask(
                gt_centroids[b],
                chain_ids[b],
                threshold=self.threshold,
                min_seq_sep=self.min_seq_sep,
                include_intra=self.include_intra,
                include_inter=self.include_inter,
            )
        return masks

    def forward(
        self,
        pred_centroids: Optional[Tensor] = None,
        gt_centroids: Optional[Tensor] = None,
        pred_atoms: Optional[Tensor] = None,
        gt_atoms: Optional[Tensor] = None,
        chain_ids: Optional[Tensor] = None,
        contact_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute contact loss.

        Args:
            pred_centroids: [B, L, 3] predicted centroids (for stage1)
            gt_centroids: [B, L, 3] ground truth centroids
            pred_atoms: [B, L, 4, 3] predicted atoms (for stage2)
            gt_atoms: [B, L, 4, 3] ground truth atoms
            chain_ids: [B, L] chain IDs
            contact_mask: [B, L, L] precomputed contact mask (optional)
            mask: [B, L] residue mask

        Returns:
            dict with 'stage1', 'stage2', 'total' losses
        """
        losses = {}
        device = gt_centroids.device if gt_centroids is not None else pred_atoms.device

        # Compute contact mask if not provided
        if contact_mask is None:
            if gt_centroids is None:
                # Compute centroids from gt_atoms
                gt_centroids = gt_atoms.mean(dim=2)  # [B, L, 3]
            contact_mask = self.compute_contact_masks(gt_centroids, chain_ids)

        # Stage 1 loss (centroids)
        if self.stage in ["stage1", "both"] and pred_centroids is not None:
            losses["stage1"] = contact_loss_centroids(
                pred_centroids, gt_centroids, contact_mask,
                chain_ids, self.inter_chain_weight, mask
            )
        else:
            losses["stage1"] = torch.tensor(0.0, device=device)

        # Stage 2 loss (atoms)
        if self.stage in ["stage2", "both"] and pred_atoms is not None:
            losses["stage2"] = contact_loss_atoms(
                pred_atoms, gt_atoms, contact_mask,
                chain_ids, self.inter_chain_weight, mask,
                self.atom_distance_type
            )
        else:
            losses["stage2"] = torch.tensor(0.0, device=device)

        # Total
        losses["total"] = losses["stage1"] + losses["stage2"]
        losses["n_contacts"] = contact_mask.float().sum().item()

        return losses

    def __repr__(self):
        return (
            f"ContactLoss(threshold={self.threshold}, min_seq_sep={self.min_seq_sep}, "
            f"inter_weight={self.inter_chain_weight}, stage={self.stage})"
        )
