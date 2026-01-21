"""Geometry auxiliary losses for protein backbone structure.

Enforces chemically reasonable configurations:
- Bond lengths (N-CA, CA-C, C-O, C-N peptide)
- Bond angles (tetrahedral CA, planar peptide)
- Omega dihedral (peptide planarity)
- O chirality (carbonyl on correct side)
- Virtual CB chirality (L-amino acid handedness)

All losses can be disabled by setting their weight to 0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Expected bond lengths in Angstroms
BOND_LENGTHS_ANGSTROM = {
    'N_CA': 1.458,   # Within residue
    'CA_C': 1.525,   # Within residue
    'C_O': 1.229,    # Within residue
    'C_N': 1.329,    # Peptide bond (between residues)
}

# Default normalization scale (typical std of protein coords)
DEFAULT_COORD_SCALE = 10.0

# Normalized bond lengths (for models using normalized coords)
BOND_LENGTHS = {k: v / DEFAULT_COORD_SCALE for k, v in BOND_LENGTHS_ANGSTROM.items()}

# Expected bond angles in degrees
BOND_ANGLES = {
    'N_CA_C': 111.0,    # Tetrahedral at CA
    'CA_C_O': 121.0,    # sp2 carbonyl
    'CA_C_N': 117.0,    # sp2 peptide
    'C_N_CA': 121.0,    # sp2 peptide
}


def safe_normalize(v: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Normalize with safety for zero vectors."""
    return v / (v.norm(dim=dim, keepdim=True) + eps)


def bond_angle(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Compute angle at p1 in degrees.

    Args:
        p0, p1, p2: [..., 3] tensors

    Returns:
        [...] angle in degrees
    """
    v1 = safe_normalize(p0 - p1, dim=-1)
    v2 = safe_normalize(p2 - p1, dim=-1)
    cos_angle = (v1 * v2).sum(dim=-1).clamp(-0.9999, 0.9999)
    return torch.acos(cos_angle) * 180 / math.pi


def dihedral_angle(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Compute dihedral angle p0-p1-p2-p3 in radians.

    Args:
        p0, p1, p2, p3: [..., 3] tensors

    Returns:
        [...] dihedral angle in radians (-pi to pi)
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Normalize b2 for the cross product
    b2_norm = safe_normalize(b2, dim=-1)
    m1 = torch.cross(n1, b2_norm, dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    return torch.atan2(y, x)


def bond_length_loss(coords: Tensor, mask: Tensor = None, valid_peptide: Tensor = None) -> Tensor:
    """Compute bond length loss.

    Args:
        coords: [B, L, 4, 3] - N=0, CA=1, C=2, O=3
        mask: [B, L] residue mask (optional)
        valid_peptide: [B, L-1] precomputed mask of valid peptide bonds from GT (optional)

    Returns:
        Scalar loss
    """
    N = coords[..., 0, :]   # [B, L, 3]
    CA = coords[..., 1, :]
    C = coords[..., 2, :]
    O = coords[..., 3, :]

    # Within-residue bonds
    d_N_CA = (N - CA).norm(dim=-1)   # [B, L]
    d_CA_C = (CA - C).norm(dim=-1)
    d_C_O = (C - O).norm(dim=-1)

    # Peptide bonds (C[i] to N[i+1])
    d_C_N = (C[:, :-1] - N[:, 1:]).norm(dim=-1)  # [B, L-1]

    # MSE from expected values
    loss_within = (d_N_CA - BOND_LENGTHS['N_CA'])**2 + \
                  (d_CA_C - BOND_LENGTHS['CA_C'])**2 + \
                  (d_C_O - BOND_LENGTHS['C_O'])**2
    loss_peptide = (d_C_N - BOND_LENGTHS['C_N'])**2

    if mask is not None:
        loss_within = loss_within * mask.float()
        # For peptide bonds, need both residues valid AND not a structure gap (from GT)
        mask_peptide = mask[:, :-1] & mask[:, 1:]
        if valid_peptide is not None:
            mask_peptide = mask_peptide & valid_peptide
        loss_peptide = loss_peptide * mask_peptide.float()

        n_valid = mask.sum() + 1e-8
        n_valid_pep = mask_peptide.sum() + 1e-8
        return loss_within.sum() / n_valid + loss_peptide.sum() / n_valid_pep

    # Without mask, still filter broken peptide bonds if provided
    if valid_peptide is not None:
        loss_peptide = loss_peptide * valid_peptide.float()
        n_valid_pep = valid_peptide.sum() + 1e-8
        return loss_within.mean() + loss_peptide.sum() / n_valid_pep

    return loss_within.mean() + loss_peptide.mean()


def bond_angle_loss(coords: Tensor, mask: Tensor = None, valid_peptide: Tensor = None) -> Tensor:
    """Compute bond angle loss.

    Args:
        coords: [B, L, 4, 3]
        mask: [B, L] residue mask (optional)
        valid_peptide: [B, L-1] precomputed mask of valid peptide bonds from GT (optional)

    Returns:
        Scalar loss
    """
    N = coords[..., 0, :]
    CA = coords[..., 1, :]
    C = coords[..., 2, :]
    O = coords[..., 3, :]
    N_next = N[:, 1:]
    CA_next = CA[:, 1:]

    # Within-residue angles
    angle_N_CA_C = bond_angle(N, CA, C)      # [B, L]
    angle_CA_C_O = bond_angle(CA, C, O)      # [B, L]

    # Cross-residue angles (peptide)
    angle_CA_C_N = bond_angle(CA[:, :-1], C[:, :-1], N_next)  # [B, L-1]
    angle_C_N_CA = bond_angle(C[:, :-1], N_next, CA_next)     # [B, L-1]

    # MSE from expected (scale down since angles are in degrees)
    loss_within = ((angle_N_CA_C - BOND_ANGLES['N_CA_C'])**2 +
                   (angle_CA_C_O - BOND_ANGLES['CA_C_O'])**2) / 100.0
    loss_peptide = ((angle_CA_C_N - BOND_ANGLES['CA_C_N'])**2 +
                    (angle_C_N_CA - BOND_ANGLES['C_N_CA'])**2) / 100.0

    if mask is not None:
        loss_within = loss_within * mask.float()
        mask_peptide = mask[:, :-1] & mask[:, 1:]
        if valid_peptide is not None:
            mask_peptide = mask_peptide & valid_peptide
        loss_peptide = loss_peptide * mask_peptide.float()

        n_valid = mask.sum() + 1e-8
        n_valid_pep = mask_peptide.sum() + 1e-8
        return loss_within.sum() / n_valid + loss_peptide.sum() / n_valid_pep

    if valid_peptide is not None:
        loss_peptide = loss_peptide * valid_peptide.float()
        n_valid_pep = valid_peptide.sum() + 1e-8
        return loss_within.mean() + loss_peptide.sum() / n_valid_pep

    return loss_within.mean() + loss_peptide.mean()


def omega_loss(coords: Tensor, mask: Tensor = None, valid_peptide: Tensor = None) -> Tensor:
    """Compute omega dihedral loss (peptide planarity).

    Omega = CA(i) - C(i) - N(i+1) - CA(i+1)
    Should be ~180 deg (trans) or ~0 deg (cis)

    Args:
        coords: [B, L, 4, 3]
        mask: [B, L] residue mask (optional)
        valid_peptide: [B, L-1] precomputed mask of valid peptide bonds from GT (optional)

    Returns:
        Scalar loss
    """
    CA = coords[..., 1, :]
    C = coords[..., 2, :]
    N_next = coords[:, 1:, 0, :]
    CA_next = coords[:, 1:, 1, :]

    omega = dihedral_angle(CA[:, :-1], C[:, :-1], N_next, CA_next)  # [B, L-1]

    # Loss: distance from +/- pi (trans) or 0 (cis)
    # Most bonds are trans, so we allow either
    trans_dev = (torch.abs(omega) - math.pi)**2
    cis_dev = omega**2

    # Take min to allow either trans or cis
    loss = torch.min(trans_dev, cis_dev)

    if mask is not None:
        mask_peptide = mask[:, :-1] & mask[:, 1:]
        if valid_peptide is not None:
            mask_peptide = mask_peptide & valid_peptide
        loss = loss * mask_peptide.float()
        n_valid = mask_peptide.sum() + 1e-8
        return loss.sum() / n_valid

    if valid_peptide is not None:
        loss = loss * valid_peptide.float()
        n_valid = valid_peptide.sum() + 1e-8
        return loss.sum() / n_valid

    return loss.mean()


def o_chirality_loss(coords: Tensor, mask: Tensor = None, valid_peptide: Tensor = None) -> Tensor:
    """Compute O chirality loss (carbonyl on correct side of peptide plane).

    Ensure O is on correct side of peptide plane.
    Plane defined by CA, C, N_next.
    O should be trans to CA_next (opposite side).

    Args:
        coords: [B, L, 4, 3]
        mask: [B, L] residue mask (optional)
        valid_peptide: [B, L-1] precomputed mask of valid peptide bonds from GT (optional)

    Returns:
        Scalar loss
    """
    CA = coords[..., 1, :]
    C = coords[..., 2, :]
    O = coords[..., 3, :]
    N_next = coords[:, 1:, 0, :]
    CA_next = coords[:, 1:, 1, :]

    # Plane normal: (C - CA) x (N_next - C)
    v1 = C[:, :-1] - CA[:, :-1]
    v2 = N_next - C[:, :-1]
    normal = torch.cross(v1, v2, dim=-1)  # [B, L-1, 3]

    # O position relative to plane
    o_vec = O[:, :-1] - C[:, :-1]
    o_side = (o_vec * normal).sum(dim=-1)  # [B, L-1]

    # CA_next position relative to plane
    ca_vec = CA_next - C[:, :-1]
    ca_side = (ca_vec * normal).sum(dim=-1)  # [B, L-1]

    # O and CA_next should be on OPPOSITE sides (trans)
    # If same sign, penalize (positive product means same side)
    loss = F.relu(o_side * ca_side)

    if mask is not None:
        mask_peptide = mask[:, :-1] & mask[:, 1:]
        if valid_peptide is not None:
            mask_peptide = mask_peptide & valid_peptide
        loss = loss * mask_peptide.float()
        n_valid = mask_peptide.sum() + 1e-8
        return loss.sum() / n_valid

    if valid_peptide is not None:
        loss = loss * valid_peptide.float()
        n_valid = valid_peptide.sum() + 1e-8
        return loss.sum() / n_valid

    return loss.mean()


def virtual_cb_loss(coords: Tensor, mask: Tensor = None,
                    glycine_mask: Tensor = None) -> Tensor:
    """Compute virtual CB chirality loss (L-amino acid handedness).

    Compute virtual CB position and check chirality.
    CB should be at ~-34 deg improper dihedral for L-amino acids.
    Skip for glycine (no CB).

    Args:
        coords: [B, L, 4, 3]
        mask: [B, L] residue mask (optional)
        glycine_mask: [B, L] True for glycine residues (optional)

    Returns:
        Scalar loss
    """
    N = coords[..., 0, :]
    CA = coords[..., 1, :]
    C = coords[..., 2, :]

    # Virtual CB: perpendicular to N-CA-C plane, tetrahedral geometry
    v1 = safe_normalize(N - CA, dim=-1)
    v2 = safe_normalize(C - CA, dim=-1)

    # Bisector of N-CA-C
    bisector = safe_normalize(v1 + v2, dim=-1)

    # Perpendicular to plane
    perp = torch.cross(v1, v2, dim=-1)
    perp = safe_normalize(perp, dim=-1)

    # CB direction: rotate bisector away from plane
    # For L-amino acids, CB is on specific side
    cb_dir = -bisector * 0.5 + perp * 0.866  # ~60 deg from bisector
    CB_virtual = CA + cb_dir * 1.52  # CB-CA bond length

    # Improper dihedral: N - CA - C - CB
    chi = dihedral_angle(N, CA, C, CB_virtual)

    # Expected: ~-34 deg = -0.59 rad for L-amino acids
    expected = -0.59
    loss = (chi - expected)**2

    # Apply masks
    if glycine_mask is not None:
        loss = loss * (~glycine_mask).float()

    if mask is not None:
        loss = loss * mask.float()
        if glycine_mask is not None:
            n_valid = (mask & ~glycine_mask).sum() + 1e-8
        else:
            n_valid = mask.sum() + 1e-8
        return loss.sum() / n_valid

    return loss.mean()


def bounded_loss(loss: Tensor, scale: float = 1.0) -> Tensor:
    """Bound loss to [0, 1] range using x/(1+x) transform.

    This prevents any single loss from exploding and dominating.
    The scale parameter controls the "softness" - larger scale means
    the loss saturates slower.

    Args:
        loss: Raw loss value (must be >= 0)
        scale: Scale factor (default 1.0). Loss of `scale` maps to 0.5.

    Returns:
        Bounded loss in [0, 1]
    """
    return loss / (scale + loss)


class GeometryLoss(nn.Module):
    """Combined geometry loss with configurable weights.

    Set any weight to 0 to disable that loss component.
    All losses are bounded to [0, 1] range for stability.

    Args:
        bond_length_weight: Weight for bond length loss (default 1.0)
        bond_angle_weight: Weight for bond angle loss (default 1.0)
        omega_weight: Weight for omega dihedral loss (default 1.0)
        o_chirality_weight: Weight for O chirality loss (default 1.0)
        cb_chirality_weight: Weight for virtual CB chirality loss (default 0.0)
        bound_losses: Whether to bound losses to [0,1] (default True)
    """

    def __init__(
        self,
        bond_length_weight: float = 1.0,
        bond_angle_weight: float = 1.0,
        omega_weight: float = 1.0,
        o_chirality_weight: float = 1.0,
        cb_chirality_weight: float = 0.0,
        bound_losses: bool = True,
    ):
        super().__init__()
        self.weights = {
            'bond_length': bond_length_weight,
            'bond_angle': bond_angle_weight,
            'omega': omega_weight,
            'o_chirality': o_chirality_weight,
            'cb_chirality': cb_chirality_weight,
        }
        self.bound_losses = bound_losses
        # Scale factors for bounding (loss of this value maps to 0.5)
        # Tuned based on typical raw loss values for random predictions
        self.scales = {
            'bond_length': 1.0,    # raw loss ~1 for random → 0.5
            'bond_angle': 50.0,    # raw loss ~50 for random (after /100) → 0.5
            'omega': 1.0,          # raw loss ~1 for random → 0.5
            'o_chirality': 1.0,    # raw loss ~1 for random → 0.5
            'cb_chirality': 1.0,   # raw loss ~1 for random → 0.5
        }

    def forward(
        self,
        coords: Tensor,
        mask: Tensor = None,
        glycine_mask: Tensor = None,
        gt_coords: Tensor = None,
        peptide_threshold: float = 0.2,
    ) -> dict:
        """Compute geometry losses.

        Args:
            coords: [B, L, 4, 3] predicted atom positions (N, CA, C, O)
            mask: [B, L] residue mask (optional)
            glycine_mask: [B, L] True for glycine residues (optional)
            gt_coords: [B, L, 4, 3] ground truth coords for peptide bond detection (optional)
            peptide_threshold: Max C-N distance (normalized) for valid peptide bond (default 0.2 = 2Å)

        Returns:
            dict with individual losses + 'total'
        """
        losses = {}

        # Compute valid peptide mask from GT coords (not predicted)
        # This avoids penalizing chain breaks that exist in the structure
        valid_peptide = None
        if gt_coords is not None:
            C_gt = gt_coords[..., 2, :]  # [B, L, 3]
            N_next_gt = gt_coords[:, 1:, 0, :]  # [B, L-1, 3]
            d_C_N_gt = (C_gt[:, :-1] - N_next_gt).norm(dim=-1)  # [B, L-1]
            valid_peptide = d_C_N_gt < peptide_threshold

        # Only compute losses with non-zero weights
        if self.weights['bond_length'] > 0:
            losses['bond_length'] = bond_length_loss(coords, mask, valid_peptide)
        else:
            losses['bond_length'] = torch.tensor(0.0, device=coords.device)

        if self.weights['bond_angle'] > 0:
            losses['bond_angle'] = bond_angle_loss(coords, mask, valid_peptide)
        else:
            losses['bond_angle'] = torch.tensor(0.0, device=coords.device)

        if self.weights['omega'] > 0:
            losses['omega'] = omega_loss(coords, mask, valid_peptide)
        else:
            losses['omega'] = torch.tensor(0.0, device=coords.device)

        if self.weights['o_chirality'] > 0:
            losses['o_chirality'] = o_chirality_loss(coords, mask, valid_peptide)
        else:
            losses['o_chirality'] = torch.tensor(0.0, device=coords.device)

        if self.weights['cb_chirality'] > 0:
            losses['cb_chirality'] = virtual_cb_loss(coords, mask, glycine_mask)
        else:
            losses['cb_chirality'] = torch.tensor(0.0, device=coords.device)

        # Apply bounding if enabled
        if self.bound_losses:
            for k in losses:
                if k in self.scales:
                    losses[k] = bounded_loss(losses[k], self.scales[k])

        # Weighted sum (now all losses are in [0, 1] range if bounded)
        total = sum(self.weights[k] * v for k, v in losses.items())
        losses['total'] = total

        return losses

    def __repr__(self):
        active = [k for k, v in self.weights.items() if v > 0]
        return f"GeometryLoss(active={active})"


# =============================================================================
# Contact-Based Loss
# =============================================================================

# Default contact parameters (in Angstroms, will be normalized)
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
    chain_ids: Tensor = None,
    inter_chain_weight: float = 1.0,
    mask: Tensor = None,
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
    B = pred_centroids.shape[0]

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
    chain_ids: Tensor = None,
    inter_chain_weight: float = 1.0,
    mask: Tensor = None,
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
        # Reshape to [B, L, 4, 3] -> compute pairwise for all atoms
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
        pred_centroids: Tensor = None,
        gt_centroids: Tensor = None,
        pred_atoms: Tensor = None,
        gt_atoms: Tensor = None,
        chain_ids: Tensor = None,
        contact_mask: Tensor = None,
        mask: Tensor = None,
    ) -> dict:
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
            losses['stage1'] = contact_loss_centroids(
                pred_centroids, gt_centroids, contact_mask,
                chain_ids, self.inter_chain_weight, mask
            )
        else:
            losses['stage1'] = torch.tensor(0.0, device=device)

        # Stage 2 loss (atoms)
        if self.stage in ["stage2", "both"] and pred_atoms is not None:
            losses['stage2'] = contact_loss_atoms(
                pred_atoms, gt_atoms, contact_mask,
                chain_ids, self.inter_chain_weight, mask,
                self.atom_distance_type
            )
        else:
            losses['stage2'] = torch.tensor(0.0, device=device)

        # Total
        losses['total'] = losses['stage1'] + losses['stage2']
        losses['n_contacts'] = contact_mask.float().sum().item()

        return losses

    def __repr__(self):
        return (f"ContactLoss(threshold={self.threshold}, min_seq_sep={self.min_seq_sep}, "
                f"inter_weight={self.inter_chain_weight}, stage={self.stage})")


# =============================================================================
# lDDT Metrics (Local Distance Difference Test)
# =============================================================================

def compute_lddt(
    pred_ca: Tensor,
    gt_ca: Tensor,
    mask: Tensor = None,
    inclusion_radius: float = 15.0,
    thresholds: list = [0.5, 1.0, 2.0, 4.0],
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
    # Per-residue score: mean of valid pairs involving that residue
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
    inter_chain = (chain_a.unsqueeze(-1) & chain_b.unsqueeze(-2)) | \
                  (chain_b.unsqueeze(-1) & chain_a.unsqueeze(-2))  # [B, L, L]

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
    mask: Tensor = None,
    interface_threshold: float = 8.0,
    inclusion_radius: float = 15.0,
    thresholds: list = [0.5, 1.0, 2.0, 4.0],
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
    mask: Tensor = None,
    interface_threshold: float = 8.0,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> dict:
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
        'lddt': lddt.item(),
        'ilddt': ilddt.item(),
        'n_interface': int(n_interface),
    }
