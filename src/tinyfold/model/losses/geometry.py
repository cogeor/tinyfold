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
from typing import Optional, Dict


# Expected bond lengths in Angstroms
BOND_LENGTHS_ANGSTROM = {
    "N_CA": 1.458,   # Within residue
    "CA_C": 1.525,   # Within residue
    "C_O": 1.229,    # Within residue
    "C_N": 1.329,    # Peptide bond (between residues)
}

# Default normalization scale (typical std of protein coords)
DEFAULT_COORD_SCALE = 10.0

# Normalized bond lengths (for models using normalized coords)
BOND_LENGTHS = {k: v / DEFAULT_COORD_SCALE for k, v in BOND_LENGTHS_ANGSTROM.items()}

# Expected bond angles in degrees
BOND_ANGLES = {
    "N_CA_C": 111.0,    # Tetrahedral at CA
    "CA_C_O": 121.0,    # sp2 carbonyl
    "CA_C_N": 117.0,    # sp2 peptide
    "C_N_CA": 121.0,    # sp2 peptide
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


def bond_length_loss(
    coords: Tensor,
    mask: Optional[Tensor] = None,
    valid_peptide: Optional[Tensor] = None,
) -> Tensor:
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
    loss_within = (
        (d_N_CA - BOND_LENGTHS["N_CA"]) ** 2 +
        (d_CA_C - BOND_LENGTHS["CA_C"]) ** 2 +
        (d_C_O - BOND_LENGTHS["C_O"]) ** 2
    )
    loss_peptide = (d_C_N - BOND_LENGTHS["C_N"]) ** 2

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


def bond_angle_loss(
    coords: Tensor,
    mask: Optional[Tensor] = None,
    valid_peptide: Optional[Tensor] = None,
) -> Tensor:
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
    loss_within = (
        (angle_N_CA_C - BOND_ANGLES["N_CA_C"]) ** 2 +
        (angle_CA_C_O - BOND_ANGLES["CA_C_O"]) ** 2
    ) / 100.0
    loss_peptide = (
        (angle_CA_C_N - BOND_ANGLES["CA_C_N"]) ** 2 +
        (angle_C_N_CA - BOND_ANGLES["C_N_CA"]) ** 2
    ) / 100.0

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


def omega_loss(
    coords: Tensor,
    mask: Optional[Tensor] = None,
    valid_peptide: Optional[Tensor] = None,
) -> Tensor:
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
    trans_dev = (torch.abs(omega) - math.pi) ** 2
    cis_dev = omega ** 2

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


def o_chirality_loss(
    coords: Tensor,
    mask: Optional[Tensor] = None,
    valid_peptide: Optional[Tensor] = None,
) -> Tensor:
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


def virtual_cb_loss(
    coords: Tensor,
    mask: Optional[Tensor] = None,
    glycine_mask: Optional[Tensor] = None,
) -> Tensor:
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
    loss = (chi - expected) ** 2

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
            "bond_length": bond_length_weight,
            "bond_angle": bond_angle_weight,
            "omega": omega_weight,
            "o_chirality": o_chirality_weight,
            "cb_chirality": cb_chirality_weight,
        }
        self.bound_losses = bound_losses
        # Scale factors for bounding (loss of this value maps to 0.5)
        self.scales = {
            "bond_length": 1.0,
            "bond_angle": 50.0,
            "omega": 1.0,
            "o_chirality": 1.0,
            "cb_chirality": 1.0,
        }

    def forward(
        self,
        coords: Tensor,
        mask: Optional[Tensor] = None,
        glycine_mask: Optional[Tensor] = None,
        gt_coords: Optional[Tensor] = None,
        peptide_threshold: float = 0.2,
    ) -> Dict[str, Tensor]:
        """Compute geometry losses.

        Args:
            coords: [B, L, 4, 3] predicted atom positions (N, CA, C, O)
            mask: [B, L] residue mask (optional)
            glycine_mask: [B, L] True for glycine residues (optional)
            gt_coords: [B, L, 4, 3] ground truth coords for peptide bond detection (optional)
            peptide_threshold: Max C-N distance (normalized) for valid peptide bond (default 0.2 = 2A)

        Returns:
            dict with individual losses + 'total'
        """
        losses = {}

        # Compute valid peptide mask from GT coords (not predicted)
        valid_peptide = None
        if gt_coords is not None:
            C_gt = gt_coords[..., 2, :]  # [B, L, 3]
            N_next_gt = gt_coords[:, 1:, 0, :]  # [B, L-1, 3]
            d_C_N_gt = (C_gt[:, :-1] - N_next_gt).norm(dim=-1)  # [B, L-1]
            valid_peptide = d_C_N_gt < peptide_threshold

        # Only compute losses with non-zero weights
        if self.weights["bond_length"] > 0:
            losses["bond_length"] = bond_length_loss(coords, mask, valid_peptide)
        else:
            losses["bond_length"] = torch.tensor(0.0, device=coords.device)

        if self.weights["bond_angle"] > 0:
            losses["bond_angle"] = bond_angle_loss(coords, mask, valid_peptide)
        else:
            losses["bond_angle"] = torch.tensor(0.0, device=coords.device)

        if self.weights["omega"] > 0:
            losses["omega"] = omega_loss(coords, mask, valid_peptide)
        else:
            losses["omega"] = torch.tensor(0.0, device=coords.device)

        if self.weights["o_chirality"] > 0:
            losses["o_chirality"] = o_chirality_loss(coords, mask, valid_peptide)
        else:
            losses["o_chirality"] = torch.tensor(0.0, device=coords.device)

        if self.weights["cb_chirality"] > 0:
            losses["cb_chirality"] = virtual_cb_loss(coords, mask, glycine_mask)
        else:
            losses["cb_chirality"] = torch.tensor(0.0, device=coords.device)

        # Apply bounding if enabled
        if self.bound_losses:
            for k in losses:
                if k in self.scales:
                    losses[k] = bounded_loss(losses[k], self.scales[k])

        # Weighted sum
        total = sum(self.weights[k] * v for k, v in losses.items())
        losses["total"] = total

        return losses

    def __repr__(self):
        active = [k for k, v in self.weights.items() if v > 0]
        return f"GeometryLoss(active={active})"
