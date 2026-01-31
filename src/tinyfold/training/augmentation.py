"""Augmentation utilities for TinyFold training.

Provides:
- random_rotation_matrix: Generate random SO(3) rotation matrices
- apply_rigid_augment: Apply random SE(3) transformation
- apply_rotation_augment: Apply random SO(3) rotation only
"""

import torch
from torch import Tensor


def random_rotation_matrix(batch_size: int, device: torch.device) -> Tensor:
    """Generate random rotation matrices using QR decomposition.
    
    Args:
        batch_size: Number of rotation matrices to generate
        device: Target device
        
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    # Random matrix
    A = torch.randn(batch_size, 3, 3, device=device)
    # QR decomposition gives orthogonal matrix
    Q, R = torch.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    signs = torch.sign(torch.diagonal(R, dim1=1, dim2=2))
    Q = Q * signs.unsqueeze(1)
    # Fix determinant
    det = torch.det(Q)
    Q[:, :, 0] *= det.unsqueeze(1)
    return Q


def apply_rigid_augment(
    coords_res: Tensor, 
    centroids: Tensor, 
    translation_scale: float = 2.0
) -> tuple[Tensor, Tensor]:
    """Apply random SE(3) transformation (rotation + translation).

    Args:
        coords_res: [B, L, 4, 3] atom coordinates
        centroids: [B, L, 3] residue centroids
        translation_scale: Scale for random translation

    Returns:
        coords_res_aug: [B, L, 4, 3] transformed atom coordinates
        centroids_aug: [B, L, 3] transformed centroids
    """
    B = coords_res.shape[0]
    device = coords_res.device

    # Generate random rotation matrices
    R = random_rotation_matrix(B, device)

    # Apply rotation to atom coordinates
    coords_flat = coords_res.view(B, -1, 3)
    coords_rot = torch.bmm(coords_flat, R.transpose(1, 2))
    
    # Apply rotation to centroids
    centroids_rot = torch.bmm(centroids, R.transpose(1, 2))

    # Generate random translation
    translation = torch.randn(B, 1, 3, device=device) * translation_scale

    # Apply translation
    coords_aug = coords_rot + translation
    centroids_aug = centroids_rot + translation

    return coords_aug.view_as(coords_res), centroids_aug


def apply_rotation_augment(coords_res: Tensor, centroids: Tensor) -> tuple[Tensor, Tensor]:
    """Apply random SO(3) rotation only (no translation).
    
    Alias for apply_rigid_augment with translation_scale=0.
    """
    return apply_rigid_augment(coords_res, centroids, translation_scale=0.0)
