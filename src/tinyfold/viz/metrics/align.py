"""Kabsch alignment utilities."""

import numpy as np


def kabsch_align(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align predicted coordinates to reference using Kabsch algorithm.

    Finds optimal rotation R and translation t that minimizes RMSD.

    Args:
        pred_xyz: [N, 3] predicted coordinates
        ref_xyz: [N, 3] reference coordinates
        mask: [N] boolean mask for atoms to use in alignment (optional)

    Returns:
        aligned_pred: [N, 3] aligned predicted coordinates
        R: [3, 3] rotation matrix
        t: [3] translation vector
    """
    if mask is not None:
        pred_masked = pred_xyz[mask]
        ref_masked = ref_xyz[mask]
    else:
        pred_masked = pred_xyz
        ref_masked = ref_xyz

    # Center both point clouds
    pred_center = pred_masked.mean(axis=0)
    ref_center = ref_masked.mean(axis=0)

    pred_centered = pred_masked - pred_center
    ref_centered = ref_masked - ref_center

    # Compute covariance matrix
    H = pred_centered.T @ ref_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation (handle reflection case)
    R = Vt.T @ U.T

    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = ref_center - R @ pred_center

    # Apply transformation to all points
    aligned_pred = (R @ pred_xyz.T).T + t

    return aligned_pred, R, t


def align_on_subset(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    align_mask: np.ndarray,
) -> np.ndarray:
    """Align on a subset of atoms, apply transformation to all.

    Useful for receptor-aligned LRMSD: align on chain A, evaluate on chain B.

    Args:
        pred_xyz: [N, 3] predicted coordinates
        ref_xyz: [N, 3] reference coordinates
        align_mask: [N] boolean mask for atoms to use in alignment

    Returns:
        aligned_pred: [N, 3] aligned predicted coordinates
    """
    aligned_pred, _, _ = kabsch_align(pred_xyz, ref_xyz, mask=align_mask)
    return aligned_pred
