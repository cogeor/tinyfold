"""Unit tests for ``compute_c_rmsd`` (Complex-RMSD).

Complex-RMSD Kabsch-aligns predicted chain A onto ground-truth chain A, then
applies that SINGLE rigid transform to the whole prediction (chain A and
chain B together), and finally returns RMSD over all valid CA. This test
file pins three behaviours:

1. When pred chain A matches GT chain A but chain B is moved off-pose, the
   returned RMSD reflects the chain-B displacement (not zero, not aligned-away).
2. When the WHOLE complex (both chains) is jointly rotated+translated by the
   same rigid transform, C-RMSD is ~0 (the chain-A fit recovers the transform).
3. Fewer than 3 chain-A residues triggers an ``AssertionError`` (the assert
   guards against a degenerate Kabsch SVD).
"""

import math

import pytest
import torch

from tinyfold.model.losses import compute_c_rmsd


def _rot_z(angle_rad: float) -> torch.Tensor:
    """3x3 rotation around z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)


def _make_two_chain_gt(n_a: int = 10, n_b: int = 10, seed: int = 0):
    """Build ground-truth CA tensor with chain A and chain B."""
    g = torch.Generator().manual_seed(seed)
    chain_a = torch.randn(n_a, 3, generator=g, dtype=torch.float64)
    chain_b = torch.randn(n_b, 3, generator=g, dtype=torch.float64) + torch.tensor([5.0, 0.0, 0.0])
    gt = torch.cat([chain_a, chain_b], dim=0).unsqueeze(0)  # [1, n_a+n_b, 3]
    chain_ids = torch.cat([
        torch.zeros(n_a, dtype=torch.long),
        torch.ones(n_b, dtype=torch.long),
    ]).unsqueeze(0)
    return gt, chain_ids


def test_chain_b_offset_gives_nontrivial_rmsd():
    """Chain A matches; chain B is rotated/translated off its native pose.

    Because the Kabsch fit on chain A is the identity (chain A pred == chain A
    gt), the transform applied to chain B is also the identity, so the residual
    on chain B equals its raw displacement. With a 30-degree z-rotation and a
    [5, -2, 3] shift, the per-chain-B residual is several Angstroms. The
    full-complex RMSD is bounded below by ``sqrt(n_b / (n_a + n_b)) *
    rmsd_chain_b`` (chain A contributes 0). We assert a permissive band:
    clearly above noise and clearly below the raw chain-B displacement.
    """
    gt, chain_ids = _make_two_chain_gt(n_a=10, n_b=10, seed=42)
    n_a = (chain_ids == 0).sum().item()

    R = _rot_z(math.radians(30.0))
    t = torch.tensor([5.0, -2.0, 3.0], dtype=torch.float64)

    pred = gt.clone()
    pred[0, n_a:] = pred[0, n_a:] @ R.T + t  # move chain B only

    rmsd = compute_c_rmsd(pred.float(), gt.float(), chain_ids).item()

    # Compute the raw chain-B displacement RMSD as a reference upper bound.
    diff = (pred[0, n_a:] - gt[0, n_a:]).float()
    rmsd_b_only = torch.sqrt((diff ** 2).sum(-1).mean()).item()

    # The C-RMSD over both chains (chain A residual = 0) should be ~half of
    # the chain-B-only residual since we average over twice as many atoms.
    assert rmsd > 1.0, f"expected RMSD comfortably above noise, got {rmsd}"
    assert rmsd < rmsd_b_only, (
        f"complex RMSD ({rmsd}) cannot exceed pure chain-B residual "
        f"({rmsd_b_only}) since chain A contributes 0"
    )
    # And it must be close to sqrt(n_b/(n_a+n_b)) * rmsd_b_only (within 1e-3).
    expected = math.sqrt(diff.shape[0] / pred.shape[1]) * rmsd_b_only
    assert abs(rmsd - expected) < 1e-3, (
        f"complex RMSD {rmsd} != expected {expected} = "
        f"sqrt(n_b/L) * chain_b_rmsd"
    )


def test_whole_complex_rigid_transform_is_zero():
    """Rigid-transforming BOTH chains by the same R, t -> C-RMSD ~ 0.

    The chain-A Kabsch fit recovers the exact transform; applying it back to
    the prediction perfectly undoes the rotation+translation, so the residual
    on both chains is numerically zero (up to float32 precision).
    """
    gt, chain_ids = _make_two_chain_gt(n_a=12, n_b=12, seed=7)

    R = _rot_z(math.radians(45.0))
    t = torch.tensor([1.5, -0.7, 2.2], dtype=torch.float64)

    # Rotate + translate the entire complex.
    pred = gt @ R.T + t  # broadcasts over the L axis

    rmsd = compute_c_rmsd(pred.float(), gt.float(), chain_ids).item()
    assert rmsd < 1e-4, f"rigid-transformed complex should fit exactly, got {rmsd}"


def test_too_few_chain_a_residues_raises():
    """compute_c_rmsd needs >=3 chain-A residues per sample for a stable SVD."""
    # Build a 5-residue 'complex' with only 2 chain-A residues.
    gt = torch.randn(1, 5, 3)
    chain_ids = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    pred = gt.clone()

    with pytest.raises(AssertionError):
        compute_c_rmsd(pred, gt, chain_ids)
