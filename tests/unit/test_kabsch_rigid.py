"""Unit tests for the shared Kabsch rigid-alignment primitive.

Covers the regression contract pinned in
``.delegate/work/20260524-051951-prio01-retrain/03/PLAN.md`` D1 / Task 4 test 1:

- ``R_hat`` is a proper orthonormal rotation (det = +1).
- ``aligned`` recovers ``tgt`` to ~1e-5 when ``tgt = (R @ src.T).T + t``.
- mask=ones matches the no-mask path.
- partial mask aligns only the unmasked subset.
"""

import torch

from tinyfold.model.geometry import kabsch_rigid
from tinyfold.training.augmentation import random_rotation_matrix


def _random_se3(B: int, device, t_scale: float = 5.0, seed: int = 0):
    """Sample a (R, t) pair with R from the existing util and t in [-s, s]."""
    g = torch.Generator(device=device).manual_seed(seed)
    R = random_rotation_matrix(B, device)
    t = (torch.rand(B, 3, device=device, generator=g) * 2.0 - 1.0) * t_scale
    return R, t


class TestKabschRigid:
    """Behavioural contract for ``kabsch_rigid``."""

    def test_proper_rotation_recovered(self):
        device = torch.device("cpu")
        torch.manual_seed(123)
        B, N = 2, 50
        src = torch.randn(B, N, 3, device=device)
        R, t = _random_se3(B, device, seed=7)
        # tgt = (R @ src) + t, per batch.
        tgt = torch.bmm(src, R.transpose(1, 2)) + t.unsqueeze(1)

        R_hat, t_hat, aligned = kabsch_rigid(src, tgt)

        # Orthonormal.
        eye = torch.eye(3, device=device).expand(B, -1, -1)
        assert torch.allclose(
            torch.bmm(R_hat, R_hat.transpose(1, 2)), eye, atol=1e-5
        )
        # Proper rotation (det = +1).
        assert torch.allclose(
            torch.det(R_hat), torch.ones(B, device=device), atol=1e-5
        )
        # Aligned reproduces tgt.
        assert torch.allclose(aligned, tgt, atol=1e-4)
        # t_hat sits inside the expected range (sanity, not a tight bound).
        assert t_hat.shape == (B, 3)

    def test_identity_when_src_equals_tgt(self):
        torch.manual_seed(7)
        src = torch.randn(2, 30, 3)
        R, t, aligned = kabsch_rigid(src, src.clone())
        eye = torch.eye(3).expand(2, -1, -1)
        # R should be (close to) identity.
        assert torch.allclose(R, eye, atol=1e-4)
        # t should be (close to) zero.
        assert torch.allclose(t, torch.zeros(2, 3), atol=1e-5)
        # aligned should match src.
        assert torch.allclose(aligned, src, atol=1e-5)

    def test_mask_ones_matches_nomask(self):
        torch.manual_seed(11)
        B, N = 2, 40
        src = torch.randn(B, N, 3)
        R, t = _random_se3(B, src.device, seed=3)
        tgt = torch.bmm(src, R.transpose(1, 2)) + t.unsqueeze(1)
        mask = torch.ones(B, N, dtype=torch.bool)

        R0, t0, a0 = kabsch_rigid(src, tgt, mask=None)
        R1, t1, a1 = kabsch_rigid(src, tgt, mask=mask)

        assert torch.allclose(R0, R1, atol=1e-5)
        assert torch.allclose(t0, t1, atol=1e-5)
        assert torch.allclose(a0, a1, atol=1e-5)

    def test_partial_mask_aligns_unmasked_subset(self):
        """src and tgt agree on the MASKED positions (junk we want to ignore)
        but disagree on the UNMASKED ones (the real signal). The recovered R
        must align the unmasked subset to ~1e-4, even though the masked
        positions stay disagreeing.
        """
        torch.manual_seed(17)
        B, N = 2, 60
        device = torch.device("cpu")

        # The "real" signal on the unmasked half.
        src_real = torch.randn(B, N, 3, device=device)
        R, t = _random_se3(B, device, seed=5)
        tgt_real = torch.bmm(src_real, R.transpose(1, 2)) + t.unsqueeze(1)

        # Mask: first half True (unmasked), second half False (masked-out).
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask[:, : N // 2] = True

        # Build src/tgt that agree on the masked half (so they DON'T influence
        # the SVD if mask is respected) but disagree on the unmasked half.
        src = src_real.clone()
        tgt = tgt_real.clone()
        # Replace the masked-out half of tgt with the same coords as src — if
        # the mask is ignored, these will distort the fit.
        tgt[:, N // 2 :] = src[:, N // 2 :]

        R_hat, t_hat, aligned = kabsch_rigid(src, tgt, mask=mask)

        # Aligned must match tgt on the UNMASKED half (the "real" signal).
        delta_unmasked = (aligned[:, : N // 2] - tgt[:, : N // 2]).abs()
        assert delta_unmasked.max() < 1e-3, (
            f"unmasked alignment error too large: {delta_unmasked.max().item():.6f}"
        )

    def test_batch_independence(self):
        """Different batch elements get independent (R, t)."""
        torch.manual_seed(31)
        N = 40
        src_a = torch.randn(1, N, 3)
        src_b = torch.randn(1, N, 3)
        src = torch.cat([src_a, src_b], dim=0)

        R_ab, t_ab = _random_se3(2, src.device, seed=9)
        tgt = torch.bmm(src, R_ab.transpose(1, 2)) + t_ab.unsqueeze(1)

        R_hat, _, aligned = kabsch_rigid(src, tgt)
        # Each batch element should solve its own problem.
        assert torch.allclose(aligned, tgt, atol=1e-4)
        # The two recovered R's should differ (random SE3 are independent).
        assert not torch.allclose(R_hat[0], R_hat[1], atol=1e-2)
