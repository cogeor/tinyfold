"""Tests for live model components.

Following the testing philosophy from CLAUDE.md: focus on shape invariants and
diffusion-schedule properties. (Tests for the retired EGNN/Pairformer/TokenEmbedder
lineage were removed when those modules were deleted.)
"""

import torch

from tinyfold.model.diffusion.schedule import DiffusionSchedule


class TestDiffusionSchedule:
    """Tests for the diffusion schedule."""

    def test_alpha_bar_monotonic(self):
        """alpha_bar should be monotonically decreasing."""
        schedule = DiffusionSchedule(T=16)
        for t in range(1, schedule.T):
            assert schedule.alpha_bar[t] < schedule.alpha_bar[t - 1], \
                f"alpha_bar not decreasing at t={t}"

    def test_alpha_bar_bounds(self):
        """alpha_bar should stay in (0, 1]."""
        schedule = DiffusionSchedule(T=16)
        assert (schedule.alpha_bar > 0).all(), "alpha_bar <= 0"
        assert (schedule.alpha_bar <= 1).all(), "alpha_bar > 1"

    def test_q_sample_variance(self):
        """q_sample should produce roughly unit variance for unit-variance x0."""
        schedule = DiffusionSchedule(T=16)
        x0 = torch.randn(1000, 3)
        for t in [0, 7, 15]:
            var = schedule.q_sample(x0, t).var()
            assert 0.5 < var < 2.0, f"Unexpected variance at t={t}: {var}"
