"""Atom-diffusion loss-weight schedule.

The atom loss rides on the SAME trunk as the centroid stage, so its gradient
slightly slows centroid convergence (measured in the 1baz overfit). The fix is a
loss-BALANCE knob, not detaching the atom gradient -- detaching was measured
decisively worse for atoms (0.32 A -> 1.14 A), see
notes/2026-07-11-crop-diffusion-2stage-SPEC.md.

``atom_loss_ramp`` is the pure schedule extracted from train_resfold.py so it can
be tested without a GPU or a training run.
"""

from itertools import pairwise

from tinyfold.training.objective import atom_loss_ramp


class TestLegacyBehaviourPreserved:
    """start_step=0 must reproduce the original inline ramp exactly.

    Original: ``ramp = min(1.0, step / warmup) if warmup > 0 else 1.0``
    then ``alpha = ramp * atom_weight``.
    """

    def test_matches_original_formula(self):
        weight, warmup = 0.5, 500
        for step in [0, 1, 100, 250, 499, 500, 501, 5000]:
            expected = min(1.0, step / warmup) * weight
            assert atom_loss_ramp(
                step, weight=weight, warmup_steps=warmup, start_step=0
            ) == expected

    def test_zero_warmup_is_full_weight_immediately(self):
        # warmup <= 0 disabled the ramp entirely in the original.
        for warmup in (0, -1):
            assert atom_loss_ramp(0, weight=0.5, warmup_steps=warmup) == 0.5
            assert atom_loss_ramp(9999, weight=0.5, warmup_steps=warmup) == 0.5

    def test_default_start_step_is_zero(self):
        # Callers that omit start_step get the legacy schedule.
        assert atom_loss_ramp(250, weight=0.5, warmup_steps=500) == 0.25


class TestDelayedStart:
    """start_step > 0 lets centroids converge BEFORE atoms enter the loss."""

    def test_zero_before_start(self):
        for step in [0, 100, 1999]:
            assert atom_loss_ramp(
                step, weight=0.5, warmup_steps=500, start_step=2000
            ) == 0.0

    def test_ramps_from_start_step(self):
        # Ramp is measured from start_step, not from step 0.
        assert atom_loss_ramp(2000, weight=0.5, warmup_steps=500, start_step=2000) == 0.0
        assert atom_loss_ramp(2250, weight=0.5, warmup_steps=500, start_step=2000) == 0.25
        assert atom_loss_ramp(2500, weight=0.5, warmup_steps=500, start_step=2000) == 0.5

    def test_saturates_at_weight(self):
        assert atom_loss_ramp(10_000, weight=0.5, warmup_steps=500, start_step=2000) == 0.5

    def test_delayed_start_with_no_warmup_is_a_step_function(self):
        def f(s):
            return atom_loss_ramp(s, weight=0.5, warmup_steps=0, start_step=2000)

        assert f(1999) == 0.0
        assert f(2000) == 0.5


class TestEdgeCases:
    def test_zero_weight_stays_zero(self):
        assert atom_loss_ramp(5000, weight=0.0, warmup_steps=500) == 0.0

    def test_monotonic_non_decreasing(self):
        vals = [
            atom_loss_ramp(s, weight=0.5, warmup_steps=500, start_step=1000)
            for s in range(0, 3000, 50)
        ]
        assert all(b >= a for a, b in pairwise(vals))

    def test_never_exceeds_weight(self):
        for s in range(0, 5000, 37):
            assert atom_loss_ramp(s, weight=0.5, warmup_steps=500, start_step=1000) <= 0.5

    def test_negative_start_step_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            atom_loss_ramp(0, weight=0.5, warmup_steps=500, start_step=-1)
