"""Pin the Kabsch adapters to the single canonical implementation.

`model.geometry.kabsch.kabsch_rigid` is the one torch Kabsch. Two thin adapters
wrap it — `losses.mse.kabsch_align` (origin-centred convention) and
`diffusion.utils.kabsch_align_to_target` (target-frame convention). These tests
guard against the adapters silently diverging from the canonical helper.
"""

import torch

from tinyfold.model.diffusion.utils import kabsch_align_to_target
from tinyfold.model.geometry.kabsch import kabsch_rigid
from tinyfold.model.losses.mse import kabsch_align


def _random_case(seed: int):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(3, 20, 3, generator=g)
    target = torch.randn(3, 20, 3, generator=g)
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[:, 15:] = False  # exercise the masked path
    return pred, target, mask


def test_diffusion_adapter_matches_canonical():
    pred, target, mask = _random_case(0)
    _, _, aligned = kabsch_rigid(pred, target, mask)
    got = kabsch_align_to_target(pred, target, mask)
    assert torch.allclose(got, aligned, atol=1e-6)


def test_mse_adapter_is_canonical_in_centred_frame():
    pred, target, mask = _random_case(1)
    _, _, aligned = kabsch_rigid(pred, target, mask)
    pred_aligned, target_centred = kabsch_align(pred, target, mask)
    # mse.kabsch_align returns both tensors origin-centred; the residual
    # pred_aligned - target_centred must equal the canonical aligned - target.
    m = mask.unsqueeze(-1)
    lhs = (pred_aligned - target_centred) * m
    rhs = (aligned - target) * m
    assert torch.allclose(lhs, rhs, atol=1e-6)
