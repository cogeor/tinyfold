"""Tests for tinyfold.model.resfold.relpos.RelposBias.

The bias must:
  - return a finite [B, n_heads, L, L] tensor for valid inputs
  - clip absolute index differences to ±clip
  - place the "same chain" and "different chain" buckets in distinct slots
  - back-propagate gradients into the learned table
  - initialize at zero so it's a no-op on freshly-constructed models
"""

from __future__ import annotations

import pytest
import torch

from tinyfold.model.resfold.relpos import RelposBias


def test_init_zero():
    layer = RelposBias(n_heads=8, clip=32)
    assert torch.all(layer.table == 0)


def test_output_shape_and_finite():
    layer = RelposBias(n_heads=4, clip=16)
    B, L = 2, 10
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_ids = torch.zeros(B, L, dtype=torch.long)
    out = layer(res_idx, chain_ids)
    assert out.shape == (B, 4, L, L)
    assert torch.isfinite(out).all()


def test_clip_saturates():
    """Pair distances beyond +clip and -clip must map to the same bucket."""
    layer = RelposBias(n_heads=1, clip=4)
    # Set distinct values in the table so different buckets are distinguishable.
    with torch.no_grad():
        layer.table.copy_(torch.arange(2 * 4 + 1).float().view(-1, 1, 1).expand(-1, 2, 1))

    # Three positions: 0, 5, 100. Same chain.
    res_idx = torch.tensor([[0, 5, 100]])
    chain_ids = torch.zeros(1, 3, dtype=torch.long)
    out = layer(res_idx, chain_ids)  # [1, 1, 3, 3]
    out = out.squeeze(0).squeeze(0)
    # diff[1, 2] = 5 - 100 = -95 -> clipped to -4 -> bucket 0 -> value 0
    # diff[2, 1] = 100 - 5 = +95 -> clipped to +4 -> bucket 8 -> value 8
    # diff[0, 2] = 0 - 100 = -100 -> clipped to -4 -> bucket 0 -> value 0
    # diff[2, 0] = +100 -> clipped to +4 -> bucket 8 -> value 8
    # diff[0, 1] = -5 -> clipped to -4 -> bucket 0 -> value 0
    # diff[1, 0] = +5 -> clipped to +4 -> bucket 8 -> value 8
    assert out[1, 2].item() == 0
    assert out[2, 1].item() == 8
    assert out[0, 2].item() == 0
    assert out[2, 0].item() == 8
    assert out[0, 1].item() == 0
    assert out[1, 0].item() == 8
    # Self-attention diagonal: diff=0 -> bucket = clip = 4 -> value 4
    for i in range(3):
        assert out[i, i].item() == 4


def test_same_vs_different_chain():
    layer = RelposBias(n_heads=1, clip=2)
    # Different value in the two same/diff slots so we can identify which
    # column was hit.
    with torch.no_grad():
        layer.table.zero_()
        # bucket 2 (diff=0), same=0 -> 7; same=1 -> 13
        layer.table[2, 0, 0] = 7.0
        layer.table[2, 1, 0] = 13.0

    res_idx = torch.tensor([[0, 0, 0]])
    chain_ids = torch.tensor([[0, 0, 1]])  # third token on different chain
    out = layer(res_idx, chain_ids).squeeze()  # [3, 3]
    # (0, 0) and (1, 1) and (2, 2): all self, same chain when chain[i]==chain[j].
    assert out[0, 0].item() == 13.0  # same chain (both 0)
    assert out[0, 2].item() == 7.0   # chain[0]=0, chain[2]=1 -> different
    assert out[2, 2].item() == 13.0  # same chain (both 1)


def test_gradient_flows_through_bias():
    layer = RelposBias(n_heads=2, clip=4)
    # Non-zero init so gradient is non-trivial.
    with torch.no_grad():
        layer.table.add_(0.1 * torch.randn_like(layer.table))

    res_idx = torch.arange(6).unsqueeze(0)
    chain_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    out = layer(res_idx, chain_ids)
    loss = out.pow(2).sum()
    loss.backward()
    assert layer.table.grad is not None
    assert torch.isfinite(layer.table.grad).all()
    assert (layer.table.grad.abs() > 0).any(), "expected some non-zero gradient"


def test_invalid_clip_raises():
    with pytest.raises(ValueError, match="clip"):
        RelposBias(n_heads=1, clip=0)
