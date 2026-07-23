"""Tests for unclamped inter-chain loss weighting (C6).

Two independent pieces, both off at their defaults:
- interchain_weight up-weights inter-chain pairs in the distance-consistency
  loss (1.0 = bitwise-unchanged).
- FAPE clamps intra-chain pairs at fape_clamp but leaves inter-chain pairs
  unclamped, so a badly-docked interface keeps producing gradient.
"""

import torch

from tinyfold.model.losses import (
    compute_distance_consistency_loss,
    compute_fape_loss,
)
from tinyfold.training import random_rotation_matrix


def _backbone(B, L, seed=0):
    torch.manual_seed(seed)
    # Random but non-degenerate N/CA/C/O per residue.
    return torch.randn(B, L, 4, 3)


# --- interchain_weight ------------------------------------------------------

def test_interchain_weight_1_is_bitwise_identical():
    torch.manual_seed(0)
    pred = torch.randn(2, 6, 3)
    target = torch.randn(2, 6, 3)
    mask = torch.ones(2, 6, dtype=torch.bool)
    chain = torch.tensor([[0, 0, 0, 1, 1, 1]] * 2)
    base = compute_distance_consistency_loss(pred, target, mask)
    with_chain = compute_distance_consistency_loss(
        pred, target, mask, chain_ids=chain, interchain_weight=1.0
    )
    assert torch.equal(base, with_chain)


def test_interchain_weight_up_weights_interface_error():
    """Perfect intra-chain distances, wrong inter-chain distances: raising the
    inter-chain weight must raise the loss."""
    L = 6
    chain = torch.tensor([[0, 0, 0, 1, 1, 1]])
    mask = torch.ones(1, L, dtype=torch.bool)
    torch.manual_seed(1)
    target = torch.randn(1, L, 3)
    # pred: keep each chain's internal geometry, but move chain B far away so
    # only inter-chain distances are wrong.
    pred = target.clone()
    pred[:, 3:] += torch.tensor([20.0, 0.0, 0.0])
    lo = compute_distance_consistency_loss(pred, target, mask, chain_ids=chain, interchain_weight=1.0)
    hi = compute_distance_consistency_loss(pred, target, mask, chain_ids=chain, interchain_weight=5.0)
    assert hi > lo


# --- FAPE -------------------------------------------------------------------

def test_fape_is_invariant_to_global_rigid_transform():
    pred = _backbone(1, 5, seed=2)
    target = _backbone(1, 5, seed=3)
    chain = torch.tensor([[0, 0, 0, 1, 1]])
    base = compute_fape_loss(pred, target, chain)

    R = random_rotation_matrix(1, pred.device)[0]      # [3,3]
    t = torch.tensor([3.0, -2.0, 1.0])
    def rigid(x):
        return x @ R.T + t
    pred_r = rigid(pred.reshape(-1, 3)).reshape(pred.shape)
    target_r = rigid(target.reshape(-1, 3)).reshape(target.shape)
    moved = compute_fape_loss(pred_r, target_r, chain)
    assert torch.allclose(base, moved, atol=1e-4)


def test_interchain_pairs_are_unclamped_while_intrachain_saturate():
    """A rigidly-displaced chain B produces a frame-relative error == the shift
    on every cross-chain pair. Labelled inter-chain it grows unbounded with the
    shift; labelled intra-chain it saturates at the clamp."""
    pred = _backbone(1, 4, seed=4)
    chain_inter = torch.tensor([[0, 0, 1, 1]])
    chain_intra = torch.tensor([[0, 0, 0, 0]])
    clamp = 10.0

    def fape_for(shift, chain):
        target = pred.clone()
        target[:, 2:] += torch.tensor([shift, 0.0, 0.0])
        return compute_fape_loss(pred, target, chain, clamp=clamp)

    inter_20 = fape_for(20.0, chain_inter)
    inter_40 = fape_for(40.0, chain_inter)
    intra_20 = fape_for(20.0, chain_intra)
    intra_40 = fape_for(40.0, chain_intra)

    # Inter-chain: unbounded growth with the displacement.
    assert inter_40 > inter_20 * 1.5
    # Intra-chain: saturated at the clamp, so doubling the shift does not move it.
    assert torch.allclose(intra_20, intra_40, atol=1e-4)
    # And unclamped inter-chain exceeds the clamped intra-chain at the same shift.
    assert inter_20 > intra_20
