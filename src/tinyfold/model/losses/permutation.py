"""Chain-permutation-aware training loss (C5).

34.5% of DIPS-Plus is exact homodimers. For such a complex the two chains are
indistinguishable on the input side, while the ground-truth assignment of which
chain is "A" is an arbitrary crystallographic labelling. Scoring the model
against one fixed labelling hands it mutually contradictory targets across
samples, and the minimum-loss response is to regress toward the mean -- the
homodimer failure mode.

AF-Multimer makes permutation alignment mandatory for homomers; Boltz-1 applies
greedy symmetry correction. Here, for sequence-identical chains, we pick the
chain assignment (identity vs A<->B swap) that minimises the Kabsch-aligned
centroid MSE, and apply that SAME assignment to every coordinate loss so they
stay consistent within a step.

The homodimer predicate reuses ``chains_are_interchangeable`` (landed in A4) so
train and eval agree on what counts as interchangeable.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tinyfold.model.losses.mse import compute_mse_loss
from tinyfold.model.metrics import chains_are_interchangeable


def _chain_positions(chain_ids_b: Tensor, mask_b: Tensor) -> tuple[Tensor, Tensor]:
    """Valid residue positions of chain A and chain B, each ascending."""
    a_pos = torch.nonzero((chain_ids_b == 0) & mask_b, as_tuple=False).squeeze(-1)
    b_pos = torch.nonzero((chain_ids_b == 1) & mask_b, as_tuple=False).squeeze(-1)
    return a_pos, b_pos


def choose_chain_permutation(
    pred: Tensor,          # [B, L, 3] predicted centroids (detached upstream)
    target: Tensor,        # [B, L, 3] target centroids
    chain_ids: Tensor,     # [B, L]
    aa_seq: Tensor,        # [B, L]
    mask: Tensor | None = None,  # [B, L]
) -> Tensor:
    """Return a ``[B]`` bool tensor: True where swapping chains A<->B lowers the
    Kabsch-aligned centroid MSE for a sequence-identical (homodimer) sample.
    Always False for heterodimers and for samples a crop left with unequal chain
    lengths (not swappable). Evaluated ONCE per step so all coordinate losses use
    a consistent assignment.
    """
    B, L = chain_ids.shape
    device = chain_ids.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    swap = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        m = mask[b]
        if not chains_are_interchangeable(aa_seq[b][m], chain_ids[b][m]):
            continue
        a_pos, b_pos = _chain_positions(chain_ids[b], m)
        if a_pos.numel() == 0 or a_pos.numel() != b_pos.numel():
            continue
        swapped = target[b].clone()
        swapped[a_pos] = target[b][b_pos]
        swapped[b_pos] = target[b][a_pos]
        mse_id = compute_mse_loss(
            pred[b:b + 1], target[b:b + 1], mask[b:b + 1], reduction="per_sample"
        )
        mse_sw = compute_mse_loss(
            pred[b:b + 1], swapped.unsqueeze(0), mask[b:b + 1], reduction="per_sample"
        )
        if mse_sw < mse_id:
            swap[b] = True
    return swap


def apply_chain_swap(
    x: Tensor,             # [B, L, ...] per-residue tensor
    chain_ids: Tensor,     # [B, L]
    mask: Tensor,          # [B, L]
    swap: Tensor,          # [B] bool
) -> Tensor:
    """Exchange chain-A and chain-B residues in per-residue tensor ``x`` for the
    samples flagged in ``swap``. Returns a new tensor; unflagged samples and
    padding are untouched (bitwise). Works for any trailing shape ([B,L,3]
    centroids, [B,L,4,3] atoms, [B,L,14,3] atom14, ...).
    """
    out = x.clone()
    for b in range(x.shape[0]):
        if not bool(swap[b]):
            continue
        a_pos, b_pos = _chain_positions(chain_ids[b], mask[b])
        if a_pos.numel() != b_pos.numel():
            continue
        out[b, a_pos] = x[b, b_pos]
        out[b, b_pos] = x[b, a_pos]
    return out


def permutation_aligned_target(
    pred: Tensor,
    target: Tensor,
    chain_ids: Tensor,
    aa_seq: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Target under the loss-minimising chain assignment (identity or A<->B
    swap), per sample, for sequence-identical chains only. Heterodimers are
    returned unchanged (bitwise). Convenience wrapper over
    :func:`choose_chain_permutation` + :func:`apply_chain_swap`.
    """
    B, L = chain_ids.shape
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=chain_ids.device)
    swap = choose_chain_permutation(pred, target, chain_ids, aa_seq, mask)
    return apply_chain_swap(target, chain_ids, mask, swap)
