"""Tests for chain-permutation-aware training loss (C5).

For sequence-identical (homodimer) chains the GT A/B labelling is arbitrary, so
the loss should score against whichever assignment the prediction is closer to,
applied consistently across every coordinate loss. Heterodimers are untouched.
"""

import numpy as np
import pyarrow as pa
import torch

from tinyfold.model.losses import (
    apply_chain_swap,
    choose_chain_permutation,
    compute_mse_loss,
    permutation_aligned_target,
)
from tinyfold.model.metrics import chains_are_interchangeable
from tinyfold.training.data import load_sample


def _homodimer(L=6):
    la = lb = L // 2
    chain = torch.tensor([0] * la + [1] * lb)
    seq = torch.tensor([5, 6, 7][:la] * 1 + [5, 6, 7][:lb] * 1)  # A == B
    mask = torch.ones(L, dtype=torch.bool)
    return chain, seq, mask


# --- swap invariance of the loss -------------------------------------------

def test_homodimer_label_swapped_target_gives_same_loss():
    """The permutation-aligned loss must not depend on which arbitrary labelling
    the GT arrived in."""
    torch.manual_seed(0)
    chain, seq, mask = _homodimer()
    chain, seq, mask = chain[None], seq[None], mask[None]
    pred = torch.randn(1, 6, 3)
    target = torch.randn(1, 6, 3)
    swapped = apply_chain_swap(target, chain, mask, torch.tensor([True]))

    t1 = permutation_aligned_target(pred, target, chain, seq, mask)
    t2 = permutation_aligned_target(pred, swapped, chain, seq, mask)
    loss1 = compute_mse_loss(pred, t1, mask)
    loss2 = compute_mse_loss(pred, t2, mask)
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_swap_is_chosen_when_it_lowers_the_loss():
    chain, seq, mask = _homodimer()
    chain, seq, mask = chain[None], seq[None], mask[None]
    target = torch.randn(1, 6, 3)
    swapped = apply_chain_swap(target, chain, mask, torch.tensor([True]))
    # Predict exactly the SWAPPED arrangement -> the swap assignment must win.
    pred = swapped.clone()
    swap = choose_chain_permutation(pred, target, chain, seq, mask)
    assert bool(swap[0]) is True
    aligned = permutation_aligned_target(pred, target, chain, seq, mask)
    assert torch.allclose(aligned, swapped, atol=1e-6)


# --- heterodimers are untouched --------------------------------------------

def test_heterodimer_target_is_bitwise_unchanged():
    L = 6
    chain = torch.tensor([0, 0, 0, 1, 1, 1])[None]
    seq = torch.tensor([1, 2, 3, 9, 8, 7])[None]   # A != B
    mask = torch.ones(1, L, dtype=torch.bool)
    pred = torch.randn(1, L, 3)
    target = torch.randn(1, L, 3)
    out = permutation_aligned_target(pred, target, chain, seq, mask)
    assert torch.equal(out, target)
    assert not bool(choose_chain_permutation(pred, target, chain, seq, mask).any())


# --- one decision, applied consistently across terms -----------------------

def test_same_permutation_applies_to_centroids_and_atoms():
    """The swap decision is chosen once and must map centroids and atoms by the
    identical A<->B index mapping."""
    chain, _, mask = _homodimer()
    chain, mask = chain[None], mask[None]
    swap = torch.tensor([True])
    centroids = torch.randn(1, 6, 3)
    atoms = torch.randn(1, 6, 4, 3)

    c_sw = apply_chain_swap(centroids, chain, mask, swap)
    a_sw = apply_chain_swap(atoms, chain, mask, swap)

    # chain A residues (0,1,2) must now hold chain B residues (3,4,5), for both.
    assert torch.equal(c_sw[0, [0, 1, 2]], centroids[0, [3, 4, 5]])
    assert torch.equal(a_sw[0, [0, 1, 2]], atoms[0, [3, 4, 5]])
    assert torch.equal(c_sw[0, [3, 4, 5]], centroids[0, [0, 1, 2]])
    assert torch.equal(a_sw[0, [3, 4, 5]], atoms[0, [0, 1, 2]])


def test_swap_false_is_identity():
    chain, _, mask = _homodimer()
    chain, mask = chain[None], mask[None]
    x = torch.randn(1, 6, 3)
    assert torch.equal(apply_chain_swap(x, chain, mask, torch.tensor([False])), x)


# --- is_homomer cached at load time ----------------------------------------

def _table(seqs, chains):
    """Minimal parquet-shaped table load_sample can decode."""
    rows = {"sample_id": [], "seq": [], "chain_id_res": [], "res_idx": [],
            "atom_coords": [], "atom_to_res": [], "atom_type": [],
            "LA": [], "LB": []}
    rng = np.random.default_rng(0)
    for k, (seq, chain) in enumerate(zip(seqs, chains)):
        L = len(seq)
        la = int(sum(1 for c in chain if c == 0))
        rows["sample_id"].append(f"s{k}")
        rows["seq"].append(list(seq))
        rows["chain_id_res"].append(list(chain))
        rows["res_idx"].append(list(range(la)) + list(range(L - la)))
        rows["atom_coords"].append(rng.normal(size=4 * L * 3).astype(np.float32).tolist())
        rows["atom_to_res"].append([r for r in range(L) for _ in range(4)])
        rows["atom_type"].append([0, 1, 2, 3] * L)
        rows["LA"].append(la)
        rows["LB"].append(L - la)
    return pa.table(rows)


def test_is_homomer_matches_sequence_equality_at_load_time():
    table = _table(
        seqs=[[5, 6, 7, 5, 6, 7], [1, 2, 3, 9, 8, 7]],
        chains=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
    )
    homo = load_sample(table, 0)
    hetero = load_sample(table, 1)
    assert homo["is_homomer"] is True
    assert hetero["is_homomer"] is False
    # Agrees with the independent predicate on the decoded tensors.
    assert homo["is_homomer"] == chains_are_interchangeable(
        homo["aa_seq"], homo["chain_ids"]
    )
