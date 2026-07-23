"""Tests for chain-permutation-aware DockQ.

For a homodimer the two chains are indistinguishable on the input side, so
which one the ground truth calls "A" is an arbitrary crystallographic
labelling. Standard DockQ and the CAPRI criteria both allow optimal chain
mapping; without it a structurally correct prediction with A/B exchanged scores
as a failure. 34.5% of DIPS-Plus complexes are exact homodimers, so this
silently penalised a third of every evaluation.

Skips if the optional DockQ backend is not importable.
"""

import numpy as np
import pytest
import torch

from tinyfold.model.metrics.dockq import chains_are_interchangeable, compute_dockq


def _strand(n: int, offset: np.ndarray) -> np.ndarray:
    coords = np.zeros((n, 4, 3), np.float32)
    for i in range(n):
        ca = np.array([i * 3.8, 0.0, 0.0], np.float32) + offset
        coords[i, 0] = ca + [-1.0, 0.3, 0.0]  # N
        coords[i, 1] = ca                      # CA
        coords[i, 2] = ca + [1.0, 0.3, 0.0]    # C
        coords[i, 3] = ca + [1.4, 1.2, 0.0]    # O
    return coords


def _homodimer():
    """Two identical 12-residue strands 4.5 A apart -> real interface."""
    a = _strand(12, np.array([0, 0, 0], np.float32))
    b = _strand(12, np.array([0, 4.5, 0], np.float32))
    coords = torch.tensor(np.concatenate([a, b], 0))
    aa = torch.zeros(24, dtype=torch.long)          # identical sequences
    chain_ids = torch.tensor([0] * 12 + [1] * 12)
    return coords, aa, chain_ids


def _heterodimer():
    """Same geometry, but the two chains carry different sequences."""
    coords, _, chain_ids = _homodimer()
    aa = torch.tensor([0] * 12 + [1] * 12, dtype=torch.long)
    return coords, aa, chain_ids


def _swap_chains(coords: torch.Tensor) -> torch.Tensor:
    """Exchange the coordinates of the two 12-residue chains."""
    return torch.cat([coords[12:], coords[:12]], dim=0)


def _require_backend(result):
    if result["dockq"] is None:
        pytest.skip("DockQ backend not available")


# --- the interchangeability predicate -------------------------------------

def test_identical_sequences_are_interchangeable():
    _, aa, chain_ids = _homodimer()
    assert chains_are_interchangeable(aa, chain_ids)


def test_different_sequences_are_not_interchangeable():
    _, aa, chain_ids = _heterodimer()
    assert not chains_are_interchangeable(aa, chain_ids)


def test_different_lengths_are_not_interchangeable():
    aa = torch.zeros(24, dtype=torch.long)
    chain_ids = torch.tensor([0] * 10 + [1] * 14)
    assert not chains_are_interchangeable(aa, chain_ids)


def test_single_chain_is_not_interchangeable():
    aa = torch.zeros(12, dtype=torch.long)
    chain_ids = torch.zeros(12, dtype=torch.long)
    assert not chains_are_interchangeable(aa, chain_ids)


# --- the behaviour that matters -------------------------------------------

def test_label_swapped_homodimer_scores_the_same():
    """The headline fix: relabelling a correct homodimer must not lose points."""
    coords, aa, chain_ids = _homodimer()
    ref = compute_dockq(coords, coords, aa, chain_ids)
    _require_backend(ref)

    swapped = _swap_chains(coords)
    r = compute_dockq(swapped, coords, aa, chain_ids)
    assert r["dockq"] == pytest.approx(ref["dockq"], abs=1e-3)
    assert r["chain_perm_used"] is True


def test_label_swapped_homodimer_is_penalised_without_the_fix():
    """Guards the fix: with permutation off, the swap is scored as a failure.

    If this test ever stops failing to reproduce the penalty, the swap fixture
    is not actually exercising the assignment and the test above is vacuous.
    """
    coords, aa, chain_ids = _homodimer()
    ref = compute_dockq(coords, coords, aa, chain_ids)
    _require_backend(ref)

    swapped = _swap_chains(coords)
    r = compute_dockq(swapped, coords, aa, chain_ids,
                      allow_chain_permutation=False)
    assert r["dockq"] < ref["dockq"]
    assert r["chain_perm_used"] is False


def test_unswapped_prediction_does_not_report_a_permutation():
    coords, aa, chain_ids = _homodimer()
    r = compute_dockq(coords, coords, aa, chain_ids)
    _require_backend(r)
    assert r["dockq"] == pytest.approx(1.0, abs=1e-3)
    assert r["chain_perm_used"] is False


def test_heterodimer_is_bitwise_unaffected():
    """A heterodimer admits only one assignment, so nothing may change."""
    coords, aa, chain_ids = _heterodimer()
    on = compute_dockq(coords, coords, aa, chain_ids, allow_chain_permutation=True)
    _require_backend(on)
    off = compute_dockq(coords, coords, aa, chain_ids, allow_chain_permutation=False)
    assert on["dockq"] == off["dockq"]
    assert on["fnat"] == off["fnat"]
    assert on["chain_perm_used"] is False


def test_permutation_never_lowers_the_score():
    """Taking the max over assignments is monotone: >= the fixed assignment."""
    coords, aa, chain_ids = _homodimer()
    pred = coords.clone()
    pred[12:] += torch.tensor([0.0, 1.5, 0.0])  # partially wrong pose

    on = compute_dockq(pred, coords, aa, chain_ids, allow_chain_permutation=True)
    _require_backend(on)
    off = compute_dockq(pred, coords, aa, chain_ids, allow_chain_permutation=False)
    assert on["dockq"] >= off["dockq"] - 1e-9


def test_result_always_carries_the_permutation_key():
    """Downstream code may read the key unconditionally."""
    coords, aa, chain_ids = _homodimer()
    r = compute_dockq(coords, coords, aa, chain_ids)
    assert "chain_perm_used" in r
    for key in ("dockq", "fnat", "irms", "lrms"):
        assert key in r, f"missing legacy key {key}"
