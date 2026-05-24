"""Tests for the configurable chain-length filter in the data pipeline.

Pins the post-fix behaviour:
  - Default ``max_chain_length`` is ``None`` (no upper cap).
  - When ``max_chain_length`` is ``None``, arbitrarily long chains pass.
  - When ``max_chain_length`` is an int, oversized chains are rejected.
  - The ``min_chain_length`` floor still rejects parsing fragments.

Pre-fix, the function silently capped both chains at 300 residues (the
old ``MAX_CHAIN_LENGTH`` constant), which truncated DIPS-Plus during
data prep and made downstream training filter look more lenient than
it actually was. These tests guard against regression.
"""

import pytest

from tinyfold.constants import MAX_CHAIN_LENGTH, MIN_CHAIN_LENGTH
from tinyfold.data.processing.filters import (
    FilterReason,
    validate_chain_length,
)


def test_default_max_chain_length_is_none():
    """Default upper cap is disabled — faithful pass-through."""
    assert MAX_CHAIN_LENGTH is None


def test_default_min_chain_length_is_40():
    """Default lower bound still rejects fragments."""
    assert MIN_CHAIN_LENGTH == 40


def test_passes_long_chains_when_no_max():
    """1000 + 1000 residue chains pass with the default (no cap)."""
    result = validate_chain_length(LA=1000, LB=1000)
    assert result.passed, result.details


def test_rejects_when_explicit_max_exceeded():
    """Caller-supplied max cap still rejects oversized chains."""
    result = validate_chain_length(LA=350, LB=200, max_len=300)
    assert not result.passed
    assert result.reason == FilterReason.CHAIN_A_TOO_LONG


def test_rejects_chain_b_too_long():
    result = validate_chain_length(LA=200, LB=350, max_len=300)
    assert not result.passed
    assert result.reason == FilterReason.CHAIN_B_TOO_LONG


def test_min_chain_length_still_rejects_fragments():
    """Default min=40 still rejects short fragments."""
    result = validate_chain_length(LA=30, LB=200)
    assert not result.passed
    assert result.reason == FilterReason.CHAIN_A_TOO_SHORT


def test_empty_chains_rejected_regardless():
    result = validate_chain_length(LA=0, LB=200)
    assert not result.passed
    assert result.reason == FilterReason.CHAIN_A_EMPTY

    result = validate_chain_length(LA=200, LB=0)
    assert not result.passed
    assert result.reason == FilterReason.CHAIN_B_EMPTY


def test_explicit_none_max_disables_cap():
    """Passing max_len=None explicitly behaves like the default."""
    result = validate_chain_length(LA=5000, LB=5000, max_len=None)
    assert result.passed, result.details


@pytest.mark.parametrize("la,lb", [(40, 40), (40, 1000), (1000, 40), (300, 600), (600, 300)])
def test_at_or_above_default_min_passes(la: int, lb: int):
    """Various pairs at or above the default min, with no cap, all pass."""
    result = validate_chain_length(LA=la, LB=lb)
    assert result.passed, (la, lb, result.details)
