"""Tests for make_template_inputs source dispatch (oracle / monomer / retrieved)."""

import pytest
import torch

from tinyfold.retrieval import make_template_inputs


def _batch(B=2, L=6):
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    return {
        "coords_res": torch.randn(B, L, 4, 3),
        "mask_res": torch.ones(B, L, dtype=torch.bool),
        "chain_ids": chain,
    }


def test_none_returns_nones():
    assert make_template_inputs(_batch(), source="none") == (None, None, None)


def test_oracle_single_frame_all_pairs():
    b = _batch()
    coords, mask, frame = make_template_inputs(b, source="oracle")
    assert torch.equal(coords, b["coords_res"])
    assert torch.equal(frame, torch.zeros_like(b["chain_ids"]))   # one frame


def test_oracle_monomer_per_chain_frames():
    b = _batch()
    _, _, frame = make_template_inputs(b, source="oracle_monomer")
    assert torch.equal(frame, b["chain_ids"])                     # per-chain frames


def test_retrieved_uses_batch_template_tensors():
    b = _batch()
    b["template_coords_res"] = torch.randn_like(b["coords_res"])
    b["template_mask"] = torch.ones_like(b["mask_res"])
    coords, mask, frame = make_template_inputs(b, source="retrieved")
    assert torch.equal(coords, b["template_coords_res"])
    assert torch.equal(frame, b["chain_ids"])                     # per-chain frames


def test_retrieved_missing_cache_is_template_free():
    # No template tensors in the batch -> treated as template-free (Nones).
    assert make_template_inputs(_batch(), source="retrieved") == (None, None, None)


def test_dropout_reduces_coverage():
    b = _batch(B=1, L=200)
    g = torch.Generator().manual_seed(0)
    _, mask, _ = make_template_inputs(b, source="oracle", dropout=0.5, generator=g)
    frac = mask.float().mean().item()
    assert 0.3 < frac < 0.7   # ~half dropped


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        make_template_inputs(_batch(), source="bogus")
