"""Tests for the Loop 05 ESM-2 cache + projection pathway (Task F).

The cache prep is run out-of-band (``scripts/prepare_esm2_embeddings.py``);
these tests are skip-on-missing so CI can run them lazily without first
downloading ~150 MB of ESM-2 weights.

Coverage:

* ``test_esm2_cache_shape``: every cached NPZ has shape ``(LA+LB, 480)`` and
  dtype ``float16``, with ``LA``/``LB`` matching the parquet row. Spot-checks
  the first 20 cached samples; fails if there are zero (mis-pointed dir).
* ``test_esm2_residue_encoder_construction``: ``ResidueEncoder`` in ESM mode
  has ``aa_embed=None`` and a trainable ``esm_proj`` of the right shape.
* ``test_esm2_residue_encoder_forward``: forward pass over a cached sample
  produces a finite trunk token tensor of shape ``[1, L, c_token]``.
* ``test_esm2_param_delta``: the only added trainable params relative to the
  learned-mode baseline are ``trunk.esm_proj.weight`` and ``.bias`` (i.e.
  exactly ``esm_dim * c_token + c_token`` extras), and NO ``EsmModel``
  parameters are part of the state dict.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from tinyfold.model.resfold.denoiser import ESM_DIMS, ResidueEncoder
from tinyfold.model.resfold.onestep import ResFoldOneStep


CACHE_DIR = Path("data/processed/esm2_35M")
PARQUET_PATH = Path("data/processed/samples.parquet")


def _skip_if_no_cache() -> None:
    if not CACHE_DIR.exists():
        pytest.skip(
            f"ESM-2 cache not built at {CACHE_DIR}; run "
            "scripts/prepare_esm2_embeddings.py first"
        )
    if not PARQUET_PATH.exists():
        pytest.skip(f"Parquet not found at {PARQUET_PATH}")


# ---------------------------------------------------------------------------
# Cache shape contract
# ---------------------------------------------------------------------------


def test_esm2_cache_shape():
    _skip_if_no_cache()

    table = pq.read_table(
        str(PARQUET_PATH), columns=["sample_id", "LA", "LB"]
    )
    df = table.to_pandas()

    n_checked = 0
    for _, row in df.iterrows():
        p = CACHE_DIR / f"{row['sample_id']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        assert d["embeddings"].shape == (int(row["LA"]) + int(row["LB"]), 480), (
            f"{row['sample_id']}: embedding shape {d['embeddings'].shape} != "
            f"({row['LA']}+{row['LB']}, 480)"
        )
        assert d["embeddings"].dtype == np.float16, (
            f"{row['sample_id']}: dtype {d['embeddings'].dtype} != float16"
        )
        assert int(d["LA"]) == int(row["LA"])
        assert int(d["LB"]) == int(row["LB"])
        n_checked += 1
        if n_checked >= 20:
            break

    assert n_checked > 0, (
        f"no cached samples found in {CACHE_DIR}; check that the prep run "
        "actually wrote files (not just tmp files)"
    )


# ---------------------------------------------------------------------------
# Model construction & forward
# ---------------------------------------------------------------------------


def test_esm2_residue_encoder_construction_learned():
    """Default 'learned' mode keeps the historical nn.Embedding lookup."""
    enc = ResidueEncoder(c_token=64, n_layers=2, aa_embed="learned")
    assert enc.aa_embed_mode == "learned"
    assert enc.aa_embed is not None
    assert isinstance(enc.aa_embed, torch.nn.Embedding)
    assert enc.esm_proj is None


def test_esm2_residue_encoder_construction_esm():
    """ESM mode replaces the AA lookup with a frozen-input projection."""
    enc = ResidueEncoder(c_token=64, n_layers=2, aa_embed="esm2_35M")
    assert enc.aa_embed_mode == "esm2_35M"
    assert enc.aa_embed is None
    assert isinstance(enc.esm_proj, torch.nn.Linear)
    assert enc.esm_proj.in_features == 480
    assert enc.esm_proj.out_features == 64
    assert enc.esm_proj.weight.requires_grad
    # Bias init should be zero (xavier on weight, zeros on bias).
    assert torch.allclose(enc.esm_proj.bias, torch.zeros_like(enc.esm_proj.bias))


def test_esm2_residue_encoder_forward_runs():
    """ESM-mode forward returns finite trunk tokens of the right shape."""
    torch.manual_seed(0)
    enc = ResidueEncoder(c_token=64, n_layers=2, aa_embed="esm2_35M")
    B, L = 2, 10
    aa = torch.randint(0, 21, (B, L))
    chain = torch.randint(0, 2, (B, L))
    res_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, L, dtype=torch.bool)
    esm = torch.randn(B, L, 480)
    tokens = enc(aa, chain, res_idx, mask, esm_embed=esm)
    assert tokens.shape == (B, L, 64)
    assert torch.isfinite(tokens).all()


def test_esm2_residue_encoder_forward_requires_esm_embed():
    """ESM mode without esm_embed must fail fast (assert)."""
    enc = ResidueEncoder(c_token=64, n_layers=2, aa_embed="esm2_35M")
    B, L = 1, 5
    aa = torch.zeros(B, L, dtype=torch.long)
    chain = torch.zeros(B, L, dtype=torch.long)
    res_idx = torch.arange(L).unsqueeze(0)
    mask = torch.ones(B, L, dtype=torch.bool)
    with pytest.raises(AssertionError):
        _ = enc(aa, chain, res_idx, mask, esm_embed=None)


# ---------------------------------------------------------------------------
# Param-delta sanity check
# ---------------------------------------------------------------------------


def test_esm2_param_delta_vs_learned():
    """Switching learned -> esm2_35M removes the aa_embed lookup
    (n_aa_types * c_token params) and adds an esm_proj Linear
    (esm_dim * c_token + c_token params). NO EsmModel weights end up in
    the state dict — ESM-2 itself is not part of the model."""
    c_token = 64
    common = dict(
        c_token=c_token, trunk_layers=2, denoiser_blocks=2,
        atom_head_layers=2, atom_head_heads=4,
    )
    m_learned = ResFoldOneStep(**common, aa_embed="learned")
    m_esm = ResFoldOneStep(**common, aa_embed="esm2_35M")

    # Param counts
    p_learned = sum(p.numel() for p in m_learned.parameters())
    p_esm = sum(p.numel() for p in m_esm.parameters())

    n_aa_types = 21
    esm_dim = ESM_DIMS["esm2_35M"]  # 480
    expected_delta = (esm_dim * c_token + c_token) - (n_aa_types * c_token)
    assert p_esm - p_learned == expected_delta, (
        f"param delta {p_esm - p_learned} != expected {expected_delta}"
    )

    # No EsmModel param: every state-dict key must come from our own modules.
    keys = list(m_esm.state_dict().keys())
    # Quick sanity: esm_proj weight + bias appear, aa_embed lookup does not.
    assert any(k.endswith("esm_proj.weight") for k in keys)
    assert any(k.endswith("esm_proj.bias") for k in keys)
    assert not any("aa_embed.weight" in k for k in keys), (
        "aa_embed.weight should not be in ESM-mode state dict"
    )
    # EsmModel itself never goes into the model.
    assert not any("EsmModel" in k or "esm.encoder" in k for k in keys), (
        f"unexpected ESM-model param leaked into state dict: "
        f"{[k for k in keys if 'esm' in k.lower()]}"
    )
