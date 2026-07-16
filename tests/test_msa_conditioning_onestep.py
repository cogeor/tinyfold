"""F4: coevolution (msa_cond) threads through the full ResFoldOneStep.

Mirrors test_template_conditioning.py at the whole-model level: a msa_cond model
builds with a small overhead, is a strict no-op at zero-init (so a coev run starts
byte-equivalent to its control), moves once the projection is non-zero (whole path
connected: msa_feats -> proj -> pair -> attention bias -> tokens), and requires
pair_repr. The [B,L,L,F] cache dtype is fp16; the model runs fp32.
"""

import pytest
import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.msa.features import MSA_FEAT_DIM


def _tiny_model(msa_cond):
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32, trunk_layers=2, trunk_heads=4, denoiser_blocks=2,
        denoiser_heads=4, n_timesteps=10, relpos_bias=True, pair_repr=True,
        c_pair=16, pair_layers=2, pair_hidden=16, msa_cond=msa_cond,
    )


def _fake_batch(B=2, L=8):
    torch.manual_seed(1)
    aa = torch.randint(0, 20, (B, L))
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    msa_feats = torch.randn(B, L, L, MSA_FEAT_DIM)
    return aa, chain, res_idx, mask, msa_feats


def test_builds_and_param_overhead_small():
    base = sum(p.numel() for p in _tiny_model(False).parameters())
    coev = sum(p.numel() for p in _tiny_model(True).parameters())
    extra = coev - base
    # Only msa_proj (F_msa -> c_pair) is new.
    assert extra == MSA_FEAT_DIM * 16 + 16


def test_zero_init_msa_is_noop():
    m = _tiny_model(True).eval()
    aa, chain, res_idx, mask, msa_feats = _fake_batch()
    with torch.no_grad():
        no_msa = m.get_trunk_tokens(aa, chain, res_idx, mask)
        with_msa = m.get_trunk_tokens(aa, chain, res_idx, mask, msa_feats=msa_feats)
    assert torch.allclose(no_msa, with_msa, atol=1e-6)


def test_nonzero_projection_changes_output():
    m = _tiny_model(True).eval()
    with torch.no_grad():
        torch.manual_seed(2)
        m.trunk.pair_track.msa_proj.weight.normal_(0, 0.5)
        m.trunk.pair_track.to_bias.weight.normal_(0, 0.5)
    aa, chain, res_idx, mask, msa_feats = _fake_batch()
    with torch.no_grad():
        no_msa = m.get_trunk_tokens(aa, chain, res_idx, mask)
        with_msa = m.get_trunk_tokens(aa, chain, res_idx, mask, msa_feats=msa_feats)
    assert (no_msa - with_msa).abs().max().item() > 1e-4


def test_forward_sigma_runs_with_msa_fp16_cache():
    m = _tiny_model(True).eval()
    aa, chain, res_idx, mask, msa_feats = _fake_batch()
    B, L = aa.shape
    x_t = torch.randn(B, L, 3)
    sigma = torch.full((B,), 1.5)
    with torch.no_grad():
        cen, atoms, _ = m.forward_sigma(
            x_t, aa, chain, res_idx, sigma, mask=mask,
            msa_feats=msa_feats.half(),  # cache is fp16; model upcasts
        )
    assert cen.shape == (B, L, 3) and atoms.shape == (B, L, 4, 3)
    assert torch.isfinite(cen).all() and torch.isfinite(atoms).all()


def test_msa_cond_requires_pair_repr():
    with pytest.raises(ValueError):
        ResFoldOneStep(c_token=16, pair_repr=False, msa_cond=True)


def test_config_roundtrip_carries_msa_cond():
    from tinyfold.model.resfold.config import ResFoldConfig

    cfg = ResFoldConfig(pair_repr=True, msa_cond=True)
    assert cfg.to_kwargs()["msa_cond"] is True
    assert ResFoldConfig.from_config({"c_token_s1": 32, "trunk_layers": 2,
        "denoiser_blocks": 2, "atom_head_layers": 2, "atom_head_heads": 4,
        "pair_repr": True, "msa_cond": True}).msa_cond is True
