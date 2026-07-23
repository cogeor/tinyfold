"""Coevolution conditioning through the pair track (spec §5).

The design claim is that coevolution is "one more pair-track prior" -- a MIRROR
of the template path, additive on the same bus, zero-init, independently
ablatable. These tests hold that claim to account on the real module.
"""

import torch

from tinyfold.model.resfold.pair_track import PairTrack
from tinyfold.msa.features import MSA_FEAT_DIM


def _inputs(B=2, L=6):
    s = torch.randn(B, L, 32)
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    return s, res_idx, chain, mask


def _track(**kw):
    return PairTrack(c_token=32, n_heads=4, c_pair=16, n_layers=2, c_hidden=16, **kw)


def _msa_feats(B=2, L=6):
    return torch.randn(B, L, L, MSA_FEAT_DIM)


class TestDisabledIsAPureNoOp:
    def test_no_msa_params_when_disabled(self):
        assert _track(msa_cond=False).msa_proj is None

    def test_disabled_track_ignores_supplied_feats(self):
        pt = _track(msa_cond=False)
        s, res_idx, chain, mask = _inputs()
        a, _, _ = pt(s, res_idx, chain, mask)
        b, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats())
        torch.testing.assert_close(a, b)


class TestZeroInitNoOp:
    """A coevolution arm must start byte-identical to its no-coevolution arm."""

    def test_proj_is_zero_init(self):
        pt = _track(msa_cond=True)
        assert torch.count_nonzero(pt.msa_proj.weight) == 0
        assert torch.count_nonzero(pt.msa_proj.bias) == 0

    def test_feats_do_not_change_output_at_init(self):
        pt = _track(msa_cond=True)
        s, res_idx, chain, mask = _inputs()
        a, _, _ = pt(s, res_idx, chain, mask)
        b, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats())
        torch.testing.assert_close(a, b)

    def test_enabled_arm_matches_disabled_arm_at_init(self):
        torch.manual_seed(0)
        off = _track(msa_cond=False)
        torch.manual_seed(0)
        on = _track(msa_cond=True)
        s, res_idx, chain, mask = _inputs()
        a, _, _ = off(s, res_idx, chain, mask)
        b, _, _ = on(s, res_idx, chain, mask, msa_feats=_msa_feats())
        torch.testing.assert_close(a, b)


class TestFeatsActuallyReachTheOutput:
    """The E2a lesson: prove the channel is wired BEFORE trusting a null result.

    A zero-init projection that never learns is indistinguishable from a
    disconnected one, so perturb the weights and require the output to move.
    """

    def test_trained_proj_makes_feats_matter(self):
        pt = _track(msa_cond=True)
        torch.nn.init.normal_(pt.msa_proj.weight, std=0.5)
        torch.nn.init.normal_(pt.to_bias.weight, std=0.5)
        s, res_idx, chain, mask = _inputs()
        a, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats())
        b, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats() * 3.0)
        assert not torch.allclose(a, b)

    def test_gradients_reach_msa_proj(self):
        pt = _track(msa_cond=True)
        torch.nn.init.normal_(pt.to_bias.weight, std=0.1)
        s, res_idx, chain, mask = _inputs()
        bias, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats())
        bias.sum().backward()
        assert pt.msa_proj.weight.grad is not None
        assert torch.count_nonzero(pt.msa_proj.weight.grad) > 0


class TestComposesWithTemplates:
    """Spec §3: keep BOTH priors; they cover disjoint failure modes."""

    def test_both_priors_can_be_enabled_together(self):
        pt = _track(template_cond=True, msa_cond=True)
        assert pt.template_proj is not None
        assert pt.msa_proj is not None

    def test_each_prior_is_independently_ablatable(self):
        pt = _track(template_cond=True, msa_cond=True)
        s, res_idx, chain, mask = _inputs()
        B, L = 2, 6
        tmpl = torch.randn(B, L, 4, 3)
        # Every combination must run: neither, either, both.
        for kw in [
            {},
            {"template_coords_res": tmpl},
            {"msa_feats": _msa_feats()},
            {"template_coords_res": tmpl, "msa_feats": _msa_feats()},
        ]:
            bias, _, _ = pt(s, res_idx, chain, mask, **kw)
            assert bias.shape == (B, 4, L, L)

    def test_param_overhead_is_one_small_projection(self):
        base = sum(p.numel() for p in _track().parameters())
        with_msa = sum(p.numel() for p in _track(msa_cond=True).parameters())
        # c_pair=16, F_msa=4 -> 16*4 weights + 16 bias.
        assert with_msa - base == MSA_FEAT_DIM * 16 + 16


class TestDtypeAndShape:
    def test_accepts_fp16_cache_dtype(self):
        # The cache is stored fp16; the model runs fp32.
        pt = _track(msa_cond=True)
        torch.nn.init.normal_(pt.msa_proj.weight, std=0.1)
        s, res_idx, chain, mask = _inputs()
        bias, _, _ = pt(s, res_idx, chain, mask, msa_feats=_msa_feats().half())
        assert bias.dtype == torch.float32
        assert torch.isfinite(bias).all()
