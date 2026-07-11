"""Unit tests for the Phase H pair-representation track.

Covers the standalone PairTrack module and its integration into
ResFoldOneStep (the active model). Keeps to the project's integration-leaning
philosophy: shapes, masking, no-op-at-init, and gradient flow on the real
model path rather than mocked internals.
"""

import torch

from tinyfold.model.resfold.pair_track import PairTrack
from tinyfold.model.resfold.onestep import ResFoldOneStep


def _toy_inputs(B=2, L=7, device="cpu"):
    aa = torch.zeros(B, L, dtype=torch.long, device=device)
    chain = torch.zeros(B, L, dtype=torch.long, device=device)
    chain[:, L // 2:] = 1  # two chains
    res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    return aa, chain, res_idx, mask


class TestPairTrack:
    def test_bias_shape(self):
        n_heads = 8
        pt = PairTrack(c_token=32, n_heads=n_heads, c_pair=16, n_layers=2, c_hidden=16)
        B, L = 2, 7
        s = torch.randn(B, L, 32)
        _, chain, res_idx, mask = _toy_inputs(B, L)
        bias, _ = pt(s, res_idx, chain, mask)
        assert bias.shape == (B, n_heads, L, L)

    def test_zero_at_init(self):
        """to_bias is zero-init -> the track is a no-op additive bias at step 0."""
        pt = PairTrack(c_token=32, n_heads=4, c_pair=16, n_layers=2, c_hidden=16)
        B, L = 2, 7
        s = torch.randn(B, L, 32)
        _, chain, res_idx, mask = _toy_inputs(B, L)
        bias, _ = pt(s, res_idx, chain, mask)
        assert torch.allclose(bias, torch.zeros_like(bias))

    def test_padding_does_not_leak(self):
        """Padded rows/cols of the pair tensor must stay masked through tri-mul."""
        pt = PairTrack(c_token=16, n_heads=2, c_pair=16, n_layers=2, c_hidden=16)
        # Force a non-trivial bias by perturbing the output head.
        torch.nn.init.normal_(pt.to_bias.weight, std=0.5)
        B, L = 1, 6
        s = torch.randn(B, L, 16)
        chain = torch.zeros(B, L, dtype=torch.long)
        res_idx = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[0, -2:] = False  # last two residues are padding

        bias_full, _ = pt(s, res_idx, chain, mask)
        # Changing the *padded* single-rep entries must not change the bias on
        # the valid block — i.e. padding does not leak into the pair update.
        s2 = s.clone()
        s2[0, -2:] = torch.randn(2, 16)
        bias_pert, _ = pt(s2, res_idx, chain, mask)
        valid = mask[0]
        vb_full = bias_full[0, :, valid][:, :, valid]
        vb_pert = bias_pert[0, :, valid][:, :, valid]
        assert torch.allclose(vb_full, vb_pert, atol=1e-5)


class TestOneStepIntegration:
    def test_builds_and_forward(self):
        model = ResFoldOneStep(
            c_token=64, trunk_layers=2, denoiser_blocks=2,
            atom_head_heads=4, aa_embed="learned",
            relpos_bias=True, pair_repr=True, c_pair=32, pair_layers=2, pair_hidden=32,
        )
        aa, chain, res_idx, mask = _toy_inputs(B=2, L=8)
        x = torch.randn(2, 8, 3)
        sigma = torch.full((2,), 0.5)
        c, a, lddt = model.forward_sigma(x, aa, chain, res_idx, sigma, mask)
        assert c.shape == (2, 8, 3)
        assert a.shape == (2, 8, 4, 3)

    def test_param_overhead_is_small(self):
        # Use the production trunk width (c_token=256, 6 layers) where the
        # ~12M-param backbone makes the pair track's absolute cost a small
        # fraction. (At toy widths the same track is naturally a large %.)
        common = dict(c_token=256, trunk_layers=6, denoiser_blocks=6, relpos_bias=True)
        base = ResFoldOneStep(pair_repr=False, **common)
        pair = ResFoldOneStep(pair_repr=True, c_pair=64, pair_layers=3, pair_hidden=64, **common)
        nb = base.count_parameters()["total"]
        npp = pair.count_parameters()["total"]
        assert npp > nb  # pair track adds params
        assert (npp - nb) / nb < 0.05  # but a small fraction at production scale

    def test_gradients_flow_to_pair_track(self):
        model = ResFoldOneStep(
            c_token=64, trunk_layers=2, denoiser_blocks=2,
            relpos_bias=True, pair_repr=True, c_pair=32, pair_layers=2, pair_hidden=32,
        )
        aa, chain, res_idx, mask = _toy_inputs(B=2, L=8)
        x = torch.randn(2, 8, 3)
        sigma = torch.full((2,), 0.5)
        c, a, _ = model.forward_sigma(x, aa, chain, res_idx, sigma, mask)
        loss = c.pow(2).mean() + a.pow(2).mean()
        loss.backward()
        # The triangle-mul gate params should receive gradient even though the
        # output head is zero-init (gradient reaches them through the residual).
        pt = model.trunk.pair_track
        grads = [p.grad for p in pt.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum() > 0 for g in grads)
