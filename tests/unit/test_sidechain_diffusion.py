"""Stage-3 sidechain diffusion head + local-frame transforms.

The load-bearing claim of spec §4 is that expressing sidechains in the fixed
backbone frame buys rotation/translation invariance FOR FREE -- which is what
makes rotation augmentation "just work" here, unlike the backbone offset head
that needed explicit aug_R bookkeeping. That claim is tested directly.
"""

import torch

from tinyfold.constants import AA_TO_IDX
from tinyfold.model.resfold.sidechain_diffusion import (
    NUM_SIDECHAIN_SLOTS,
    SidechainDiffusionHead,
    assemble_atom14,
    sidechain_to_global,
    sidechain_to_local,
)


def _atom14(B=2, L=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Realistic-ish backbone spread so the Gram-Schmidt frame is well-conditioned.
    return torch.randn(B, L, 14, 3, generator=g)


def _rand_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestLocalFrameTransforms:
    def test_roundtrip_is_identity(self):
        x = _atom14()
        local = sidechain_to_local(x)
        back = sidechain_to_global(local, x[..., :4, :])
        torch.testing.assert_close(back, x[..., 4:, :], atol=1e-5, rtol=1e-5)

    def test_shape(self):
        assert sidechain_to_local(_atom14()).shape == (2, 5, NUM_SIDECHAIN_SLOTS, 3)

    def test_invariant_to_global_rotation(self):
        # THE property: rotate the whole structure, local coords do not move.
        x = _atom14()
        R = _rand_rotation()
        x_rot = x @ R.T
        torch.testing.assert_close(
            sidechain_to_local(x_rot), sidechain_to_local(x), atol=1e-4, rtol=1e-4
        )

    def test_invariant_to_global_translation(self):
        x = _atom14()
        shift = torch.tensor([3.0, -2.0, 7.0])
        torch.testing.assert_close(
            sidechain_to_local(x + shift), sidechain_to_local(x), atol=1e-5, rtol=1e-5
        )

    def test_equivariant_reconstruction_under_rotation(self):
        # Same local coords + rotated backbone -> rotated global sidechain.
        x = _atom14()
        R = _rand_rotation(1)
        local = sidechain_to_local(x)
        g1 = sidechain_to_global(local, x[..., :4, :]) @ R.T
        g2 = sidechain_to_global(local, x[..., :4, :] @ R.T)
        torch.testing.assert_close(g1, g2, atol=1e-4, rtol=1e-4)

    def test_ca_is_the_origin(self):
        # An atom sitting exactly on CA maps to the local origin.
        x = _atom14()
        x[..., 4, :] = x[..., 1, :]
        local = sidechain_to_local(x)
        torch.testing.assert_close(local[..., 0, :], torch.zeros_like(local[..., 0, :]),
                                   atol=1e-5, rtol=1e-5)


class TestAssemble:
    def test_splices_to_atom14(self):
        x = _atom14()
        out = assemble_atom14(x[..., :4, :], x[..., 4:, :])
        assert out.shape == (2, 5, 14, 3)
        torch.testing.assert_close(out, x)

    def test_backbone_slots_are_preserved_exactly(self):
        # Stage 3 must never perturb the frozen backbone (spec §2).
        x = _atom14()
        sc = torch.randn(2, 5, NUM_SIDECHAIN_SLOTS, 3)
        out = assemble_atom14(x[..., :4, :], sc)
        torch.testing.assert_close(out[..., :4, :], x[..., :4, :])


def _head(**kw):
    return SidechainDiffusionHead(c_token=32, n_layers=1, n_heads=2, **kw)


def _batch(B=2, L=5):
    x_t = torch.randn(B, L, NUM_SIDECHAIN_SLOTS, 3)
    aatype = torch.full((B, L), AA_TO_IDX["D"])
    sigma = torch.full((B,), 0.3)
    mask = torch.ones(B, L, dtype=torch.bool)
    return x_t, aatype, sigma, mask


class TestHeadShapes:
    def test_output_shape(self):
        h = _head()
        x_t, aatype, sigma, mask = _batch()
        assert h(x_t, aatype, sigma, mask=mask).shape == x_t.shape

    def test_runs_without_tokens(self):
        h = _head()
        x_t, aatype, sigma, mask = _batch()
        assert torch.isfinite(h(x_t, aatype, sigma, mask=mask)).all()

    def test_accepts_trunk_tokens(self):
        h = _head()
        x_t, aatype, sigma, mask = _batch()
        tokens = torch.randn(2, 5, 32)
        assert torch.isfinite(h(x_t, aatype, sigma, tokens=tokens, mask=mask)).all()


class TestEdmPreconditioning:
    def test_coefficients_shapes(self):
        h = _head()
        c_skip, c_out, c_in, c_noise = h.edm_coefficients(torch.tensor([0.1, 1.0]))
        assert c_skip.shape == (2, 1, 1, 1)
        assert c_noise.shape == (2,)

    def test_low_sigma_keeps_the_input(self):
        # As sigma -> 0 the denoiser must approach the identity (c_skip -> 1).
        h = _head()
        c_skip, c_out, _, _ = h.edm_coefficients(torch.tensor([1e-6]))
        assert c_skip.item() > 0.999
        assert c_out.item() < 1e-4

    def test_high_sigma_discards_the_input(self):
        h = _head()
        c_skip, _, _, _ = h.edm_coefficients(torch.tensor([100.0]))
        assert c_skip.item() < 1e-3

    def test_near_zero_sigma_output_matches_input(self):
        h = _head()
        x_t, aatype, _, mask = _batch()
        out = h(x_t, aatype, torch.full((2,), 1e-6), mask=mask)
        torch.testing.assert_close(out, x_t, atol=1e-3, rtol=1e-3)


class TestConditioning:
    def test_residue_identity_changes_the_prediction(self):
        # Without identity conditioning the head cannot know GLY from TRP.
        h = _head()
        torch.nn.init.normal_(h.proj.weight, std=0.5)
        x_t, _, sigma, mask = _batch()
        a = h(x_t, torch.full((2, 5), AA_TO_IDX["G"]), sigma, mask=mask)
        b = h(x_t, torch.full((2, 5), AA_TO_IDX["W"]), sigma, mask=mask)
        assert not torch.allclose(a, b)

    def test_sigma_changes_the_prediction(self):
        h = _head()
        torch.nn.init.normal_(h.proj.weight, std=0.5)
        x_t, aatype, _, mask = _batch()
        a = h(x_t, aatype, torch.full((2,), 0.05), mask=mask)
        b = h(x_t, aatype, torch.full((2,), 2.0), mask=mask)
        assert not torch.allclose(a, b)

    def test_same_type_residues_are_distinguishable_by_position(self):
        # Without positional encoding a residue transformer is permutation-
        # invariant, so two same-type residues get identical predictions -- the
        # bug that floored the S5 overfit at ~2.5 A. Position must break the tie.
        h = _head()
        torch.nn.init.normal_(h.proj.weight, std=0.5)
        # Identical noised state + identical restype at every position.
        x_t = torch.zeros(1, 5, NUM_SIDECHAIN_SLOTS, 3)
        aatype = torch.full((1, 5), AA_TO_IDX["L"])
        out = h(x_t, aatype, torch.full((1,), 0.5), mask=torch.ones(1, 5, dtype=torch.bool))
        # Predictions for residue 0 and residue 4 must differ purely from position.
        assert not torch.allclose(out[:, 0], out[:, 4])

    def test_residues_are_coupled_not_independent(self):
        # §5 rejects independent per-residue packing: perturbing residue 0 must
        # be able to move residue 4, or clashes can never resolve.
        h = _head()
        torch.nn.init.normal_(h.proj.weight, std=0.5)
        x_t, aatype, sigma, mask = _batch()
        a = h(x_t, aatype, sigma, mask=mask)
        x2 = x_t.clone()
        x2[:, 0] += 5.0
        b = h(x2, aatype, sigma, mask=mask)
        assert not torch.allclose(a[:, 4], b[:, 4])


class TestMaskingAndGradients:
    def test_padded_residues_are_zeroed(self):
        h = _head()
        x_t, aatype, sigma, _ = _batch()
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 3:] = False
        out = h(x_t, aatype, sigma, mask=mask)
        assert torch.count_nonzero(out[:, 3:]) == 0

    def test_gradients_flow(self):
        h = _head()
        x_t, aatype, sigma, mask = _batch()
        h(x_t, aatype, sigma, mask=mask).sum().backward()
        grads = [p.grad for p in h.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_finite_across_the_sigma_range(self):
        h = _head()
        x_t, aatype, _, mask = _batch()
        for s in [1e-4, 0.01, 0.35, 1.0, 10.0]:
            out = h(x_t, aatype, torch.full((2,), s), mask=mask)
            assert torch.isfinite(out).all(), s
