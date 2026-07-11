"""C7 tests: template conditioning via the pair track.

Verifies the SPEC done-when criteria for C7:
  * a template_cond model builds and reports its (small) extra param count;
  * the trunk forward runs both WITH and WITHOUT a template;
  * at zero-init the template path is a strict no-op (template arm == no-template
    arm), so a template run starts byte-equivalent to its control;
  * once the template projection is non-zero, feeding a template CHANGES the
    trunk output -- proving the whole signal path is connected
    (coords -> features -> proj -> pair -> attention bias -> tokens).
"""

import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep


def _tiny_model(template_cond):
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32,
        trunk_layers=2,
        trunk_heads=4,
        denoiser_blocks=2,
        denoiser_heads=4,
        n_timesteps=10,
        relpos_bias=True,
        pair_repr=True,
        c_pair=16,
        pair_layers=2,
        pair_hidden=16,
        template_cond=template_cond,
        template_rbf=16,
    )


def _fake_batch(B=2, L=9):
    torch.manual_seed(1)
    aa = torch.randint(0, 20, (B, L))
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    coords_res = torch.randn(B, L, 4, 3)
    return aa, chain, res_idx, mask, coords_res


def test_builds_and_param_overhead_small():
    base = _tiny_model(template_cond=False)
    tmpl = _tiny_model(template_cond=True)
    n_base = sum(p.numel() for p in base.parameters())
    n_tmpl = sum(p.numel() for p in tmpl.parameters())
    extra = n_tmpl - n_base
    assert extra > 0, "template_cond added no parameters"
    # Only the template_proj (feat_dim -> c_pair) is new; must be a small share.
    assert extra < 0.15 * n_base, f"template overhead too large: {extra}/{n_base}"


def test_zero_init_template_is_noop():
    m = _tiny_model(template_cond=True).eval()
    aa, chain, res_idx, mask, coords_res = _fake_batch()
    frame = torch.zeros_like(chain)
    with torch.no_grad():
        no_tmpl = m.get_trunk_tokens(aa, chain, res_idx, mask)
        with_tmpl = m.get_trunk_tokens(
            aa, chain, res_idx, mask,
            template_coords_res=coords_res, template_mask=mask, template_frame_id=frame,
        )
    assert torch.allclose(no_tmpl, with_tmpl, atol=1e-6), (
        "template path is not a no-op at zero-init"
    )


def test_nonzero_projection_changes_output():
    m = _tiny_model(template_cond=True).eval()
    # Break BOTH zero-inits so the template can influence the output: the
    # template projection (template -> pair channel) AND the pair track's
    # bias head (pair channel -> attention bias, zero-init by Phase H design).
    with torch.no_grad():
        torch.manual_seed(2)
        m.trunk.pair_track.template_proj.weight.normal_(0, 0.5)
        m.trunk.pair_track.to_bias.weight.normal_(0, 0.5)
    aa, chain, res_idx, mask, coords_res = _fake_batch()
    frame = torch.zeros_like(chain)
    with torch.no_grad():
        no_tmpl = m.get_trunk_tokens(aa, chain, res_idx, mask)
        with_tmpl = m.get_trunk_tokens(
            aa, chain, res_idx, mask,
            template_coords_res=coords_res, template_mask=mask, template_frame_id=frame,
        )
    diff = (no_tmpl - with_tmpl).abs().max().item()
    assert diff > 1e-4, f"template did not change trunk output (max diff {diff:.2e})"


def test_forward_sigma_runs_with_template():
    m = _tiny_model(template_cond=True).eval()
    aa, chain, res_idx, mask, coords_res = _fake_batch()
    frame = torch.zeros_like(chain)
    B, L = aa.shape
    x_t = torch.randn(B, L, 3)
    sigma = torch.full((B,), 1.5)
    with torch.no_grad():
        cen, atoms, _ = m.forward_sigma(
            x_t, aa, chain, res_idx, sigma, mask=mask,
            template_coords_res=coords_res, template_mask=mask, template_frame_id=frame,
        )
    assert cen.shape == (B, L, 3)
    assert atoms.shape == (B, L, 4, 3)
    assert torch.isfinite(cen).all() and torch.isfinite(atoms).all()


def test_template_cond_requires_pair_repr():
    import pytest
    with pytest.raises(ValueError):
        ResFoldOneStep(c_token=16, pair_repr=False, template_cond=True)
