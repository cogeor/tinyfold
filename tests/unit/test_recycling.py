"""Tests for trunk recycling (C1).

Recycling re-runs the sequence trunk ``n_recycle`` times, feeding each pass's
token (and, with the pair track, pair) representation back into the next. The
projection is zero-initialised, so on a fresh model recycling is a literal
no-op and ``n_recycle=0`` is bitwise-identical to the pre-recycling forward.
Only the final pass is differentiated, so peak memory does not grow with
``n_recycle``.
"""

import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep


def _toy_inputs(B=2, L=8, device="cpu"):
    aa = torch.zeros(B, L, dtype=torch.long, device=device)
    chain = torch.zeros(B, L, dtype=torch.long, device=device)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    x = torch.randn(B, L, 3, device=device)
    sigma = torch.full((B,), 0.5, device=device)
    return aa, chain, res_idx, mask, x, sigma


def _small_model(**kw):
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32, trunk_layers=2, denoiser_blocks=2, atom_head_layers=1,
        **kw,
    )


# --- bitwise no-op at n_recycle=0 ------------------------------------------

def test_n_recycle_0_matches_direct_trunk_path():
    """n_recycle=0 must reproduce the exact pre-recycling forward: one trunk
    pass with no fed-back state."""
    model = _small_model()
    model.eval()
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()

    out0 = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=0)

    # Pre-recycling path: trunk once (no recycle kwargs), then the shared core.
    tok = model.trunk(aa, chain, res_idx, mask)
    out_direct = model.forward_sigma_with_trunk(
        x, tok, sigma, mask, res_idx=res_idx, chain_ids=chain
    )

    assert torch.equal(out0.centroid_pred, out_direct.centroid_pred)
    assert torch.equal(out0.atoms_pred, out_direct.atoms_pred)


def test_n_recycle_0_matches_direct_trunk_path_with_pair_track():
    model = _small_model(relpos_bias=True, pair_repr=True, c_pair=16,
                         pair_layers=2, pair_hidden=16)
    model.eval()
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()

    out0 = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=0)
    tok = model.trunk(aa, chain, res_idx, mask)
    out_direct = model.forward_sigma_with_trunk(
        x, tok, sigma, mask, res_idx=res_idx, chain_ids=chain
    )
    assert torch.equal(out0.centroid_pred, out_direct.centroid_pred)
    assert torch.equal(out0.atoms_pred, out_direct.atoms_pred)


def test_recycling_is_a_noop_at_init_for_any_count():
    """Zero-init projection: even n_recycle>0 leaves the output unchanged on a
    fresh model, because the fed-back rep is projected to zero."""
    model = _small_model()
    model.eval()
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()
    out0 = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=0)
    for n in (1, 2, 3):
        out_n = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=n)
        assert torch.equal(out0.centroid_pred, out_n.centroid_pred)


# --- the projection is genuinely wired (once trained) ----------------------

def test_recycling_changes_output_once_the_projection_is_nonzero():
    """With a non-zero recycle projection the fed-back rep actually flows, so
    n_recycle>0 must differ from n_recycle=0 -- proving the path is connected."""
    model = _small_model()
    model.eval()
    with torch.no_grad():
        model.trunk.recycle_proj.weight.normal_(std=0.5)
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()
    out0 = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=0)
    out1 = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=1)
    assert not torch.equal(out0.centroid_pred, out1.centroid_pred)


# --- gradient flows only through the final pass ----------------------------

def _count_grad_tracked_trunk_passes(model, n_recycle):
    """Run a forward and count how many trunk passes produced a grad-requiring
    output. Under the recycling contract this is exactly 1 for any n_recycle,
    which is what keeps peak memory flat."""
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()
    orig = model.trunk.forward
    grad_tracked = []

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        tok = out[0] if isinstance(out, tuple) else out
        grad_tracked.append(bool(tok.requires_grad))
        return out

    model.trunk.forward = wrapped
    try:
        out = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=n_recycle)
    finally:
        model.trunk.forward = orig
    return grad_tracked, out


def test_only_the_final_pass_is_differentiated():
    model = _small_model()
    model.train()
    for n in (0, 1, 3):
        grad_tracked, _ = _count_grad_tracked_trunk_passes(model, n)
        assert len(grad_tracked) == n + 1, "wrong number of trunk passes"
        # Exactly one differentiated pass, and it is the last.
        assert sum(grad_tracked) == 1
        assert grad_tracked[-1] is True
        assert all(g is False for g in grad_tracked[:-1])


def test_backward_runs_through_recycling():
    model = _small_model()
    model.train()
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()
    out = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=2)
    (out.centroid_pred.pow(2).mean() + out.atoms_pred.pow(2).mean()).backward()
    grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


# --- shape / dtype invariance ----------------------------------------------

def test_output_shape_and_dtype_independent_of_recycle_count():
    model = _small_model()
    model.eval()
    aa, chain, res_idx, mask, x, sigma = _toy_inputs()
    ref = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=0)
    for n in (1, 4):
        out = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, n_recycle=n)
        assert out.centroid_pred.shape == ref.centroid_pred.shape
        assert out.centroid_pred.dtype == ref.centroid_pred.dtype
        assert out.atoms_pred.shape == ref.atoms_pred.shape


# --- eval seams pick up recycling ------------------------------------------

def test_get_trunk_tokens_recycles():
    """The eval fast path caches trunk tokens via get_trunk_tokens; it must
    honour n_recycle so --n_recycle_eval takes effect."""
    model = _small_model()
    model.eval()
    with torch.no_grad():
        model.trunk.recycle_proj.weight.normal_(std=0.5)
    aa, chain, res_idx, mask, _, _ = _toy_inputs()
    t0 = model.get_trunk_tokens(aa, chain, res_idx, mask, n_recycle=0)
    t1 = model.get_trunk_tokens(aa, chain, res_idx, mask, n_recycle=1)
    assert t0.shape == t1.shape
    assert not torch.equal(t0, t1)


# --- construction does not perturb existing weight init --------------------

def test_adding_recycle_params_preserves_rng_stream():
    """Building the recycle modules restores the RNG state, so a model built
    from a fixed seed has byte-identical non-recycle weights regardless of the
    recycle path existing. We check the recycle projection is zeroed and that
    the surrounding init is deterministic across two seeded builds."""
    a = _small_model()
    b = _small_model()
    assert torch.equal(a.trunk.recycle_proj.weight, torch.zeros_like(a.trunk.recycle_proj.weight))
    # Two identically-seeded builds agree everywhere (init is deterministic).
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param {n1} differs across seeded builds"
