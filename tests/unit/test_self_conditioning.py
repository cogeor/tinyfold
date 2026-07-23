"""Tests for AF3 detached mini-rollout self-conditioning (C4).

self_cond_rollout runs a short reverse-diffusion rollout from the current noised
state at train time and returns the DETACHED clean-coord estimate to feed back
as x0_prev. It lives in inference/samplers.py so train and inference share one
code path. The trained step must receive no gradient from the rollout.
"""

import torch

from tinyfold.inference import samplers as samplers_mod
from tinyfold.inference import self_cond_rollout
from tinyfold.model.resfold.onestep import ResFoldOneStep


def _model():
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32, trunk_layers=2, denoiser_blocks=2, atom_head_layers=1,
    )


def _batch(B=2, L=8):
    aa = torch.zeros(B, L, dtype=torch.long)
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    return {"aa_seq": aa, "chain_ids": chain, "res_idx": res_idx, "mask_res": mask}


def test_rollout_result_is_detached():
    """Even inside an autograd-enabled context with grad-requiring params, the
    fed-back x0_prev must carry no grad_fn -- the rollout contributes no gradient
    to the trained step."""
    model = _model()
    batch = _batch()
    x_t = torch.randn(2, 8, 3)
    sigma = torch.full((2,), 5.0)
    with torch.enable_grad():
        x0 = self_cond_rollout(model, batch, x_t, sigma, n_steps=2, is_onestep=True)
    assert not x0.requires_grad
    assert x0.grad_fn is None


def test_rollout_shape_matches_x_t():
    model = _model()
    batch = _batch()
    x_t = torch.randn(2, 8, 3)
    sigma = torch.full((2,), 3.0)
    x0 = self_cond_rollout(model, batch, x_t, sigma, n_steps=3, is_onestep=True)
    assert x0.shape == x_t.shape


def test_n_steps_controls_number_of_denoiser_calls():
    model = _model()
    batch = _batch()
    x_t = torch.randn(2, 8, 3)
    sigma = torch.full((2,), 4.0)
    for n in (1, 3, 5):
        calls = {"n": 0}
        orig = model.forward_sigma

        def counting(*a, _orig=orig, _calls=calls, **kw):
            _calls["n"] += 1
            return _orig(*a, **kw)

        model.forward_sigma = counting
        try:
            self_cond_rollout(model, batch, x_t, sigma, n_steps=n, is_onestep=True)
        finally:
            model.forward_sigma = orig
        assert calls["n"] == n


def test_rollout_output_is_clamped():
    model = _model()
    batch = _batch()
    x_t = torch.randn(2, 8, 3) * 50.0
    sigma = torch.full((2,), 8.0)
    x0 = self_cond_rollout(model, batch, x_t, sigma, n_steps=2, is_onestep=True,
                           clamp_val=3.0)
    assert x0.abs().max() <= 3.0 + 1e-5


def test_rollout_is_the_shared_inference_function():
    """The train loop imports self_cond_rollout from the inference package, so
    it is exactly the sampler-module function (they cannot drift)."""
    assert self_cond_rollout is samplers_mod.self_cond_rollout
