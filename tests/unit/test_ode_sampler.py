"""Tests for the few-step ODE sampler (C7).

Protenix-Mini: AF3-class models tolerate 2-step ODE sampling with no retraining,
but only with gamma0=0 (no noise re-injection) and step scale eta=1.0. The ODE
path routes through the same sample_k_centroids plumbing so K-sample eval picks
it up unchanged. With gamma0=0, eta=1.0 and the noiser's own step count it
reduces to the deterministic VE Euler trajectory.
"""

import torch

from tinyfold.inference import sample_centroids_ode, sample_k_centroids
from tinyfold.inference.samplers import sample_centroids_ve
from tinyfold.model.diffusion.noise import VENoiser
from tinyfold.model.diffusion.schedule import KarrasSchedule
from tinyfold.model.resfold.onestep import ResFoldOneStep


def _model():
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32, trunk_layers=2, denoiser_blocks=2, atom_head_layers=1,
    )


def _batch(B=1, L=8):
    aa = torch.zeros(B, L, dtype=torch.long)
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    return {"aa_seq": aa, "chain_ids": chain, "res_idx": res_idx, "mask_res": mask}


def _noiser(n_steps=20):
    return VENoiser(KarrasSchedule(n_steps=n_steps, sigma_min=0.002, sigma_max=10.0, rho=7.0))


# --- convergence to the VE trajectory --------------------------------------

def test_ode_matches_ve_with_gamma0_and_eta_defaults():
    """gamma0=0, eta=1.0, same step count -> identical to the deterministic VE
    Euler trajectory (self-conditioning off)."""
    model = _model()
    model.eval()
    batch = _batch()
    N = 24
    noiser = _noiser(N)

    gen1 = torch.Generator().manual_seed(123)
    ve, ve_atoms = sample_centroids_ve(
        model, batch, noiser, torch.device("cpu"),
        align_per_step=True, recenter=True, self_cond=False,
        is_onestep=True, generator=gen1,
    )
    gen2 = torch.Generator().manual_seed(123)
    ode, ode_atoms = sample_centroids_ode(
        model, batch, noiser, torch.device("cpu"),
        n_steps=N, gamma0=0.0, eta=1.0,
        align_per_step=True, recenter=True, is_onestep=True, generator=gen2,
    )
    assert torch.allclose(ve, ode, atol=1e-4)
    assert torch.allclose(ve_atoms, ode_atoms, atol=1e-4)


# --- step count honoured ----------------------------------------------------

def test_step_count_is_honoured():
    model = _model()
    model.eval()
    batch = _batch()
    noiser = _noiser(50)
    for n in (2, 5, 8):
        calls = {"n": 0}
        orig = model.forward_sigma_with_trunk

        def counting(*a, _orig=orig, _c=calls, **kw):
            _c["n"] += 1
            return _orig(*a, **kw)

        model.forward_sigma_with_trunk = counting
        try:
            sample_centroids_ode(model, batch, noiser, torch.device("cpu"),
                                 n_steps=n, is_onestep=True)
        finally:
            model.forward_sigma_with_trunk = orig
        # n Euler steps + 1 final forward at sigma_min for the atom read-out.
        assert calls["n"] == n + 1


def test_different_step_counts_give_different_results():
    model = _model()
    model.eval()
    batch = _batch()
    noiser = _noiser(50)
    gen_a = torch.Generator().manual_seed(7)
    a, _ = sample_centroids_ode(model, batch, noiser, torch.device("cpu"),
                                n_steps=2, is_onestep=True, generator=gen_a)
    gen_b = torch.Generator().manual_seed(7)
    b, _ = sample_centroids_ode(model, batch, noiser, torch.device("cpu"),
                                n_steps=8, is_onestep=True, generator=gen_b)
    assert not torch.allclose(a, b)


# --- K-sample plumbing ------------------------------------------------------

def test_k_sample_ode_returns_k_distinct_samples():
    model = _model()
    model.eval()
    batch = _batch()
    noiser = _noiser(50)
    K = 4
    cents, atoms, lddts = sample_k_centroids(
        model, batch, noiser, torch.device("cpu"),
        K=K, base_seed=1, target_idx=0, is_onestep=True, one_shot=False,
        sampler="ode", ode_steps=2,
    )
    assert cents.shape[0] == K
    assert atoms.shape[0] == K
    # Distinct draws: no two identical.
    for i in range(K):
        for j in range(i + 1, K):
            assert not torch.allclose(cents[i], cents[j])
