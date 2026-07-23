"""Tests for the esm2_650M frozen-pLM option (C9).

Embeddings are frozen and cached, so a bigger pLM costs one preprocessing pass +
disk and zero training time. The plumbing is generic over esm_dim; C9 only adds
the enum entry (1280d) and the prep-script variant.
"""

import torch

from tinyfold.cli.prepare_esm2 import ESM_VARIANTS
from tinyfold.model.resfold.config import ResFoldConfig
from tinyfold.model.resfold.denoiser import ESM_DIMS
from tinyfold.model.resfold.onestep import ResFoldOneStep


def test_esm2_650M_is_a_known_dim():
    assert ESM_DIMS["esm2_650M"] == 1280


def test_prep_variant_maps_650M_to_1280():
    model_id, dim = ESM_VARIANTS["650M"]
    assert dim == 1280
    assert "650M" in model_id


def test_config_round_trips_to_model_construction():
    cfg = ResFoldConfig(c_token=32, trunk_layers=2, denoiser_blocks=2,
                        aa_embed="esm2_650M")
    model = ResFoldOneStep(**cfg.to_kwargs())
    # esm_dim resolves from the enum; the projection maps 1280 -> c_token.
    assert model.trunk.esm_dim == 1280
    assert model.trunk.esm_proj.in_features == 1280
    assert model.trunk.esm_proj.out_features == 32


def test_from_config_round_trips_the_enum():
    run_cfg = {
        "c_token_s1": 32, "trunk_layers": 2, "denoiser_blocks": 2,
        "atom_head_layers": 2, "atom_head_heads": 4,
        "aa_embed": "esm2_650M", "_esm_dim": 1280,
    }
    model = ResFoldOneStep(**ResFoldConfig.from_config(run_cfg).to_kwargs())
    assert model.trunk.esm_dim == 1280


def test_synthetic_650M_cache_projects_and_runs():
    """A [B, L, 1280] embedding must project to c_token and run a full forward."""
    torch.manual_seed(0)
    model = ResFoldOneStep(c_token=32, trunk_layers=2, denoiser_blocks=2,
                           atom_head_layers=1, aa_embed="esm2_650M")
    model.eval()
    B, L = 2, 8
    aa = torch.zeros(B, L, dtype=torch.long)
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    x = torch.randn(B, L, 3)
    sigma = torch.full((B,), 0.5)
    esm = torch.randn(B, L, 1280)   # synthetic 650M cache
    out = model.forward_sigma(x, aa, chain, res_idx, sigma, mask, esm_embed=esm)
    assert out.centroid_pred.shape == (B, L, 3)
    assert out.atoms_pred.shape == (B, L, 4, 3)
