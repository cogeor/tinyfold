"""ResFoldConfig contract (L30).

- The dataclass field set (and defaults) must stay in lock-step with
  ResFoldOneStep.__init__ (signature-drift guard).
- Building via config.to_kwargs() must be identical to direct kwargs.
- from_config must reproduce build_onestep_from_config's key translation.
"""

import inspect

from tinyfold.inference.build import build_onestep_from_config
from tinyfold.model.resfold.config import ResFoldConfig
from tinyfold.model.resfold.onestep import ResFoldOneStep


def _ctor_params():
    sig = inspect.signature(ResFoldOneStep.__init__)
    return {name: p for name, p in sig.parameters.items() if name != "self"}


def test_config_fields_match_constructor_signature():
    ctor = _ctor_params()
    cfg_fields = set(ResFoldConfig().to_kwargs().keys())
    assert cfg_fields == set(ctor), (
        "ResFoldConfig drifted from ResFoldOneStep.__init__: "
        f"missing={set(ctor) - cfg_fields}, extra={cfg_fields - set(ctor)}"
    )


def test_config_defaults_match_constructor_defaults():
    ctor = _ctor_params()
    defaults = ResFoldConfig().to_kwargs()
    for name, p in ctor.items():
        if p.default is inspect.Parameter.empty:
            continue
        assert defaults[name] == p.default, f"default mismatch for {name}"


def test_build_via_config_matches_direct_kwargs():
    """ResFoldOneStep(**cfg.to_kwargs()) == ResFoldOneStep(**same kwargs)."""
    kwargs = dict(
        c_token=32, trunk_layers=1, trunk_heads=2,
        denoiser_blocks=1, denoiser_heads=2,
        atom_head_layers=1, atom_head_heads=2, n_timesteps=10,
    )
    direct = ResFoldOneStep(**kwargs)
    via_cfg = ResFoldOneStep(**ResFoldConfig(**kwargs).to_kwargs())
    assert (
        direct.count_parameters()["total"]
        == via_cfg.count_parameters()["total"]
    )


def test_from_config_translates_run_config_keys():
    """build_onestep_from_config must match a hand-translated direct build."""
    cfg = {
        "model_kind": "onestep",
        "c_token_s1": 32,
        "trunk_layers": 1,
        "denoiser_blocks": 1,
        "atom_head_layers": 1,
        "atom_head_heads": 2,
        "T": 10,
        "aa_embed": "learned",
        "esm_dim": 16,
        "confidence_head": True,
    }
    model_a = build_onestep_from_config(cfg)
    # Hand-translate the same keys the way from_config does.
    model_b = ResFoldOneStep(
        c_token=32, trunk_layers=1, denoiser_blocks=1,
        atom_head_layers=1, atom_head_heads=2, n_timesteps=10,
        dropout=0.0, aa_embed="learned", esm_dim=16,
        confidence_head=True,
    )
    assert (
        model_a.count_parameters()["total"]
        == model_b.count_parameters()["total"]
    )
    # from_config honors the falsy-global_scale -> 11.0 convention.
    assert ResFoldConfig.from_config({**cfg, "global_scale": 0}).global_scale == 11.0
