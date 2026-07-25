"""load_onestep_run tolerates missing off-by-default params, not real mismatches.

C1 added zero-init recycling params (recycle_norm/proj) to ResidueEncoder. Every
checkpoint trained before C1 lacks those four keys, but the recycle path is inert
at eval (n_recycle=0), so such a checkpoint must still load. A genuinely missing
or unexpected key is a real architecture mismatch and must still raise.
"""

import json

import pytest
import torch

from tinyfold.inference.build import _is_offdefault_param, load_onestep_run
from tinyfold.model.resfold.config import ResFoldConfig
from tinyfold.model.resfold.onestep import ResFoldOneStep

_CFG = {
    "model_kind": "onestep",
    "c_token_s1": 32, "trunk_layers": 2, "denoiser_blocks": 2,
    "atom_head_layers": 1, "atom_head_heads": 4, "aa_embed": "esm2_35M",
}


def _write_run(tmp_path, state):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_CFG), encoding="utf-8")
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": state}, ckpt_path)
    return ckpt_path


def _fresh_state():
    model = ResFoldOneStep(**ResFoldConfig.from_config(_CFG).to_kwargs())
    return model.state_dict()


def test_offdefault_predicate():
    assert _is_offdefault_param("trunk.recycle_proj.weight")
    assert _is_offdefault_param("trunk.recycle_norm.bias")
    assert not _is_offdefault_param("trunk.input_proj.weight")


def test_pre_c1_checkpoint_loads(tmp_path):
    # Simulate a pre-C1 checkpoint: drop the recycle_* keys entirely.
    state = {k: v for k, v in _fresh_state().items() if "recycle" not in k}
    assert not any("recycle" in k for k in state)
    ckpt = _write_run(tmp_path, state)
    model, cfg = load_onestep_run(ckpt, torch.device("cpu"))
    # recycle_proj stayed at its zero-init (no-op) constructed value.
    assert float(model.trunk.recycle_proj.weight.abs().sum()) == 0.0
    assert cfg["model_kind"] == "onestep"


def test_real_missing_key_still_raises(tmp_path):
    # Drop a genuine (non-off-by-default) param -> real mismatch -> raise.
    state = {k: v for k, v in _fresh_state().items() if "input_proj" not in k}
    ckpt = _write_run(tmp_path, state)
    with pytest.raises(RuntimeError, match="mismatch"):
        load_onestep_run(ckpt, torch.device("cpu"))


def test_unexpected_key_still_raises(tmp_path):
    state = _fresh_state()
    state["trunk.bogus_extra.weight"] = torch.zeros(3)
    ckpt = _write_run(tmp_path, state)
    with pytest.raises(RuntimeError, match="mismatch"):
        load_onestep_run(ckpt, torch.device("cpu"))
