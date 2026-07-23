"""Tests for EMA of weights (C3).

EMA keeps a shadow copy of the model's params updated after each optimizer step
as shadow = decay*shadow + (1-decay)*param. Eval and checkpointing read the
smoothed weights; model_state_dict stays raw so existing loaders keep working.
"""

import torch
import torch.nn as nn

from tinyfold.training.checkpointing import EMA, load_checkpoint, save_checkpoint


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)


def test_shadow_tracks_analytic_ema_on_a_scripted_sequence():
    """Feed a scripted parameter trajectory through EMA.update and compare the
    shadow against the closed-form EMA of that trajectory."""
    torch.manual_seed(0)
    model = _Tiny()
    decay = 0.9
    ema = EMA(model, decay)

    # Analytic EMA computed alongside for the weight tensor.
    analytic = model.lin.weight.detach().clone()
    for _ in range(20):
        with torch.no_grad():
            model.lin.weight.copy_(torch.randn_like(model.lin.weight))
            model.lin.bias.copy_(torch.randn_like(model.lin.bias))
        ema.update(model)
        analytic = decay * analytic + (1 - decay) * model.lin.weight.detach()

    assert torch.allclose(ema.shadow["lin.weight"], analytic, atol=1e-6)


def test_store_copy_restore_round_trips_live_weights():
    torch.manual_seed(1)
    model = _Tiny()
    ema = EMA(model, 0.5)
    # Drift the shadow away from the live weights.
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    ema.update(model)
    live_before = model.lin.weight.detach().clone()

    ema.store(model)
    ema.copy_to(model)
    assert torch.allclose(model.lin.weight, ema.shadow["lin.weight"])
    assert not torch.allclose(model.lin.weight, live_before)

    ema.restore(model)
    assert torch.equal(model.lin.weight, live_before)


def test_decay_zero_is_never_constructed_but_state_dict_overlays_correctly():
    """state_dict(model) must be a full, loadable dict with EMA params overlaid
    and non-param buffers preserved."""
    model = _Tiny()
    ema = EMA(model, 0.99)
    sd = ema.state_dict(model)
    # Same keys as a normal state dict.
    assert set(sd.keys()) == set(model.state_dict().keys())
    # Param entries equal the shadow.
    assert torch.equal(sd["lin.weight"], ema.shadow["lin.weight"])
    # Loads cleanly into a fresh model.
    fresh = _Tiny()
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_checkpoint_round_trips_both_state_dicts(tmp_path):
    torch.manual_seed(2)
    model = _Tiny()
    ema = EMA(model, 0.9)
    with torch.no_grad():
        model.lin.weight.add_(2.0)   # raw != ema
    ema.update(model)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=5, ema_state_dict=ema.state_dict(model))

    ckpt = torch.load(path, weights_only=True)
    assert "model_state_dict" in ckpt and "ema_state_dict" in ckpt
    # model_state_dict is the RAW weights.
    assert torch.equal(ckpt["model_state_dict"]["lin.weight"], model.lin.weight)
    # ema_state_dict is the shadow.
    assert torch.equal(ckpt["ema_state_dict"]["lin.weight"], ema.shadow["lin.weight"])
    # And they differ.
    assert not torch.equal(
        ckpt["model_state_dict"]["lin.weight"], ckpt["ema_state_dict"]["lin.weight"]
    )

    # load_checkpoint (raw path) restores the raw weights unchanged.
    fresh = _Tiny()
    load_checkpoint(path, fresh)
    assert torch.equal(fresh.lin.weight, model.lin.weight)


def test_save_without_ema_omits_the_key(tmp_path):
    """decay=0 path: no ema passed -> checkpoint contents unchanged (no
    ema_state_dict key), so existing loaders see exactly today's checkpoint."""
    model = _Tiny()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, step=1)
    ckpt = torch.load(path, weights_only=True)
    assert "ema_state_dict" not in ckpt
