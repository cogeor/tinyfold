import torch

from tinyfold.training.objective import LossComposer, LossRegistry, LossTerm


def test_loss_registry_register_and_get():
    registry = LossRegistry()
    registry.register("mse", lambda pred, target: ((pred - target) ** 2).mean())
    assert registry.has("mse")
    assert "mse" in registry.names()

    fn = registry.get("mse")
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 0.0])
    value = fn(pred=pred, target=target)
    assert torch.is_tensor(value)
    assert value.item() > 0.0


def test_loss_composer_combines_weighted_terms():
    terms = [
        LossTerm("a", lambda x: x.mean(), weight=1.0),
        LossTerm("b", lambda x: x.mean() * 2.0, weight=0.5),
    ]
    composer = LossComposer(terms)
    x = torch.tensor([2.0, 4.0])
    total, metrics = composer(x=x)

    # a = 3.0, b = 6.0 -> weighted total = 3.0 + 3.0 = 6.0
    assert abs(total.item() - 6.0) < 1e-6
    assert abs(metrics["a"] - 3.0) < 1e-6
    assert abs(metrics["b"] - 6.0) < 1e-6
    assert abs(metrics["total"] - 6.0) < 1e-6


def test_loss_composer_skips_disabled_terms():
    terms = [
        LossTerm("a", lambda x: x.mean(), enabled=False),
        LossTerm("b", lambda x: x.mean(), weight=0.0),
        LossTerm("c", lambda x: x.mean(), weight=2.0),
    ]
    composer = LossComposer(terms)
    x = torch.tensor([1.0, 3.0])
    total, metrics = composer(x=x)

    assert abs(total.item() - 4.0) < 1e-6
    assert "a" not in metrics
    assert "b" not in metrics
    assert abs(metrics["c"] - 2.0) < 1e-6

