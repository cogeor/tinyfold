"""The continuous-sigma sampler dispatch picks the right sampler + kwargs."""

import tinyfold.training.eval as ev
from tinyfold.training.eval import sample_centroids_continuous


def _fakes(monkeypatch):
    calls = {}

    def fake_one_shot(*a, **k):
        calls["one_shot"] = k
        return "OS"

    def fake_ve(*a, **k):
        calls["ve"] = k
        return "VE"

    monkeypatch.setattr(ev, "sample_centroids_one_shot", fake_one_shot)
    monkeypatch.setattr(ev, "sample_centroids_ve", fake_ve)
    return calls


def test_one_shot_calls_one_shot_sampler(monkeypatch):
    calls = _fakes(monkeypatch)
    out = sample_centroids_continuous(
        None, None, None, None, one_shot=True, align_per_step=True,
        recenter=False, kabsch_interp=True, is_onestep=True,
    )
    assert out == "OS"
    assert "ve" not in calls
    assert calls["one_shot"] == {"is_onestep": True}


def test_multistep_calls_ve_sampler_with_flags(monkeypatch):
    calls = _fakes(monkeypatch)
    out = sample_centroids_continuous(
        None, None, None, None, one_shot=False, align_per_step=True,
        recenter=True, kabsch_interp=False, is_onestep=False,
    )
    assert out == "VE"
    assert "one_shot" not in calls
    assert calls["ve"] == {
        "align_per_step": True, "recenter": True,
        "kabsch_interp": False, "is_onestep": False,
    }
