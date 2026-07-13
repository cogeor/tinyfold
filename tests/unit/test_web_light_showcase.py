import json
from pathlib import Path

import pytest

# Anchor to the repo root from this file's location so the test does not depend
# on the current working directory. tests/unit/<file> -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_JSON = REPO_ROOT / "assets" / "showcase_samples.json"
WEB_STATIC = REPO_ROOT / "web-light" / "static"


def test_web_light_static_files_exist():
    if not WEB_STATIC.exists():
        pytest.skip("web-light static assets not present (generated artifact)")
    for rel in ("index.html", "css/style.css", "js/viewer.js", "js/app.js"):
        assert (WEB_STATIC / rel).exists(), f"missing web-light/static/{rel}"


def test_web_light_showcase_payload_shape():
    if not SHOWCASE_JSON.exists():
        pytest.skip("showcase_samples.json not present (generated artifact)")
    payload = json.loads(SHOWCASE_JSON.read_text(encoding="utf-8"))
    assert "samples" in payload
    samples = payload["samples"]
    assert isinstance(samples, list)
    assert len(samples) > 0

    # The showcase is curated held-out TEST complexes (best DockQ).
    splits = {s["split"] for s in samples}
    assert splits == {"test"}

    first = samples[0]
    for key in (
        "sample_id",
        "split",
        "n_atoms",
        "n_residues",
        "dockq",
        "capri",
        "c_rmsd",
        "rmsd",
        "inference_time",
        "ground_truth_pdb",
        "prediction_pdb",
    ):
        assert key in first

    assert "ATOM" in first["ground_truth_pdb"]
    assert "ATOM" in first["prediction_pdb"]
    assert "END" in first["ground_truth_pdb"]
    assert "END" in first["prediction_pdb"]
