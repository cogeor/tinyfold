import json
from pathlib import Path


def test_web_light_static_files_exist():
    root = Path("web-light/static")
    assert (root / "index.html").exists()
    assert (root / "css/style.css").exists()
    assert (root / "js/viewer.js").exists()
    assert (root / "js/app.js").exists()
    assert Path("assets/showcase_samples.json").exists()


def test_web_light_showcase_payload_shape():
    path = Path("assets/showcase_samples.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "samples" in payload
    samples = payload["samples"]
    assert isinstance(samples, list)
    assert len(samples) > 0

    splits = {s["split"] for s in samples}
    assert "train" in splits
    assert "test" in splits

    first = samples[0]
    for key in (
        "sample_id",
        "split",
        "n_atoms",
        "n_residues",
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
