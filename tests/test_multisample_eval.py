"""Tests for Loop 02's multi-sample inference path.

Covers:
  - ``sample_k_centroids`` returns K reproducible, distinct samples.
  - Re-running with the same seeds produces byte-identical output.
  - (Integration, skipped without the Phase C checkpoint) Re-eval with
    K=5 emits ``oracle@5``, ``mean@5``, ``ranked@5`` tokens and obeys
    ``oracle@5 <= mean@5``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.resfold.onestep import ResFoldOneStep

import train_resfold


REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_C_CKPT = (
    REPO_ROOT
    / "outputs"
    / "resfold"
    / "phase_c_n8600"
    / "resfold_s1_8K_20260524_020409"
    / "best_model.pt"
)


def _make_tiny_model() -> ResFoldOneStep:
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32,
        trunk_layers=1,
        trunk_heads=2,
        denoiser_blocks=1,
        denoiser_heads=2,
        atom_head_layers=1,
        atom_head_heads=2,
        n_timesteps=10,
        dropout=0.0,
    )


def _make_batch(L: int = 10) -> dict:
    aa_seq = torch.randint(0, 20, (1, L), dtype=torch.long)
    chain_ids = torch.zeros(1, L, dtype=torch.long)
    chain_ids[:, L // 2:] = 1
    res_idx = torch.arange(L, dtype=torch.long).unsqueeze(0)
    mask_res = torch.ones(1, L, dtype=torch.bool)
    centroids = torch.randn(1, L, 3)
    return {
        "aa_seq": aa_seq,
        "chain_ids": chain_ids,
        "res_idx": res_idx,
        "mask_res": mask_res,
        "centroids": centroids,
    }


def _make_noiser() -> VENoiser:
    schedule = KarrasSchedule(n_steps=5, sigma_min=0.01, sigma_max=5.0)
    return VENoiser(schedule, sigma_data=1.0)


def test_sample_k_centroids_one_shot_returns_distinct_samples() -> None:
    """K samples must differ pairwise; same seed -> bit-identical re-run."""
    model = _make_tiny_model().eval()
    batch = _make_batch(L=10)
    noiser = _make_noiser()
    device = torch.device("cpu")

    # Loop 06: sample_k_centroids now returns a 3-tuple where the third slot
    # is per-sample predicted lDDT (None when the model has no confidence head).
    centroids, atoms, pred_lddts = train_resfold.sample_k_centroids(
        model, batch, noiser, device,
        K=4, base_seed=42, target_idx=0,
        is_onestep=True, one_shot=True,
    )
    assert centroids.shape == (4, 1, 10, 3)
    assert atoms is not None and atoms.shape == (4, 1, 10, 4, 3)
    assert pred_lddts is None, "no confidence_head -> pred_lddts must be None"

    # Pairwise distinctness on the centroid output.
    for i in range(4):
        for j in range(i + 1, 4):
            diff = (centroids[i] - centroids[j]).norm().item()
            assert diff > 1e-3, f"samples {i} and {j} are too close: {diff:.6e}"

    # Reproducibility: same seeds -> identical tensors.
    centroids2, atoms2, pred_lddts2 = train_resfold.sample_k_centroids(
        model, batch, noiser, device,
        K=4, base_seed=42, target_idx=0,
        is_onestep=True, one_shot=True,
    )
    assert torch.equal(centroids, centroids2)
    assert torch.equal(atoms, atoms2)
    assert pred_lddts2 is None


def test_sample_k_centroids_ve_path_distinct() -> None:
    """The non-one-shot VE path also produces distinct seeded samples."""
    model = _make_tiny_model().eval()
    batch = _make_batch(L=10)
    noiser = _make_noiser()
    device = torch.device("cpu")

    centroids, _, pred_lddts = train_resfold.sample_k_centroids(
        model, batch, noiser, device,
        K=3, base_seed=7, target_idx=2,
        is_onestep=True, one_shot=False,
        align_per_step=False, recenter=True, self_cond=False,
    )
    assert centroids.shape == (3, 1, 10, 3)
    assert pred_lddts is None  # no confidence head on this tiny model
    for i in range(3):
        for j in range(i + 1, 3):
            assert (centroids[i] - centroids[j]).norm().item() > 1e-3


def test_different_target_idx_gives_independent_samples() -> None:
    """target_idx is part of the seed mix; different targets -> different RNG."""
    model = _make_tiny_model().eval()
    batch = _make_batch(L=10)
    noiser = _make_noiser()
    device = torch.device("cpu")

    c_a, _, _ = train_resfold.sample_k_centroids(
        model, batch, noiser, device,
        K=1, base_seed=42, target_idx=0,
        is_onestep=True, one_shot=True,
    )
    c_b, _, _ = train_resfold.sample_k_centroids(
        model, batch, noiser, device,
        K=1, base_seed=42, target_idx=1,
        is_onestep=True, one_shot=True,
    )
    assert not torch.equal(c_a, c_b)


@pytest.mark.slow
@pytest.mark.skipif(
    not PHASE_C_CKPT.exists(),
    reason=f"Phase C checkpoint not present at {PHASE_C_CKPT}; CI-safe skip.",
)
def test_eval_only_k5_integration(tmp_path: Path) -> None:
    """End-to-end smoke: --n_samples 5 --eval_K_list 1,5 emits tokens.

    Asserts ``oracle@5 <= mean@5`` (oracle is a per-target min over the same
    pool the mean averages) and that the ranked@5 token is present.
    """
    cfg = REPO_ROOT / "configs" / "train" / "resfold" / "phase_c_n8600.yaml"
    out = tmp_path / "multisample_smoke"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_resfold.py"),
        "--config", str(cfg),
        "--eval_only",
        "--checkpoint", str(PHASE_C_CKPT),
        "--n_test", "4",
        "--n_samples", "5",
        "--eval_K_list", "1,5",
        "--output_dir", str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
    )

    # Find the train.log and grep for the eval summary line.
    log_files = list(out.rglob("train.log"))
    assert log_files, f"no train.log under {out}"
    log_text = log_files[0].read_text(encoding="utf-8")
    assert "oracle@5" in log_text
    assert "mean@5" in log_text
    assert "ranked@5" in log_text

    # Pull the numeric values out of the most recent summary line.
    summary_line = [ln for ln in log_text.splitlines() if "oracle@5" in ln][-1]
    def _val(token: str) -> float:
        marker = f"{token}: "
        i = summary_line.index(marker) + len(marker)
        j = summary_line.index(" A", i)
        return float(summary_line[i:j])
    o5 = _val("oracle@5")
    m5 = _val("mean@5")
    assert o5 <= m5 + 1e-6, f"oracle@5 ({o5}) must be <= mean@5 ({m5})"
