"""Tests for the DockQ metric wrapper (model/metrics/dockq.py).

DockQ is the headline metric but had no coverage. Uses a synthetic 2-chain
complex with a guaranteed interface so the score is deterministic. Skips if the
optional DockQ backend is not importable.
"""

import numpy as np
import pytest
import torch

from tinyfold.model.metrics.dockq import compute_dockq


def _strand(n: int, offset: np.ndarray) -> np.ndarray:
    coords = np.zeros((n, 4, 3), np.float32)
    for i in range(n):
        ca = np.array([i * 3.8, 0.0, 0.0], np.float32) + offset
        coords[i, 0] = ca + [-1.0, 0.3, 0.0]  # N
        coords[i, 1] = ca                      # CA
        coords[i, 2] = ca + [1.0, 0.3, 0.0]    # C
        coords[i, 3] = ca + [1.4, 1.2, 0.0]    # O
    return coords


def _complex():
    a = _strand(12, np.array([0, 0, 0], np.float32))
    b = _strand(12, np.array([0, 4.5, 0], np.float32))  # 4.5 A -> interface contacts
    coords = torch.tensor(np.concatenate([a, b], 0))
    aa = torch.zeros(24, dtype=torch.long)
    chain_ids = torch.tensor([0] * 12 + [1] * 12)
    return coords, aa, chain_ids


def test_identical_pose_scores_one():
    coords, aa, chain_ids = _complex()
    r = compute_dockq(coords, coords, aa, chain_ids)
    if r["dockq"] is None:
        pytest.skip("DockQ backend not available")
    assert r["dockq"] == pytest.approx(1.0, abs=1e-3)
    assert r["fnat"] == pytest.approx(1.0, abs=1e-3)


def test_perturbed_pose_scores_lower():
    coords, aa, chain_ids = _complex()
    ref = compute_dockq(coords, coords, aa, chain_ids)
    if ref["dockq"] is None:
        pytest.skip("DockQ backend not available")
    pred = coords.clone()
    pred[12:] += torch.tensor([0.0, 6.0, 0.0])  # pull chain B out of the interface
    r = compute_dockq(pred, coords, aa, chain_ids)
    assert r["dockq"] < ref["dockq"]
    assert 0.0 <= r["dockq"] <= 1.0
