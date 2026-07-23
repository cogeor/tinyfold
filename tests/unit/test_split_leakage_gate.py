"""Tests for the split-leakage audit and its hard gate.

Every headline number this project produced before 2026-07-23 came from a
`test_strategy="random"` split. On the le200 split that meant 183 of 200 test
complexes shared a sequence cluster with training, and re-scoring the same
checkpoint by stratum gave DockQ 0.251 (leaked, n=183) vs 0.044 (clean, n=17)
with 0% medium-quality. The cluster machinery existed the whole time; it simply
was not wired to the configs that produced the results, and nothing in the run
output recorded which kind of split was used.

These tests pin the instrument, the gate, and the escape hatch.
"""

import argparse
import json

import pyarrow as pa
import pytest

from tinyfold.training.cluster_split import audit_split_leakage
from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices
from tinyfold.training.setup import SplitLeakageError, audit_and_gate_split


class _Logger:
    """Minimal stand-in for the training Logger."""

    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(str(msg))

    @property
    def text(self):
        return "\n".join(self.lines)


def _table(n: int) -> pa.Table:
    return pa.Table.from_pylist([
        {"sample_id": f"s{i:05d}", "atom_type": [0] * 100, "LA": 25, "LB": 25}
        for i in range(n)
    ])


def _clusters_file(tmp_path, mapping: dict[str, int]) -> str:
    p = tmp_path / "clusters.json"
    p.write_text(json.dumps({"sample_to_cluster": mapping}))
    return str(p)


def _args(**kw):
    base = dict(load_split=None, require_clean_split=True, clusters=None)
    base.update(kw)
    return argparse.Namespace(**base)


# --- the audit instrument --------------------------------------------------

def test_audit_counts_leaked_test_samples():
    clusters = {"a": 1, "b": 1, "c": 2, "d": 3}
    audit = audit_split_leakage(train_ids=["a"], test_ids=["b", "c", "d"], clusters=clusters)
    assert audit["n_test_leaked"] == 1          # "b" shares cluster 1 with "a"
    assert audit["frac_test_leaked"] == pytest.approx(1 / 3)
    assert audit["n_train_clusters"] == 1
    assert audit["n_test_clusters"] == 3


def test_audit_reports_zero_for_a_clean_split():
    clusters = {"a": 1, "b": 2, "c": 3}
    audit = audit_split_leakage(["a"], ["b", "c"], clusters)
    assert audit["n_test_leaked"] == 0
    assert audit["frac_test_leaked"] == 0.0


def test_audit_flags_unclustered_samples():
    """Samples missing from clusters.json cannot be checked -- say so."""
    clusters = {"a": 1, "b": 2}
    audit = audit_split_leakage(["a"], ["b", "unknown"], clusters)
    assert audit["n_test_unclustered"] == 1
    assert audit["n_test_leaked"] == 0


def test_audit_reports_test_cluster_concentration():
    clusters = {"t0": 9, "t1": 9, "t2": 9, "t3": 5, "x": 1}
    audit = audit_split_leakage(["x"], ["t0", "t1", "t2", "t3"], clusters)
    assert audit["n_test_clusters"] == 2
    assert audit["max_test_cluster_share"] == pytest.approx(0.75)


# --- the gate --------------------------------------------------------------

def test_gate_raises_on_a_leaky_split(tmp_path):
    table = _table(4)
    # s00000 and s00001 share cluster 1 -> putting one each side is leakage.
    path = _clusters_file(tmp_path, {"s00000": 1, "s00001": 1, "s00002": 2, "s00003": 3})
    with pytest.raises(SplitLeakageError, match="LEAKY SPLIT"):
        audit_and_gate_split(
            _args(clusters=path), table,
            train_indices=[0], test_indices=[1], logger=_Logger(),
        )


def test_gate_passes_a_clean_split(tmp_path):
    table = _table(4)
    path = _clusters_file(tmp_path, {"s00000": 1, "s00001": 2, "s00002": 3, "s00003": 4})
    audit = audit_and_gate_split(
        _args(clusters=path), table,
        train_indices=[0], test_indices=[1, 2], logger=_Logger(),
    )
    assert audit["split_audited"] is True
    assert audit["n_test_leaked"] == 0


def test_escape_hatch_proceeds_but_labels_the_run(tmp_path):
    table = _table(4)
    path = _clusters_file(tmp_path, {"s00000": 1, "s00001": 1, "s00002": 2, "s00003": 3})
    logger = _Logger()
    audit = audit_and_gate_split(
        _args(clusters=path, require_clean_split=False), table,
        train_indices=[0], test_indices=[1], logger=logger,
    )
    assert audit["n_test_leaked"] == 1
    assert "LEAKY SPLIT" in logger.text, "a leaky run must be labelled in its log"


def test_missing_clusters_file_raises_by_default(tmp_path):
    """Not being able to measure leakage is itself a failure.

    Silently not measuring is exactly how this went unnoticed for six months.
    """
    table = _table(2)
    missing = str(tmp_path / "nope.json")
    with pytest.raises(SplitLeakageError, match="Cannot audit split leakage"):
        audit_and_gate_split(
            _args(clusters=missing), table,
            train_indices=[0], test_indices=[1], logger=_Logger(),
        )


def test_missing_clusters_file_is_a_warning_when_not_required(tmp_path):
    table = _table(2)
    logger = _Logger()
    audit = audit_and_gate_split(
        _args(clusters=str(tmp_path / "nope.json"), require_clean_split=False),
        table, train_indices=[0], test_indices=[1], logger=logger,
    )
    assert audit["split_audited"] is False
    assert "WARNING" in logger.text


def test_concentrated_but_clean_split_warns(tmp_path):
    """Disjoint is not the same as informative -- effective n is the cluster count."""
    table = _table(6)
    path = _clusters_file(tmp_path, {
        "s00000": 1, "s00001": 7, "s00002": 7, "s00003": 7, "s00004": 7, "s00005": 8,
    })
    logger = _Logger()
    audit = audit_and_gate_split(
        _args(clusters=path), table,
        train_indices=[0], test_indices=[1, 2, 3, 4, 5], logger=logger,
    )
    assert audit["n_test_leaked"] == 0          # genuinely clean
    assert audit["max_test_cluster_share"] == pytest.approx(0.8)
    assert "Effective n is the cluster count" in logger.text


# --- test_strategy="cluster" as a first-class strategy ---------------------

def test_cluster_strategy_produces_a_clean_split(tmp_path):
    table = _table(60)
    # 30 clusters of 2 samples each.
    mapping = {f"s{i:05d}": i // 2 for i in range(60)}
    path = _clusters_file(tmp_path, mapping)

    cfg = DataSplitConfig(
        n_train=30, n_test=10, min_atoms=0, max_atoms=99999,
        test_strategy="cluster", clusters_path=path, per_cluster_cap=1, seed=42,
    )
    train, test = get_train_test_indices(table, cfg)
    assert set(train).isdisjoint(set(test))

    ids = table["sample_id"]
    audit = audit_split_leakage(
        [ids[i].as_py() for i in train], [ids[i].as_py() for i in test], mapping,
    )
    assert audit["n_test_leaked"] == 0
    assert audit["n_test_clusters"] == len(test), "cap=1 -> one cluster per sample"


def test_cluster_strategy_requires_a_clusters_path():
    with pytest.raises(ValueError, match="requires clusters_path"):
        DataSplitConfig(n_train=10, n_test=5, test_strategy="cluster")


def test_random_strategy_still_accepted_and_default():
    cfg = DataSplitConfig(n_train=10, n_test=5)
    assert cfg.test_strategy == "random"
    assert cfg.per_cluster_cap == 1  # only consulted by the cluster strategy


def test_unknown_strategy_still_raises():
    with pytest.raises(ValueError, match="must be 'random', 'stratified' or 'cluster'"):
        DataSplitConfig(n_train=10, n_test=5, test_strategy="oracle")
