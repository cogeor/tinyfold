"""Tests for the per-cluster test cap in cluster_holdout_indices.

Cluster DISJOINTNESS was already guaranteed; what was missing was cluster
DIVERSITY. Without a cap, whole clusters are appended to the test set until
n_test samples are collected, so one large cluster can supply most of the test
set and n_test samples carry only a handful of clusters' worth of independent
information. The committed splits show exactly this:

    clean_le240: 200 samples /  41 clusters (top-5 = 52% of test)
    clean_le400: 200 samples /  44 clusters (top-5 = 55% of test)
    clean_le600: 200 samples /  21 clusters (ONE cluster = 137 of 200)

These tests pin the cap's guarantees and, critically, that the default path is
unchanged so previously generated splits still reproduce.
"""

import pyarrow as pa
import pytest

from tinyfold.training.cluster_split import cluster_holdout_indices


def _table(n: int) -> pa.Table:
    """Minimal parquet-shaped table: cluster_holdout_indices needs only
    `sample_id` and `atom_type` (for the atom-count filter)."""
    return pa.Table.from_pylist([
        {"sample_id": f"s{i:05d}", "atom_type": [0] * 100} for i in range(n)
    ])


def _clusters(sizes: list[int]) -> dict[str, int]:
    """Assign sample ids to clusters of the given sizes, in order."""
    out, k = {}, 0
    for cid, size in enumerate(sizes):
        for _ in range(size):
            out[f"s{k:05d}"] = cid
            k += 1
    return out


# --- the pathology, and its fix -------------------------------------------

def test_uncapped_reproduces_the_clean_le600_pathology():
    """One dominant cluster swamps the test set when uncapped.

    The bug, pinned as a regression fixture so it cannot silently come back.
    Every cluster here is larger than n_test, so whichever the shuffle visits
    first fills the entire test set on its own -- deterministic without
    depending on the shuffle order.
    """
    sizes = [250] * 30
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    _, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=200, n_test=200, seed=0,
    )
    assert len(test_idx) == 200
    # Every test sample came from a single cluster.
    assert info["n_test_clusters"] == 1
    assert info["max_test_cluster_share"] == pytest.approx(1.0)


def test_cap_of_one_fixes_the_pathology_on_the_same_fixture():
    """Same clusters, cap=1: the test set now spans 30 independent clusters."""
    sizes = [250] * 30
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    _, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=200, n_test=200, seed=0, per_cluster_cap=1,
    )
    # Only 30 clusters exist, so cap=1 yields 30 test samples rather than 200.
    # Fewer samples, but 30x the independent information.
    assert info["n_test_clusters"] == 30
    assert len(test_idx) == 30
    assert info["max_test_cluster_share"] == pytest.approx(1 / 30)


def test_cap_of_one_gives_one_sample_per_cluster():
    sizes = [300] + [2] * 400
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    _, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=200, n_test=150, seed=0, per_cluster_cap=1,
    )
    assert len(test_idx) == 150
    assert info["n_test_clusters"] == 150, "n_test samples must carry n_test clusters"
    assert info["max_test_cluster_share"] == pytest.approx(1 / 150)


@pytest.mark.parametrize("cap", [1, 2, 5])
def test_no_cluster_exceeds_the_cap(cap):
    sizes = [50, 40, 30] + [7] * 100
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    _, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=100, n_test=60, seed=3, per_cluster_cap=cap,
    )
    counts: dict[int, int] = {}
    for ix in test_idx:
        cid = clusters[table["sample_id"][ix].as_py()]
        counts[cid] = counts.get(cid, 0) + 1
    assert counts, "test set must not be empty"
    assert max(counts.values()) <= cap
    # The reported share must match the realised split, not the pre-trim one.
    assert info["max_test_cluster_share"] == pytest.approx(
        max(counts.values()) / len(test_idx)
    )
    assert info["max_test_cluster_share"] <= cap / len(test_idx) + 1e-9


# --- invariants that must survive the change ------------------------------

@pytest.mark.parametrize("cap", [None, 1, 3])
def test_cluster_disjointness_holds_for_every_cap(cap):
    sizes = [20, 15, 12] + [4] * 60
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    train_idx, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=80, n_test=40, seed=7, per_cluster_cap=cap,
    )
    assert not info["cluster_overlap"]
    train_c = {clusters[table["sample_id"][i].as_py()] for i in train_idx}
    test_c = {clusters[table["sample_id"][i].as_py()] for i in test_idx}
    assert train_c.isdisjoint(test_c)
    assert set(train_idx).isdisjoint(set(test_idx))


def test_default_path_is_bit_for_bit_unchanged():
    """cap=None must leave the RNG stream untouched.

    Subsampling inside a cluster draws from a separate stream precisely so
    that adding the cap does not silently re-roll every split generated
    before it existed. This pins the historical result.
    """
    sizes = [9, 7, 5, 3] + [2] * 50
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    kw = dict(n_train=60, n_test=30, seed=42)
    a_train, a_test, _ = cluster_holdout_indices(table, clusters, **kw)
    b_train, b_test, _ = cluster_holdout_indices(
        table, clusters, per_cluster_cap=None, **kw
    )
    assert a_train == b_train
    assert a_test == b_test


def test_seed_is_reproducible_with_cap():
    sizes = [30, 20] + [3] * 80
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    kw = dict(n_train=100, n_test=50, seed=11, per_cluster_cap=2)
    t1, e1, _ = cluster_holdout_indices(table, clusters, **kw)
    t2, e2, _ = cluster_holdout_indices(table, clusters, **kw)
    assert t1 == t2 and e1 == e2


# --- n_test_clusters and train weights ------------------------------------

def test_n_test_clusters_controls_the_stopping_rule():
    sizes = [6] * 100
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    _, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=200, n_test=999,
        seed=5, per_cluster_cap=2, n_test_clusters=20,
    )
    assert info["n_test_clusters"] == 20
    assert len(test_idx) == 40  # 20 clusters x cap 2


def test_train_weights_sum_to_one_per_cluster():
    sizes = [10, 5, 2] + [1] * 40
    table, clusters = _table(sum(sizes)), _clusters(sizes)

    train_idx, _, info = cluster_holdout_indices(
        table, clusters, n_train=40, n_test=10,
        seed=2, per_cluster_cap=1, train_cluster_uniform=True,
    )
    weights = info["train_weights"]
    assert len(weights) == len(train_idx)

    per_cluster: dict[int, float] = {}
    for ix, w in zip(train_idx, weights):
        cid = clusters[table["sample_id"][ix].as_py()]
        per_cluster[cid] = per_cluster.get(cid, 0.0) + w
    for cid, total in per_cluster.items():
        assert total == pytest.approx(1.0), f"cluster {cid} mass {total}"


def test_train_weights_absent_unless_requested():
    sizes = [4] * 20
    table, clusters = _table(sum(sizes)), _clusters(sizes)
    _, _, info = cluster_holdout_indices(
        table, clusters, n_train=40, n_test=10, seed=1,
    )
    assert "train_weights" not in info


# --- argument validation ---------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"per_cluster_cap": 0},
    {"per_cluster_cap": -1},
    {"n_test_clusters": 0},
])
def test_invalid_arguments_raise(kwargs):
    table, clusters = _table(20), _clusters([2] * 10)
    with pytest.raises(ValueError):
        cluster_holdout_indices(table, clusters, n_train=5, n_test=5, **kwargs)
