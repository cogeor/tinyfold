"""C2 test: cluster-holdout split has ZERO train/test cluster overlap."""

import pyarrow as pa

from tinyfold.training.cluster_split import cluster_holdout_indices


def _toy_table(n=60):
    # n samples of 1 residue each -> 4 atoms, so they all pass the atom filter.
    sample_ids = [f"s{i}" for i in range(n)]
    atom_type = [[0, 1, 2, 3] for _ in range(n)]
    return pa.table({
        "sample_id": sample_ids,
        "LA": [1] * n,
        "LB": [0] * n,
        "atom_type": atom_type,
    })


def test_zero_cluster_overlap():
    n = 60
    table = _toy_table(n)
    # 12 clusters of 5 samples each.
    clusters = {f"s{i}": i % 12 for i in range(n)}
    train_idx, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=30, n_test=10, seed=1,
    )
    assert info["cluster_overlap"] == []
    train_c = {clusters[f"s{i}"] for i in train_idx}
    test_c = {clusters[f"s{i}"] for i in test_idx}
    assert train_c.isdisjoint(test_c)
    assert len(set(train_idx) & set(test_idx)) == 0


def test_forced_test_clusters_go_test_side():
    n = 60
    table = _toy_table(n)
    clusters = {f"s{i}": i % 12 for i in range(n)}
    forced = {7}
    train_idx, test_idx, info = cluster_holdout_indices(
        table, clusters, n_train=30, n_test=10, seed=3, forced_test_clusters=forced,
    )
    test_c = {clusters[f"s{i}"] for i in test_idx}
    assert 7 in test_c
    train_c = {clusters[f"s{i}"] for i in train_idx}
    assert 7 not in train_c


def test_determinism():
    table = _toy_table(60)
    clusters = {f"s{i}": i % 12 for i in range(60)}
    a = cluster_holdout_indices(table, clusters, n_train=30, n_test=10, seed=5)
    b = cluster_holdout_indices(table, clusters, n_train=30, n_test=10, seed=5)
    assert a[0] == b[0] and a[1] == b[1]
