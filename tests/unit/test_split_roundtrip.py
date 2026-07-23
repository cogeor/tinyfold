"""One canonical split serialization format (data_split), one loader.

Guards the collision cleanup: `cache.load_split` (list[str]) is gone, and the
cluster producer is `save_cluster_split`, writing the same JSON `load_split` reads.
"""

import pyarrow as pa

from tinyfold.training.cluster_split import save_cluster_split
from tinyfold.training.data_split import DataSplitConfig, load_split, save_split


def _info():
    return {
        "train_indices": [0, 2, 4],
        "test_indices": [1, 3],
        "train_ids": ["a", "c", "e"],
        "test_ids": ["b", "d"],
        "n_train": 3,
        "n_test": 2,
        "train_atom_range": (40, 80),
        "test_atom_range": (44, 60),
        "config": DataSplitConfig(n_train=3, n_test=2),
    }


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "split.json")
    save_split(_info(), path)
    train, test, data = load_split(path)
    assert train == [0, 2, 4]
    assert test == [1, 3]
    assert data["n_train"] == 3 and data["n_test"] == 2


def test_cluster_split_writes_loadable_file(tmp_path):
    # Minimal table: 4 samples of 1 residue each (LA+LB=1 -> 4 atoms).
    table = pa.table({
        "sample_id": ["s0", "s1", "s2", "s3"],
        "LA": [1, 1, 1, 1],
        "LB": [0, 0, 0, 0],
        "atom_type": [[0, 1, 2, 3]] * 4,
    })
    path = str(tmp_path / "cluster_split.json")
    save_cluster_split(table, [0, 2], [1, 3], {"strategy": "cluster"}, path)
    train, test, _ = load_split(path)
    assert train == [0, 2]
    assert test == [1, 3]


def test_cache_has_no_load_split():
    import tinyfold.data.cache as cache
    assert not hasattr(cache, "load_split")
