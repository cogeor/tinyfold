"""Tests for SampleStore (B1) -- bounded-memory sample access.

Training used to build ``{idx: load_sample_raw(...)}`` for the whole split
before the first step. Measured on this dataset: decoded samples cost
~2.1 KB/residue, so the full 41,883-complex / 22.0 M-residue dataset comes to
~46 GB of sample dicts on top of the parquet table itself.

The store must bound that WITHOUT changing what training sees, so the central
test here is equivalence: an lru store returns exactly what an eager store
returns. Cropping stays in collate_batch precisely so this holds.
"""

import threading

import numpy as np
import pyarrow as pa
import pytest
import torch

from tinyfold.training.data import (
    TRAIN_COLUMNS,
    MergedSampleStore,
    SampleStore,
    load_sample,
    sample_nbytes,
)


def make_table(n_res_per_sample):
    """Parquet-shaped table with `len(n_res_per_sample)` samples."""
    rows = {"sample_id": [], "seq": [], "chain_id_res": [], "res_idx": [],
            "atom_coords": [], "atom_to_res": [], "atom_type": [],
            "LA": [], "LB": []}
    rng = np.random.default_rng(0)
    for k, L in enumerate(n_res_per_sample):
        la = L // 2
        lb = L - la
        rows["sample_id"].append(f"s{k:04d}")
        rows["seq"].append(rng.integers(0, 20, size=L).tolist())
        rows["chain_id_res"].append([0] * la + [1] * lb)
        rows["res_idx"].append(list(range(la)) + list(range(lb)))
        rows["atom_coords"].append(
            rng.normal(size=4 * L * 3).astype(np.float32).tolist()
        )
        rows["atom_to_res"].append([r for r in range(L) for _ in range(4)])
        rows["atom_type"].append([0, 1, 2, 3] * L)
        rows["LA"].append(la)
        rows["LB"].append(lb)
    return pa.table(rows)


def assert_same_sample(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), f"tensor {k} differs"
        else:
            assert a[k] == b[k], f"value {k} differs"


# --- equivalence: the property that makes the change safe ------------------

def test_lru_returns_exactly_what_eager_returns():
    table = make_table([10, 20, 30, 40, 50])
    idxs = list(range(5))
    eager = SampleStore(table, idxs, mode="eager")
    # Budget far below the working set, so nearly every access is a miss.
    lru = SampleStore(table, idxs, mode="lru", cache_mb=0.001)
    for i in idxs:
        assert_same_sample(eager[i], lru[i])


def test_repeated_access_after_eviction_is_still_identical():
    table = make_table([10, 20, 30])
    lru = SampleStore(table, [0, 1, 2], mode="lru", cache_mb=0.001)
    first = {i: {k: (v.clone() if torch.is_tensor(v) else v)
                 for k, v in lru[i].items()} for i in [0, 1, 2]}
    for _ in range(3):
        for i in [0, 1, 2]:
            assert_same_sample(first[i], lru[i])


def test_loader_kwargs_are_honoured():
    table = make_table([10])
    store = SampleStore(table, [0], mode="lru",
                        loader_kwargs={"global_scale": 11.0})
    direct = load_sample(table, 0, global_scale=11.0)
    assert_same_sample(direct, store[0])


# --- the memory bound -------------------------------------------------------

def test_lru_respects_its_byte_budget():
    table = make_table([200] * 20)
    one = sample_nbytes(load_sample(table, 0))
    budget_mb = (3.5 * one) / 1e6
    store = SampleStore(table, list(range(20)), mode="lru", cache_mb=budget_mb)
    for i in range(20):
        _ = store[i]
        assert store.resident_mb <= budget_mb * 1.001, "budget exceeded"
    assert store.n_evictions > 0


def test_residency_does_not_grow_with_split_size():
    """The whole point: a 10x bigger split must not cost 10x the RAM."""
    budget = 0.5
    table_big = make_table([100] * 100)
    eager = SampleStore(table_big, list(range(100)), mode="eager")

    lru = SampleStore(table_big, list(range(100)), mode="lru", cache_mb=budget)
    for i in range(100):
        _ = lru[i]

    small = SampleStore(make_table([100] * 10), list(range(10)),
                        mode="lru", cache_mb=budget)
    for i in range(10):
        _ = small[i]

    # Bounded by the budget, not by how many complexes the split holds -- and
    # strictly cheaper than materialising them all.
    assert lru.resident_mb <= budget * 1.001
    assert lru.resident_mb < eager.resident_mb
    # 10x the split, same ceiling: residency tracks the budget, not the split.
    assert lru.resident_mb <= max(small.resident_mb, budget) * 1.001


def test_oversized_sample_is_retained_rather_than_self_evicted():
    # One complex bigger than the whole budget must not evict itself, or every
    # access to it would miss and it would be decoded forever.
    table = make_table([500])
    store = SampleStore(table, [0], mode="lru", cache_mb=0.0001)
    store[0]
    assert len(store._cache) == 1
    n_after_first = store.n_decodes
    store[0]
    assert store.n_decodes == n_after_first, "cached entry was dropped"


def test_eager_decodes_everything_once_up_front():
    table = make_table([10] * 6)
    store = SampleStore(table, list(range(6)), mode="eager")
    assert store.n_decodes == 6
    for i in range(6):
        _ = store[i]
    assert store.n_decodes == 6, "eager mode re-decoded"


def test_lru_decodes_lazily():
    table = make_table([10] * 6)
    store = SampleStore(table, list(range(6)), mode="lru")
    assert store.n_decodes == 0, "lru decoded at construction"
    store[3]
    assert store.n_decodes == 1


def test_lru_hit_does_not_redecode():
    store = SampleStore(make_table([10] * 3), [0, 1, 2], mode="lru", cache_mb=100)
    store[1]
    store[1]
    store[1]
    assert store.n_decodes == 1


# --- decode-free metadata ---------------------------------------------------

def test_lengths_come_from_la_lb_without_decoding():
    table = make_table([10, 25, 60])
    store = SampleStore(table, [0, 1, 2], mode="lru")
    assert [store.n_res(i) for i in range(3)] == [10, 25, 60]
    assert store.length_index() == {0: {"n_res": 10}, 1: {"n_res": 25},
                                    2: {"n_res": 60}}
    assert store.n_decodes == 0, "metadata access forced a decode"


def test_length_index_agrees_with_the_decoded_sample():
    table = make_table([13, 41])
    store = SampleStore(table, [0, 1], mode="lru")
    for i in [0, 1]:
        assert store.n_res(i) == store[i]["n_res"]


def test_subset_is_bounded_and_deterministic():
    table = make_table([10] * 50)
    store = SampleStore(table, list(range(50)), mode="lru")
    a = store.subset(8, seed=1)
    assert len(a) == 8
    b = SampleStore(table, list(range(50)), mode="lru").subset(8, seed=1)
    assert [s["sample_id"] for s in a] == [s["sample_id"] for s in b]


def test_subset_larger_than_split_returns_everything():
    store = SampleStore(make_table([10] * 3), [0, 1, 2], mode="lru")
    assert len(store.subset(100)) == 3


# --- overlay (Stage 1 predictions) -----------------------------------------

def test_overlay_survives_eviction():
    table = make_table([100] * 8)
    store = SampleStore(table, list(range(8)), mode="lru", cache_mb=0.001)
    pred = torch.arange(3.0)
    store.set_overlay(0, "centroids_pred", pred)
    for i in range(8):          # force index 0 out of the cache
        _ = store[i]
    assert torch.equal(store[0]["centroids_pred"], pred)


def test_overlay_is_visible_on_an_already_cached_entry():
    store = SampleStore(make_table([10] * 2), [0, 1], mode="eager")
    s = store[0]
    store.set_overlay(0, "centroids_pred", torch.ones(2))
    assert "centroids_pred" in s
    assert torch.equal(store[0]["centroids_pred"], torch.ones(2))


# --- mapping interface ------------------------------------------------------

def test_membership_and_len_do_not_decode():
    store = SampleStore(make_table([10] * 4), [0, 2], mode="lru")
    assert 0 in store and 2 in store
    assert 1 not in store and 99 not in store
    assert len(store) == 2
    assert store.n_decodes == 0


def test_unknown_index_raises_keyerror():
    store = SampleStore(make_table([10] * 4), [0, 1], mode="lru")
    with pytest.raises(KeyError):
        store[3]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="eager"):
        SampleStore(make_table([10]), [0], mode="crop")


def test_merged_store_dispatches_to_both():
    table = make_table([10] * 6)
    a = SampleStore(table, [0, 1], mode="lru")
    b = SampleStore(table, [4, 5], mode="lru")
    merged = MergedSampleStore(a, b)
    assert len(merged) == 4
    assert 0 in merged and 5 in merged and 3 not in merged
    assert merged[5]["sample_id"] == "s0005"


# --- concurrency ------------------------------------------------------------

def test_concurrent_access_is_safe_and_consistent():
    table = make_table([50] * 12)
    store = SampleStore(table, list(range(12)), mode="lru", cache_mb=0.02)
    errors, ids = [], []
    lock = threading.Lock()

    def worker():
        try:
            for _ in range(20):
                for i in range(12):
                    s = store[i]
                    with lock:
                        ids.append((i, s["sample_id"]))
        except Exception as exc:      # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors}"
    assert all(sid == f"s{i:04d}" for i, sid in ids), "wrong sample returned"
    assert store.resident_mb <= 0.02 * 1.001


# --- column projection ------------------------------------------------------

def test_train_columns_cover_everything_load_sample_reads():
    """Guard: a new `table['...']` in load_sample must be added to
    TRAIN_COLUMNS, or the projected read will KeyError at runtime."""
    import re
    from pathlib import Path
    src = Path("src/tinyfold/training/data.py").read_text()
    body = src[src.index("def load_sample("):src.index("def collate_batch(")]
    referenced = set(re.findall(r"table\[['\"]([a-z_0-9]+)['\"]\]", body))
    assert referenced <= set(TRAIN_COLUMNS), (
        f"load_sample reads {sorted(referenced - set(TRAIN_COLUMNS))} which "
        "TRAIN_COLUMNS omits"
    )


def test_projected_table_is_enough_to_decode_a_sample():
    table = make_table([20])
    projected = table.select(TRAIN_COLUMNS)
    store = SampleStore(projected, [0], mode="lru")
    assert_same_sample(load_sample(table, 0), store[0])
