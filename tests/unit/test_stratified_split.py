"""Tests for the stratified test-split strategy in DataSplitConfig.

Pins these guarantees:
  - Random strategy is the default and matches the pre-fix behaviour.
  - Stratified strategy puts equal counts in each residue-size bin.
  - When a bin is undersized, the deficit is filled from the largest bin
    that has capacity (so |test| == n_test exactly).
  - train/test are disjoint and stable under the same seed.
"""

from collections import Counter

import pyarrow as pa
import pytest

from tinyfold.training.data_split import (
    DataSplitConfig,
    _residues_for_sample,
    get_train_test_indices,
)


def _make_table(la_lb_pairs: list[tuple[int, int]]) -> pa.Table:
    """Build a minimal parquet-shaped table with `n_atoms=LA+LB` style fields.

    The split logic only needs `sample_id`, `LA`, `LB`, and `atom_type`
    (for the atom-count filter). We pass `atom_type` as a list per row of
    length LA+LB so `len(atom_type[i].as_py()) == LA+LB`.
    """
    rows = []
    for i, (la, lb) in enumerate(la_lb_pairs):
        rows.append({
            "sample_id": f"sample_{i:04d}",
            "LA": la,
            "LB": lb,
            "atom_type": [0] * (la + lb),
        })
    return pa.Table.from_pylist(rows)


def test_random_strategy_is_default():
    cfg = DataSplitConfig(n_train=10, n_test=5)
    assert cfg.test_strategy == "random"
    assert cfg.test_size_bins == [0, 400, 600, 1000, 1500]


def test_random_strategy_unchanged_behaviour():
    """Random strategy is pre-fix behaviour: shuffle eligible, take prefix."""
    pairs = [(100, 100)] * 50 + [(300, 300)] * 30
    table = _make_table(pairs)
    cfg = DataSplitConfig(
        n_train=40, n_test=20,
        min_atoms=0, max_atoms=99999,
        test_strategy="random",
        seed=42,
    )
    train, test = get_train_test_indices(table, cfg)
    assert len(train) == 40
    assert len(test) == 20
    assert set(train).isdisjoint(set(test))


def test_stratified_equal_per_bin():
    """Stratified split puts equal share in each non-empty bin."""
    # 10 samples per bin x 5 bins = 50 samples. n_test=10 -> 2 per bin.
    bin_centres = [100, 250, 400, 600, 900]  # LA = LB = centre/2
    pairs = []
    for centre in bin_centres:
        half = centre // 2
        pairs.extend([(half, half)] * 10)
    table = _make_table(pairs)
    cfg = DataSplitConfig(
        n_train=30, n_test=10,
        min_atoms=0, max_atoms=99999,
        test_strategy="stratified",
        test_size_bins=[0, 200, 300, 500, 800],  # 5 bins
        seed=42,
    )
    train, test = get_train_test_indices(table, cfg)
    assert len(train) == 30
    assert len(test) == 10
    assert set(train).isdisjoint(set(test))

    # Count test samples per bin and assert equal distribution.
    test_bin_counts = Counter()
    bins = cfg.test_size_bins
    for row_idx in test:
        L = _residues_for_sample(table, row_idx)
        for b in range(len(bins) - 1, -1, -1):
            if L >= bins[b]:
                test_bin_counts[b] += 1
                break
    assert sum(test_bin_counts.values()) == 10
    for b in range(len(bins)):
        assert test_bin_counts[b] == 2, f"bin {b}: got {test_bin_counts[b]}"


def test_stratified_deficit_filled_from_largest_bin():
    """When a bin is too small, extras are pulled from the largest bin."""
    # bin 0 has only 1 sample; bins 1..4 have 10 each. n_test=10 means
    # target=2 per bin; bin 0 yields 1 -> deficit=1 -> fill from largest bin.
    pairs = [(50, 50)]  # 1 sample at L=100 (bin 0)
    bin_centres = [250, 400, 600, 900]
    for centre in bin_centres:
        half = centre // 2
        pairs.extend([(half, half)] * 10)
    table = _make_table(pairs)
    cfg = DataSplitConfig(
        n_train=30, n_test=10,
        min_atoms=0, max_atoms=99999,
        test_strategy="stratified",
        test_size_bins=[0, 200, 300, 500, 800],
        seed=42,
    )
    train, test = get_train_test_indices(table, cfg)
    assert len(test) == 10  # deficit filled
    # Bin 0 should have 1 (all of it), other bins should have 2 except one
    # bin gets an extra. Check the biggest bin (4) absorbed it.
    bin_counts = Counter()
    bins = cfg.test_size_bins
    for row_idx in test:
        L = _residues_for_sample(table, row_idx)
        for b in range(len(bins) - 1, -1, -1):
            if L >= bins[b]:
                bin_counts[b] += 1
                break
    assert bin_counts[0] == 1
    assert bin_counts[4] == 3  # biggest bin took the extra


def test_stratified_seed_reproducible():
    pairs = [(100, 100), (150, 150), (250, 250), (400, 400), (700, 700)] * 4
    table = _make_table(pairs)
    cfg = DataSplitConfig(
        n_train=10, n_test=5,
        min_atoms=0, max_atoms=99999,
        test_strategy="stratified",
        test_size_bins=[0, 200, 300, 500, 800],
        seed=123,
    )
    train1, test1 = get_train_test_indices(table, cfg)
    train2, test2 = get_train_test_indices(table, cfg)
    assert train1 == train2
    assert test1 == test2


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        DataSplitConfig(test_strategy="oracle")
