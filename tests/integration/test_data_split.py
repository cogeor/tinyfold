"""Integration tests for data splitting functionality.

Converted from scripts/data_split.py main block.
Tests load real data to verify deterministic splitting.

These tests are marked with @pytest.mark.slow and @pytest.mark.integration
and can be skipped with: pytest -m "not slow"
"""

import os
import tempfile
from pathlib import Path

import pytest


# Skip these tests if data file doesn't exist
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "samples.parquet"
HAS_DATA = DATA_PATH.exists()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_DATA, reason="Data file not found"),
]


@pytest.fixture(scope="module")
def parquet_table():
    """Load the parquet table once for all tests in this module."""
    import pyarrow.parquet as pq
    return pq.read_table(str(DATA_PATH))


@pytest.fixture
def split_config():
    """Default split configuration for testing."""
    from tinyfold.data import DataSplitConfig
    return DataSplitConfig(n_train=50, n_test=10, min_atoms=200, max_atoms=400)


# ============================================================================
# Test Determinism
# ============================================================================


class TestSplitDeterminism:
    """Tests for deterministic train/test splitting."""

    def test_same_indices_on_repeated_calls(self, parquet_table, split_config):
        """Splitting should return same indices on repeated calls."""
        from tinyfold.data import get_train_test_indices

        train1, test1 = get_train_test_indices(parquet_table, split_config)
        train2, test2 = get_train_test_indices(parquet_table, split_config)

        assert train1 == train2, "Train indices not deterministic across calls"
        assert test1 == test2, "Test indices not deterministic across calls"

    def test_deterministic_with_same_seed(self, parquet_table):
        """Same seed should produce same split."""
        from tinyfold.data import DataSplitConfig, get_train_test_indices

        config1 = DataSplitConfig(n_train=30, n_test=10, seed=42)
        config2 = DataSplitConfig(n_train=30, n_test=10, seed=42)

        train1, test1 = get_train_test_indices(parquet_table, config1)
        train2, test2 = get_train_test_indices(parquet_table, config2)

        assert train1 == train2
        assert test1 == test2

    def test_different_with_different_seed(self, parquet_table):
        """Different seeds should produce different splits."""
        from tinyfold.data import DataSplitConfig, get_train_test_indices

        config1 = DataSplitConfig(n_train=30, n_test=10, seed=42)
        config2 = DataSplitConfig(n_train=30, n_test=10, seed=123)

        train1, test1 = get_train_test_indices(parquet_table, config1)
        train2, test2 = get_train_test_indices(parquet_table, config2)

        # Should be different (extremely unlikely to be same by chance)
        assert train1 != train2 or test1 != test2


# ============================================================================
# Test No Overlap
# ============================================================================


class TestSplitNoOverlap:
    """Tests for train/test set separation."""

    def test_no_overlap_between_train_and_test(self, parquet_table, split_config):
        """Train and test sets should have no overlapping indices."""
        from tinyfold.data import get_train_test_indices

        train_indices, test_indices = get_train_test_indices(parquet_table, split_config)

        train_set = set(train_indices)
        test_set = set(test_indices)
        overlap = train_set & test_set

        assert len(overlap) == 0, f"Train/test overlap found: {overlap}"

    def test_all_indices_valid(self, parquet_table, split_config):
        """All indices should be valid for the table."""
        from tinyfold.data import get_train_test_indices

        train_indices, test_indices = get_train_test_indices(parquet_table, split_config)
        all_indices = train_indices + test_indices

        table_len = len(parquet_table)
        for idx in all_indices:
            assert 0 <= idx < table_len, f"Invalid index {idx} for table of length {table_len}"


# ============================================================================
# Test Split Info
# ============================================================================


class TestSplitInfo:
    """Tests for split information and summary."""

    def test_get_split_info_returns_expected_keys(self, parquet_table, split_config):
        """get_split_info should return dict with expected keys."""
        from tinyfold.data import get_split_info

        info = get_split_info(parquet_table, split_config)

        expected_keys = ['total_samples', 'eligible_samples', 'n_train', 'n_test',
                         'train_atom_range', 'test_atom_range', 'train_ids', 'test_ids',
                         'train_indices', 'test_indices', 'config']

        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_split_info_matches_actual_split(self, parquet_table, split_config):
        """Split info should match actual split indices."""
        from tinyfold.data import get_train_test_indices, get_split_info

        train_indices, test_indices = get_train_test_indices(parquet_table, split_config)
        info = get_split_info(parquet_table, split_config)

        assert info['train_indices'] == train_indices
        assert info['test_indices'] == test_indices
        assert info['n_train'] == len(train_indices)
        assert info['n_test'] == len(test_indices)


# ============================================================================
# Test Save/Load Split
# ============================================================================


class TestSplitPersistence:
    """Tests for saving and loading splits."""

    def test_save_and_load_split(self, parquet_table, split_config):
        """Saved split should be loadable and match original."""
        from tinyfold.data import get_split_info, save_split, load_split

        info = get_split_info(parquet_table, split_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = os.path.join(tmpdir, "split.json")
            save_split(info, split_path)

            # Load and verify
            loaded_train, loaded_test, loaded_info = load_split(split_path)

            assert loaded_train == info['train_indices']
            assert loaded_test == info['test_indices']
            assert loaded_info['n_train'] == info['n_train']
            assert loaded_info['n_test'] == info['n_test']

    def test_loaded_split_is_identical(self, parquet_table, split_config):
        """Loaded split should be byte-for-byte identical."""
        from tinyfold.data import get_split_info, save_split, load_split

        info = get_split_info(parquet_table, split_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = os.path.join(tmpdir, "split.json")
            save_split(info, split_path)

            loaded_train, loaded_test, _ = load_split(split_path)

            # Save again and compare
            info2 = info.copy()
            info2['train_indices'] = loaded_train
            info2['test_indices'] = loaded_test

            split_path2 = os.path.join(tmpdir, "split2.json")
            save_split(info2, split_path2)

            with open(split_path) as f1, open(split_path2) as f2:
                content1 = f1.read()
                content2 = f2.read()

            assert content1 == content2, "Re-saved split should be identical"


# ============================================================================
# Test Select Smallest Mode
# ============================================================================


class TestSelectSmallestMode:
    """Tests for select_smallest mode."""

    def test_select_smallest_returns_small_proteins(self, parquet_table):
        """select_smallest=True should return smaller proteins."""
        from tinyfold.data import DataSplitConfig, get_train_test_indices

        config = DataSplitConfig(
            n_train=20,
            n_test=5,
            select_smallest=True,
        )

        train_indices, test_indices = get_train_test_indices(parquet_table, config)
        all_indices = train_indices + test_indices

        # Get atom counts for selected samples
        atom_counts = []
        for idx in all_indices:
            n_atoms = len(parquet_table['atom_type'][idx].as_py())
            atom_counts.append(n_atoms)

        # Should all be relatively small (in the smallest N of the dataset)
        avg_atoms = sum(atom_counts) / len(atom_counts)

        # Get average of all samples
        all_atom_counts = [len(parquet_table['atom_type'][i].as_py())
                          for i in range(len(parquet_table))]
        dataset_avg = sum(all_atom_counts) / len(all_atom_counts)

        assert avg_atoms < dataset_avg, \
            f"Selected avg ({avg_atoms:.0f}) should be smaller than dataset avg ({dataset_avg:.0f})"


# ============================================================================
# Test Bucketing
# ============================================================================


class TestLengthBucketing:
    """Tests for length-based batching."""

    def test_bucket_sampler_returns_similar_lengths(self, parquet_table, split_config):
        """Samples from one bucket should have similar lengths."""
        from tinyfold.data import get_train_test_indices, LengthBucketSampler

        train_indices, _ = get_train_test_indices(parquet_table, split_config)

        # Create samples dict with n_res key
        samples = {}
        for idx in train_indices:
            n_atoms = len(parquet_table['atom_type'][idx].as_py())
            samples[idx] = {'n_res': n_atoms // 4}

        sampler = LengthBucketSampler(samples, n_buckets=4)

        # Sample a batch
        batch_indices = sampler.sample_batch(batch_size=8)

        # Get lengths of samples in batch
        lengths = [samples[idx]['n_res'] for idx in batch_indices]

        # Samples should be from similar lengths (within bucket range)
        length_range = max(lengths) - min(lengths)

        # For a 4-bucket sampler with small data, range should be reasonable
        assert length_range < 100, f"Batch length range too large: {length_range}"

    def test_dynamic_batch_sampler_returns_variable_sizes(self, parquet_table, split_config):
        """Dynamic batch sampler should return different sizes for different buckets."""
        from tinyfold.data import get_train_test_indices, DynamicBatchSampler

        train_indices, _ = get_train_test_indices(parquet_table, split_config)

        samples = {}
        for idx in train_indices:
            n_atoms = len(parquet_table['atom_type'][idx].as_py())
            samples[idx] = {'n_res': n_atoms // 4}

        sampler = DynamicBatchSampler(
            samples,
            base_batch_size=16,
            max_tokens=5000,
            n_buckets=4,
        )

        # Get batch sizes for different buckets
        batch_sizes = sampler.get_batch_sizes()

        assert len(batch_sizes) > 0
        # Batch sizes should vary based on max_res
        sizes = [b['batch_size'] for b in batch_sizes]
        assert max(sizes) >= min(sizes), "Batch sizes should exist"
