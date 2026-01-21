"""Tests for length bucketing and dynamic batch sampling."""

import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold/scripts')

import pytest
import random
from data_split import LengthBucketSampler, DynamicBatchSampler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_samples():
    """Create mock samples with varying residue counts."""
    random.seed(42)
    samples = {}
    # Create 1000 samples with n_res from 80 to 450
    for i in range(1000):
        n_res = random.randint(80, 450)
        samples[i] = {'n_res': n_res, 'sample_id': f'sample_{i}'}
    return samples


@pytest.fixture
def mock_samples_skewed():
    """Create samples with skewed distribution (many small, few large)."""
    random.seed(42)
    samples = {}
    idx = 0
    # 800 small proteins (80-150 residues)
    for _ in range(800):
        samples[idx] = {'n_res': random.randint(80, 150), 'sample_id': f'sample_{idx}'}
        idx += 1
    # 150 medium proteins (200-300 residues)
    for _ in range(150):
        samples[idx] = {'n_res': random.randint(200, 300), 'sample_id': f'sample_{idx}'}
        idx += 1
    # 50 large proteins (400-450 residues)
    for _ in range(50):
        samples[idx] = {'n_res': random.randint(400, 450), 'sample_id': f'sample_{idx}'}
        idx += 1
    return samples


# =============================================================================
# LengthBucketSampler Tests
# =============================================================================

class TestLengthBucketSampler:
    """Tests for LengthBucketSampler."""

    def test_buckets_created(self, mock_samples):
        """Test that buckets are created correctly."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=8)

        assert len(sampler.buckets) == 8
        assert sum(len(b['indices']) for b in sampler.buckets) == len(mock_samples)

    def test_buckets_sorted_by_length(self, mock_samples):
        """Test that buckets are sorted by sequence length."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=8)

        for i in range(len(sampler.buckets) - 1):
            assert sampler.buckets[i]['max_res'] <= sampler.buckets[i + 1]['min_res']

    def test_bucket_stats(self, mock_samples):
        """Test bucket statistics."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=4)
        stats = sampler.get_bucket_stats()

        assert len(stats) == 4
        for stat in stats:
            assert 'bucket' in stat
            assert 'count' in stat
            assert 'min_res' in stat
            assert 'max_res' in stat
            assert stat['min_res'] <= stat['max_res']

    def test_sample_batch_returns_correct_size(self, mock_samples):
        """Test that sample_batch returns correct number of samples."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=8)

        batch = sampler.sample_batch(batch_size=64)
        assert len(batch) == 64

        batch = sampler.sample_batch(batch_size=32)
        assert len(batch) == 32

    def test_batch_samples_from_same_bucket(self, mock_samples):
        """Test that batch samples come from similar-length proteins."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=8)

        # Sample many batches and check length variance
        for _ in range(20):
            batch = sampler.sample_batch(batch_size=64)
            lengths = [mock_samples[idx]['n_res'] for idx in batch]

            # Within a bucket, max variance should be limited
            # (bucket range is roughly (450-80)/8 = ~46 residues)
            length_range = max(lengths) - min(lengths)
            assert length_range < 100, f"Batch has too much length variance: {length_range}"

    def test_deterministic_with_seed(self, mock_samples):
        """Test that same seed produces same results."""
        sampler1 = LengthBucketSampler(mock_samples, n_buckets=8, seed=42)
        sampler2 = LengthBucketSampler(mock_samples, n_buckets=8, seed=42)

        batch1 = sampler1.sample_batch(64)
        batch2 = sampler2.sample_batch(64)

        assert batch1 == batch2

    def test_different_seeds_different_results(self, mock_samples):
        """Test that different seeds produce different results."""
        sampler1 = LengthBucketSampler(mock_samples, n_buckets=8, seed=42)
        sampler2 = LengthBucketSampler(mock_samples, n_buckets=8, seed=123)

        batch1 = sampler1.sample_batch(64)
        batch2 = sampler2.sample_batch(64)

        assert batch1 != batch2

    def test_handles_small_buckets(self):
        """Test sampling from buckets smaller than batch size."""
        # Create tiny dataset
        samples = {i: {'n_res': 100 + i} for i in range(10)}
        sampler = LengthBucketSampler(samples, n_buckets=4)

        # Should still work with replacement
        batch = sampler.sample_batch(batch_size=64)
        assert len(batch) == 64


# =============================================================================
# DynamicBatchSampler Tests
# =============================================================================

class TestDynamicBatchSampler:
    """Tests for DynamicBatchSampler."""

    def test_creation(self, mock_samples):
        """Test sampler creation."""
        sampler = DynamicBatchSampler(
            mock_samples,
            base_batch_size=64,
            max_tokens=30000,
            n_buckets=8,
        )
        assert sampler is not None
        assert sampler.base_batch_size == 64
        assert sampler.max_tokens == 30000

    def test_dynamic_batch_sizes(self, mock_samples):
        """Test that batch sizes vary by bucket."""
        sampler = DynamicBatchSampler(
            mock_samples,
            base_batch_size=64,
            max_tokens=20000,
            n_buckets=8,
        )

        batch_sizes = sampler.get_batch_sizes()

        # Larger proteins should have smaller batch sizes
        for i in range(len(batch_sizes) - 1):
            if batch_sizes[i]['max_res'] < batch_sizes[i + 1]['max_res']:
                assert batch_sizes[i]['batch_size'] >= batch_sizes[i + 1]['batch_size']

    def test_sample_batch_returns_indices_and_size(self, mock_samples):
        """Test that sample_batch returns both indices and size."""
        sampler = DynamicBatchSampler(mock_samples, max_tokens=20000)

        indices, batch_size = sampler.sample_batch()

        assert isinstance(indices, list)
        assert isinstance(batch_size, int)
        assert len(indices) == batch_size

    def test_token_budget_respected(self, mock_samples):
        """Test that batches stay within token budget."""
        max_tokens = 15000
        sampler = DynamicBatchSampler(mock_samples, max_tokens=max_tokens, n_buckets=8)

        for _ in range(50):
            indices, batch_size = sampler.sample_batch()
            max_res = max(mock_samples[idx]['n_res'] for idx in indices)
            total_tokens = batch_size * max_res

            # Allow some slack (2x) since we use bucket max_res for sizing
            assert total_tokens <= max_tokens * 2, \
                f"Token budget exceeded: {total_tokens} > {max_tokens * 2}"

    def test_minimum_batch_size(self, mock_samples):
        """Test that batch size doesn't go below minimum."""
        sampler = DynamicBatchSampler(
            mock_samples,
            max_tokens=1000,  # Very restrictive
            n_buckets=8,
        )

        for _ in range(20):
            indices, batch_size = sampler.sample_batch()
            assert batch_size >= 8, f"Batch size too small: {batch_size}"

    def test_maximum_batch_size(self, mock_samples):
        """Test that batch size doesn't exceed maximum."""
        base_batch_size = 64
        sampler = DynamicBatchSampler(
            mock_samples,
            base_batch_size=base_batch_size,
            max_tokens=100000,  # Very permissive
            n_buckets=8,
        )

        for _ in range(20):
            indices, batch_size = sampler.sample_batch()
            assert batch_size <= base_batch_size * 2, \
                f"Batch size too large: {batch_size}"

    def test_skewed_distribution(self, mock_samples_skewed):
        """Test handling of skewed sample distribution."""
        sampler = DynamicBatchSampler(
            mock_samples_skewed,
            max_tokens=20000,
            n_buckets=8,
        )

        # Should work without errors
        batch_sizes = sampler.get_batch_sizes()
        assert len(batch_sizes) == 8

        # Sample several batches
        for _ in range(20):
            indices, batch_size = sampler.sample_batch()
            assert len(indices) == batch_size


# =============================================================================
# Integration Tests
# =============================================================================

class TestBucketingIntegration:
    """Integration tests for bucketing with real-world scenarios."""

    def test_padding_efficiency_improvement(self, mock_samples):
        """Test that bucketing reduces padding waste."""
        # Without bucketing: random sampling
        random.seed(42)
        random_batches = []
        for _ in range(100):
            indices = random.sample(list(mock_samples.keys()), 64)
            lengths = [mock_samples[idx]['n_res'] for idx in indices]
            max_len = max(lengths)
            total_actual = sum(lengths)
            total_padded = max_len * 64
            efficiency = total_actual / total_padded
            random_batches.append(efficiency)

        random_efficiency = sum(random_batches) / len(random_batches)

        # With bucketing
        sampler = LengthBucketSampler(mock_samples, n_buckets=8, seed=42)
        bucket_batches = []
        for _ in range(100):
            indices = sampler.sample_batch(64)
            lengths = [mock_samples[idx]['n_res'] for idx in indices]
            max_len = max(lengths)
            total_actual = sum(lengths)
            total_padded = max_len * 64
            efficiency = total_actual / total_padded
            bucket_batches.append(efficiency)

        bucket_efficiency = sum(bucket_batches) / len(bucket_batches)

        # Bucketing should improve efficiency
        assert bucket_efficiency > random_efficiency, \
            f"Bucketing ({bucket_efficiency:.2%}) should be better than random ({random_efficiency:.2%})"

        # Should improve by at least 5%
        improvement = bucket_efficiency - random_efficiency
        assert improvement > 0.05, f"Expected >5% improvement, got {improvement:.2%}"

    def test_all_samples_reachable(self, mock_samples):
        """Test that all samples can be reached through sampling."""
        sampler = LengthBucketSampler(mock_samples, n_buckets=8, seed=42)

        # Track which samples we've seen
        seen = set()
        for _ in range(1000):
            batch = sampler.sample_batch(64)
            seen.update(batch)

        # With 1000 batches of 64, we should see most samples
        coverage = len(seen) / len(mock_samples)
        assert coverage > 0.9, f"Only {coverage:.1%} of samples reachable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
