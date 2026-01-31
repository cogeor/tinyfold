"""Deterministic train/test split for TinyFold.

Selection Logic:
1. Filter samples by atom count (default: 200-400 atoms for medium complexity)
   OR select N smallest proteins if select_smallest=True
2. Sort samples by sample_id (alphabetical, deterministic)
3. Shuffle with fixed seed (42) for reproducibility
4. Split: first n_train for training, next n_test for testing

This guarantees that n_train=100 always selects the SAME 100 training samples
and the SAME test samples, regardless of when/where you run.

Usage:
    from data_split import get_train_test_indices, DataSplitConfig

    config = DataSplitConfig(n_train=100, n_test=20)
    train_idx, test_idx = get_train_test_indices(table, config)

    # Select smallest proteins instead of range:
    config = DataSplitConfig(n_train=20000, n_test=500, select_smallest=True)
    train_idx, test_idx = get_train_test_indices(table, config)

    # Save split for Stage 2 reuse:
    save_split(info, "outputs/stage1/split.json")

    # Load in Stage 2:
    train_idx, test_idx = load_split("outputs/stage1/split.json")
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional
import pyarrow as pa


@dataclass
class DataSplitConfig:
    """Configuration for deterministic train/test split."""

    # Number of training samples
    n_train: int = 100

    # Number of test samples (default: 20% of n_train, min 10)
    n_test: Optional[int] = None

    # Atom count range for filtering (ignored if select_smallest=True)
    min_atoms: int = 200
    max_atoms: int = 400

    # If True, select N smallest proteins instead of filtering by atom range
    # Total proteins selected = n_train + n_test
    select_smallest: bool = False

    # Random seed for shuffling (fixed for reproducibility)
    seed: int = 42

    def __post_init__(self):
        if self.n_test is None:
            self.n_test = max(10, self.n_train // 5)


def get_eligible_samples(table: pa.Table, config: DataSplitConfig) -> list[tuple[int, str, int]]:
    """Get samples that meet the atom count criteria.

    Returns:
        List of (original_index, sample_id, n_atoms) tuples, sorted by sample_id
        (or by n_atoms if select_smallest=True)
    """
    if config.select_smallest:
        # Collect all samples with their atom counts
        all_samples = []
        for i in range(len(table)):
            n_atoms = len(table['atom_type'][i].as_py())
            sample_id = table['sample_id'][i].as_py()
            all_samples.append((i, sample_id, n_atoms))

        # Sort by atom count (smallest first), then by sample_id for tie-breaking
        all_samples.sort(key=lambda x: (x[2], x[1]))

        # Take N smallest where N = n_train + n_test
        total_needed = config.n_train + config.n_test
        eligible = all_samples[:total_needed]

        # Re-sort by sample_id for deterministic shuffling
        eligible.sort(key=lambda x: x[1])
        return eligible
    else:
        # Original behavior: filter by atom range
        eligible = []
        for i in range(len(table)):
            n_atoms = len(table['atom_type'][i].as_py())
            if config.min_atoms <= n_atoms <= config.max_atoms:
                sample_id = table['sample_id'][i].as_py()
                eligible.append((i, sample_id, n_atoms))

        # Sort by sample_id for deterministic ordering
        eligible.sort(key=lambda x: x[1])
        return eligible


def get_train_test_indices(
    table: pa.Table,
    config: DataSplitConfig,
) -> tuple[list[int], list[int]]:
    """Get deterministic train and test indices.

    Args:
        table: PyArrow table with samples
        config: Split configuration

    Returns:
        (train_indices, test_indices) - indices into the original table

    Raises:
        ValueError: If not enough samples available
    """
    # Get eligible samples sorted by sample_id
    eligible = get_eligible_samples(table, config)

    total_needed = config.n_train + config.n_test
    if len(eligible) < total_needed:
        if config.select_smallest:
            raise ValueError(
                f"Not enough samples: need {total_needed} (train={config.n_train}, test={config.n_test}), "
                f"but only {len(eligible)} samples in dataset"
            )
        else:
            raise ValueError(
                f"Not enough samples: need {total_needed} (train={config.n_train}, test={config.n_test}), "
                f"but only {len(eligible)} samples have {config.min_atoms}-{config.max_atoms} atoms"
            )

    # Shuffle with fixed seed
    rng = random.Random(config.seed)
    samples = eligible.copy()
    rng.shuffle(samples)

    # Split: first n_train for training, next n_test for testing
    train_samples = samples[:config.n_train]
    test_samples = samples[config.n_train:config.n_train + config.n_test]

    # Extract just the original indices (first element of tuple)
    train_indices = [s[0] for s in train_samples]
    test_indices = [s[0] for s in test_samples]

    return train_indices, test_indices


def get_split_info(table: pa.Table, config: DataSplitConfig) -> dict:
    """Get detailed information about the split for logging/debugging.

    Returns:
        Dictionary with split statistics, sample IDs, and indices
    """
    eligible = get_eligible_samples(table, config)
    train_idx, test_idx = get_train_test_indices(table, config)

    # Get sample IDs for train and test
    train_ids = [table['sample_id'][i].as_py() for i in train_idx]
    test_ids = [table['sample_id'][i].as_py() for i in test_idx]

    # Get atom counts
    train_atoms = [len(table['atom_type'][i].as_py()) for i in train_idx]
    test_atoms = [len(table['atom_type'][i].as_py()) for i in test_idx]

    return {
        'total_samples': len(table),
        'eligible_samples': len(eligible),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'train_atom_range': (min(train_atoms), max(train_atoms)),
        'test_atom_range': (min(test_atoms), max(test_atoms)),
        'train_ids': train_ids,
        'test_ids': test_ids,
        'train_indices': train_idx,
        'test_indices': test_idx,
        'config': config,
    }


def save_split(info: dict, path: str):
    """Save train/test split to JSON for Stage 2 reuse.

    Args:
        info: Split info dict from get_split_info()
        path: Path to save JSON file
    """
    # Convert config to dict for JSON serialization
    save_data = {
        'train_indices': info['train_indices'],
        'test_indices': info['test_indices'],
        'train_ids': info['train_ids'],
        'test_ids': info['test_ids'],
        'n_train': info['n_train'],
        'n_test': info['n_test'],
        'train_atom_range': list(info['train_atom_range']),
        'test_atom_range': list(info['test_atom_range']),
        'config': asdict(info['config']),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Saved split to {path}")


def load_split(path: str) -> tuple[list[int], list[int], dict]:
    """Load train/test split from JSON.

    Args:
        path: Path to JSON file saved by save_split()

    Returns:
        (train_indices, test_indices, split_info)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    print(f"Loaded split from {path}")
    print(f"  Train: {data['n_train']} samples, atoms {data['train_atom_range'][0]}-{data['train_atom_range'][1]}")
    print(f"  Test: {data['n_test']} samples, atoms {data['test_atom_range'][0]}-{data['test_atom_range'][1]}")

    return data['train_indices'], data['test_indices'], data


def print_split_summary(info: dict):
    """Print a human-readable summary of the split."""
    config = info['config']
    print("=" * 60)
    print("Data Split Summary")
    print("=" * 60)
    print(f"Total samples in dataset: {info['total_samples']}")
    if config.select_smallest:
        print(f"Selection: {info['eligible_samples']} smallest proteins")
    else:
        print(f"Eligible samples ({config.min_atoms}-{config.max_atoms} atoms): {info['eligible_samples']}")
    print()
    print(f"Training samples: {info['n_train']}")
    print(f"  Atom range: {info['train_atom_range'][0]}-{info['train_atom_range'][1]}")
    print(f"  First 5 IDs: {info['train_ids'][:5]}")
    print()
    print(f"Test samples: {info['n_test']}")
    print(f"  Atom range: {info['test_atom_range'][0]}-{info['test_atom_range'][1]}")
    print(f"  First 5 IDs: {info['test_ids'][:5]}")
    print()
    print(f"Random seed: {config.seed}")
    print("=" * 60)


# =============================================================================
# Length Bucketing for Efficient Batching
# =============================================================================

class LengthBucketSampler:
    """Sample batches from similar-length proteins to minimize padding waste.

    Proteins are grouped into buckets by residue count. Each batch is sampled
    entirely from one bucket, ensuring similar sequence lengths and minimal
    padding overhead.

    Usage:
        sampler = LengthBucketSampler(train_samples, n_buckets=8)
        batch_indices = sampler.sample_batch(batch_size=64)
    """

    def __init__(
        self,
        samples: dict,  # {idx: sample_dict} with 'n_res' key
        n_buckets: int = 8,
        seed: int = 42,
    ):
        """Initialize buckets based on sample lengths.

        Args:
            samples: Dict mapping index to sample dict (must have 'n_res' key)
            n_buckets: Number of length buckets to create
            seed: Random seed for reproducibility
        """
        self.samples = samples
        self.n_buckets = n_buckets
        self.rng = random.Random(seed)

        # Get lengths for all samples
        lengths = [(idx, s['n_res']) for idx, s in samples.items()]
        lengths.sort(key=lambda x: x[1])

        # Create buckets with roughly equal sample counts
        self.buckets = []
        bucket_size = len(lengths) // n_buckets

        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < n_buckets - 1 else len(lengths)
            bucket_indices = [idx for idx, _ in lengths[start:end]]
            if bucket_indices:
                min_len = lengths[start][1]
                max_len = lengths[end - 1][1]
                self.buckets.append({
                    'indices': bucket_indices,
                    'min_res': min_len,
                    'max_res': max_len,
                })

        # Compute bucket weights proportional to size
        total = sum(len(b['indices']) for b in self.buckets)
        self.bucket_weights = [len(b['indices']) / total for b in self.buckets]

    def sample_batch(self, batch_size: int) -> list[int]:
        """Sample a batch of indices from a single bucket.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of sample indices (all from same length bucket)
        """
        # Choose bucket weighted by size
        bucket = self.rng.choices(self.buckets, weights=self.bucket_weights, k=1)[0]

        # Sample from bucket (with replacement if needed)
        if len(bucket['indices']) >= batch_size:
            return self.rng.sample(bucket['indices'], batch_size)
        else:
            return self.rng.choices(bucket['indices'], k=batch_size)

    def get_bucket_stats(self) -> list[dict]:
        """Return statistics about each bucket."""
        return [
            {
                'bucket': i,
                'count': len(b['indices']),
                'min_res': b['min_res'],
                'max_res': b['max_res'],
            }
            for i, b in enumerate(self.buckets)
        ]


class DynamicBatchSampler:
    """Sample batches with size adjusted based on sequence length.

    Uses larger batches for small proteins and smaller batches for large
    proteins to maintain consistent GPU memory usage.

    Usage:
        sampler = DynamicBatchSampler(train_samples, base_batch_size=64, max_tokens=30000)
        batch_indices = sampler.sample_batch()
    """

    def __init__(
        self,
        samples: dict,  # {idx: sample_dict} with 'n_res' key
        base_batch_size: int = 64,
        max_tokens: int = 30000,  # Max total tokens (batch_size * max_res)
        n_buckets: int = 8,
        seed: int = 42,
    ):
        """Initialize with length bucketing and dynamic sizing.

        Args:
            samples: Dict mapping index to sample dict (must have 'n_res' key)
            base_batch_size: Default batch size for medium-length proteins
            max_tokens: Maximum total tokens per batch (batch_size * max_seq_len)
            n_buckets: Number of length buckets
            seed: Random seed
        """
        self.samples = samples
        self.base_batch_size = base_batch_size
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)

        # Create length buckets
        self.bucket_sampler = LengthBucketSampler(samples, n_buckets, seed)

    def _compute_batch_size(self, max_res: int) -> int:
        """Compute batch size based on max sequence length in bucket."""
        # batch_size * max_res <= max_tokens
        dynamic_size = self.max_tokens // max_res
        # Clamp to reasonable range
        return max(8, min(dynamic_size, self.base_batch_size * 2))

    def sample_batch(self) -> tuple[list[int], int]:
        """Sample a batch with dynamically adjusted size.

        Returns:
            (batch_indices, actual_batch_size)
        """
        # Choose bucket
        bucket = self.rng.choices(
            self.bucket_sampler.buckets,
            weights=self.bucket_sampler.bucket_weights,
            k=1
        )[0]

        # Compute batch size for this bucket
        batch_size = self._compute_batch_size(bucket['max_res'])

        # Sample from bucket
        if len(bucket['indices']) >= batch_size:
            indices = self.rng.sample(bucket['indices'], batch_size)
        else:
            indices = self.rng.choices(bucket['indices'], k=batch_size)

        return indices, batch_size

    def get_batch_sizes(self) -> list[dict]:
        """Return batch sizes for each bucket."""
        return [
            {
                'bucket': i,
                'max_res': b['max_res'],
                'batch_size': self._compute_batch_size(b['max_res']),
                'tokens': self._compute_batch_size(b['max_res']) * b['max_res'],
            }
            for i, b in enumerate(self.bucket_sampler.buckets)
        ]


if __name__ == "__main__":
    # Demo: verify determinism
    import pyarrow.parquet as pq
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data/processed/samples.parquet")
    table = pq.read_table(data_path)

    # Use 80 train, 14 test to fit in 94 available medium samples
    print("Testing determinism with n_train=80, n_test=14...")
    print()

    config = DataSplitConfig(n_train=80, n_test=14)

    # Run twice to verify same results
    train1, test1 = get_train_test_indices(table, config)
    train2, test2 = get_train_test_indices(table, config)

    assert train1 == train2, "Train indices not deterministic!"
    assert test1 == test2, "Test indices not deterministic!"
    print("[OK] Determinism verified: same indices on repeated calls")
    print()

    # Print summary
    info = get_split_info(table, config)
    print_split_summary(info)

    # Verify no overlap
    train_set = set(train1)
    test_set = set(test1)
    overlap = train_set & test_set
    assert len(overlap) == 0, f"Train/test overlap: {overlap}"
    print("[OK] No overlap between train and test sets")
