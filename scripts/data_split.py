"""Deterministic train/test split for TinyFold.

Selection Logic:
1. Filter samples by atom count (default: 200-400 atoms for medium complexity)
2. Sort samples by sample_id (alphabetical, deterministic)
3. Shuffle with fixed seed (42) for reproducibility
4. Split: first n_train for training, next n_test for testing

This guarantees that n_train=100 always selects the SAME 100 training samples
and the SAME test samples, regardless of when/where you run.

Usage:
    from data_split import get_train_test_indices, DataSplitConfig

    config = DataSplitConfig(n_train=100, n_test=20)
    train_idx, test_idx = get_train_test_indices(table, config)
"""

import random
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa


@dataclass
class DataSplitConfig:
    """Configuration for deterministic train/test split."""

    # Number of training samples
    n_train: int = 100

    # Number of test samples (default: 20% of n_train, min 10)
    n_test: Optional[int] = None

    # Atom count range for filtering
    min_atoms: int = 200
    max_atoms: int = 400

    # Random seed for shuffling (fixed for reproducibility)
    seed: int = 42

    def __post_init__(self):
        if self.n_test is None:
            self.n_test = max(10, self.n_train // 5)


def get_eligible_samples(table: pa.Table, config: DataSplitConfig) -> list[tuple[int, str]]:
    """Get samples that meet the atom count criteria.

    Returns:
        List of (original_index, sample_id) tuples, sorted by sample_id
    """
    eligible = []
    for i in range(len(table)):
        n_atoms = len(table['atom_type'][i].as_py())
        if config.min_atoms <= n_atoms <= config.max_atoms:
            sample_id = table['sample_id'][i].as_py()
            eligible.append((i, sample_id))

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
        raise ValueError(
            f"Not enough samples: need {total_needed} (train={config.n_train}, test={config.n_test}), "
            f"but only {len(eligible)} samples have {config.min_atoms}-{config.max_atoms} atoms"
        )

    # Shuffle with fixed seed
    rng = random.Random(config.seed)
    indices_and_ids = eligible.copy()
    rng.shuffle(indices_and_ids)

    # Split: first n_train for training, next n_test for testing
    train_pairs = indices_and_ids[:config.n_train]
    test_pairs = indices_and_ids[config.n_train:config.n_train + config.n_test]

    # Extract just the original indices
    train_indices = [idx for idx, _ in train_pairs]
    test_indices = [idx for idx, _ in test_pairs]

    return train_indices, test_indices


def get_split_info(table: pa.Table, config: DataSplitConfig) -> dict:
    """Get detailed information about the split for logging/debugging.

    Returns:
        Dictionary with split statistics and sample IDs
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
        'config': config,
    }


def print_split_summary(info: dict):
    """Print a human-readable summary of the split."""
    print("=" * 60)
    print("Data Split Summary")
    print("=" * 60)
    print(f"Total samples in dataset: {info['total_samples']}")
    print(f"Eligible samples ({info['config'].min_atoms}-{info['config'].max_atoms} atoms): {info['eligible_samples']}")
    print()
    print(f"Training samples: {info['n_train']}")
    print(f"  Atom range: {info['train_atom_range'][0]}-{info['train_atom_range'][1]}")
    print(f"  First 5 IDs: {info['train_ids'][:5]}")
    print()
    print(f"Test samples: {info['n_test']}")
    print(f"  Atom range: {info['test_atom_range'][0]}-{info['test_atom_range'][1]}")
    print(f"  First 5 IDs: {info['test_ids'][:5]}")
    print()
    print(f"Random seed: {info['config'].seed}")
    print("=" * 60)


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
