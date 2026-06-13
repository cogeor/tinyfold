"""Data loading and preprocessing modules.

Provides:
- PPIDataset: PyTorch dataset for protein-protein interactions
- Data splitting utilities for deterministic train/test splits
- Length bucketing for efficient batching
"""

from tinyfold.data.datasets.ppi_dataset import PPIDataset
from tinyfold.data.collate import collate_ppi
# Canonical split implementation lives in tinyfold.training.data_split (a strict
# superset of the old data/split.py: adds stratified test sampling + size bins,
# random-path indices are byte-identical). Re-exported here so the public
# `tinyfold.data` API is unchanged.
from tinyfold.training.data_split import (
    DataSplitConfig,
    get_train_test_indices,
    get_split_info,
    save_split,
    load_split,
    print_split_summary,
    LengthBucketSampler,
    DynamicBatchSampler,
)

__all__ = [
    # Dataset
    "PPIDataset",
    "collate_ppi",
    # Splitting
    "DataSplitConfig",
    "get_train_test_indices",
    "get_split_info",
    "save_split",
    "load_split",
    "print_split_summary",
    # Batching
    "LengthBucketSampler",
    "DynamicBatchSampler",
]
