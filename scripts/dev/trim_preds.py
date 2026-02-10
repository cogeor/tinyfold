#!/usr/bin/env python
"""Trim predictions.json to first 100 train + 100 test samples."""

import json
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import pyarrow.parquet as pq
from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices

# Load predictions
with open("web/predictions.json") as f:
    preds = json.load(f)

print(f"Original: {len(preds)} predictions")

# Get train/test split to know which are which
table = pq.read_table("data/processed/samples.parquet")
config = DataSplitConfig(n_train=5000, n_test=1000, min_atoms=100, max_atoms=1600)
train_indices, test_indices = get_train_test_indices(table, config)

# Get sample IDs
train_ids = [table['sample_id'][i].as_py() for i in train_indices]
test_ids = [table['sample_id'][i].as_py() for i in test_indices]

# Filter to first 100 of each that have predictions
train_with_preds = [sid for sid in train_ids if sid in preds][:100]
test_with_preds = [sid for sid in test_ids if sid in preds][:100]

keep_ids = set(train_with_preds + test_with_preds)

# Filter predictions
trimmed = {k: v for k, v in preds.items() if k in keep_ids}

print(f"Trimmed: {len(trimmed)} predictions")
print(f"  Train: {len([k for k in trimmed if k in train_ids])}")
print(f"  Test: {len([k for k in trimmed if k in test_ids])}")

# Save
with open("web/predictions.json", "w") as f:
    json.dump(trimmed, f)

print("Saved trimmed predictions.json")
