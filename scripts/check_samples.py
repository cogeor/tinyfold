#!/usr/bin/env python
"""Check if samples in the dataset are duplicates."""

import torch
import pyarrow.parquet as pq
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data/processed/samples.parquet")

print("Loading data...")
table = pq.read_table(data_path)

# Filter to small proteins
small_indices = [i for i in range(min(5000, len(table)))
                if len(table['atom_type'][i].as_py()) <= 1200]
print(f"Found {len(small_indices)} small proteins")

# Check first 10 samples
print("\n=== Sample details ===")
for i in range(10):
    idx = small_indices[i]
    seq = table['seq'][idx].as_py()
    sample_id = table['sample_id'][idx].as_py()
    n_atoms = len(table['atom_type'][idx].as_py())
    n_res = n_atoms // 4
    print(f"  {i}: idx={idx}, id={sample_id}, n_res={n_res}, seq[:10]={seq[:10]}")

# Check for duplicates in sequences
print("\n=== Checking for duplicate sequences ===")
seqs = {}
for i in range(10):
    idx = small_indices[i]
    seq = tuple(table['seq'][idx].as_py())
    if seq in seqs:
        print(f"  Sample {i} (idx={idx}) is DUPLICATE of sample {seqs[seq]}")
    else:
        seqs[seq] = i

# Check pairwise sequence similarity for samples 3-8
print("\n=== Pairwise sequence comparison for samples 3-8 ===")
for i in range(3, 9):
    for j in range(i+1, 9):
        idx_i = small_indices[i]
        idx_j = small_indices[j]
        seq_i = table['seq'][idx_i].as_py()
        seq_j = table['seq'][idx_j].as_py()

        if seq_i == seq_j:
            print(f"  {i} vs {j}: IDENTICAL sequences")
        else:
            # Check how many positions match
            min_len = min(len(seq_i), len(seq_j))
            matches = sum(1 for a, b in zip(seq_i[:min_len], seq_j[:min_len]) if a == b)
            print(f"  {i} vs {j}: {matches}/{min_len} matches ({100*matches/min_len:.1f}%)")
