#!/usr/bin/env python
"""Check the std normalization values for training samples."""

import torch
import pyarrow.parquet as pq
import numpy as np
import json
import sys
sys.path.insert(0, 'scripts')

# Load split
with open('outputs/overfit_rotation_fixed/split.json', 'r') as f:
    split = json.load(f)

train_indices = split['train_indices'][:5]

# Load data
table = pq.read_table('data/processed/samples.parquet')

print('Checking normalization std values:')
print('='*60)
for idx in train_indices:
    row = table.slice(idx, 1)
    pdb_id = row.column('pdb_id')[0].as_py()

    atom_coords = np.array(row.column('atom_coords')[0].as_py()).reshape(-1, 3)
    atom_to_res = np.array(row.column('atom_to_res')[0].as_py())
    n_res = atom_to_res.max() + 1

    # Compute centroids per residue
    centroids = np.zeros((n_res, 3))
    counts = np.zeros(n_res)
    for i, res_idx in enumerate(atom_to_res):
        centroids[res_idx] += atom_coords[i]
        counts[res_idx] += 1
    centroids = centroids / counts[:, None]

    # Center
    centroids = centroids - centroids.mean(axis=0)

    # Compute std (same as training)
    std = centroids.std()

    # Normalize
    centroids_norm = centroids / std

    print(f'{pdb_id}:')
    print(f'  std = {std:.2f}')
    print(f'  Raw X range: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}]')
    print(f'  Raw Y range: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]')
    print(f'  Raw Z range: [{centroids[:, 2].min():.1f}, {centroids[:, 2].max():.1f}]')
    print(f'  Norm X range: [{centroids_norm[:, 0].min():.2f}, {centroids_norm[:, 0].max():.2f}]')
    print(f'  Norm Y range: [{centroids_norm[:, 1].min():.2f}, {centroids_norm[:, 1].max():.2f}]')
    print(f'  Norm Z range: [{centroids_norm[:, 2].min():.2f}, {centroids_norm[:, 2].max():.2f}]')
    print()
