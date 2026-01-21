"""Analyze distribution of distances between residues in proteins.

Measures:
1. Sequential distances (i to i+1, i to i+2, etc.)
2. Distances by amino acid pair type
"""

import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold/scripts')

import torch
import numpy as np
from collections import defaultdict
import pyarrow.parquet as pq

# Amino acid mapping
AA_NAMES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def load_protein(table, idx):
    """Load a single protein and return residue centroids + AA types."""
    coords_flat = torch.tensor(table['atom_coords'][idx].as_py(), dtype=torch.float32)
    atom_types = table['atom_type'][idx].as_py()
    seq = table['seq'][idx].as_py()  # Residue types (one per residue)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4

    # Reshape to [L, 4, 3] and compute centroids
    coords = coords_flat.reshape(n_res, 4, 3)
    centroids = coords.mean(dim=1)  # [L, 3]

    # seq is already one entry per residue
    aa_types = seq

    return centroids, aa_types, n_res


def compute_sequential_distances(centroids):
    """Compute distances between sequential residues (i to i+k for k=1,2,3,...)."""
    L = centroids.shape[0]
    distances = {}

    for k in range(1, min(11, L)):  # i to i+1, i+2, ..., i+10
        # Distance from residue i to residue i+k
        diff = centroids[k:] - centroids[:-k]  # [L-k, 3]
        dist = diff.norm(dim=-1)  # [L-k]
        distances[k] = dist.numpy()

    return distances


def compute_pairwise_by_aa(centroids, aa_types, max_seq_sep=5):
    """Compute distances grouped by amino acid pair type.

    Only considers pairs within max_seq_sep sequence positions.
    """
    L = centroids.shape[0]
    pair_distances = defaultdict(list)

    for i in range(L):
        for j in range(i + 1, min(i + max_seq_sep + 1, L)):
            aa_i = aa_types[i]
            aa_j = aa_types[j]

            # Canonical ordering (smaller index first)
            if aa_i > aa_j:
                aa_i, aa_j = aa_j, aa_i

            dist = (centroids[i] - centroids[j]).norm().item()
            pair_key = (AA_NAMES[aa_i], AA_NAMES[aa_j])
            pair_distances[pair_key].append(dist)

    return pair_distances


def main():
    print("Loading data...")
    table = pq.read_table("C:/Users/costa/src/tinyfold/data/processed/samples.parquet")
    print(f"Total proteins: {len(table)}")

    # Sample a few proteins of different sizes
    n_samples = 50
    sample_indices = np.linspace(0, len(table) - 1, n_samples, dtype=int)

    print(f"\nAnalyzing {n_samples} proteins...")
    print("=" * 60)

    # Collect all sequential distances
    all_seq_distances = defaultdict(list)
    all_pair_distances = defaultdict(list)

    for idx in sample_indices:
        centroids, aa_types, n_res = load_protein(table, idx)

        # Sequential distances
        seq_dists = compute_sequential_distances(centroids)
        for k, dists in seq_dists.items():
            all_seq_distances[k].extend(dists)

        # Pair distances by AA type
        pair_dists = compute_pairwise_by_aa(centroids, aa_types, max_seq_sep=3)
        for pair, dists in pair_dists.items():
            all_pair_distances[pair].extend(dists)

    # Report sequential distances
    print("\n1. SEQUENTIAL DISTANCES (residue i to i+k)")
    print("-" * 60)
    print(f"{'k':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 60)

    for k in sorted(all_seq_distances.keys()):
        dists = np.array(all_seq_distances[k])
        print(f"{k:4d} {dists.mean():8.2f} {dists.std():8.2f} {dists.min():8.2f} {dists.max():8.2f} {len(dists):8d}")

    # Report pair distances by AA type (top 20 most common pairs)
    print("\n\n2. DISTANCES BY AMINO ACID PAIR (within 3 residues)")
    print("-" * 60)

    # Sort by count
    sorted_pairs = sorted(all_pair_distances.items(), key=lambda x: -len(x[1]))[:20]

    print(f"{'Pair':>10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 60)

    for (aa1, aa2), dists in sorted_pairs:
        dists = np.array(dists)
        pair_name = f"{aa1}-{aa2}"
        print(f"{pair_name:>10} {dists.mean():8.2f} {dists.std():8.2f} {dists.min():8.2f} {dists.max():8.2f} {len(dists):8d}")

    # Specific analysis: consecutive residue distances (k=1) by AA pair
    print("\n\n3. CONSECUTIVE DISTANCES (i to i+1) BY AMINO ACID PAIR")
    print("-" * 60)

    consec_by_pair = defaultdict(list)
    for idx in sample_indices:
        centroids, aa_types, n_res = load_protein(table, idx)
        for i in range(n_res - 1):
            aa_i = AA_NAMES[aa_types[i]]
            aa_j = AA_NAMES[aa_types[i + 1]]
            dist = (centroids[i] - centroids[i + 1]).norm().item()

            # Canonical order
            if aa_i > aa_j:
                pair = f"{aa_j}-{aa_i}"
            else:
                pair = f"{aa_i}-{aa_j}"
            consec_by_pair[pair].append(dist)

    # Sort by count
    sorted_consec = sorted(consec_by_pair.items(), key=lambda x: -len(x[1]))[:15]

    print(f"{'Pair':>10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 60)

    for pair, dists in sorted_consec:
        dists = np.array(dists)
        print(f"{pair:>10} {dists.mean():8.2f} {dists.std():8.2f} {dists.min():8.2f} {dists.max():8.2f} {len(dists):8d}")

    # Summary statistics
    print("\n\n4. SUMMARY")
    print("-" * 60)
    all_consec = np.array(all_seq_distances[1])
    print(f"Consecutive residue distance (CA-CA ~3.8A expected):")
    print(f"  Centroid distance: {all_consec.mean():.2f} +/- {all_consec.std():.2f} A")
    print(f"  Range: {all_consec.min():.2f} - {all_consec.max():.2f} A")

    # Check for chain breaks (large gaps)
    breaks = (all_consec > 6.0).sum()
    print(f"  Potential chain breaks (>6A): {breaks} ({100*breaks/len(all_consec):.1f}%)")


if __name__ == "__main__":
    main()
