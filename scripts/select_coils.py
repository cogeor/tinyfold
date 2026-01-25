"""Select coiled-coil samples for experiment."""

import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw


def compute_linearity(coords):
    """Compute linearity score (0-100%)."""
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def main():
    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    n_samples = len(table)
    print(f"Total samples in dataset: {n_samples}")

    # Analyze all samples for linearity
    print("Analyzing samples for coiled-coil detection...")
    coiled_coils = []

    for idx in range(n_samples):
        sample = load_sample_raw(table, idx)
        centroids = sample['centroids'].numpy()
        chain_ids = sample['chain_ids'].numpy()

        # Overall linearity
        linearity = compute_linearity(centroids)

        # Per-chain linearity
        chain_a = chain_ids == 0
        chain_b = chain_ids == 1
        lin_a = compute_linearity(centroids[chain_a]) if chain_a.any() else 0
        lin_b = compute_linearity(centroids[chain_b]) if chain_b.any() else 0

        # Coiled-coil criteria: high overall + both chains linear
        if linearity > 70 and lin_a > 65 and lin_b > 65:
            coiled_coils.append({
                'idx': idx,
                'sample_id': sample['sample_id'],
                'n_res': sample['n_res'],
                'linearity': linearity,
                'lin_chain_a': lin_a,
                'lin_chain_b': lin_b,
            })

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{n_samples}...")

    print(f"\nFound {len(coiled_coils)} coiled-coil samples")

    # Sort by linearity
    coiled_coils.sort(key=lambda x: x['linearity'], reverse=True)

    # Select 100 samples: 80 train, 20 test
    # Ensure diversity by picking from different PDB entries
    selected = []
    seen_pdb = {}

    for cc in coiled_coils:
        pdb_id = cc['sample_id'].split('.')[0]
        # Allow max 5 variants per PDB
        if seen_pdb.get(pdb_id, 0) < 5:
            selected.append(cc)
            seen_pdb[pdb_id] = seen_pdb.get(pdb_id, 0) + 1
        if len(selected) >= 100:
            break

    print(f"Selected {len(selected)} diverse coiled-coil samples")
    print(f"From {len(seen_pdb)} unique PDB entries")

    # Split: 80 train, 20 test
    np.random.seed(42)
    indices = np.random.permutation(len(selected))
    train_indices = [selected[i]['idx'] for i in indices[:80]]
    test_indices = [selected[i]['idx'] for i in indices[80:]]

    # Save
    output = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'all_samples': selected,
        'stats': {
            'total_coiled_coils': len(coiled_coils),
            'selected': len(selected),
            'unique_pdbs': len(seen_pdb),
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            'linearity_range': [selected[-1]['linearity'], selected[0]['linearity']],
        }
    }

    output_path = project_root / "outputs/coil_experiment/coil_samples.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"\nStats:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    print(f"  Linearity range: {selected[-1]['linearity']:.1f}% - {selected[0]['linearity']:.1f}%")

    # Show sample distribution
    print(f"\nSample PDB distribution:")
    for pdb, count in sorted(seen_pdb.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pdb}: {count} samples")


if __name__ == "__main__":
    main()
