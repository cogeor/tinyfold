"""Plot coiled-coil examples and estimate count in dataset."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw


def compute_linearity(coords):
    """Compute linearity score (0-100%). Higher = more linear/extended."""
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    # Ratio of first singular value to sum (how much variance along principal axis)
    return S[0] / (S.sum() + 1e-8) * 100


def compute_aspect_ratio(coords):
    """Compute aspect ratio (length / width)."""
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    if S[1] < 1e-8:
        return float('inf')
    return S[0] / S[1]


def plot_structure(coords, chain_ids, title, ax, std=1.0):
    """Plot a single structure."""
    coords_A = coords * std  # Convert to Angstroms

    chain_a = chain_ids == 0
    chain_b = chain_ids == 1

    if chain_a.any():
        ax.scatter(coords_A[chain_a, 0], coords_A[chain_a, 1], coords_A[chain_a, 2],
                   c='blue', s=20, alpha=0.7, label='Chain A')
        # Connect consecutive residues
        ca = coords_A[chain_a]
        ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], 'b-', alpha=0.3, linewidth=1)

    if chain_b.any():
        ax.scatter(coords_A[chain_b, 0], coords_A[chain_b, 1], coords_A[chain_b, 2],
                   c='red', s=20, alpha=0.7, label='Chain B')
        cb = coords_A[chain_b]
        ax.plot(cb[:, 0], cb[:, 1], cb[:, 2], 'r-', alpha=0.3, linewidth=1)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')


def main():
    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    # Load eval results to get test indices
    eval_path = project_root / "outputs/train_10k_continuous/eval_results.json"
    with open(eval_path) as f:
        eval_results = json.load(f)

    # Get test results with variance info
    test_results = eval_results['results']['test']

    # Load split to get all indices
    split_path = project_root / "outputs/train_10k_continuous/split.json"
    with open(split_path) as f:
        split = json.load(f)

    all_indices = split['train_indices'] + split['test_indices']

    print("Analyzing structures for coiled-coil detection...")
    print("=" * 70)

    # Analyze all samples for linearity
    linearity_data = []
    for idx in all_indices:
        sample = load_sample_raw(table, idx)
        centroids = sample['centroids'].numpy()
        chain_ids = sample['chain_ids'].numpy()

        linearity = compute_linearity(centroids)
        aspect_ratio = compute_aspect_ratio(centroids)

        # Per-chain linearity
        lin_a = compute_linearity(centroids[chain_ids == 0]) if (chain_ids == 0).any() else 0
        lin_b = compute_linearity(centroids[chain_ids == 1]) if (chain_ids == 1).any() else 0

        linearity_data.append({
            'idx': idx,
            'sample_id': sample['sample_id'],
            'n_res': sample['n_res'],
            'linearity': linearity,
            'aspect_ratio': aspect_ratio,
            'lin_chain_a': lin_a,
            'lin_chain_b': lin_b,
            'std': sample['std'],
        })

    # Sort by linearity
    linearity_data.sort(key=lambda x: x['linearity'], reverse=True)

    # Coiled-coil heuristic: linearity > 70% AND both chains linear
    coiled_coil_candidates = [
        x for x in linearity_data
        if x['linearity'] > 70 and x['lin_chain_a'] > 65 and x['lin_chain_b'] > 65
    ]

    print(f"\nLinearity distribution:")
    linearities = [x['linearity'] for x in linearity_data]
    print(f"  Min: {min(linearities):.1f}%")
    print(f"  Max: {max(linearities):.1f}%")
    print(f"  Mean: {np.mean(linearities):.1f}%")
    print(f"  Median: {np.median(linearities):.1f}%")

    print(f"\nCoiled-coil candidates (linearity > 70%, both chains > 65%):")
    print(f"  Count: {len(coiled_coil_candidates)} / {len(linearity_data)} ({100*len(coiled_coil_candidates)/len(linearity_data):.1f}%)")

    # Show top 10 most linear
    print(f"\nTop 10 most linear structures:")
    for i, x in enumerate(linearity_data[:10]):
        print(f"  {i+1}. {x['sample_id']}: lin={x['linearity']:.1f}%, "
              f"A={x['lin_chain_a']:.1f}%, B={x['lin_chain_b']:.1f}%, n_res={x['n_res']}")

    # Check if bad predictions correlate with linearity
    print(f"\nCorrelation with prediction quality (test set):")
    test_sample_ids = {r['sample_id'] for r in test_results}
    test_lin_data = [x for x in linearity_data if x['sample_id'] in test_sample_ids]

    # Match with eval results
    for x in test_lin_data:
        for r in test_results:
            if r['sample_id'] == x['sample_id']:
                x['rmse'] = r.get('rmse_mean', r.get('rmse', 0))
                x['variance'] = r.get('variance', 0)
                break

    # Correlation
    test_lin = [x for x in test_lin_data if 'rmse' in x]
    if test_lin:
        lins = np.array([x['linearity'] for x in test_lin])
        rmses = np.array([x['rmse'] for x in test_lin])
        corr = np.corrcoef(lins, rmses)[0, 1]
        print(f"  Linearity vs RMSE correlation: {corr:.3f}")

        variances = np.array([x['variance'] for x in test_lin])
        corr_var = np.corrcoef(lins, variances)[0, 1]
        print(f"  Linearity vs Variance correlation: {corr_var:.3f}")

    # Plot top 3 coiled-coils
    print(f"\nPlotting top 3 coiled-coil candidates...")

    fig = plt.figure(figsize=(15, 5))

    for i, x in enumerate(coiled_coil_candidates[:3]):
        sample = load_sample_raw(table, x['idx'])
        centroids = sample['centroids'].numpy()
        chain_ids = sample['chain_ids'].numpy()

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_structure(centroids, chain_ids,
                      f"{x['sample_id']}\nLinearity: {x['linearity']:.1f}%\nn_res: {x['n_res']}",
                      ax, std=x['std'])
        ax.legend(fontsize=8)

    plt.suptitle("Coiled-Coil Examples (High Linearity)", fontsize=12)
    plt.tight_layout()

    output_path = project_root / "outputs/coiled_coils.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")

    # Also plot some globular examples for comparison
    fig2 = plt.figure(figsize=(15, 5))

    # Get least linear (most globular)
    globular = linearity_data[-10:][::-1]  # Least linear

    print(f"\nTop 3 most globular (least linear) structures:")
    for i, x in enumerate(globular[:3]):
        print(f"  {i+1}. {x['sample_id']}: lin={x['linearity']:.1f}%, n_res={x['n_res']}")

        sample = load_sample_raw(table, x['idx'])
        centroids = sample['centroids'].numpy()
        chain_ids = sample['chain_ids'].numpy()

        ax = fig2.add_subplot(1, 3, i+1, projection='3d')
        plot_structure(centroids, chain_ids,
                      f"{x['sample_id']}\nLinearity: {x['linearity']:.1f}%\nn_res: {x['n_res']}",
                      ax, std=x['std'])
        ax.legend(fontsize=8)

    plt.suptitle("Globular Examples (Low Linearity)", fontsize=12)
    plt.tight_layout()

    output_path2 = project_root / "outputs/globular_examples.png"
    plt.savefig(output_path2, dpi=150)
    print(f"Saved plot to: {output_path2}")

    plt.close('all')


if __name__ == "__main__":
    main()
