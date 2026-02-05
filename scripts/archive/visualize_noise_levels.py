"""Visualize what high-noise looks like for different STD molecules."""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw

def main():
    output_dir = Path("outputs/train_1k")

    # Load split
    with open(output_dir / "split.json") as f:
        split = json.load(f)
    train_indices = split["train_indices"]

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    # Find samples with different STD values
    samples_info = []
    for idx in train_indices[:500]:  # Check first 500
        sample = load_sample_raw(table, idx)
        samples_info.append({
            'idx': idx,
            'std': sample['std'],
            'n_res': sample['n_res'],
            'name': sample.get('name', f'idx_{idx}')
        })

    # Sort by STD
    samples_info.sort(key=lambda x: x['std'])

    # Pick: low STD, medium STD, high STD
    low_std = samples_info[10]  # ~8
    med_std = samples_info[len(samples_info)//2]  # ~10-12
    high_std = samples_info[-10]  # ~18+

    print(f"Low STD sample:  {low_std['name']} - STD={low_std['std']:.2f}")
    print(f"Med STD sample:  {med_std['name']} - STD={med_std['std']:.2f}")
    print(f"High STD sample: {high_std['name']} - STD={high_std['std']:.2f}")

    # Noise levels to visualize (in normalized space)
    sigma_levels = [0, 1, 3, 5, 10]

    fig, axes = plt.subplots(3, len(sigma_levels), figsize=(4*len(sigma_levels), 12))

    for row, sample_info in enumerate([low_std, med_std, high_std]):
        sample = load_sample_raw(table, sample_info['idx'])

        # Get centroids (normalized)
        coords = np.array(sample['coords']).reshape(-1, 4, 3)
        centroids = coords.mean(axis=1)  # (n_res, 3)
        chain_ids = np.array(sample['chain_ids'])

        for col, sigma in enumerate(sigma_levels):
            ax = axes[row, col]

            # Add noise
            np.random.seed(42)
            noise = np.random.randn(*centroids.shape)
            noisy = centroids + sigma * noise

            # Plot in 2D (X-Y projection)
            mask_a = chain_ids == 0
            mask_b = chain_ids == 1

            ax.scatter(noisy[mask_a, 0], noisy[mask_a, 1], c='blue', s=10, alpha=0.7, label='Chain A')
            ax.scatter(noisy[mask_b, 0], noisy[mask_b, 1], c='red', s=10, alpha=0.7, label='Chain B')

            # Compute actual noise in Angstroms
            noise_angstrom = sigma * sample_info['std']

            if row == 0:
                ax.set_title(f"σ={sigma}\n({noise_angstrom:.0f}Å actual)")
            else:
                ax.set_title(f"({noise_angstrom:.0f}Å actual)")

            if col == 0:
                ax.set_ylabel(f"STD={sample_info['std']:.1f}\n{sample_info['name']}")

            ax.set_aspect('equal')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.suptitle("Effect of noise σ on different STD molecules\n(X-Y projection, normalized coords)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "noise_visualization.png", dpi=150)
    print(f"\nSaved to {output_dir / 'noise_visualization.png'}")

    # Also print SNR analysis
    print("\n=== Signal-to-Noise Ratio at σ=10 ===")
    print("(SNR = signal_std / noise_std = 1 / sigma)")
    print(f"In normalized space: SNR = 1/10 = 0.1 (same for all)")
    print(f"\nBut in Angstrom space at σ=10:")
    for info in [low_std, med_std, high_std]:
        actual_signal = info['std']  # STD of coordinates in Angstroms
        actual_noise = 10 * info['std']  # noise in Angstroms
        print(f"  STD={info['std']:.1f}: signal~{actual_signal:.1f}Å, noise~{actual_noise:.0f}Å")

    # Key insight
    print("\n=== KEY INSIGHT ===")
    print("The NORMALIZED noise σ=10 means:")
    print(f"  - Low STD ({low_std['std']:.1f}):  {10*low_std['std']:.0f}Å noise - structure completely destroyed")
    print(f"  - High STD ({high_std['std']:.1f}): {10*high_std['std']:.0f}Å noise - structure completely destroyed")
    print("\nBut the RELATIVE signal is the same (normalized to unit variance)!")
    print("So the issue might not be noise level but something else...")

    # Let's also check the actual coordinate ranges
    print("\n=== Coordinate ranges in ANGSTROMS ===")
    for info in [low_std, med_std, high_std]:
        sample = load_sample_raw(table, info['idx'])
        coords = np.array(sample['coords']).reshape(-1, 3) * info['std']
        print(f"  STD={info['std']:.1f}: X=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], "
              f"Y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}], "
              f"Z=[{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")
        print(f"           Range: {coords.max() - coords.min():.1f}Å")

if __name__ == "__main__":
    main()
