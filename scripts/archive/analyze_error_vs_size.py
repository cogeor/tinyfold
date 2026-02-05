"""Analyze error vs protein size for the 1K model on train set."""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import (
    load_sample_raw, collate_batch, sample_centroids_ve,
    create_schedule, create_noiser
)
from models.resfold_pipeline import ResFoldPipeline
from tinyfold.model.losses.mse import compute_rmse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Load model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=256,
        trunk_layers=9,
        denoiser_blocks=7,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create VE noiser (continuous sigma, AF3-style)
    from models.diffusion import KarrasSchedule, VENoiser
    schedule = KarrasSchedule(
        sigma_min=0.0004,
        sigma_max=10.0,
        rho=7.0,
        n_steps=50,
    )
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Evaluate on random subset of train set
    import random
    random.seed(42)
    sample_indices = random.sample(train_indices, min(100, len(train_indices)))

    sizes = []
    errors = []
    n_residues_list = []
    spatial_extents = []  # max - min for each axis
    sample_names = []
    stds = []
    chain_lengths = []  # (chain_a_len, chain_b_len)

    print(f"Evaluating {len(sample_indices)} random train samples...")

    for i, idx in enumerate(sample_indices):
        sample = load_sample_raw(table, idx)
        n_atoms = sample["n_atoms"]
        n_res = sample["n_res"]

        # Compute spatial extent from raw coords (before normalization)
        coords = np.array(sample["coords"]).reshape(-1, 3)
        coords_unnorm = coords * sample["std"]  # undo normalization
        x_range = coords_unnorm[:, 0].max() - coords_unnorm[:, 0].min()
        y_range = coords_unnorm[:, 1].max() - coords_unnorm[:, 1].min()
        z_range = coords_unnorm[:, 2].max() - coords_unnorm[:, 2].min()
        max_extent = max(x_range, y_range, z_range)

        # Chain lengths
        chain_ids = np.array(sample["chain_ids"])
        chain_a_len = (chain_ids == 0).sum()
        chain_b_len = (chain_ids == 1).sum()

        # Run inference
        with torch.no_grad():
            batch = collate_batch([sample], device)

            # VE sampling
            centroids_pred = sample_centroids_ve(
                model, batch, noiser, device,
                align_per_step=True,
                recenter=True,
                self_cond=True,
            )

            # Compute RMSE
            rmse = compute_rmse(
                centroids_pred, batch["centroids"], batch["mask_res"]
            ).item() * sample["std"]

            sizes.append(n_atoms)
            n_residues_list.append(n_res)
            errors.append(rmse)
            spatial_extents.append(max_extent)
            sample_names.append(sample.get("name", f"idx_{idx}"))
            stds.append(sample["std"])
            chain_lengths.append((chain_a_len, chain_b_len))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_indices)} - RMSE: {rmse:.2f} A | extent: {max_extent:.1f} A | atoms: {n_atoms}")

    # Convert to numpy
    sizes = np.array(sizes)
    errors = np.array(errors)
    n_residues = np.array(n_residues_list)
    spatial_extents = np.array(spatial_extents)
    stds = np.array(stds)
    chain_lengths = np.array(chain_lengths)

    # Plot error vs various features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Atoms
    ax = axes[0, 0]
    ax.scatter(sizes, errors, alpha=0.5, s=20)
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Centroid RMSE (A)")
    ax.set_title("Error vs Number of Atoms")
    z = np.polyfit(sizes, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes), max(sizes), 100)
    ax.plot(x_line, p(x_line), "r--", label=f"slope={z[0]:.4f}")
    ax.legend()

    # Spatial extent
    ax = axes[0, 1]
    ax.scatter(spatial_extents, errors, alpha=0.5, s=20)
    ax.set_xlabel("Max spatial extent (A)")
    ax.set_ylabel("Centroid RMSE (A)")
    ax.set_title("Error vs Spatial Extent")
    z = np.polyfit(spatial_extents, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(spatial_extents), max(spatial_extents), 100)
    ax.plot(x_line, p(x_line), "r--", label=f"slope={z[0]:.4f}")
    ax.legend()

    # STD (normalization factor)
    ax = axes[1, 0]
    ax.scatter(stds, errors, alpha=0.5, s=20)
    ax.set_xlabel("STD (normalization factor)")
    ax.set_ylabel("Centroid RMSE (A)")
    ax.set_title("Error vs STD")
    z = np.polyfit(stds, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(stds), max(stds), 100)
    ax.plot(x_line, p(x_line), "r--", label=f"slope={z[0]:.4f}")
    ax.legend()

    # Chain length ratio
    chain_ratios = chain_lengths[:, 0] / (chain_lengths[:, 0] + chain_lengths[:, 1])
    ax = axes[1, 1]
    ax.scatter(chain_ratios, errors, alpha=0.5, s=20)
    ax.set_xlabel("Chain A fraction")
    ax.set_ylabel("Centroid RMSE (A)")
    ax.set_title("Error vs Chain Balance")
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_size.png", dpi=150)
    print(f"\nSaved plot to {output_dir / 'error_vs_size.png'}")

    # Print stats
    print(f"\n=== Correlations ===")
    print(f"Atoms vs error:    {np.corrcoef(sizes, errors)[0,1]:.3f}")
    print(f"Extent vs error:   {np.corrcoef(spatial_extents, errors)[0,1]:.3f}")
    print(f"STD vs error:      {np.corrcoef(stds, errors)[0,1]:.3f}")
    print(f"Chain ratio vs error: {np.corrcoef(chain_ratios, errors)[0,1]:.3f}")
    print(f"\nMean RMSE: {np.mean(errors):.2f} A")

    # Find bad samples (top 10 highest error)
    bad_indices = np.argsort(errors)[-10:][::-1]
    print(f"\n=== TOP 10 WORST SAMPLES ===")
    print(f"{'Name':<20} {'RMSE':>8} {'Atoms':>6} {'Extent':>8} {'STD':>8} {'ChainA':>7} {'ChainB':>7}")
    print("-" * 75)
    for i in bad_indices:
        print(f"{sample_names[i]:<20} {errors[i]:>8.2f} {sizes[i]:>6} {spatial_extents[i]:>8.1f} {stds[i]:>8.2f} {chain_lengths[i][0]:>7} {chain_lengths[i][1]:>7}")

    # Find good samples (bottom 10 lowest error)
    good_indices = np.argsort(errors)[:10]
    print(f"\n=== TOP 10 BEST SAMPLES ===")
    print(f"{'Name':<20} {'RMSE':>8} {'Atoms':>6} {'Extent':>8} {'STD':>8} {'ChainA':>7} {'ChainB':>7}")
    print("-" * 75)
    for i in good_indices:
        print(f"{sample_names[i]:<20} {errors[i]:>8.2f} {sizes[i]:>6} {spatial_extents[i]:>8.1f} {stds[i]:>8.2f} {chain_lengths[i][0]:>7} {chain_lengths[i][1]:>7}")

    # Binned stats by extent
    print(f"\n=== RMSE by spatial extent ===")
    extent_bins = [(0, 40), (40, 60), (60, 80), (80, 150)]
    for lo, hi in extent_bins:
        mask = (spatial_extents >= lo) & (spatial_extents < hi)
        if mask.sum() > 0:
            print(f"  {lo}-{hi} A extent: {np.mean(errors[mask]):.2f} A (n={mask.sum()})")

if __name__ == "__main__":
    main()
