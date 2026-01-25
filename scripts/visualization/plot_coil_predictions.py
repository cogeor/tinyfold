"""Plot predictions vs ground truth for coiled-coil examples."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw, collate_batch, sample_centroids_ve
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses import kabsch_align


def compute_linearity(coords):
    """Compute linearity score (0-100%)."""
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def plot_pred_vs_gt(pred, gt, chain_ids, title, ax, std=1.0):
    """Plot prediction vs ground truth overlaid."""
    pred_A = pred * std
    gt_A = gt * std

    chain_a = chain_ids == 0
    chain_b = chain_ids == 1

    # Ground truth (solid)
    if chain_a.any():
        ax.scatter(gt_A[chain_a, 0], gt_A[chain_a, 1], gt_A[chain_a, 2],
                   c='blue', s=15, alpha=0.8, label='GT Chain A')
        ca = gt_A[chain_a]
        ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], 'b-', alpha=0.5, linewidth=1)

    if chain_b.any():
        ax.scatter(gt_A[chain_b, 0], gt_A[chain_b, 1], gt_A[chain_b, 2],
                   c='red', s=15, alpha=0.8, label='GT Chain B')
        cb = gt_A[chain_b]
        ax.plot(cb[:, 0], cb[:, 1], cb[:, 2], 'r-', alpha=0.5, linewidth=1)

    # Prediction (hollow markers)
    if chain_a.any():
        ax.scatter(pred_A[chain_a, 0], pred_A[chain_a, 1], pred_A[chain_a, 2],
                   c='cyan', s=15, alpha=0.6, marker='^', label='Pred Chain A')

    if chain_b.any():
        ax.scatter(pred_A[chain_b, 0], pred_A[chain_b, 1], pred_A[chain_b, 2],
                   c='orange', s=15, alpha=0.6, marker='^', label='Pred Chain B')

    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    # Load model
    checkpoint_path = project_root / "outputs/train_10k_continuous/best_model.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=256,
        trunk_layers=14,
        denoiser_blocks=10,
        n_timesteps=50,
        stage1_only=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Create noiser with more steps for better quality
    schedule = KarrasSchedule(n_steps=200, sigma_min=0.002, sigma_max=10.0, rho=7.0)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Load split
    split_path = project_root / "outputs/train_10k_continuous/split.json"
    with open(split_path) as f:
        split = json.load(f)

    all_indices = split['train_indices'] + split['test_indices']

    # Find coiled-coil examples
    print("Finding coiled-coil examples...")
    coiled_coils = []
    for idx in all_indices:
        sample = load_sample_raw(table, idx)
        centroids = sample['centroids'].numpy()
        chain_ids = sample['chain_ids'].numpy()

        linearity = compute_linearity(centroids)
        lin_a = compute_linearity(centroids[chain_ids == 0]) if (chain_ids == 0).any() else 0
        lin_b = compute_linearity(centroids[chain_ids == 1]) if (chain_ids == 1).any() else 0

        if linearity > 70 and lin_a > 65 and lin_b > 65:
            is_test = idx in split['test_indices']
            coiled_coils.append({
                'idx': idx,
                'sample_id': sample['sample_id'],
                'linearity': linearity,
                'is_test': is_test,
            })

    # Sort by linearity and pick diverse examples
    coiled_coils.sort(key=lambda x: x['linearity'], reverse=True)

    # Pick 3: highest linearity, one from test if available
    selected = []
    seen_pdb = set()
    for cc in coiled_coils:
        pdb_id = cc['sample_id'].split('.')[0]
        if pdb_id not in seen_pdb:
            selected.append(cc)
            seen_pdb.add(pdb_id)
        if len(selected) >= 3:
            break

    print(f"\nSelected coiled-coils for prediction:")
    for cc in selected:
        split_name = "TEST" if cc['is_test'] else "TRAIN"
        print(f"  {cc['sample_id']}: linearity={cc['linearity']:.1f}% [{split_name}]")

    # Generate predictions
    fig = plt.figure(figsize=(15, 5))

    for i, cc in enumerate(selected):
        sample = load_sample_raw(table, cc['idx'])
        batch = collate_batch([sample], device)

        # Run inference
        torch.manual_seed(42)
        with torch.no_grad():
            pred = sample_centroids_ve(model, batch, noiser, device,
                                       align_per_step=True, recenter=True)

        # Get tensors
        n_res = sample['n_res']
        pred_np = pred[0, :n_res].cpu()
        gt_np = batch['centroids'][0, :n_res].cpu()
        chain_ids = batch['chain_ids'][0, :n_res].cpu().numpy()
        std = sample['std']

        # Kabsch align prediction to GT
        pred_aligned, gt_aligned = kabsch_align(pred_np.unsqueeze(0), gt_np.unsqueeze(0))
        pred_aligned = pred_aligned.squeeze(0).numpy()
        gt_aligned = gt_aligned.squeeze(0).numpy()

        # Compute RMSE
        rmse = np.sqrt(((pred_aligned - gt_aligned) ** 2).mean()) * std

        # Plot
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        split_name = "TEST" if cc['is_test'] else "TRAIN"
        plot_pred_vs_gt(pred_aligned, gt_aligned, chain_ids,
                       f"{cc['sample_id']} [{split_name}]\n"
                       f"Linearity: {cc['linearity']:.1f}%, RMSE: {rmse:.1f}Å",
                       ax, std=std)
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle("Coiled-Coil Predictions: GT (solid) vs Pred (triangles)", fontsize=12)
    plt.tight_layout()

    output_path = project_root / "outputs/coil_predictions.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")

    plt.close('all')


if __name__ == "__main__":
    main()
