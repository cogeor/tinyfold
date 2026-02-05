"""Debug 3mq7 alignment issue."""

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    # Load model
    checkpoint_path = project_root / "outputs/train_10k_continuous/best_model.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=256, trunk_layers=14, denoiser_blocks=10,
        n_timesteps=50, stage1_only=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    schedule = KarrasSchedule(n_steps=200, sigma_min=0.002, sigma_max=10.0, rho=7.0)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Load split and find 3mq7
    split_path = project_root / "outputs/train_10k_continuous/split.json"
    with open(split_path) as f:
        split = json.load(f)

    all_indices = split['train_indices'] + split['test_indices']

    # Find 3mq7
    target_idx = None
    for idx in all_indices:
        sample = load_sample_raw(table, idx)
        if '3mq7' in sample['sample_id']:
            target_idx = idx
            break

    if target_idx is None:
        print("3mq7 not found!")
        return

    sample = load_sample_raw(table, target_idx)
    print(f"Sample: {sample['sample_id']}")
    print(f"n_res: {sample['n_res']}")
    print(f"std: {sample['std']}")

    batch = collate_batch([sample], device)

    # Run inference
    torch.manual_seed(42)
    with torch.no_grad():
        pred = sample_centroids_ve(model, batch, noiser, device,
                                   align_per_step=True, recenter=True)

    n_res = sample['n_res']
    pred_raw = pred[0, :n_res].cpu()
    gt_raw = batch['centroids'][0, :n_res].cpu()
    chain_ids = batch['chain_ids'][0, :n_res].cpu().numpy()

    print(f"\n=== RAW (before Kabsch) ===")
    print(f"Pred shape: {pred_raw.shape}")
    print(f"GT shape: {gt_raw.shape}")
    print(f"Pred range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    print(f"GT range: [{gt_raw.min():.2f}, {gt_raw.max():.2f}]")
    print(f"Pred std: {pred_raw.std():.3f}")
    print(f"GT std: {gt_raw.std():.3f}")

    # Check per-chain stats
    chain_a = chain_ids == 0
    chain_b = chain_ids == 1
    print(f"\nChain A: {chain_a.sum()} residues")
    print(f"Chain B: {chain_b.sum()} residues")

    print(f"\nPred Chain A center: {pred_raw[chain_a].mean(dim=0).numpy()}")
    print(f"Pred Chain B center: {pred_raw[chain_b].mean(dim=0).numpy()}")
    print(f"GT Chain A center: {gt_raw[chain_a].mean(dim=0).numpy()}")
    print(f"GT Chain B center: {gt_raw[chain_b].mean(dim=0).numpy()}")

    # Raw RMSE (no alignment)
    raw_rmse = torch.sqrt(((pred_raw - gt_raw) ** 2).mean()).item() * sample['std']
    print(f"\nRaw RMSE (no alignment): {raw_rmse:.1f} Å")

    # Kabsch alignment
    pred_aligned, gt_aligned = kabsch_align(pred_raw.unsqueeze(0), gt_raw.unsqueeze(0))
    pred_aligned = pred_aligned.squeeze(0)
    gt_aligned = gt_aligned.squeeze(0)

    aligned_rmse = torch.sqrt(((pred_aligned - gt_aligned) ** 2).mean()).item() * sample['std']
    print(f"Aligned RMSE (Kabsch): {aligned_rmse:.1f} Å")

    # Per-chain alignment
    pred_a = pred_raw[chain_a]
    gt_a = gt_raw[chain_a]
    pred_a_aligned, gt_a_aligned = kabsch_align(pred_a.unsqueeze(0), gt_a.unsqueeze(0))
    rmse_a = torch.sqrt(((pred_a_aligned - gt_a_aligned) ** 2).mean()).item() * sample['std']
    print(f"Chain A RMSE (aligned separately): {rmse_a:.1f} Å")

    pred_b = pred_raw[chain_b]
    gt_b = gt_raw[chain_b]
    pred_b_aligned, gt_b_aligned = kabsch_align(pred_b.unsqueeze(0), gt_b.unsqueeze(0))
    rmse_b = torch.sqrt(((pred_b_aligned - gt_b_aligned) ** 2).mean()).item() * sample['std']
    print(f"Chain B RMSE (aligned separately): {rmse_b:.1f} Å")

    # Check if chains are swapped - try aligning pred_A to gt_B
    pred_a_to_gt_b, gt_b_c = kabsch_align(pred_a.unsqueeze(0), gt_b.unsqueeze(0))
    rmse_swap_a = torch.sqrt(((pred_a_to_gt_b - gt_b_c) ** 2).mean()).item() * sample['std']
    print(f"\nChain swap check:")
    print(f"  Pred A -> GT B RMSE: {rmse_swap_a:.1f} Å")

    pred_b_to_gt_a, gt_a_c = kabsch_align(pred_b.unsqueeze(0), gt_a.unsqueeze(0))
    rmse_swap_b = torch.sqrt(((pred_b_to_gt_a - gt_a_c) ** 2).mean()).item() * sample['std']
    print(f"  Pred B -> GT A RMSE: {rmse_swap_b:.1f} Å")

    # Plot comparison
    fig = plt.figure(figsize=(20, 5))

    # 1. Raw (no alignment)
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(gt_raw[chain_a, 0], gt_raw[chain_a, 1], gt_raw[chain_a, 2], c='blue', s=10, alpha=0.7, label='GT A')
    ax1.scatter(gt_raw[chain_b, 0], gt_raw[chain_b, 1], gt_raw[chain_b, 2], c='red', s=10, alpha=0.7, label='GT B')
    ax1.scatter(pred_raw[chain_a, 0], pred_raw[chain_a, 1], pred_raw[chain_a, 2], c='cyan', s=10, alpha=0.5, marker='^', label='Pred A')
    ax1.scatter(pred_raw[chain_b, 0], pred_raw[chain_b, 1], pred_raw[chain_b, 2], c='orange', s=10, alpha=0.5, marker='^', label='Pred B')
    ax1.set_title(f'Raw (no alignment)\nRMSE: {raw_rmse:.1f} Å')
    ax1.legend(fontsize=7)

    # 2. Global Kabsch alignment
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(gt_aligned[chain_a, 0], gt_aligned[chain_a, 1], gt_aligned[chain_a, 2], c='blue', s=10, alpha=0.7, label='GT A')
    ax2.scatter(gt_aligned[chain_b, 0], gt_aligned[chain_b, 1], gt_aligned[chain_b, 2], c='red', s=10, alpha=0.7, label='GT B')
    ax2.scatter(pred_aligned[chain_a, 0], pred_aligned[chain_a, 1], pred_aligned[chain_a, 2], c='cyan', s=10, alpha=0.5, marker='^', label='Pred A')
    ax2.scatter(pred_aligned[chain_b, 0], pred_aligned[chain_b, 1], pred_aligned[chain_b, 2], c='orange', s=10, alpha=0.5, marker='^', label='Pred B')
    ax2.set_title(f'Global Kabsch\nRMSE: {aligned_rmse:.1f} Å')
    ax2.legend(fontsize=7)

    # 3. Per-chain alignment
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(gt_a_aligned[0, :, 0], gt_a_aligned[0, :, 1], gt_a_aligned[0, :, 2], c='blue', s=10, alpha=0.7, label='GT A')
    ax3.scatter(pred_a_aligned[0, :, 0], pred_a_aligned[0, :, 1], pred_a_aligned[0, :, 2], c='cyan', s=10, alpha=0.5, marker='^', label='Pred A')
    ax3.scatter(gt_b_aligned[0, :, 0] + 5, gt_b_aligned[0, :, 1], gt_b_aligned[0, :, 2], c='red', s=10, alpha=0.7, label='GT B (shifted)')
    ax3.scatter(pred_b_aligned[0, :, 0] + 5, pred_b_aligned[0, :, 1], pred_b_aligned[0, :, 2], c='orange', s=10, alpha=0.5, marker='^', label='Pred B (shifted)')
    ax3.set_title(f'Per-chain alignment\nA: {rmse_a:.1f} Å, B: {rmse_b:.1f} Å')
    ax3.legend(fontsize=7)

    # 4. Chain swap test
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.scatter(gt_b_c[0, :, 0], gt_b_c[0, :, 1], gt_b_c[0, :, 2], c='red', s=10, alpha=0.7, label='GT B')
    ax4.scatter(pred_a_to_gt_b[0, :, 0], pred_a_to_gt_b[0, :, 1], pred_a_to_gt_b[0, :, 2], c='cyan', s=10, alpha=0.5, marker='^', label='Pred A')
    ax4.scatter(gt_a_c[0, :, 0] + 5, gt_a_c[0, :, 1], gt_a_c[0, :, 2], c='blue', s=10, alpha=0.7, label='GT A (shifted)')
    ax4.scatter(pred_b_to_gt_a[0, :, 0] + 5, pred_b_to_gt_a[0, :, 1], pred_b_to_gt_a[0, :, 2], c='orange', s=10, alpha=0.5, marker='^', label='Pred B (shifted)')
    ax4.set_title(f'Chain SWAP test\nA→B: {rmse_swap_a:.1f} Å, B→A: {rmse_swap_b:.1f} Å')
    ax4.legend(fontsize=7)

    plt.suptitle(f"3mq7 Alignment Debug", fontsize=12)
    plt.tight_layout()

    output_path = project_root / "outputs/debug_3mq7.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved debug plot to: {output_path}")

    plt.close('all')


if __name__ == "__main__":
    main()
