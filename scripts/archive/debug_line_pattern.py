"""Debug why residues appear in a line pattern during denoising."""
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
    load_sample_raw, collate_batch,
    create_schedule, create_noiser
)
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser


def visualize_trajectory(model, batch, noiser, device, sample_name, output_dir):
    """Run denoising and visualize each step."""
    model.eval()

    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Get sigma schedule
    sigmas = noiser.sigmas.to(device)

    # Initialize at highest noise level
    x = sigmas[0] * torch.randn(B, L, 3, device=device)

    # Store trajectory
    trajectory = [x.cpu().numpy().copy()]
    sigma_values = [sigmas[0].item()]
    x0_predictions = []  # Store x0 predictions to see what model outputs

    x0_prev = None

    # Run denoising
    print(f"Denoising {sample_name}...")
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = sigma.expand(B)

        with torch.no_grad():
            # Predict x0 using the correct API
            x0_pred = model.stage1.forward_sigma(
                x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                sigma_batch, mask, x0_prev=x0_prev
            )
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            # Store x0 prediction
            x0_predictions.append(x0_pred.cpu().numpy().copy())

            # Euler step
            d = (x - x0_pred) / sigma
            x = x + d * (sigma_next - sigma)

            # Re-center
            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            x = x - centroid

            x0_prev = x0_pred.detach()

        trajectory.append(x.cpu().numpy().copy())
        sigma_values.append(sigma_next.item())

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{len(sigmas)-1}, sigma={sigma.item():.4f}")

    # Plot x0 predictions over time (what model thinks the clean structure is)
    print("\nPlotting x0 predictions...")
    x0_steps = [0, len(x0_predictions)//4, len(x0_predictions)//2, 3*len(x0_predictions)//4, len(x0_predictions)-1]

    fig, axes = plt.subplots(1, len(x0_steps), figsize=(20, 4))
    chain_ids_np = batch['chain_ids'].cpu().numpy()

    for idx, step in enumerate(x0_steps):
        ax = axes[idx]
        x0 = x0_predictions[step][0]
        mask_a = chain_ids_np[0] == 0
        mask_b = chain_ids_np[0] == 1

        ax.scatter(x0[mask_a, 0], x0[mask_a, 1], c='blue', s=15, alpha=0.7)
        ax.scatter(x0[mask_b, 0], x0[mask_b, 1], c='red', s=15, alpha=0.7)
        ax.set_title(f"x0 at step {step}\nsigma={sigma_values[step]:.3f}")
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    plt.suptitle(f"Model's x0 predictions (what it thinks clean structure is): {sample_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"x0_predictions_{sample_name.replace('.', '_')}.png", dpi=150)
    print(f"Saved x0 predictions plot")

    # Get ground truth
    gt_centroids = batch['centroids'].cpu().numpy()
    chain_ids = batch['chain_ids'].cpu().numpy()

    # Plot trajectory at key steps
    n_steps = len(trajectory)
    steps_to_plot = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]

    fig = plt.figure(figsize=(20, 8))

    for idx, step in enumerate(steps_to_plot):
        # 2D projection (X-Y)
        ax = fig.add_subplot(2, 5, idx + 1)
        coords = trajectory[step][0]  # First sample
        mask_a = chain_ids[0] == 0
        mask_b = chain_ids[0] == 1

        ax.scatter(coords[mask_a, 0], coords[mask_a, 1], c='blue', s=15, alpha=0.7, label='Chain A')
        ax.scatter(coords[mask_b, 0], coords[mask_b, 1], c='red', s=15, alpha=0.7, label='Chain B')
        ax.set_title(f"Step {step}\nsigma={sigma_values[step]:.3f}")
        ax.set_aspect('equal')
        if idx == 0:
            ax.legend(fontsize=8)

        # 2D projection (X-Z)
        ax = fig.add_subplot(2, 5, idx + 6)
        ax.scatter(coords[mask_a, 0], coords[mask_a, 2], c='blue', s=15, alpha=0.7)
        ax.scatter(coords[mask_b, 0], coords[mask_b, 2], c='red', s=15, alpha=0.7)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal')

    plt.suptitle(f"Denoising trajectory: {sample_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"trajectory_{sample_name.replace('.', '_')}.png", dpi=150)
    print(f"Saved trajectory plot")

    # Also plot final vs ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    final = trajectory[-1][0]
    gt = gt_centroids[0]

    # X-Y
    ax = axes[0]
    ax.scatter(gt[mask_a, 0], gt[mask_a, 1], c='blue', s=20, alpha=0.5, marker='o', label='GT Chain A')
    ax.scatter(gt[mask_b, 0], gt[mask_b, 1], c='red', s=20, alpha=0.5, marker='o', label='GT Chain B')
    ax.scatter(final[mask_a, 0], final[mask_a, 1], c='cyan', s=20, alpha=0.7, marker='x', label='Pred Chain A')
    ax.scatter(final[mask_b, 0], final[mask_b, 1], c='orange', s=20, alpha=0.7, marker='x', label='Pred Chain B')
    ax.set_title("X-Y projection")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # X-Z
    ax = axes[1]
    ax.scatter(gt[mask_a, 0], gt[mask_a, 2], c='blue', s=20, alpha=0.5, marker='o')
    ax.scatter(gt[mask_b, 0], gt[mask_b, 2], c='red', s=20, alpha=0.5, marker='o')
    ax.scatter(final[mask_a, 0], final[mask_a, 2], c='cyan', s=20, alpha=0.7, marker='x')
    ax.scatter(final[mask_b, 0], final[mask_b, 2], c='orange', s=20, alpha=0.7, marker='x')
    ax.set_title("X-Z projection")
    ax.set_aspect('equal')

    # Y-Z
    ax = axes[2]
    ax.scatter(gt[mask_a, 1], gt[mask_a, 2], c='blue', s=20, alpha=0.5, marker='o')
    ax.scatter(gt[mask_b, 1], gt[mask_b, 2], c='red', s=20, alpha=0.5, marker='o')
    ax.scatter(final[mask_a, 1], final[mask_a, 2], c='cyan', s=20, alpha=0.7, marker='x')
    ax.scatter(final[mask_b, 1], final[mask_b, 2], c='orange', s=20, alpha=0.7, marker='x')
    ax.set_title("Y-Z projection")
    ax.set_aspect('equal')

    plt.suptitle(f"Final prediction vs Ground Truth: {sample_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"final_vs_gt_{sample_name.replace('.', '_')}.png", dpi=150)
    print(f"Saved final vs GT plot")

    # Analyze linearity
    print(f"\n=== Linearity Analysis ===")
    analyze_linearity(final, chain_ids[0], "Prediction")
    analyze_linearity(gt, chain_ids[0], "Ground Truth")

    return trajectory, sigma_values


def analyze_linearity(coords, chain_ids, name):
    """Check if coords are linear (lie on a line)."""
    from numpy.linalg import svd

    for chain_id, chain_name in [(0, 'A'), (1, 'B')]:
        mask = chain_ids == chain_id
        pts = coords[mask]

        if len(pts) < 3:
            continue

        # Center points
        centered = pts - pts.mean(axis=0)

        # SVD to get principal components
        U, S, Vt = svd(centered)

        # Singular values indicate spread along each axis
        total_var = S.sum()
        explained = S / total_var * 100

        # Check if most variance is in first PC (linear)
        linearity = S[0] / (S[0] + S[1] + 1e-8) * 100

        print(f"  {name} Chain {chain_name}: "
              f"S=[{S[0]:.2f}, {S[1]:.2f}, {S[2]:.2f}] "
              f"Linearity={linearity:.1f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/train_1k")

    # Load split to find sample
    with open(output_dir / "split.json") as f:
        split = json.load(f)
    train_indices = split["train_indices"]

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    # Find the specific sample
    target_name = "3mxg.pdb2_3"
    target_idx = None

    print(f"Looking for sample: {target_name}")
    for idx in train_indices:
        sample_id = table['sample_id'][idx].as_py()
        if target_name in sample_id:
            target_idx = idx
            print(f"Found: idx={idx}, sample_id={sample_id}")
            break

    if target_idx is None:
        print(f"Sample {target_name} not found in train set, searching all...")
        for idx in range(len(table)):
            sample_id = table['sample_id'][idx].as_py()
            if target_name in sample_id:
                target_idx = idx
                print(f"Found: idx={idx}, sample_id={sample_id}")
                break

    if target_idx is None:
        print(f"Sample {target_name} not found!")
        return

    # Load sample
    sample = load_sample_raw(table, target_idx)
    print(f"\nSample info:")
    print(f"  n_res: {sample['n_res']}")
    print(f"  n_atoms: {sample['n_atoms']}")
    print(f"  std: {sample['std']:.2f}")

    # Load model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=256,
        trunk_layers=9,
        denoiser_blocks=7,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create VE noiser
    schedule = KarrasSchedule(
        sigma_min=0.0004,
        sigma_max=10.0,
        rho=7.0,
        n_steps=50,
    )
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Prepare batch
    batch = collate_batch([sample], device)

    # Visualize trajectory
    visualize_trajectory(model, batch, noiser, device, target_name, output_dir)


if __name__ == "__main__":
    main()
