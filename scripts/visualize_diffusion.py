#!/usr/bin/env python
"""Visualize the diffusion process from extended chain to folded protein.

Usage:
    python visualize_diffusion.py --model_path outputs/overfit_1M_linear/best_model.pt
"""

import argparse
import os
import sys
import torch
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import create_model, create_schedule, create_noiser
from models.diffusion import generate_extended_chain
from data_split import DataSplitConfig, get_train_test_indices


def kabsch_align(pred, target):
    """Align pred to target using Kabsch algorithm.

    Args:
        pred: [N, 3] predicted coordinates
        target: [N, 3] target coordinates

    Returns:
        pred_aligned: [N, 3] aligned predicted coordinates
    """
    # Center both
    pred_center = pred.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    pred_c = pred - pred_center
    target_c = target - target_center

    # Compute rotation
    H = pred_c.T @ target_c
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection
    d = torch.det(Vt.T @ U.T)
    D = torch.eye(3, device=pred.device)
    D[2, 2] = d

    R = Vt.T @ D @ U.T
    pred_aligned = pred_c @ R + target_center

    return pred_aligned


def load_sample_raw(table, i):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    coords = coords.reshape(n_atoms, 3)
    aa_seq = seq_res[atom_to_res]
    chain_ids = chain_res[atom_to_res]

    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    return {
        'coords': coords,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'std': std.item(),
        'n_atoms': n_atoms,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_single(sample, device):
    """Collate a single sample into batch format."""
    n = sample['n_atoms']
    return {
        'coords': sample['coords'].unsqueeze(0).to(device),
        'atom_types': sample['atom_types'].unsqueeze(0).to(device),
        'atom_to_res': sample['atom_to_res'].unsqueeze(0).to(device),
        'aa_seq': sample['aa_seq'].unsqueeze(0).to(device),
        'chain_ids': sample['chain_ids'].unsqueeze(0).to(device),
        'mask': torch.ones(1, n, dtype=torch.bool, device=device),
    }


@torch.no_grad()
def ddpm_sample_trajectory(model, atom_types, atom_to_res, aa_seq, chain_ids,
                           noiser, x_start, x_linear, mask=None, clamp_val=3.0,
                           save_every=5, use_linear_chain=False):
    """Diffusion sampling loop that saves intermediate states.

    Args:
        x_start: Starting point (extended chain for linear_chain noise)
        x_linear: Extended chain coordinates (needed for linear_chain reverse)
        save_every: Save state every N steps
        use_linear_chain: Use linear chain reverse step instead of DDPM

    Returns:
        trajectory: List of (t, x) tuples
    """
    device = atom_types.device
    B, N = atom_types.shape
    x = x_start.clone()

    trajectory = [(noiser.T, x.clone())]

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if use_linear_chain:
            # Use linear chain reverse step (interpolation, not DDPM noise)
            x = noiser.reverse_step(x, x0_pred, t, x_linear)
        else:
            # Standard DDPM reverse step
            if t > 0:
                ab_t = noiser.alpha_bar[t]
                ab_prev = noiser.alpha_bar[t - 1]
                beta = noiser.betas[t]
                alpha = noiser.alphas[t]

                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x

                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

        if t % save_every == 0 or t == 0:
            trajectory.append((t, x.clone()))

    return trajectory


def plot_structure(ax, coords, chain_ids, title):
    """Plot 3D structure colored by chain."""
    coords = coords.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    if mask_a.any():
        ax.scatter(coords[mask_a, 0], coords[mask_a, 1], coords[mask_a, 2],
                   c='blue', s=15, alpha=0.7, label='Chain A')
    if mask_b.any():
        ax.scatter(coords[mask_b, 0], coords[mask_b, 1], coords[mask_b, 2],
                   c='red', s=15, alpha=0.7, label='Chain B')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    max_range = max(coords.max() - coords.min(), 1.0) / 2
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0, help="Test sample index")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_frames", type=int, default=10, help="Number of frames to show")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']

    print(f"Model config: {model_args['model']}, h_dim={model_args['h_dim']}, n_layers={model_args['n_layers']}")
    print(f"Noise type: {model_args['noise_type']}")
    print(f"Best test RMSE: {checkpoint['test_rmse']:.2f} A")

    # Create model
    model = create_model(
        model_args['model'],
        h_dim=model_args['h_dim'],
        n_heads=8,
        n_layers=model_args['n_layers'],
        n_timesteps=model_args['T'],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create noiser
    schedule = create_schedule(model_args['schedule'], T=model_args['T'])
    if model_args['noise_type'] == "linear_chain":
        noiser = create_noiser(model_args['noise_type'], schedule, noise_scale=model_args['noise_scale'])
    else:
        noiser = create_noiser(model_args['noise_type'], schedule)
    noiser = noiser.to(device)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    table = pq.read_table(data_path)

    # Get test samples using same split as training
    split_config = DataSplitConfig(
        n_train=model_args['n_train'],
        n_test=model_args['n_test'],
        min_atoms=model_args['min_atoms'],
        max_atoms=model_args['max_atoms'],
        seed=42,
    )
    _, test_indices = get_train_test_indices(table, split_config)

    # Load test sample
    test_idx = test_indices[args.sample_idx]
    sample = load_sample_raw(table, test_idx)
    batch = collate_single(sample, device)

    print(f"\nTest sample: {sample['sample_id']}")
    print(f"  Atoms: {sample['n_atoms']}")

    # Generate extended chain as starting point
    n_atoms = sample['n_atoms']
    x_chain = generate_extended_chain(
        n_atoms,
        batch['atom_to_res'][0],
        batch['atom_types'][0],
        batch['chain_ids'][0],
        device
    )
    # Normalize like training data
    x_chain = x_chain - x_chain.mean(dim=0, keepdim=True)
    x_chain = x_chain / x_chain.std()
    x_chain = x_chain.unsqueeze(0)  # Add batch dim

    # Run diffusion with trajectory
    use_linear_chain = (model_args['noise_type'] == 'linear_chain')
    print(f"\nRunning diffusion sampling ({noiser.T} steps, linear_chain={use_linear_chain})...")
    save_every = max(1, noiser.T // args.n_frames)
    trajectory = ddpm_sample_trajectory(
        model, batch['atom_types'], batch['atom_to_res'],
        batch['aa_seq'], batch['chain_ids'], noiser, x_chain,
        x_linear=x_chain,  # Extended chain for reverse interpolation
        mask=batch['mask'], save_every=save_every,
        use_linear_chain=use_linear_chain
    )

    print(f"Saved {len(trajectory)} frames")

    # Align all trajectory frames to ground truth
    ground_truth = batch['coords'][0]
    trajectory_aligned = []
    for t, x in trajectory:
        x_aligned = kabsch_align(x[0], ground_truth)
        trajectory_aligned.append((t, x_aligned))

    print("Aligned trajectory to ground truth")

    # Create visualization
    n_plots = min(len(trajectory_aligned) + 1, args.n_frames + 2)  # +1 for ground truth
    cols = min(5, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    # Select frames to show
    frame_indices = [0]  # Start with extended chain
    step = max(1, (len(trajectory_aligned) - 1) // (args.n_frames - 1))
    for i in range(1, len(trajectory_aligned)):
        if i % step == 0 or i == len(trajectory_aligned) - 1:
            frame_indices.append(i)
    frame_indices = frame_indices[:args.n_frames]

    # Plot trajectory frames (aligned)
    for plot_idx, frame_idx in enumerate(frame_indices):
        t, x = trajectory_aligned[frame_idx]
        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        plot_structure(ax, x, batch['chain_ids'][0], f't={t}')

    # Plot ground truth
    ax = fig.add_subplot(rows, cols, len(frame_indices) + 1, projection='3d')
    plot_structure(ax, ground_truth, batch['chain_ids'][0], 'Ground Truth')

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'diffusion_trajectory_{sample["sample_id"]}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved trajectory: {output_path}")

    # Also save individual frames for animation (aligned)
    frames_dir = os.path.join(args.output_dir, f'frames_{sample["sample_id"]}')
    os.makedirs(frames_dir, exist_ok=True)

    for i, (t, x) in enumerate(trajectory_aligned):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_structure(ax, x, batch['chain_ids'][0], f't={t}')
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f'frame_{i:03d}_t{t:03d}.png'), dpi=100)
        plt.close()

    # Save ground truth frame
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_structure(ax, batch['coords'][0], batch['chain_ids'][0], 'Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(frames_dir, 'ground_truth.png'), dpi=100)
    plt.close()

    print(f"Saved {len(trajectory)} individual frames to {frames_dir}/")


if __name__ == "__main__":
    main()
