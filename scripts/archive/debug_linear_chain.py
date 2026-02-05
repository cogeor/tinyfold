"""Debug linear chain sampling vs training."""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
from models import create_schedule, create_noiser
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import generate_extended_chain
from tinyfold.model.losses import kabsch_align, compute_mse_loss, compute_rmse
from scripts.train_coil_experiment import load_sample_raw, collate_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path("outputs/coil_experiment/linear_chain_aligned/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=128, trunk_layers=4, denoiser_blocks=4,
        n_timesteps=50, stage1_only=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data
    with open("outputs/coil_experiment/coil_samples.json") as f:
        coil_data = json.load(f)

    data_path = Path("data/processed/samples.parquet")
    table = pq.read_table(data_path)

    train_idx = coil_data['train_indices'][0]
    sample = load_sample_raw(table, train_idx)
    batch = collate_batch([sample], device)

    print(f"Sample: {sample['sample_id']}, n_res: {sample['n_res']}, std: {sample['std']:.2f}")

    schedule = create_schedule("linear", T=50)
    noiser = create_noiser("linear_chain", schedule).to(device)

    B, L = batch['aa_seq'].shape
    n_res = batch['n_res'][0]
    mask = batch['mask_res']

    # Generate extended chain
    atom_to_res = torch.arange(n_res, device=device)
    atom_type = torch.ones(n_res, dtype=torch.long, device=device)
    chain_ids = batch['chain_ids'][0, :n_res]

    x_linear = torch.zeros(1, L, 3, device=device)
    x_linear[0, :n_res] = generate_extended_chain(n_res, atom_to_res, atom_type, chain_ids, device)
    x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
    x_linear = x_linear / x_linear.std(dim=(1,2), keepdim=True).clamp(min=1e-6)

    gt = batch['centroids']

    print(f"\nGT stats: mean={gt[0,:n_res].mean():.3f}, std={gt[0,:n_res].std():.3f}")
    print(f"Extended stats: mean={x_linear[0,:n_res].mean():.3f}, std={x_linear[0,:n_res].std():.3f}")

    # Test 1: Direct prediction at different t values (like training)
    print("\n=== Test 1: Training-style forward pass ===")
    print("t  | sqrt_ab | loss    | RMSE (Å)")
    print("-" * 45)

    with torch.no_grad():
        for t_val in [0, 5, 10, 25, 40, 49]:
            t = torch.tensor([t_val], device=device)

            sqrt_ab = schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_1_minus_ab = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

            # Align extended chain to GT (like training)
            x_linear_aligned, _ = kabsch_align(x_linear, gt, mask)

            # Create x_t
            x_t = sqrt_ab * gt + sqrt_1_minus_ab * x_linear_aligned

            # Predict
            x0_pred = model.forward_stage1(
                x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t, mask
            )

            loss = compute_mse_loss(x0_pred, gt, mask).item()
            rmse = compute_rmse(x0_pred, gt, mask).item() * sample['std']

            print(f"{t_val:2d} | {sqrt_ab.item():.3f}   | {loss:.5f} | {rmse:.2f}")

    # Test 2: What happens with UN-aligned extended chain (like sampling start)
    print("\n=== Test 2: Sampling-style (un-aligned start) ===")
    print("t  | sqrt_ab | loss    | RMSE (Å)")
    print("-" * 45)

    with torch.no_grad():
        for t_val in [49, 40, 25, 10, 5, 0]:
            t = torch.tensor([t_val], device=device)

            # Use UN-aligned extended chain (like at start of sampling)
            x0_pred = model.forward_stage1(
                x_linear, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t, mask
            )

            loss = compute_mse_loss(x0_pred, gt, mask).item()
            rmse = compute_rmse(x0_pred, gt, mask).item() * sample['std']

            print(f"{t_val:2d} | {schedule.sqrt_alpha_bar[t_val].item():.3f}   | {loss:.5f} | {rmse:.2f}")

    # Test 3: Full sampling
    print("\n=== Test 3: Full sampling ===")

    with torch.no_grad():
        x = x_linear.clone()

        for t in reversed(range(50)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            x0_pred = model.forward_stage1(
                x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t_batch, mask
            )
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if t > 0:
                x_linear_aligned, _ = kabsch_align(x_linear, x0_pred, mask)
                sqrt_ab_prev = schedule.sqrt_alpha_bar[t - 1]
                sqrt_1_minus_ab_prev = schedule.sqrt_one_minus_alpha_bar[t - 1]
                x = sqrt_ab_prev * x0_pred + sqrt_1_minus_ab_prev * x_linear_aligned
            else:
                x = x0_pred

            if t in [49, 40, 25, 10, 5, 0]:
                rmse = compute_rmse(x0_pred, gt, mask).item() * sample['std']
                print(f"Step t={t:2d}: x0_pred RMSE = {rmse:.2f} Å")

        final_rmse = compute_rmse(x, gt, mask).item() * sample['std']
        print(f"\nFinal RMSE: {final_rmse:.2f} Å")

    # Test 4: What if we align x_linear to gt before starting sampling?
    print("\n=== Test 4: Sampling with pre-aligned start ===")

    with torch.no_grad():
        # Pre-align extended chain to GT (cheating, but diagnostic)
        x_start, _ = kabsch_align(x_linear, gt, mask)
        x = x_start.clone()

        for t in reversed(range(50)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            x0_pred = model.forward_stage1(
                x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t_batch, mask
            )
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if t > 0:
                # Still align to prediction (but starting point was aligned to GT)
                x_linear_aligned, _ = kabsch_align(x_linear, x0_pred, mask)
                sqrt_ab_prev = schedule.sqrt_alpha_bar[t - 1]
                sqrt_1_minus_ab_prev = schedule.sqrt_one_minus_alpha_bar[t - 1]
                x = sqrt_ab_prev * x0_pred + sqrt_1_minus_ab_prev * x_linear_aligned
            else:
                x = x0_pred

        final_rmse = compute_rmse(x, gt, mask).item() * sample['std']
        print(f"Final RMSE (pre-aligned start): {final_rmse:.2f} Å")


def test_multiple_samples():
    """Test on multiple samples like training eval."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path("outputs/coil_experiment/linear_chain_aligned/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ResFoldPipeline(
        c_token_s1=128, trunk_layers=4, denoiser_blocks=4,
        n_timesteps=50, stage1_only=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data
    with open("outputs/coil_experiment/coil_samples.json") as f:
        coil_data = json.load(f)

    data_path = Path("data/processed/samples.parquet")
    table = pq.read_table(data_path)

    schedule = create_schedule("linear", T=50)

    print("=== Testing on first 20 train samples ===")
    print("(Matching the training eval)")

    rmses = []
    for i, idx in enumerate(coil_data['train_indices'][:20]):
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)

        B, L = batch['aa_seq'].shape
        n_res = batch['n_res'][0]
        mask = batch['mask_res']
        gt = batch['centroids']

        # Generate extended chain
        atom_to_res = torch.arange(n_res, device=device)
        atom_type = torch.ones(n_res, dtype=torch.long, device=device)
        chain_ids = batch['chain_ids'][0, :n_res]

        x_linear = torch.zeros(1, L, 3, device=device)
        x_linear[0, :n_res] = generate_extended_chain(n_res, atom_to_res, atom_type, chain_ids, device)
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_linear = x_linear / x_linear.std(dim=(1,2), keepdim=True).clamp(min=1e-6)

        # Full sampling
        with torch.no_grad():
            x = x_linear.clone()

            for t in reversed(range(50)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)

                x0_pred = model.forward_stage1(
                    x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'], t_batch, mask
                )
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

                if t > 0:
                    x_linear_aligned, _ = kabsch_align(x_linear, x0_pred, mask)
                    sqrt_ab_prev = schedule.sqrt_alpha_bar[t - 1]
                    sqrt_1_minus_ab_prev = schedule.sqrt_one_minus_alpha_bar[t - 1]
                    x = sqrt_ab_prev * x0_pred + sqrt_1_minus_ab_prev * x_linear_aligned
                else:
                    x = x0_pred

            rmse = compute_rmse(x, gt, mask).item() * sample['std']
            rmses.append(rmse)

        print(f"Sample {i+1}: {sample['sample_id'][:15]:15s} | RMSE: {rmse:.2f} Å")

    print(f"\nAverage RMSE: {np.mean(rmses):.2f} Å")
    print(f"Min: {np.min(rmses):.2f} Å, Max: {np.max(rmses):.2f} Å")


def compare_gaussian_vs_linear():
    """Compare Gaussian vs Linear Chain on same samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models
    ckpt_lc = torch.load("outputs/coil_experiment/linear_chain_aligned/best_model.pt",
                         map_location=device, weights_only=False)
    ckpt_g = torch.load("outputs/coil_experiment/gaussian/best_model.pt",
                        map_location=device, weights_only=False)

    model_lc = ResFoldPipeline(c_token_s1=128, trunk_layers=4, denoiser_blocks=4,
                                n_timesteps=50, stage1_only=True).to(device)
    model_lc.load_state_dict(ckpt_lc['model_state_dict'])
    model_lc.eval()

    model_g = ResFoldPipeline(c_token_s1=128, trunk_layers=4, denoiser_blocks=4,
                               n_timesteps=50, stage1_only=True).to(device)
    model_g.load_state_dict(ckpt_g['model_state_dict'])
    model_g.eval()

    # Load data
    with open("outputs/coil_experiment/coil_samples.json") as f:
        coil_data = json.load(f)
    table = pq.read_table("data/processed/samples.parquet")
    schedule = create_schedule("linear", T=50)

    print("Sample             | Gaussian | LinearChain | Diff")
    print("-" * 55)

    g_rmses, lc_rmses = [], []

    for idx in coil_data['train_indices'][:10]:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)
        B, L = batch['aa_seq'].shape
        n_res = batch['n_res'][0]
        mask = batch['mask_res']
        gt = batch['centroids']

        # Generate extended chain for LC
        atom_to_res = torch.arange(n_res, device=device)
        atom_type = torch.ones(n_res, dtype=torch.long, device=device)
        chain_ids = batch['chain_ids'][0, :n_res]
        x_linear = torch.zeros(1, L, 3, device=device)
        x_linear[0, :n_res] = generate_extended_chain(n_res, atom_to_res, atom_type, chain_ids, device)
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_linear = x_linear / x_linear.std(dim=(1,2), keepdim=True).clamp(min=1e-6)

        with torch.no_grad():
            # Gaussian sampling
            x = torch.randn(1, L, 3, device=device)
            for t in reversed(range(50)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                x0_pred = model_g.forward_stage1(x, batch['aa_seq'], batch['chain_ids'],
                                                  batch['res_idx'], t_batch, mask)
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
                if t > 0:
                    ab_t = schedule.alpha_bar[t]
                    ab_prev = schedule.alpha_bar[t - 1]
                    beta = schedule.betas[t]
                    alpha = schedule.alphas[t]
                    coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                    coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                    mean = coef1 * x0_pred + coef2 * x
                    var = beta * (1 - ab_prev) / (1 - ab_t)
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = x0_pred
                # Recenter
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
                x = x - centroid

            g_rmse = compute_rmse(x, gt, mask).item() * sample['std']
            g_rmses.append(g_rmse)

            # Linear chain sampling
            x = x_linear.clone()
            for t in reversed(range(50)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                x0_pred = model_lc.forward_stage1(x, batch['aa_seq'], batch['chain_ids'],
                                                   batch['res_idx'], t_batch, mask)
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
                if t > 0:
                    x_linear_aligned, _ = kabsch_align(x_linear, x0_pred, mask)
                    sqrt_ab_prev = schedule.sqrt_alpha_bar[t - 1]
                    sqrt_1_minus_ab_prev = schedule.sqrt_one_minus_alpha_bar[t - 1]
                    x = sqrt_ab_prev * x0_pred + sqrt_1_minus_ab_prev * x_linear_aligned
                else:
                    x = x0_pred

            lc_rmse = compute_rmse(x, gt, mask).item() * sample['std']
            lc_rmses.append(lc_rmse)

        diff = lc_rmse - g_rmse
        print(f"{sample['sample_id'][:18]:18s} | {g_rmse:7.2f}  | {lc_rmse:10.2f}  | {diff:+.1f}")

    print("-" * 55)
    print(f"{'Average':18s} | {np.mean(g_rmses):7.2f}  | {np.mean(lc_rmses):10.2f}  | {np.mean(lc_rmses)-np.mean(g_rmses):+.1f}")


if __name__ == "__main__":
    compare_gaussian_vs_linear()
