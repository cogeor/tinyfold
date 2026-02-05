"""Diagnose linearity issue: is it trunk or denoiser? All molecules or some?"""
import json
import torch
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw, collate_batch
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser


def compute_linearity(coords):
    """Compute linearity score (0-100%). Higher = more linear."""
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def run_inference(model, batch, noiser, device, seed=None):
    """Run one inference and return final coords + linearity."""
    if seed is not None:
        torch.manual_seed(seed)

    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']
    sigmas = noiser.sigmas.to(device)

    x = sigmas[0] * torch.randn(B, L, 3, device=device)
    x0_prev = None

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = sigma.expand(B)

        with torch.no_grad():
            x0_pred = model.stage1.forward_sigma(
                x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                sigma_batch, mask, x0_prev=x0_prev
            )
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            d = (x - x0_pred) / sigma
            x = x + d * (sigma_next - sigma)

            # Re-center
            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            x = x - centroid

            x0_prev = x0_pred.detach()

    return x.cpu().numpy()


def test_trunk_tokens(model, batch, device):
    """Check if trunk tokens themselves have linearity bias."""
    model.eval()
    with torch.no_grad():
        # Get trunk tokens
        trunk_tokens = model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )
        # trunk_tokens shape: [B, L, C]
        tokens = trunk_tokens[0].cpu().numpy()  # [L, C]

        # Do PCA on tokens to see if they're linear in feature space
        centered = tokens - tokens.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Check variance explained by first few PCs
        total_var = (S**2).sum()
        explained = (S**2) / total_var * 100

        return {
            'singular_values': S[:10],
            'explained_variance': explained[:10],
            'top3_explained': explained[:3].sum(),
        }


def test_single_step_denoising(model, batch, noiser, device, sigma_level=5.0):
    """Test what model predicts from pure noise at a given sigma."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Create noisy input
    torch.manual_seed(42)
    x_noisy = sigma_level * torch.randn(B, L, 3, device=device)
    sigma_batch = torch.tensor([sigma_level], device=device).expand(B)

    with torch.no_grad():
        x0_pred = model.stage1.forward_sigma(
            x_noisy, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            sigma_batch, mask, x0_prev=None
        )

    return x0_pred.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/train_1k")

    # Load data
    with open(output_dir / "split.json") as f:
        split = json.load(f)
    train_indices = split["train_indices"]

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    table = pq.read_table(project_root / "data/processed/samples.parquet")

    # Load model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model = ResFoldPipeline(c_token_s1=256, trunk_layers=9, denoiser_blocks=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create noiser
    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=50)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Test on multiple samples
    np.random.seed(42)
    test_indices = np.random.choice(train_indices, size=20, replace=False)

    print("=" * 70)
    print("LINEARITY DIAGNOSIS")
    print("=" * 70)

    # =========================================================================
    # TEST 1: Does linearity happen to all molecules?
    # =========================================================================
    print("\n[TEST 1] Linearity across different molecules (single inference each)")
    print("-" * 70)
    print(f"{'Sample':<20} {'GT Lin%':>10} {'Pred Lin%':>10} {'Delta':>10}")
    print("-" * 70)

    gt_linearities = []
    pred_linearities = []

    for idx in test_indices:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)

        # Ground truth linearity
        gt = batch['centroids'][0].cpu().numpy()
        gt_lin = compute_linearity(gt)

        # Prediction linearity
        pred = run_inference(model, batch, noiser, device, seed=42)
        pred_lin = compute_linearity(pred[0])

        gt_linearities.append(gt_lin)
        pred_linearities.append(pred_lin)

        name = sample.get('sample_id', f'idx_{idx}')[:18]
        print(f"{name:<20} {gt_lin:>10.1f} {pred_lin:>10.1f} {pred_lin-gt_lin:>+10.1f}")

    print("-" * 70)
    print(f"{'MEAN':<20} {np.mean(gt_linearities):>10.1f} {np.mean(pred_linearities):>10.1f} {np.mean(pred_linearities)-np.mean(gt_linearities):>+10.1f}")
    print(f"{'STD':<20} {np.std(gt_linearities):>10.1f} {np.std(pred_linearities):>10.1f}")

    # =========================================================================
    # TEST 2: Does linearity vary across inference runs (same molecule)?
    # =========================================================================
    print("\n[TEST 2] Linearity across multiple inference runs (same molecule)")
    print("-" * 70)

    # Pick 3 molecules
    test_samples = [train_indices[0], train_indices[10], train_indices[20]]

    for idx in test_samples:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)
        name = sample.get('sample_id', f'idx_{idx}')[:25]

        gt_lin = compute_linearity(batch['centroids'][0].cpu().numpy())

        run_linearities = []
        for seed in range(10):
            pred = run_inference(model, batch, noiser, device, seed=seed)
            run_linearities.append(compute_linearity(pred[0]))

        print(f"{name}: GT={gt_lin:.1f}% | Pred mean={np.mean(run_linearities):.1f}% +/- {np.std(run_linearities):.1f}%")
        print(f"  Runs: {[f'{x:.0f}' for x in run_linearities]}")

    # =========================================================================
    # TEST 3: Is the bias in trunk tokens or denoiser?
    # =========================================================================
    print("\n[TEST 3] Trunk token analysis")
    print("-" * 70)

    for idx in test_samples[:3]:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)
        name = sample.get('sample_id', f'idx_{idx}')[:25]

        trunk_info = test_trunk_tokens(model, batch, device)
        print(f"{name}:")
        print(f"  Top 3 PCs explain: {trunk_info['top3_explained']:.1f}% of variance")
        print(f"  Singular values: {trunk_info['singular_values'][:5]}")

    # =========================================================================
    # TEST 4: Single-step denoising at different sigma levels
    # =========================================================================
    print("\n[TEST 4] Single-step prediction linearity at different sigma levels")
    print("-" * 70)

    sample = load_sample_raw(table, train_indices[0])
    batch = collate_batch([sample], device)
    gt_lin = compute_linearity(batch['centroids'][0].cpu().numpy())
    print(f"Sample: {sample.get('sample_id', 'unknown')}, GT linearity: {gt_lin:.1f}%")

    for sigma in [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]:
        x0_pred = test_single_step_denoising(model, batch, noiser, device, sigma)
        pred_lin = compute_linearity(x0_pred[0])
        print(f"  sigma={sigma:>5.1f}: x0_pred linearity = {pred_lin:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_increased = all(p > g + 5 for p, g in zip(pred_linearities, gt_linearities))
    mean_increase = np.mean(pred_linearities) - np.mean(gt_linearities)

    print(f"Mean GT linearity: {np.mean(gt_linearities):.1f}%")
    print(f"Mean Pred linearity: {np.mean(pred_linearities):.1f}%")
    print(f"Mean increase: {mean_increase:+.1f}%")
    print(f"All samples show increased linearity: {all_increased}")

    if mean_increase > 10:
        print("\n>>> DIAGNOSIS: Model has systematic linearity bias!")
        print("    The denoiser is collapsing structures toward linear arrangements.")


if __name__ == "__main__":
    main()
