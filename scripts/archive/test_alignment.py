"""Test if alignment causes the linearity issue."""
import json
import torch
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw, collate_batch, kabsch_align_to_target
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses.mse import compute_rmse


def compute_linearity(coords):
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def run_inference_configurable(model, batch, noiser, device, align_per_step=True, recenter=True, seed=42):
    """Run inference with configurable alignment and recentering."""
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

            # Optional: Kabsch align x0_pred to x
            if align_per_step:
                x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            # Euler step
            d = (x - x0_pred) / sigma
            x = x + d * (sigma_next - sigma)

            # Optional: Re-center
            if recenter:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
                x = x - centroid

            x0_prev = x0_pred.detach()

    return x


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

    # Load trained model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model = ResFoldPipeline(c_token_s1=256, trunk_layers=9, denoiser_blocks=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create noiser
    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=50)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Find test samples - 3mxg and a few others
    test_samples = []
    for idx in train_indices:
        sample_id = table['sample_id'][idx].as_py()
        if "3mxg.pdb2_3" in sample_id:
            test_samples.insert(0, idx)  # Put 3mxg first
        elif len(test_samples) < 5:
            test_samples.append(idx)

    # Make sure we have 3mxg + 4 others
    test_samples = test_samples[:5]

    print("=" * 80)
    print("TESTING ALIGNMENT HYPOTHESIS")
    print("=" * 80)

    configs = [
        ("align=True, recenter=True", True, True),
        ("align=False, recenter=True", False, True),
        ("align=True, recenter=False", True, False),
        ("align=False, recenter=False", False, False),
    ]

    for idx in test_samples:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)
        name = sample.get('sample_id', f'idx_{idx}')[:20]

        gt = batch['centroids'][0].cpu().numpy()
        gt_lin = compute_linearity(gt)

        print(f"\n{name} (GT linearity: {gt_lin:.1f}%)")
        print("-" * 60)
        print(f"{'Config':<35} {'Linearity':>12} {'Delta':>10} {'RMSE':>10}")
        print("-" * 60)

        for config_name, align, recenter in configs:
            pred = run_inference_configurable(
                model, batch, noiser, device,
                align_per_step=align, recenter=recenter
            )
            pred_np = pred[0].cpu().numpy()
            pred_lin = compute_linearity(pred_np)

            # Compute RMSE (need to align for fair comparison)
            from tinyfold.model.losses.mse import kabsch_align
            pred_aligned, gt_aligned = kabsch_align(pred, batch['centroids'])
            rmse = ((pred_aligned - gt_aligned) ** 2).mean().sqrt().item() * sample['std']

            print(f"{config_name:<35} {pred_lin:>12.1f}% {pred_lin - gt_lin:>+10.1f} {rmse:>10.2f}A")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("- If 'align=False' fixes linearity: alignment is the cause")
    print("- If 'recenter=False' fixes linearity: recentering is the cause")
    print("- If neither fixes it: issue is in the model's predictions themselves")
    print("=" * 80)


if __name__ == "__main__":
    main()
