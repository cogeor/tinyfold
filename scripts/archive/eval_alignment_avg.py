"""Evaluate average performance with/without alignment on train_1k model."""
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
from tinyfold.model.losses.mse import kabsch_align


def compute_linearity(coords):
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def run_inference(model, batch, noiser, device, align_per_step=True, seed=42):
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

            if align_per_step:
                x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            d = (x - x0_pred) / sigma
            x = x + d * (sigma_next - sigma)

            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            x = x - centroid

            x0_prev = x0_pred.detach()

    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/train_1k")

    with open(output_dir / "split.json") as f:
        split = json.load(f)
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    table = pq.read_table(project_root / "data/processed/samples.parquet")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model = ResFoldPipeline(c_token_s1=256, trunk_layers=9, denoiser_blocks=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=50)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Evaluate on subset of train and all test
    np.random.seed(42)
    eval_train = np.random.choice(train_indices, size=50, replace=False)

    print("=" * 70)
    print("AVERAGE PERFORMANCE: WITH vs WITHOUT ALIGNMENT")
    print("=" * 70)

    for split_name, indices in [("Train (50 samples)", eval_train), ("Test (100 samples)", test_indices)]:
        print(f"\n{split_name}")
        print("-" * 70)

        results = {"align": [], "no_align": []}

        for i, idx in enumerate(indices):
            sample = load_sample_raw(table, idx)
            batch = collate_batch([sample], device)
            gt = batch['centroids']
            gt_lin = compute_linearity(gt[0].cpu().numpy())

            for align in [True, False]:
                pred = run_inference(model, batch, noiser, device, align_per_step=align)
                pred_aligned, gt_aligned = kabsch_align(pred, gt)
                rmse = ((pred_aligned - gt_aligned) ** 2).mean().sqrt().item() * sample['std']
                pred_lin = compute_linearity(pred[0].cpu().numpy())

                key = "align" if align else "no_align"
                results[key].append({
                    'rmse': rmse,
                    'linearity': pred_lin,
                    'gt_linearity': gt_lin,
                    'lin_delta': pred_lin - gt_lin
                })

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(indices)}...")

        # Summary
        print(f"\n{'Metric':<25} {'With Align':>15} {'Without Align':>15} {'Improvement':>15}")
        print("-" * 70)

        align_rmse = np.mean([r['rmse'] for r in results['align']])
        no_align_rmse = np.mean([r['rmse'] for r in results['no_align']])
        print(f"{'RMSE (Angstrom)':<25} {align_rmse:>15.2f} {no_align_rmse:>15.2f} {no_align_rmse - align_rmse:>+15.2f}")

        align_lin = np.mean([r['linearity'] for r in results['align']])
        no_align_lin = np.mean([r['linearity'] for r in results['no_align']])
        gt_lin_avg = np.mean([r['gt_linearity'] for r in results['align']])
        print(f"{'Pred Linearity (%)':<25} {align_lin:>15.1f} {no_align_lin:>15.1f} {'':<15}")
        print(f"{'GT Linearity (%)':<25} {gt_lin_avg:>15.1f} {gt_lin_avg:>15.1f} {'':<15}")

        align_delta = np.mean([abs(r['lin_delta']) for r in results['align']])
        no_align_delta = np.mean([abs(r['lin_delta']) for r in results['no_align']])
        print(f"{'|Linearity Delta| (%)':<25} {align_delta:>15.1f} {no_align_delta:>15.1f} {no_align_delta - align_delta:>+15.1f}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("- If RMSE is similar: alignment doesn't hurt, just fixes linearity")
    print("- If RMSE much better with align: alignment is essential")
    print("- Recommendation: Use --align_per_step, no retraining needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
