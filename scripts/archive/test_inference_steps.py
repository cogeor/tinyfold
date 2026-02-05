"""Test if linearity depends on number of inference steps."""
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
    """Compute linearity score (0-100%)."""
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def run_inference_n_steps(model, batch, n_steps, device, seed=42):
    """Run inference with specified number of steps."""
    torch.manual_seed(seed)

    # Create noiser with specified steps
    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=n_steps)
    noiser = VENoiser(schedule, sigma_data=1.0)

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

    # Test samples - include 3mxg which showed high linearity
    # Find 3mxg
    target_idx = None
    for idx in train_indices:
        if "3mxg" in table['sample_id'][idx].as_py():
            target_idx = idx
            break

    test_samples = [target_idx] + list(np.random.choice([i for i in train_indices if i != target_idx], size=4, replace=False))

    print("=" * 70)
    print("LINEARITY vs NUMBER OF INFERENCE STEPS")
    print("=" * 70)

    step_counts = [10, 25, 50, 100, 200, 500]

    for idx in test_samples:
        sample = load_sample_raw(table, idx)
        batch = collate_batch([sample], device)
        name = sample.get('sample_id', f'idx_{idx}')[:20]

        gt = batch['centroids'][0].cpu().numpy()
        gt_lin = compute_linearity(gt)

        print(f"\n{name} (GT linearity: {gt_lin:.1f}%)")
        print(f"{'Steps':>8} {'Linearity':>12} {'Delta':>10}")
        print("-" * 32)

        for n_steps in step_counts:
            pred = run_inference_n_steps(model, batch, n_steps, device)
            pred_lin = compute_linearity(pred[0])
            print(f"{n_steps:>8} {pred_lin:>12.1f}% {pred_lin - gt_lin:>+10.1f}")


if __name__ == "__main__":
    main()
