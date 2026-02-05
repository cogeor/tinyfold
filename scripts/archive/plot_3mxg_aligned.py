"""Plot 3mxg.pdb2_3 with alignment per step."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw, collate_batch, kabsch_align_to_target
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses.mse import kabsch_align


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

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    table = pq.read_table(project_root / "data/processed/samples.parquet")

    # Find 3mxg
    target_idx = None
    for idx in range(len(table)):
        if "3mxg.pdb2_3" in table['sample_id'][idx].as_py():
            target_idx = idx
            break

    sample = load_sample_raw(table, target_idx)
    batch = collate_batch([sample], device)

    # Load model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model = ResFoldPipeline(c_token_s1=256, trunk_layers=9, denoiser_blocks=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=50)
    noiser = VENoiser(schedule, sigma_data=1.0)

    # Run inference WITH alignment
    pred = run_inference(model, batch, noiser, device, align_per_step=True)

    # Align prediction to GT for visualization
    gt = batch['centroids']
    pred_aligned, gt_aligned = kabsch_align(pred, gt)

    # Convert to numpy and scale to Angstroms
    pred_np = pred_aligned[0].cpu().numpy() * sample['std']
    gt_np = gt_aligned[0].cpu().numpy() * sample['std']
    chain_ids = batch['chain_ids'][0].cpu().numpy()

    # Compute RMSE
    rmse = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=1).mean())

    # Plot
    fig = plt.figure(figsize=(15, 5))

    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    # X-Y projection
    ax = fig.add_subplot(131)
    ax.scatter(gt_np[mask_a, 0], gt_np[mask_a, 1], c='blue', s=30, alpha=0.6, label='GT Chain A')
    ax.scatter(gt_np[mask_b, 0], gt_np[mask_b, 1], c='red', s=30, alpha=0.6, label='GT Chain B')
    ax.scatter(pred_np[mask_a, 0], pred_np[mask_a, 1], c='cyan', s=30, alpha=0.8, marker='x', label='Pred Chain A')
    ax.scatter(pred_np[mask_b, 0], pred_np[mask_b, 1], c='orange', s=30, alpha=0.8, marker='x', label='Pred Chain B')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title('X-Y Projection')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # X-Z projection
    ax = fig.add_subplot(132)
    ax.scatter(gt_np[mask_a, 0], gt_np[mask_a, 2], c='blue', s=30, alpha=0.6)
    ax.scatter(gt_np[mask_b, 0], gt_np[mask_b, 2], c='red', s=30, alpha=0.6)
    ax.scatter(pred_np[mask_a, 0], pred_np[mask_a, 2], c='cyan', s=30, alpha=0.8, marker='x')
    ax.scatter(pred_np[mask_b, 0], pred_np[mask_b, 2], c='orange', s=30, alpha=0.8, marker='x')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Z (Å)')
    ax.set_title('X-Z Projection')
    ax.set_aspect('equal')

    # Y-Z projection
    ax = fig.add_subplot(133)
    ax.scatter(gt_np[mask_a, 1], gt_np[mask_a, 2], c='blue', s=30, alpha=0.6)
    ax.scatter(gt_np[mask_b, 1], gt_np[mask_b, 2], c='red', s=30, alpha=0.6)
    ax.scatter(pred_np[mask_a, 1], pred_np[mask_a, 2], c='cyan', s=30, alpha=0.8, marker='x')
    ax.scatter(pred_np[mask_b, 1], pred_np[mask_b, 2], c='orange', s=30, alpha=0.8, marker='x')
    ax.set_xlabel('Y (Å)')
    ax.set_ylabel('Z (Å)')
    ax.set_title('Y-Z Projection')
    ax.set_aspect('equal')

    plt.suptitle(f'3mxg.pdb2_3 - With Alignment Per Step\nRMSE: {rmse:.2f} Å', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "3mxg_aligned.png", dpi=150)
    print(f"Saved to {output_dir / '3mxg_aligned.png'}")
    print(f"RMSE: {rmse:.2f} Å")


if __name__ == "__main__":
    main()
