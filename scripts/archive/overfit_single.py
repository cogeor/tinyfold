"""Overfit on a single molecule to test if model can learn it."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import load_sample_raw, collate_batch
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses.mse import compute_rmse


def compute_linearity(coords):
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return S[0] / (S.sum() + 1e-8) * 100


def run_inference(model, batch, noiser, device):
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']
    sigmas = noiser.sigmas.to(device)

    torch.manual_seed(42)
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
            mask_exp = mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            x = x - centroid
            x0_prev = x0_pred.detach()

    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/overfit_3mxg")
    output_dir.mkdir(exist_ok=True)

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

    if target_idx is None:
        print("3mxg.pdb2_3 not found!")
        return

    sample = load_sample_raw(table, target_idx)
    print(f"Sample: {sample['sample_id']}")
    print(f"  n_res: {sample['n_res']}, n_atoms: {sample['n_atoms']}, std: {sample['std']:.2f}")

    batch = collate_batch([sample], device)
    gt_centroids = batch['centroids']
    gt_lin = compute_linearity(gt_centroids[0].cpu().numpy())
    print(f"  GT linearity: {gt_lin:.1f}%")

    # Create model (fresh, untrained)
    model = ResFoldPipeline(c_token_s1=256, trunk_layers=9, denoiser_blocks=7).to(device)
    model.train()

    # Create noiser
    schedule = KarrasSchedule(sigma_min=0.0004, sigma_max=10.0, rho=7.0, n_steps=50)
    noiser = VENoiser(schedule, sigma_data=1.0).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    n_steps = 5000
    print(f"\nTraining for {n_steps} steps...")
    print("-" * 60)

    losses = []
    linearities = []

    for step in range(1, n_steps + 1):
        model.train()

        # Sample sigma
        sigma = noiser.sample_sigma_stratified(1, device)

        # Add noise
        noise = torch.randn_like(gt_centroids)
        x_noisy = gt_centroids + sigma.view(-1, 1, 1) * noise

        # Predict
        x0_pred = model.stage1.forward_sigma(
            x_noisy, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            sigma, batch['mask_res'], x0_prev=None
        )

        # Loss
        loss = ((x0_pred - gt_centroids) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 500 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                pred = run_inference(model, batch, noiser, device)
                rmse = compute_rmse(pred, gt_centroids, batch['mask_res']).item() * sample['std']
                pred_lin = compute_linearity(pred[0].cpu().numpy())

            linearities.append((step, pred_lin))
            print(f"Step {step:5d} | loss: {loss.item():.6f} | RMSE: {rmse:.2f}A | linearity: {pred_lin:.1f}% (GT: {gt_lin:.1f}%)")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = run_inference(model, batch, noiser, device)
        rmse = compute_rmse(pred, gt_centroids, batch['mask_res']).item() * sample['std']
        pred_lin = compute_linearity(pred[0].cpu().numpy())

    print("-" * 60)
    print(f"Final: RMSE={rmse:.2f}A, linearity={pred_lin:.1f}% (GT: {gt_lin:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curve
    ax = axes[0]
    ax.semilogy(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    # Linearity over training
    ax = axes[1]
    steps, lins = zip(*linearities)
    ax.plot(steps, lins, 'b-o', label='Predicted')
    ax.axhline(gt_lin, color='r', linestyle='--', label='Ground Truth')
    ax.set_xlabel("Step")
    ax.set_ylabel("Linearity %")
    ax.set_title("Linearity During Training")
    ax.legend()

    # Final prediction vs GT
    ax = axes[2]
    pred_np = pred[0].cpu().numpy()
    gt_np = gt_centroids[0].cpu().numpy()
    chain_ids = batch['chain_ids'][0].cpu().numpy()
    mask_a = chain_ids == 0
    mask_b = chain_ids == 1

    ax.scatter(gt_np[mask_a, 0], gt_np[mask_a, 1], c='blue', s=20, alpha=0.5, marker='o', label='GT A')
    ax.scatter(gt_np[mask_b, 0], gt_np[mask_b, 1], c='red', s=20, alpha=0.5, marker='o', label='GT B')
    ax.scatter(pred_np[mask_a, 0], pred_np[mask_a, 1], c='cyan', s=20, alpha=0.7, marker='x', label='Pred A')
    ax.scatter(pred_np[mask_b, 0], pred_np[mask_b, 1], c='orange', s=20, alpha=0.7, marker='x', label='Pred B')
    ax.set_title(f"Final: RMSE={rmse:.2f}A")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "overfit_3mxg.png", dpi=150)
    print(f"\nSaved plot to {output_dir / 'overfit_3mxg.png'}")


if __name__ == "__main__":
    main()
