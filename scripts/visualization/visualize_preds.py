#!/usr/bin/env python
"""Visualize predictions vs ground truth for a few proteins."""

import math
import torch
import torch.nn as nn
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Copy model and utilities from train_scale.py
class AttentionDiffusionV2(nn.Module):
    def __init__(self, h_dim=64, n_heads=4, n_layers=4, n_timesteps=50, dropout=0.0, n_aa_types=21, n_chains=2):
        super().__init__()
        self.h_dim = h_dim
        self.n_timesteps = n_timesteps
        self.atom_type_embed = nn.Embedding(4, h_dim // 4)
        self.aa_embed = nn.Embedding(n_aa_types, h_dim)
        self.chain_embed = nn.Embedding(n_chains, h_dim // 4)
        self.time_embed = nn.Embedding(n_timesteps, h_dim)
        self.coord_proj = nn.Linear(3, h_dim)
        input_dim = (h_dim // 4) + h_dim + (h_dim // 4) + h_dim + h_dim + h_dim
        self.input_proj = nn.Linear(input_dim, h_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_heads, dim_feedforward=h_dim * 4,
                                                    dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(h_dim)
        self.output_proj = nn.Linear(h_dim, 3)

    def sinusoidal_pos_enc(self, positions, dim):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x_t, atom_types, atom_to_res, aa_seq, chain_ids, t, mask=None):
        B, N, _ = x_t.shape
        atom_emb = self.atom_type_embed(atom_types)
        aa_emb = self.aa_embed(aa_seq)
        chain_emb = self.chain_embed(chain_ids)
        res_emb = self.sinusoidal_pos_enc(atom_to_res, self.h_dim)
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)
        coord_emb = self.coord_proj(x_t)
        h = torch.cat([atom_emb, aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
        h = self.input_proj(h)
        h = self.transformer(h, src_key_padding_mask=(~mask if mask is not None else None))
        h = self.output_norm(h)
        return self.output_proj(h)


class CosineSchedule:
    def __init__(self, T=50, s=0.008):
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1]])
        self.betas = 1 - self.alphas

    def to(self, device):
        for attr in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'alphas', 'betas']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


@torch.no_grad()
def ddpm_sample(model, atom_types, atom_to_res, aa_seq, chain_ids, schedule, mask, clamp_val=3.0):
    device = atom_types.device
    B, N = atom_types.shape
    x = torch.randn(B, N, 3, device=device)
    for t in reversed(range(schedule.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
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
    return x


def kabsch_align(pred, target):
    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    pred_c = pred - pred_mean
    target_c = target - target_mean
    H = pred_c.T @ target_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    D = torch.eye(3, device=pred.device)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    pred_aligned = pred_c @ R.T
    return pred_aligned + target_mean, target


def load_sample(table, i, device):
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
    coords_norm = coords / std
    return {
        'coords': coords,
        'coords_norm': coords_norm.unsqueeze(0).to(device),
        'atom_types': atom_types.unsqueeze(0).to(device),
        'atom_to_res': atom_to_res.unsqueeze(0).to(device),
        'aa_seq': aa_seq.unsqueeze(0).to(device),
        'chain_ids': chain_ids.unsqueeze(0).to(device),
        'mask': torch.ones(1, n_atoms, dtype=torch.bool, device=device),
        'std': std.item(),
        'sample_id': table['sample_id'][i].as_py(),
        'chain_ids_raw': chain_res[atom_to_res],
    }


def plot_protein(ax, coords, chain_ids, title, color_a='blue', color_b='red', alpha=1.0):
    """Plot protein backbone with chain coloring."""
    coords = coords.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    # Plot chain A
    mask_a = chain_ids == 0
    if mask_a.any():
        ax.scatter(coords[mask_a, 0], coords[mask_a, 1], coords[mask_a, 2],
                   c=color_a, s=10, alpha=alpha, label='Chain A')

    # Plot chain B
    mask_b = chain_ids == 1
    if mask_b.any():
        ax.scatter(coords[mask_b, 0], coords[mask_b, 1], coords[mask_b, 2],
                   c=color_b, s=10, alpha=alpha, label='Chain B')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    table = pq.read_table("data/processed/samples.parquet")

    # Find medium-sized samples
    medium_indices = []
    for i in range(len(table)):
        n_atoms = len(table['atom_type'][i].as_py())
        if 200 <= n_atoms <= 400:
            medium_indices.append(i)

    # Create model and load from training (if exists)
    model = AttentionDiffusionV2(h_dim=64, n_heads=4, n_layers=4, n_timesteps=50, dropout=0.0).to(device)
    schedule = CosineSchedule(T=50).to(device)

    # Quick train on first 10 samples to get a model
    print("Quick training on 10 samples for visualization...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_indices = medium_indices[:10]

    import random
    for step in range(5000):
        idx = random.choice(train_indices)
        data = load_sample(table, idx, device)
        t = torch.randint(0, schedule.T, (1,), device=device)
        noise = torch.randn_like(data['coords_norm'])
        sqrt_ab = schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * data['coords_norm'] + sqrt_1_ab * noise

        x0_pred = model(x_t, data['atom_types'], data['atom_to_res'],
                        data['aa_seq'], data['chain_ids'], t, data['mask'])

        # Kabsch loss
        pred = x0_pred[0]
        target = data['coords_norm'][0]
        pred_aligned, target_c = kabsch_align(pred, target)
        loss = ((pred_aligned - target_c) ** 2).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    # Visualize predictions for a few samples
    model.eval()
    fig = plt.figure(figsize=(16, 12))

    samples_to_plot = train_indices[:4]

    for i, idx in enumerate(samples_to_plot):
        data = load_sample(table, idx, device)

        with torch.no_grad():
            pred_norm = ddpm_sample(model, data['atom_types'], data['atom_to_res'],
                                     data['aa_seq'], data['chain_ids'], schedule, data['mask'])
            pred = pred_norm[0] * data['std']

            # Align prediction to ground truth
            pred_aligned, target = kabsch_align(pred, data['coords'].to(device))

            rmse = torch.sqrt(((pred_aligned - target) ** 2).sum(dim=-1).mean()).item()

        # Ground truth
        ax1 = fig.add_subplot(2, 4, i + 1, projection='3d')
        plot_protein(ax1, data['coords'], data['chain_ids_raw'],
                     f"{data['sample_id']}\nGround Truth")

        # Prediction
        ax2 = fig.add_subplot(2, 4, i + 5, projection='3d')
        plot_protein(ax2, pred_aligned.cpu(), data['chain_ids_raw'],
                     f"Prediction\nRMSE: {rmse:.2f} Å", color_a='cyan', color_b='orange')

    plt.tight_layout()
    plt.savefig('outputs/predictions_viz.png', dpi=150)
    print("Saved to outputs/predictions_viz.png")
    plt.close()


if __name__ == "__main__":
    main()
