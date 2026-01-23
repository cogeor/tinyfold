#!/usr/bin/env python
"""Debug why convergence differs between 1 sample and many samples."""

import sys
import time
import random
import torch
import pyarrow.parquet as pq
import os

from models import create_schedule, create_noiser
from models.resfold_pipeline import ResFoldPipeline
from tinyfold.model.losses import compute_mse_loss, compute_distance_consistency_loss


def load_sample_raw(table, i):
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
    }


def collate_batch(samples, device):
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)

    centroids = torch.zeros(B, max_res, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    for i, s in enumerate(samples):
        L = s['n_res']
        centroids[i, :L] = s['centroids']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True

    return {
        'centroids': centroids.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
    }


def check_encoding_differences(samples, device):
    """Check if different samples produce different trunk encodings."""
    print("\n=== Checking encoding differences ===")

    # Create small model
    model = ResFoldPipeline(
        c_token_s1=64,  # Small
        trunk_layers=2,
        denoiser_blocks=2,
        n_timesteps=50,
        dropout=0.0,
        stage1_only=True,
    ).to(device)
    model.eval()

    encodings = []
    for idx, sample in list(samples.items())[:10]:
        batch = collate_batch([sample], device)

        with torch.no_grad():
            # Get trunk tokens (the sequence encoding)
            trunk_tokens = model.stage1.get_trunk_tokens(
                batch['aa_seq'], batch['chain_ids'],
                batch['res_idx'], batch['mask_res']
            )
            # Take mean over sequence dimension
            enc_mean = trunk_tokens[0, :sample['n_res']].mean(dim=0)
            encodings.append(enc_mean)

        print(f"  Sample {idx}: n_res={sample['n_res']}, aa_seq[:5]={sample['aa_seq'][:5].tolist()}, enc_norm={enc_mean.norm().item():.4f}")

    # Check pairwise similarities
    print("\n  Pairwise cosine similarities:")
    for i in range(min(5, len(encodings))):
        for j in range(i+1, min(5, len(encodings))):
            cos_sim = torch.nn.functional.cosine_similarity(
                encodings[i].unsqueeze(0), encodings[j].unsqueeze(0)
            ).item()
            print(f"    {i} vs {j}: {cos_sim:.4f}")

    del model
    torch.cuda.empty_cache()


def train_test(name, samples, n_steps=500, batch_size=1, c_token=64, trunk_layers=2, denoiser_blocks=2):
    """Train and report convergence."""
    device = torch.device("cuda")

    model = ResFoldPipeline(
        c_token_s1=c_token,
        trunk_layers=trunk_layers,
        denoiser_blocks=denoiser_blocks,
        n_timesteps=50,
        dropout=0.0,
        stage1_only=True,
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())

    schedule = create_schedule("linear", T=50)
    noiser = create_noiser("gaussian", schedule)
    noiser = noiser.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    indices = list(samples.keys())

    print(f"\n=== {name} ===")
    print(f"  Samples: {len(samples)}, batch_size: {batch_size}, params: {n_params/1e6:.2f}M")

    losses = []
    for step in range(1, n_steps + 1):
        batch_indices = random.choices(indices, k=batch_size)
        batch_samples = [samples[idx] for idx in batch_indices]
        batch = collate_batch(batch_samples, device)

        t = torch.randint(0, noiser.T, (batch_size,), device=device)
        noise = torch.randn_like(batch['centroids'])
        sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * batch['centroids'] + sqrt_one_minus_ab * noise

        centroids_pred = model.forward_stage1(
            x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t, batch['mask_res']
        )

        loss_mse = compute_mse_loss(centroids_pred, batch['centroids'], batch['mask_res'])
        loss_dist = compute_distance_consistency_loss(centroids_pred, batch['centroids'], batch['mask_res'])
        loss = loss_mse + 0.1 * loss_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step in [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
            print(f"  Step {step:4d}: loss={loss.item():.4f}, mse={loss_mse.item():.4f}")

    del model
    torch.cuda.empty_cache()
    return losses


def main():
    device = torch.device("cuda")
    print(f"Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")

    print("Loading data...")
    table = pq.read_table(data_path)

    # Filter to small proteins (< 300 residues = 1200 atoms)
    small_indices = [i for i in range(min(5000, len(table)))
                    if len(table['atom_type'][i].as_py()) <= 1200]
    print(f"Found {len(small_indices)} small proteins")

    # Load 10 samples
    samples_10 = {i: load_sample_raw(table, small_indices[i]) for i in range(10)}

    # Check encoding differences
    check_encoding_differences(samples_10, device)

    # Test with 1M model (small)
    print("\n" + "="*70)
    print("Testing with ~1M param model (c_token=64, trunk=2, denoiser=2)")
    print("="*70)

    # Test 1: 10 samples, bs=10 with small model
    train_test("10 samples, bs=10, ~1M model", samples_10, n_steps=500, batch_size=10,
               c_token=64, trunk_layers=2, denoiser_blocks=2)

    # Test 2: Same sample repeated 10x, bs=10 with small model
    samples_1_repeated = {i: samples_10[0] for i in range(10)}
    train_test("1 sample x10, bs=10, ~1M model", samples_1_repeated, n_steps=500, batch_size=10,
               c_token=64, trunk_layers=2, denoiser_blocks=2)


if __name__ == "__main__":
    main()
