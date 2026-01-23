#!/usr/bin/env python
"""Profile training to find bottlenecks."""

import sys
import time
import random
import torch
import pyarrow.parquet as pq
import os

# Model imports
from models import create_schedule, create_noiser
from models.resfold_pipeline import ResFoldPipeline
from tinyfold.model.losses import compute_mse_loss, compute_distance_consistency_loss

# Data loading from train_resfold
def load_sample_raw(table, i):
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
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
        'atom_to_res': atom_to_res,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)
    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)
    stds = []

    for i, s in enumerate(samples):
        L = s['n_res']
        N = s['n_atoms']
        centroids[i, :L] = s['centroids']
        coords_res[i, :L] = s['coords_res']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True
        coords[i, :N] = s['coords']
        atom_types[i, :N] = s['atom_types']
        mask_atom[i, :N] = True
        stds.append(s['std'])

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")

    print("Loading parquet table...")
    t0 = time.time()
    table = pq.read_table(data_path)
    n_total = len(table)
    print(f"  Table loaded: {time.time() - t0:.2f}s ({n_total} samples)")

    # Test sample loading at different scales
    for n_samples in [100, 1000, 5000, 10000]:
        if n_samples > n_total:
            break
        print(f"\n=== Loading {n_samples} samples ===")
        t0 = time.time()
        samples = {i: load_sample_raw(table, i) for i in range(n_samples)}
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.2f}s ({load_time/n_samples*1000:.2f}ms/sample)")

        # Check sample sizes
        sizes = [s['n_res'] for s in samples.values()]
        print(f"  Residue sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")

    # Test with full dataset loading
    print(f"\n=== Loading ALL {n_total} samples ===")
    t0 = time.time()
    samples = {i: load_sample_raw(table, i) for i in range(n_total)}
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s ({load_time/n_total*1000:.2f}ms/sample)")

    sizes = [s['n_res'] for s in samples.values()]
    print(f"  Residue sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")

    # Create model and run a few steps
    print("\n=== Training loop test ===")
    model = ResFoldPipeline(
        c_token_s1=256,
        trunk_layers=9,
        denoiser_blocks=7,
        n_timesteps=50,
        dropout=0.0,
        stage1_only=True,
    ).to(device)
    model.train()

    schedule = create_schedule("linear", T=50)
    noiser = create_noiser("gaussian", schedule)
    noiser = noiser.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    indices = list(samples.keys())
    batch_size = 32

    # Run 20 steps
    times = []
    for step in range(20):
        t0 = time.time()

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

        torch.cuda.synchronize()
        times.append(time.time() - t0)

        # Print max_res for this batch
        max_res = max(s['n_res'] for s in batch_samples)
        print(f"Step {step+1}: loss={loss.item():.4f}, time={times[-1]*1000:.0f}ms, max_res={max_res}")

    print(f"\nAvg step time: {sum(times)/len(times)*1000:.0f}ms")
    print(f"Steps/sec: {len(times)/sum(times):.1f}")


if __name__ == "__main__":
    main()
