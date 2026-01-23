#!/usr/bin/env python
"""Debug what happens during diffusion sampling."""

import torch
import pyarrow.parquet as pq
import numpy as np
import json
import sys
sys.path.insert(0, 'scripts')

from models.resfold_pipeline import ResFoldPipeline
from models import create_schedule, VENoiser, KarrasSchedule
from models.diffusion import kabsch_align_to_target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model WITH rotation augmentation
print("Loading model trained WITH rotation augmentation...")
ckpt = torch.load('outputs/overfit_rotation_fixed/best_model.pt', map_location=device)
args = ckpt['args']

model = ResFoldPipeline(
    c_token_s1=args.get('c_token_s1', 256),
    trunk_layers=args.get('trunk_layers', 9),
    denoiser_blocks=args.get('denoiser_blocks', 7),
    stage1_only=True,
)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

# Load sample
with open('outputs/overfit_rotation_fixed/split.json', 'r') as f:
    split = json.load(f)
train_indices = split['train_indices']

table = pq.read_table('data/processed/samples.parquet')

def get_sample(idx):
    row = table.slice(idx, 1)
    pdb_id = row.column('pdb_id')[0].as_py()

    atom_coords = np.array(row.column('atom_coords')[0].as_py()).reshape(-1, 3)
    atom_to_res = np.array(row.column('atom_to_res')[0].as_py())
    n_res = atom_to_res.max() + 1

    centroids = np.zeros((n_res, 3))
    counts = np.zeros(n_res)
    for i, res_idx in enumerate(atom_to_res):
        centroids[res_idx] += atom_coords[i]
        counts[res_idx] += 1
    centroids = centroids / counts[:, None]
    centroids = centroids - centroids.mean(axis=0)

    # Normalize
    std = centroids.std()
    centroids = centroids / std

    aa_seq = np.array(row.column('seq')[0].as_py())
    chain_ids = np.array(row.column('chain_id_res')[0].as_py())
    res_idx = np.arange(len(aa_seq))

    return {
        'pdb_id': pdb_id,
        'centroids': torch.tensor(centroids, dtype=torch.float32, device=device).unsqueeze(0),
        'aa_seq': torch.tensor(aa_seq, dtype=torch.long, device=device).unsqueeze(0),
        'chain_ids': torch.tensor(chain_ids, dtype=torch.long, device=device).unsqueeze(0),
        'res_idx': torch.tensor(res_idx, dtype=torch.long, device=device).unsqueeze(0),
        'mask_res': torch.ones(1, len(aa_seq), dtype=torch.bool, device=device),
        'std': std,
    }

sample = get_sample(train_indices[0])
print(f"Sample: {sample['pdb_id']}")
print(f"GT range X: [{sample['centroids'][0, :, 0].min():.2f}, {sample['centroids'][0, :, 0].max():.2f}]")
print(f"GT range Y: [{sample['centroids'][0, :, 1].min():.2f}, {sample['centroids'][0, :, 1].max():.2f}]")
print(f"GT range Z: [{sample['centroids'][0, :, 2].min():.2f}, {sample['centroids'][0, :, 2].max():.2f}]")

# Setup VE noiser
sigma_min, sigma_max = 0.002, 10.0
T = 50
karras = KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0, n_steps=T)
noiser = VENoiser(karras)

# Manual sampling with diagnostics
print("\n" + "="*60)
print("Debugging VE sampling step by step:")
print("="*60)

B, L = sample['aa_seq'].shape
sigmas = noiser.sigmas.to(device)

# Initialize
x = sigmas[0] * torch.randn(B, L, 3, device=device)
print(f"\nInitial x (sigma={sigmas[0]:.2f}):")
print(f"  X range: [{x[0, :, 0].min():.2f}, {x[0, :, 0].max():.2f}]")
print(f"  Y range: [{x[0, :, 1].min():.2f}, {x[0, :, 1].max():.2f}]")
print(f"  Z range: [{x[0, :, 2].min():.2f}, {x[0, :, 2].max():.2f}]")

x0_prev = None

# Sample every 10 steps
for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    sigma_batch = sigma.expand(B)

    with torch.no_grad():
        x0_pred = model.stage1.forward_sigma(
            x, sample['aa_seq'], sample['chain_ids'], sample['res_idx'],
            sigma_batch, sample['mask_res'], x0_prev=x0_prev
        )
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

    if i % 10 == 0:
        print(f"\nStep {i} (sigma={sigma:.4f}):")
        print(f"  x0_pred BEFORE align:")
        print(f"    X: [{x0_pred[0, :, 0].min():.2f}, {x0_pred[0, :, 0].max():.2f}]")
        print(f"    Y: [{x0_pred[0, :, 1].min():.2f}, {x0_pred[0, :, 1].max():.2f}]")
        print(f"    Z: [{x0_pred[0, :, 2].min():.2f}, {x0_pred[0, :, 2].max():.2f}]")

    # Kabsch align to current x
    x0_pred_aligned = kabsch_align_to_target(x0_pred, x, sample['mask_res'])

    if i % 10 == 0:
        print(f"  x0_pred AFTER align:")
        print(f"    X: [{x0_pred_aligned[0, :, 0].min():.2f}, {x0_pred_aligned[0, :, 0].max():.2f}]")
        print(f"    Y: [{x0_pred_aligned[0, :, 1].min():.2f}, {x0_pred_aligned[0, :, 1].max():.2f}]")
        print(f"    Z: [{x0_pred_aligned[0, :, 2].min():.2f}, {x0_pred_aligned[0, :, 2].max():.2f}]")

    x0_prev = x0_pred.detach()

    # Euler step
    d = (x - x0_pred_aligned) / sigma
    dt = sigma_next - sigma
    x = x + d * dt

    if i % 10 == 0:
        print(f"  x AFTER Euler step:")
        print(f"    X: [{x[0, :, 0].min():.2f}, {x[0, :, 0].max():.2f}]")
        print(f"    Y: [{x[0, :, 1].min():.2f}, {x[0, :, 1].max():.2f}]")
        print(f"    Z: [{x[0, :, 2].min():.2f}, {x[0, :, 2].max():.2f}]")

print(f"\nFinal x:")
print(f"  X: [{x[0, :, 0].min():.2f}, {x[0, :, 0].max():.2f}]")
print(f"  Y: [{x[0, :, 1].min():.2f}, {x[0, :, 1].max():.2f}]")
print(f"  Z: [{x[0, :, 2].min():.2f}, {x[0, :, 2].max():.2f}]")

# Compare WITHOUT rotation augmentation
print("\n" + "="*60)
print("Now testing model WITHOUT rotation augmentation:")
print("="*60)

ckpt2 = torch.load('outputs/overfit_no_rotation/best_model.pt', map_location=device)
args2 = ckpt2['args']
model2 = ResFoldPipeline(
    c_token_s1=args2.get('c_token_s1', 256),
    trunk_layers=args2.get('trunk_layers', 9),
    denoiser_blocks=args2.get('denoiser_blocks', 7),
    stage1_only=True,
)
model2.load_state_dict(ckpt2['model_state_dict'])
model2 = model2.to(device)
model2.eval()

# Same sampling
x = sigmas[0] * torch.randn(B, L, 3, device=device)
x0_prev = None

for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    sigma_batch = sigma.expand(B)

    with torch.no_grad():
        x0_pred = model2.stage1.forward_sigma(
            x, sample['aa_seq'], sample['chain_ids'], sample['res_idx'],
            sigma_batch, sample['mask_res'], x0_prev=x0_prev
        )
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

    if i % 10 == 0:
        print(f"\nStep {i} (sigma={sigma:.4f}):")
        print(f"  x0_pred (no rotation aug model):")
        print(f"    X: [{x0_pred[0, :, 0].min():.2f}, {x0_pred[0, :, 0].max():.2f}]")
        print(f"    Y: [{x0_pred[0, :, 1].min():.2f}, {x0_pred[0, :, 1].max():.2f}]")
        print(f"    Z: [{x0_pred[0, :, 2].min():.2f}, {x0_pred[0, :, 2].max():.2f}]")

    x0_pred_aligned = kabsch_align_to_target(x0_pred, x, sample['mask_res'])
    x0_prev = x0_pred.detach()

    d = (x - x0_pred_aligned) / sigma
    dt = sigma_next - sigma
    x = x + d * dt

print(f"\nFinal x (no rotation aug):")
print(f"  X: [{x[0, :, 0].min():.2f}, {x[0, :, 0].max():.2f}]")
print(f"  Y: [{x[0, :, 1].min():.2f}, {x[0, :, 1].max():.2f}]")
print(f"  Z: [{x[0, :, 2].min():.2f}, {x[0, :, 2].max():.2f}]")
