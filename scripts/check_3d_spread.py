#!/usr/bin/env python
"""Check 3D spread of training data centroids."""

import torch
import pyarrow.parquet as pq
import numpy as np
import json
import sys
sys.path.insert(0, 'scripts')

# Load split to get training indices
with open('outputs/overfit_rotation_fixed/split.json', 'r') as f:
    split = json.load(f)

train_indices = split['train_indices']

# Load data
table = pq.read_table('data/processed/samples.parquet')

def get_centroids(row):
    """Extract centroids from a row."""
    atom_coords = np.array(row.column('atom_coords')[0].as_py()).reshape(-1, 3)
    atom_to_res = np.array(row.column('atom_to_res')[0].as_py())
    n_res = atom_to_res.max() + 1

    # Compute centroids per residue
    centroids = np.zeros((n_res, 3))
    counts = np.zeros(n_res)
    for i, res_idx in enumerate(atom_to_res):
        centroids[res_idx] += atom_coords[i]
        counts[res_idx] += 1
    centroids = centroids / counts[:, None]

    # Center
    centroids = centroids - centroids.mean(axis=0)
    return centroids

print('Checking 3D spread of centroids in TRAINING data:')
print('='*60)
for idx in train_indices[:5]:
    row = table.slice(idx, 1)
    pdb_id = row.column('pdb_id')[0].as_py()
    centroids = get_centroids(row)

    x_range = centroids[:, 0].max() - centroids[:, 0].min()
    y_range = centroids[:, 1].max() - centroids[:, 1].min()
    z_range = centroids[:, 2].max() - centroids[:, 2].min()

    print(f'{pdb_id}: X={x_range:.1f}, Y={y_range:.1f}, Z={z_range:.1f} Angstrom')
    print(f'  X: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}]')
    print(f'  Y: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}]')
    print(f'  Z: [{centroids[:, 2].min():.1f}, {centroids[:, 2].max():.1f}]')

# Now let's load a model and check what it predicts
print('\n')
print('='*60)
print('Checking model predictions:')
print('='*60)

# Load model
from models.resfold_pipeline import ResFoldPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Get a sample
idx = train_indices[0]
row = table.slice(idx, 1)
pdb_id = row.column('pdb_id')[0].as_py()
centroids = get_centroids(row)

aa_seq = np.array(row.column('seq')[0].as_py())
chain_ids = np.array(row.column('chain_id_res')[0].as_py())
res_idx = np.array(row.column('res_idx')[0].as_py())

# Convert to tensors
centroids_t = torch.tensor(centroids, dtype=torch.float32, device=device).unsqueeze(0)
aa_seq_t = torch.tensor(aa_seq, dtype=torch.long, device=device).unsqueeze(0)
chain_ids_t = torch.tensor(chain_ids, dtype=torch.long, device=device).unsqueeze(0)
res_idx_t = torch.tensor(res_idx, dtype=torch.long, device=device).unsqueeze(0)

print(f'\nSample: {pdb_id}')
print(f'GT centroids shape: {centroids_t.shape}')
print(f'GT X range: [{centroids_t[0, :, 0].min():.1f}, {centroids_t[0, :, 0].max():.1f}]')
print(f'GT Y range: [{centroids_t[0, :, 1].min():.1f}, {centroids_t[0, :, 1].max():.1f}]')
print(f'GT Z range: [{centroids_t[0, :, 2].min():.1f}, {centroids_t[0, :, 2].max():.1f}]')

# Test 1: Direct prediction with sigma=0.1 (small noise)
with torch.no_grad():
    sigma = torch.tensor([0.1], device=device)
    noise = torch.randn_like(centroids_t)
    x_t = centroids_t + sigma * noise

    x0_pred = model.stage1.forward_sigma(x_t, aa_seq_t, chain_ids_t, res_idx_t, sigma)

    print(f'\nWith sigma=0.1 (input close to GT):')
    print(f'  Pred X range: [{x0_pred[0, :, 0].min():.1f}, {x0_pred[0, :, 0].max():.1f}]')
    print(f'  Pred Y range: [{x0_pred[0, :, 1].min():.1f}, {x0_pred[0, :, 1].max():.1f}]')
    print(f'  Pred Z range: [{x0_pred[0, :, 2].min():.1f}, {x0_pred[0, :, 2].max():.1f}]')

# Test 2: Direct prediction with sigma=10 (large noise)
with torch.no_grad():
    sigma = torch.tensor([10.0], device=device)
    noise = torch.randn_like(centroids_t)
    x_t = centroids_t + sigma * noise

    x0_pred = model.stage1.forward_sigma(x_t, aa_seq_t, chain_ids_t, res_idx_t, sigma)

    print(f'\nWith sigma=10 (high noise):')
    print(f'  Input X range: [{x_t[0, :, 0].min():.1f}, {x_t[0, :, 0].max():.1f}]')
    print(f'  Input Y range: [{x_t[0, :, 1].min():.1f}, {x_t[0, :, 1].max():.1f}]')
    print(f'  Input Z range: [{x_t[0, :, 2].min():.1f}, {x_t[0, :, 2].max():.1f}]')
    print(f'  Pred X range: [{x0_pred[0, :, 0].min():.1f}, {x0_pred[0, :, 0].max():.1f}]')
    print(f'  Pred Y range: [{x0_pred[0, :, 1].min():.1f}, {x0_pred[0, :, 1].max():.1f}]')
    print(f'  Pred Z range: [{x0_pred[0, :, 2].min():.1f}, {x0_pred[0, :, 2].max():.1f}]')

# Test 3: Pure noise input (sigma=10, starting from pure noise)
with torch.no_grad():
    sigma = torch.tensor([10.0], device=device)
    x_t = 10 * torch.randn_like(centroids_t)  # Pure noise, no GT

    x0_pred = model.stage1.forward_sigma(x_t, aa_seq_t, chain_ids_t, res_idx_t, sigma)

    print(f'\nWith pure noise input (x_t is just noise, no GT):')
    print(f'  Input X range: [{x_t[0, :, 0].min():.1f}, {x_t[0, :, 0].max():.1f}]')
    print(f'  Input Y range: [{x_t[0, :, 1].min():.1f}, {x_t[0, :, 1].max():.1f}]')
    print(f'  Input Z range: [{x_t[0, :, 2].min():.1f}, {x_t[0, :, 2].max():.1f}]')
    print(f'  Pred X range: [{x0_pred[0, :, 0].min():.1f}, {x0_pred[0, :, 0].max():.1f}]')
    print(f'  Pred Y range: [{x0_pred[0, :, 1].min():.1f}, {x0_pred[0, :, 1].max():.1f}]')
    print(f'  Pred Z range: [{x0_pred[0, :, 2].min():.1f}, {x0_pred[0, :, 2].max():.1f}]')

# Test 4: Check output_proj weights
print('\n')
print('='*60)
print('Checking output_proj weights:')
print('='*60)
w = model.stage1.output_proj.weight.data  # [3, c_token]
b = model.stage1.output_proj.bias.data    # [3]
print(f'output_proj weight shape: {w.shape}')
print(f'output_proj bias: {b}')
print(f'Weight norm per output dim:')
print(f'  X: {w[0].norm():.4f}')
print(f'  Y: {w[1].norm():.4f}')
print(f'  Z: {w[2].norm():.4f}')

# Test 5: Check what the trunk produces
print('\n')
print('='*60)
print('Checking trunk output:')
print('='*60)
with torch.no_grad():
    trunk_tokens = model.stage1.trunk(aa_seq_t, chain_ids_t, res_idx_t, None)
    print(f'Trunk tokens shape: {trunk_tokens.shape}')
    print(f'Trunk tokens mean: {trunk_tokens.mean():.4f}')
    print(f'Trunk tokens std: {trunk_tokens.std():.4f}')
    print(f'Trunk tokens range: [{trunk_tokens.min():.4f}, {trunk_tokens.max():.4f}]')
