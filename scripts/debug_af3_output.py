"""Debug AF3 model output to understand sphere pattern."""
import torch
import sys
import json
import pyarrow.parquet as pq
import numpy as np
sys.path.insert(0, 'scripts')

from models import create_model

# Load model
device = torch.device('cuda')
model = create_model('af3_style', c_token=256, c_atom=128, trunk_layers=5, denoiser_blocks=5)
ckpt = torch.load('outputs/atom_diffusion_10M_v7/best_model.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# Load one sample from parquet
with open('outputs/train_10k_continuous/split.json') as f:
    split = json.load(f)
sample_id = split['train_ids'][0]

table = pq.read_table('data/processed/samples.parquet')
df = table.to_pandas()
row = df[df['sample_id'] == sample_id].iloc[0]

# Extract data
coords = np.array(row['atom_coords']).reshape(-1, 3)
aa_seq_res = np.array(row['seq'])  # per-residue
chain_ids_res = np.array(row['chain_id_res'])  # per-residue
atom_types = np.array(row['atom_type'])
atom_to_res = np.array(row['atom_to_res'])
n_atoms = len(coords)

# Expand residue-level to atom-level (4 atoms per residue)
n_res = len(aa_seq_res)
aa_seq = np.repeat(aa_seq_res, 4)[:n_atoms]
chain_ids = np.repeat(chain_ids_res, 4)[:n_atoms]

# Normalize
center = coords.mean(axis=0)
coords_centered = coords - center
std = coords_centered.std()
coords_norm = coords_centered / std

# Create batch tensors
def to_batch(arr, dtype=torch.long):
    return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

batch = {
    'coords': to_batch(coords_norm, torch.float32),
    'aa_seq': to_batch(aa_seq),
    'chain_ids': to_batch(chain_ids),
    'atom_types': to_batch(atom_types),
    'atom_to_res': to_batch(atom_to_res),
    'mask': torch.ones(1, n_atoms, dtype=torch.bool, device=device),
}

print(f"Sample: {sample_id}")
print(f"N atoms: {n_atoms}")
print(f"Std: {std:.2f}")

x0 = batch['coords']  # Ground truth [B, N, 3]
print(f"\nGround truth stats (normalized):")
print(f"  Mean: {x0.mean().item():.4f}")
print(f"  Std:  {x0.std().item():.4f}")
print(f"  Min:  {x0.min().item():.4f}")
print(f"  Max:  {x0.max().item():.4f}")

# Test at different sigma levels
sigmas = [0.01, 0.1, 1.0, 5.0, 10.0]

for sigma_val in sigmas:
    sigma = torch.tensor([sigma_val], device=device)
    noise = torch.randn_like(x0)
    x_t = x0 + sigma * noise  # VE noise

    with torch.no_grad():
        x0_pred = model.forward_sigma(
            x_t, batch['atom_types'], batch['atom_to_res'],
            batch['aa_seq'], batch['chain_ids'], sigma, batch['mask']
        )

    # Compute norms of each atom position
    pred_norms = x0_pred.norm(dim=-1)  # [B, N]

    print(f"\nSigma = {sigma_val}:")
    print(f"  Pred mean: {x0_pred.mean().item():.4f}")
    print(f"  Pred std:  {x0_pred.std().item():.4f}")
    print(f"  Pred min:  {x0_pred.min().item():.4f}")
    print(f"  Pred max:  {x0_pred.max().item():.4f}")
    print(f"  Norm mean: {pred_norms.mean().item():.4f}")
    print(f"  Norm std:  {pred_norms.std().item():.4f}")

    # Check if outputs look spherical (similar norms)
    if pred_norms.std() < 0.5:
        print(f"  WARNING: Output looks spherical (low norm variance)")
