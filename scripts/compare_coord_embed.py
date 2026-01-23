#!/usr/bin/env python
"""Compare coord_embed weights between models with and without rotation augmentation."""

import torch
import sys
sys.path.insert(0, 'scripts')
from models.resfold_pipeline import ResFoldPipeline

device = 'cpu'

print("="*60)
print("Comparing coord_embed weights")
print("="*60)

# Load model WITH rotation augmentation
ckpt1 = torch.load('outputs/overfit_rotation_fixed/best_model.pt', map_location=device)
args1 = ckpt1['args']
model1 = ResFoldPipeline(
    c_token_s1=args1.get('c_token_s1', 256),
    trunk_layers=args1.get('trunk_layers', 9),
    denoiser_blocks=args1.get('denoiser_blocks', 7),
    stage1_only=True,
)
model1.load_state_dict(ckpt1['model_state_dict'])

# Load model WITHOUT rotation augmentation
ckpt2 = torch.load('outputs/overfit_no_rotation/best_model.pt', map_location=device)
args2 = ckpt2['args']
model2 = ResFoldPipeline(
    c_token_s1=args2.get('c_token_s1', 256),
    trunk_layers=args2.get('trunk_layers', 9),
    denoiser_blocks=args2.get('denoiser_blocks', 7),
    stage1_only=True,
)
model2.load_state_dict(ckpt2['model_state_dict'])

# Compare coord_embed weights
w1 = model1.stage1.coord_embed.weight.data  # [c_token, 3]
b1 = model1.stage1.coord_embed.bias.data    # [c_token]
w2 = model2.stage1.coord_embed.weight.data
b2 = model2.stage1.coord_embed.bias.data

print("\nModel WITH rotation augmentation:")
print(f"  coord_embed weight norm: {w1.norm():.4f}")
print(f"  coord_embed bias norm: {b1.norm():.4f}")
print(f"  weight per-dim norm: X={w1[:, 0].norm():.4f}, Y={w1[:, 1].norm():.4f}, Z={w1[:, 2].norm():.4f}")

print("\nModel WITHOUT rotation augmentation:")
print(f"  coord_embed weight norm: {w2.norm():.4f}")
print(f"  coord_embed bias norm: {b2.norm():.4f}")
print(f"  weight per-dim norm: X={w2[:, 0].norm():.4f}, Y={w2[:, 1].norm():.4f}, Z={w2[:, 2].norm():.4f}")

# Compare output_proj weights
w1_out = model1.stage1.output_proj.weight.data  # [3, c_token]
w2_out = model2.stage1.output_proj.weight.data

print("\nOutput projection weights:")
print(f"  WITH rotation aug: norm={w1_out.norm():.4f}")
print(f"    per-dim: X={w1_out[0].norm():.4f}, Y={w1_out[1].norm():.4f}, Z={w1_out[2].norm():.4f}")
print(f"  WITHOUT rotation aug: norm={w2_out.norm():.4f}")
print(f"    per-dim: X={w2_out[0].norm():.4f}, Y={w2_out[1].norm():.4f}, Z={w2_out[2].norm():.4f}")

# Test: what does each model output for pure sequence input (zero coordinates)?
print("\n" + "="*60)
print("Testing model output for zero coordinate input:")
print("="*60)

# Create dummy input
B, L = 1, 50
aa_seq = torch.randint(0, 20, (B, L))
chain_ids = torch.zeros(B, L, dtype=torch.long)
res_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
mask = torch.ones(B, L, dtype=torch.bool)

# Zero coordinates
x_zero = torch.zeros(B, L, 3)
sigma = torch.tensor([1.0])

with torch.no_grad():
    out1 = model1.stage1.forward_sigma(x_zero, aa_seq, chain_ids, res_idx, sigma, mask)
    out2 = model2.stage1.forward_sigma(x_zero, aa_seq, chain_ids, res_idx, sigma, mask)

print(f"Model WITH rotation aug output for zero coords:")
print(f"  range: [{out1.min():.2f}, {out1.max():.2f}]")
print(f"  std: {out1.std():.4f}")

print(f"Model WITHOUT rotation aug output for zero coords:")
print(f"  range: [{out2.min():.2f}, {out2.max():.2f}]")
print(f"  std: {out2.std():.4f}")

# Test: what does each model output for random coordinates?
x_random = torch.randn(B, L, 3) * 2
with torch.no_grad():
    out1_rand = model1.stage1.forward_sigma(x_random, aa_seq, chain_ids, res_idx, sigma, mask)
    out2_rand = model2.stage1.forward_sigma(x_random, aa_seq, chain_ids, res_idx, sigma, mask)

print(f"\nModel WITH rotation aug output for random coords:")
print(f"  range: [{out1_rand.min():.2f}, {out1_rand.max():.2f}]")
print(f"  std: {out1_rand.std():.4f}")

print(f"Model WITHOUT rotation aug output for random coords:")
print(f"  range: [{out2_rand.min():.2f}, {out2_rand.max():.2f}]")
print(f"  std: {out2_rand.std():.4f}")

# Compare how much output changes with different coord inputs
diff1 = (out1 - out1_rand).abs().mean()
diff2 = (out2 - out2_rand).abs().mean()

print(f"\nOutput sensitivity to coordinate input:")
print(f"  WITH rotation aug: {diff1:.4f}")
print(f"  WITHOUT rotation aug: {diff2:.4f}")
print(f"  (Higher = model uses coordinates more)")
