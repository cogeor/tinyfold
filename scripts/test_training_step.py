#!/usr/bin/env python
"""Test a single training step with rotation augmentation."""

import torch
import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

from models.resfold_pipeline import ResFoldPipeline
from models.training_utils import random_rotation_matrix
from tinyfold.model.losses.mse import kabsch_align, compute_mse_loss

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
model = ResFoldPipeline(
    c_token_s1=256,
    trunk_layers=9,
    denoiser_blocks=7,
    stage1_only=True,
)
model = model.to(device)
model.train()

# Create fake batch
B, L = 2, 50
gt = torch.randn(B, L, 3, device=device) * 2  # Normalized coords
aa_seq = torch.randint(0, 20, (B, L), device=device)
chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
mask = torch.ones(B, L, dtype=torch.bool, device=device)

print("="*60)
print("Testing training step WITH rotation augmentation")
print("="*60)

# Add noise
sigma = torch.tensor([1.0], device=device)
noise = torch.randn_like(gt)
x_t = gt + sigma * noise

# Apply rotation augmentation
R = random_rotation_matrix(B, device)
x_t_rotated = torch.bmm(x_t, R.transpose(1, 2))
target_rotated = torch.bmm(gt, R.transpose(1, 2))

print(f"GT range: X=[{gt[0,:,0].min():.2f}, {gt[0,:,0].max():.2f}], "
      f"Y=[{gt[0,:,1].min():.2f}, {gt[0,:,1].max():.2f}], "
      f"Z=[{gt[0,:,2].min():.2f}, {gt[0,:,2].max():.2f}]")
print(f"x_t range: X=[{x_t[0,:,0].min():.2f}, {x_t[0,:,0].max():.2f}], "
      f"Y=[{x_t[0,:,1].min():.2f}, {x_t[0,:,1].max():.2f}], "
      f"Z=[{x_t[0,:,2].min():.2f}, {x_t[0,:,2].max():.2f}]")
print(f"x_t_rotated range: X=[{x_t_rotated[0,:,0].min():.2f}, {x_t_rotated[0,:,0].max():.2f}], "
      f"Y=[{x_t_rotated[0,:,1].min():.2f}, {x_t_rotated[0,:,1].max():.2f}], "
      f"Z=[{x_t_rotated[0,:,2].min():.2f}, {x_t_rotated[0,:,2].max():.2f}]")
print(f"target_rotated range: X=[{target_rotated[0,:,0].min():.2f}, {target_rotated[0,:,0].max():.2f}], "
      f"Y=[{target_rotated[0,:,1].min():.2f}, {target_rotated[0,:,1].max():.2f}], "
      f"Z=[{target_rotated[0,:,2].min():.2f}, {target_rotated[0,:,2].max():.2f}]")

# Forward pass
pred = model.stage1.forward_sigma(x_t_rotated, aa_seq, chain_ids, res_idx, sigma.expand(B), mask)

print(f"\nModel prediction range:")
print(f"  X=[{pred[0,:,0].min():.2f}, {pred[0,:,0].max():.2f}]")
print(f"  Y=[{pred[0,:,1].min():.2f}, {pred[0,:,1].max():.2f}]")
print(f"  Z=[{pred[0,:,2].min():.2f}, {pred[0,:,2].max():.2f}]")

# Compute loss (with our fix: align target->pred)
loss = compute_mse_loss(pred, target_rotated, mask)
print(f"\nMSE loss: {loss:.4f}")

# Backward pass
loss.backward()

# Check gradients
print(f"\nGradient check:")
print(f"  output_proj.weight grad norm: {model.stage1.output_proj.weight.grad.norm():.6f}")
print(f"  coord_embed.weight grad norm: {model.stage1.coord_embed.weight.grad.norm():.6f}")

# Check gradient per output dimension
grad = model.stage1.output_proj.weight.grad
print(f"  output_proj grad per dim: X={grad[0].norm():.6f}, Y={grad[1].norm():.6f}, Z={grad[2].norm():.6f}")

print("\n" + "="*60)
print("Testing WITHOUT rotation augmentation for comparison")
print("="*60)

model.zero_grad()

# Forward without rotation
pred_no_rot = model.stage1.forward_sigma(x_t, aa_seq, chain_ids, res_idx, sigma.expand(B), mask)

print(f"Model prediction (no rotation) range:")
print(f"  X=[{pred_no_rot[0,:,0].min():.2f}, {pred_no_rot[0,:,0].max():.2f}]")
print(f"  Y=[{pred_no_rot[0,:,1].min():.2f}, {pred_no_rot[0,:,1].max():.2f}]")
print(f"  Z=[{pred_no_rot[0,:,2].min():.2f}, {pred_no_rot[0,:,2].max():.2f}]")

loss_no_rot = compute_mse_loss(pred_no_rot, gt, mask)
print(f"MSE loss (no rotation): {loss_no_rot:.4f}")

loss_no_rot.backward()
grad_no_rot = model.stage1.output_proj.weight.grad
print(f"output_proj grad per dim (no rot): X={grad_no_rot[0].norm():.6f}, Y={grad_no_rot[1].norm():.6f}, Z={grad_no_rot[2].norm():.6f}")
