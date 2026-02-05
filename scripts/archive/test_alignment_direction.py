#!/usr/bin/env python
"""Test that Kabsch alignment direction is correct."""

import torch
import sys
sys.path.insert(0, 'src')
from tinyfold.model.losses.mse import kabsch_align, compute_mse_loss

torch.manual_seed(42)

print("="*60)
print("Testing Kabsch alignment direction")
print("="*60)

# Create GT coordinates
B, L = 2, 20
gt = torch.randn(B, L, 3) * 10

# Create a random rotation
def random_rotation(B):
    q = torch.randn(B, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=1)
    return R

R = random_rotation(B)

# Create rotated target (simulating rotation augmentation)
target_rotated = torch.bmm(gt, R.transpose(1, 2))

# Simulate model prediction: model outputs in its own frame (unrotated, close to gt)
pred = gt + torch.randn_like(gt) * 0.1  # Small noise

print(f"GT range: X=[{gt[0,:,0].min():.1f}, {gt[0,:,0].max():.1f}], "
      f"Y=[{gt[0,:,1].min():.1f}, {gt[0,:,1].max():.1f}], "
      f"Z=[{gt[0,:,2].min():.1f}, {gt[0,:,2].max():.1f}]")
print(f"Pred range: X=[{pred[0,:,0].min():.1f}, {pred[0,:,0].max():.1f}], "
      f"Y=[{pred[0,:,1].min():.1f}, {pred[0,:,1].max():.1f}], "
      f"Z=[{pred[0,:,2].min():.1f}, {pred[0,:,2].max():.1f}]")
print(f"Target (rotated) range: X=[{target_rotated[0,:,0].min():.1f}, {target_rotated[0,:,0].max():.1f}], "
      f"Y=[{target_rotated[0,:,1].min():.1f}, {target_rotated[0,:,1].max():.1f}], "
      f"Z=[{target_rotated[0,:,2].min():.1f}, {target_rotated[0,:,2].max():.1f}]")

# Now test loss with our fix: kabsch_align(target, pred)
# This should align target_rotated to pred's frame
target_aligned, pred_c = kabsch_align(target_rotated, pred)

print(f"\nAfter alignment (target->pred):")
print(f"Pred_c range: X=[{pred_c[0,:,0].min():.1f}, {pred_c[0,:,0].max():.1f}], "
      f"Y=[{pred_c[0,:,1].min():.1f}, {pred_c[0,:,1].max():.1f}], "
      f"Z=[{pred_c[0,:,2].min():.1f}, {pred_c[0,:,2].max():.1f}]")
print(f"Target_aligned range: X=[{target_aligned[0,:,0].min():.1f}, {target_aligned[0,:,0].max():.1f}], "
      f"Y=[{target_aligned[0,:,1].min():.1f}, {target_aligned[0,:,1].max():.1f}], "
      f"Z=[{target_aligned[0,:,2].min():.1f}, {target_aligned[0,:,2].max():.1f}]")

# The aligned target should now be in pred's frame (similar to GT)
# Check if target_aligned is close to gt (both centered)
gt_c = gt - gt.mean(dim=1, keepdim=True)
diff_to_gt = (target_aligned - gt_c).abs().mean()
diff_pred_to_target = (pred_c - target_aligned).abs().mean()

print(f"\nTarget_aligned distance to GT_centered: {diff_to_gt:.4f}")
print(f"Pred_c distance to target_aligned: {diff_pred_to_target:.4f}")

# Compute loss
loss = compute_mse_loss(pred, target_rotated)
print(f"\nMSE loss: {loss:.4f}")

# Compare with wrong direction (aligning pred→target)
print("\n" + "="*60)
print("What if we did it WRONG (align pred->target)?")
print("="*60)
pred_aligned_wrong, target_c_wrong = kabsch_align(pred, target_rotated)
print(f"Pred_aligned range: X=[{pred_aligned_wrong[0,:,0].min():.1f}, {pred_aligned_wrong[0,:,0].max():.1f}], "
      f"Y=[{pred_aligned_wrong[0,:,1].min():.1f}, {pred_aligned_wrong[0,:,1].max():.1f}], "
      f"Z=[{pred_aligned_wrong[0,:,2].min():.1f}, {pred_aligned_wrong[0,:,2].max():.1f}]")
print(f"Target_c range: X=[{target_c_wrong[0,:,0].min():.1f}, {target_c_wrong[0,:,0].max():.1f}], "
      f"Y=[{target_c_wrong[0,:,1].min():.1f}, {target_c_wrong[0,:,1].max():.1f}], "
      f"Z=[{target_c_wrong[0,:,2].min():.1f}, {target_c_wrong[0,:,2].max():.1f}]")

# Both should give similar loss value (Kabsch is symmetric in that sense)
# But the GRADIENT direction is what matters for learning
loss_wrong = ((pred_aligned_wrong - target_c_wrong) ** 2).mean()
print(f"MSE loss (wrong way): {loss_wrong:.4f}")
