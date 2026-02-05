#!/usr/bin/env python
"""Test rotation augmentation + Kabsch alignment to find the bug."""

import torch
import sys
sys.path.insert(0, 'scripts')

from models.training_utils import random_rotation_matrix
from tinyfold.model.losses import kabsch_align, compute_mse_loss

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create fake coordinates
B, L = 2, 50
gt = torch.randn(B, L, 3, device=device) * 10  # Ground truth

print("="*60)
print("Test 1: Kabsch alignment of rotated vs original")
print("="*60)

# Rotate gt by random rotation
R = random_rotation_matrix(B, device)
gt_rotated = torch.bmm(gt, R.transpose(1, 2))

# Kabsch align rotated back to original
aligned, target_c = kabsch_align(gt_rotated, gt)

# They should match (after centering)
gt_centered = gt - gt.mean(dim=1, keepdim=True)
diff = (aligned - gt_centered).abs().mean()
print(f"gt shape: {gt.shape}")
print(f"Rotation R shape: {R.shape}")
print(f"gt_rotated shape: {gt_rotated.shape}")
print(f"After Kabsch align, diff from gt_centered: {diff:.6f}")
print(f"(Should be ~0)")

print("\n" + "="*60)
print("Test 2: MSE loss with rotation augmentation")
print("="*60)

# Simulate training scenario:
# - x_t = R @ (gt + noise)
# - target = R @ gt
# - model predicts x0_pred
# - loss = MSE(x0_pred, target) with Kabsch

noise = torch.randn_like(gt) * 1.0  # sigma=1
x_t = gt + noise
x_t_rotated = torch.bmm(x_t, R.transpose(1, 2))
target_rotated = torch.bmm(gt, R.transpose(1, 2))

# Case A: Model predicts perfect (rotated) output
pred_perfect = target_rotated.clone()
loss_perfect = compute_mse_loss(pred_perfect, target_rotated)
print(f"Perfect prediction loss: {loss_perfect:.6f} (should be ~0)")

# Case B: Model predicts UN-rotated output (wrong frame)
pred_wrong_frame = gt.clone()
loss_wrong = compute_mse_loss(pred_wrong_frame, target_rotated)
print(f"Wrong frame prediction loss: {loss_wrong:.6f}")
print(f"(This should be ~0 too because Kabsch aligns!)")

# Case C: Model predicts collapsed output (small values)
pred_collapsed = gt.clone() * 0.1  # 10x smaller
loss_collapsed = compute_mse_loss(pred_collapsed, target_rotated)
print(f"Collapsed prediction (0.1x scale) loss: {loss_collapsed:.6f}")
print(f"(This should be HIGH because Kabsch doesn't scale)")

print("\n" + "="*60)
print("Test 3: Check Kabsch doesn't scale")
print("="*60)

# Create scaled prediction
scale = 0.1
pred_scaled = gt * scale
aligned_scaled, target_c = kabsch_align(pred_scaled, gt)

print(f"Original gt std: {gt.std():.4f}")
print(f"Scaled pred std: {pred_scaled.std():.4f}")
print(f"Aligned pred std: {aligned_scaled.std():.4f}")
print(f"(Aligned should have same std as scaled pred, NOT gt)")

# The loss should be high because scales don't match
loss = compute_mse_loss(pred_scaled, gt)
print(f"Loss for 0.1x scaled pred: {loss:.4f}")

print("\n" + "="*60)
print("Test 4: Translation augmentation")
print("="*60)

# Add translation to x_t only
translation = torch.randn(B, 1, 3, device=device) * 5.0  # Large translation
x_t_translated = x_t + translation

# Model sees translated input, should predict un-translated (centered) output
# because Kabsch centers everything

# Check if translation affects what the model should predict
x_t_centered = x_t_translated - x_t_translated.mean(dim=1, keepdim=True)
gt_centered = gt - gt.mean(dim=1, keepdim=True)

print(f"x_t mean before translation: {x_t.mean(dim=1).abs().mean():.4f}")
print(f"x_t mean after translation: {x_t_translated.mean(dim=1).abs().mean():.4f}")
print(f"After centering, x_t mean: {x_t_centered.mean(dim=1).abs().mean():.4f}")
print("Translation doesn't affect centered coordinates (Kabsch centers)")

print("\n" + "="*60)
print("Test 5: Combined rotation + translation")
print("="*60)

# Full augmentation: rotate then translate
x_t_aug = torch.bmm(x_t, R.transpose(1, 2)) + translation
target_aug = torch.bmm(gt, R.transpose(1, 2))  # Only rotated, not translated

# Perfect prediction in augmented frame
loss_perfect_aug = compute_mse_loss(target_aug, target_aug)
print(f"Perfect augmented prediction loss: {loss_perfect_aug:.6f}")

# What if model outputs un-augmented prediction?
loss_unaugmented = compute_mse_loss(gt, target_aug)
print(f"Un-augmented prediction loss: {loss_unaugmented:.6f}")
print("(Should be ~0 because Kabsch handles rotation!)")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
The bug is: Kabsch alignment makes rotation augmentation USELESS!

When we:
1. Rotate x_t by R
2. Rotate target by R
3. Model predicts something
4. Kabsch aligns prediction to target

The Kabsch will align ANY rotation of the prediction to the target.
So the model has no incentive to learn the specific rotation R.

The model learns to ignore the coordinate frame and output a "mean shape"
at arbitrary scale/rotation. With many different rotations during training,
the easiest solution is to output small values near zero.

SOLUTION: Don't use Kabsch loss OR don't use rotation augmentation.
Since Kabsch is needed for rotation invariance, remove rotation aug.
""")
