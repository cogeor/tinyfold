#!/usr/bin/env python
"""Test that rotation matrices are valid."""

import torch
import sys
sys.path.insert(0, 'scripts')
from models.training_utils import random_rotation_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Testing rotation matrix generation:")
print("="*60)

# Generate rotation matrices
B = 10
R = random_rotation_matrix(B, device)

print(f"Shape: {R.shape}")

# Check orthogonality: R @ R^T = I
RRT = torch.bmm(R, R.transpose(1, 2))
I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
ortho_error = (RRT - I).abs().max()
print(f"Orthogonality error (R @ R^T - I): {ortho_error:.6f}")

# Check determinant = 1 (proper rotation, not reflection)
det = torch.det(R)
print(f"Determinants: {det}")
print(f"Det error from 1: {(det - 1).abs().max():.6f}")

# Test that rotation preserves distances
print("\n" + "="*60)
print("Testing that rotation preserves distances:")
print("="*60)

# Create random point cloud
L = 50
x = torch.randn(B, L, 3, device=device) * 10

# Rotate it
x_rotated = torch.bmm(x, R.transpose(1, 2))

# Check distances preserved
# Pairwise distances before and after
x_flat = x.view(B, L, 1, 3)
dist_before = ((x_flat - x_flat.transpose(1, 2)) ** 2).sum(-1).sqrt()

x_rot_flat = x_rotated.view(B, L, 1, 3)
dist_after = ((x_rot_flat - x_rot_flat.transpose(1, 2)) ** 2).sum(-1).sqrt()

dist_error = (dist_before - dist_after).abs().max()
print(f"Max distance error: {dist_error:.6f}")

# Check that rotation changes coordinates
coord_change = (x - x_rotated).abs().mean()
print(f"Mean coordinate change after rotation: {coord_change:.2f}")

# Check that different batches get different rotations
R_diff = (R[0] - R[1]).abs().mean()
print(f"Difference between R[0] and R[1]: {R_diff:.4f}")

print("\n" + "="*60)
print("Testing bmm rotation application:")
print("="*60)

# Test that bmm applies rotation correctly
# For row vectors, x @ R^T rotates x by R
p = torch.tensor([[[1.0, 0.0, 0.0]]], device=device)  # Point on x-axis
R_90z = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]], device=device)  # 90 deg around z

p_rot = torch.bmm(p, R_90z.transpose(1, 2))
print(f"Point [1,0,0] rotated 90 deg around Z:")
print(f"  Expected: [0, 1, 0]")
print(f"  Got:      {p_rot[0, 0].tolist()}")

# The transpose is needed because our R is for column vectors
# x @ R^T = (R @ x^T)^T
# So if R rotates column vectors, then x @ R^T rotates row vectors
p_rot2 = torch.bmm(p, R_90z)  # Without transpose
print(f"Without transpose: {p_rot2[0, 0].tolist()}")

print("\n" + "="*60)
print("Conclusion:")
print("="*60)
if ortho_error < 1e-5 and (det - 1).abs().max() < 1e-5 and dist_error < 1e-5:
    print("Rotation matrices are VALID")
else:
    print("Rotation matrices have ISSUES!")
