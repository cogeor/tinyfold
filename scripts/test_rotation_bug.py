#!/usr/bin/env python
"""Test to verify rotation augmentation bug hypothesis.

The issue: For non-anchored residues, anchor_pos = 0, so the model has
NO information about which rotation R was applied. But the target is
R @ atoms, so the model must somehow infer R from anchored residues.

This test verifies:
1. Rotation matrix is correct (orthogonal, det=1)
2. Rotation is applied consistently to atoms and centroids
3. The issue is with non-anchored residues not knowing R
"""

import torch
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from models.training_utils import random_rotation_matrix


def test_rotation_matrix_valid():
    """Verify rotation matrix is valid (orthogonal, det=1)."""
    print("Test 1: Rotation matrix validity")
    print("-" * 40)

    B = 5
    R = random_rotation_matrix(B, torch.device('cpu'))

    # Check orthogonality: R @ R.T = I
    I = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    RRT = torch.bmm(R, R.transpose(1, 2))
    ortho_error = (RRT - I).abs().max().item()
    print(f"  Orthogonality error (R @ R.T - I): {ortho_error:.2e}")

    # Check determinant = 1 (proper rotation, not reflection)
    dets = torch.linalg.det(R)
    det_error = (dets - 1.0).abs().max().item()
    print(f"  Determinant error (det - 1): {det_error:.2e}")

    if ortho_error < 1e-5 and det_error < 1e-5:
        print("  PASS: Rotation matrices are valid")
        return True
    else:
        print("  FAIL: Rotation matrices are invalid!")
        return False


def test_rotation_consistency():
    """Verify centroid of rotated atoms = rotated centroid."""
    print("\nTest 2: Rotation consistency (centroid of R@atoms = R@centroid)")
    print("-" * 40)

    B, L = 2, 10
    atoms = torch.randn(B, L, 4, 3)  # [B, L, 4, 3]
    centroids = atoms.mean(dim=2)    # [B, L, 3]

    R = random_rotation_matrix(B, torch.device('cpu'))  # [B, 3, 3]

    # Rotate atoms
    atoms_flat = atoms.view(B, -1, 3)  # [B, L*4, 3]
    atoms_rot = torch.bmm(atoms_flat, R.transpose(1, 2))
    atoms_rot = atoms_rot.view(B, L, 4, 3)

    # Centroid of rotated atoms
    centroids_from_rot = atoms_rot.mean(dim=2)  # [B, L, 3]

    # Rotate original centroids
    centroids_rot = torch.bmm(centroids, R.transpose(1, 2))

    # These should be equal
    diff = (centroids_from_rot - centroids_rot).abs().max().item()
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-5:
        print("  PASS: Rotation is consistent")
        return True
    else:
        print("  FAIL: Rotation is inconsistent!")
        return False


def test_anchor_pos_issue():
    """Demonstrate the core issue with non-anchored residues."""
    print("\nTest 3: The anchor_pos issue for non-anchored residues")
    print("-" * 40)

    B, L = 1, 10

    # Original atoms and centroids
    atoms_orig = torch.randn(B, L, 4, 3)
    centroids_orig = atoms_orig.mean(dim=2)

    # Apply two different rotations
    R1 = random_rotation_matrix(B, torch.device('cpu'))
    R2 = random_rotation_matrix(B, torch.device('cpu'))

    # Rotate
    atoms_flat = atoms_orig.view(B, -1, 3)
    atoms_r1 = torch.bmm(atoms_flat, R1.transpose(1, 2)).view(B, L, 4, 3)
    atoms_r2 = torch.bmm(atoms_flat, R2.transpose(1, 2)).view(B, L, 4, 3)

    centroids_r1 = torch.bmm(centroids_orig, R1.transpose(1, 2))
    centroids_r2 = torch.bmm(centroids_orig, R2.transpose(1, 2))

    # Anchor mask: first 3 residues anchored
    anchor_mask = torch.zeros(B, L, dtype=torch.bool)
    anchor_mask[:, :3] = True

    # Build anchor_pos
    anchor_pos_r1 = centroids_r1 * anchor_mask.unsqueeze(-1).float()
    anchor_pos_r2 = centroids_r2 * anchor_mask.unsqueeze(-1).float()

    print(f"  Anchor mask: {anchor_mask[0].tolist()}")
    print(f"  ")
    print(f"  For ANCHORED residues (0,1,2):")
    print(f"    anchor_pos carries rotation info: different for R1 vs R2")
    print(f"    anchor_pos_r1[0,0] = {anchor_pos_r1[0,0].tolist()}")
    print(f"    anchor_pos_r2[0,0] = {anchor_pos_r2[0,0].tolist()}")
    print(f"  ")
    print(f"  For NON-ANCHORED residues (3,4,5,...):")
    print(f"    anchor_pos = [0,0,0] for BOTH R1 and R2!")
    print(f"    anchor_pos_r1[0,5] = {anchor_pos_r1[0,5].tolist()}")
    print(f"    anchor_pos_r2[0,5] = {anchor_pos_r2[0,5].tolist()}")
    print(f"  ")
    print(f"  But the TARGET atoms ARE different:")
    print(f"    target_r1[0,5,0] (N atom) = {atoms_r1[0,5,0].tolist()}")
    print(f"    target_r2[0,5,0] (N atom) = {atoms_r2[0,5,0].tolist()}")
    print(f"  ")
    print(f"  THE BUG: Model receives SAME input (zeros) for non-anchored residues,")
    print(f"           but is expected to produce DIFFERENT outputs (rotated atoms)!")
    print(f"  ")
    print(f"  The model must INFER the rotation R from anchored residues,")
    print(f"  which requires learning sophisticated cross-attention patterns.")
    print(f"  For overfitting, this is nearly impossible.")

    return True


def test_offset_prediction():
    """Show that offsets are NOT rotation-invariant."""
    print("\nTest 4: Offsets are NOT rotation-invariant")
    print("-" * 40)

    B, L = 1, 5

    atoms_orig = torch.randn(B, L, 4, 3)
    centroids_orig = atoms_orig.mean(dim=2)

    # Offsets in original frame
    offsets_orig = atoms_orig - centroids_orig.unsqueeze(2)  # [B, L, 4, 3]

    # Apply rotation
    R = random_rotation_matrix(B, torch.device('cpu'))

    atoms_rot = torch.bmm(atoms_orig.view(B, -1, 3), R.transpose(1, 2)).view(B, L, 4, 3)
    centroids_rot = torch.bmm(centroids_orig, R.transpose(1, 2))

    # Offsets in rotated frame
    offsets_rot = atoms_rot - centroids_rot.unsqueeze(2)

    # These are DIFFERENT (rotated versions of each other)
    print(f"  offsets_orig[0,0,0] = {offsets_orig[0,0,0].tolist()}")
    print(f"  offsets_rot[0,0,0]  = {offsets_rot[0,0,0].tolist()}")
    print(f"  ")
    print(f"  Relationship: offsets_rot = R @ offsets_orig")

    # Verify
    offsets_orig_rotated = torch.bmm(offsets_orig.view(B, -1, 3), R.transpose(1, 2))
    offsets_orig_rotated = offsets_orig_rotated.view(B, L, 4, 3)
    diff = (offsets_rot - offsets_orig_rotated).abs().max().item()
    print(f"  Verification: max diff = {diff:.2e}")
    print(f"  ")
    print(f"  So even for ANCHORED residues, the model must learn to predict")
    print(f"  ROTATED offsets, not fixed offsets. This is learnable but harder.")

    return True


def propose_fix():
    print("\n" + "=" * 60)
    print("PROPOSED FIX")
    print("=" * 60)
    print("""
The bug: Rotation augmentation changes the target atoms but
         non-anchored residues have NO information about which rotation.

Fix options:

1. DISABLE rotation augmentation for IterFold
   - The model is NOT SE(3) equivariant
   - Rotation augmentation is meant for equivariant models
   - For non-equivariant models, it just makes training harder
   - This is what made overfitting work!

2. Use rotation-INVARIANT features instead of rotation-AUGMENTED
   - Predict inter-residue distances instead of absolute coords
   - Use relative positions in local frames
   - This requires architecture changes

3. Always provide SOME position hint for ALL residues
   - Instead of anchor_pos=0, use noisy positions
   - anchor_pos = gt_centroid + noise for non-anchored
   - Model always knows approximate global orientation

RECOMMENDATION: Option 1 (disable rotation) is simplest.
Rotation augmentation doesn't help IterFold because:
- It's not equivariant
- anchor_pos already provides orientation info for anchored residues
- Non-anchored residues lose all orientation info

For diffusion models, rotation augmentation IS useful because
the noisy coordinates x_t carry orientation info for ALL residues.
""")


if __name__ == '__main__':
    print("=" * 60)
    print("Rotation Augmentation Bug Analysis")
    print("=" * 60)
    print()

    test_rotation_matrix_valid()
    test_rotation_consistency()
    test_anchor_pos_issue()
    test_offset_prediction()
    propose_fix()
