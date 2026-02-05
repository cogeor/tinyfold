#!/usr/bin/env python
"""Debug IterFold architecture to find the bug."""

import torch
import torch.nn as nn
from models.iterfold import IterFold

def test_residual_connection():
    """Test the residual connection behavior."""
    print("=" * 60)
    print("Testing residual connection behavior")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IterFold(
        c_token=64,
        trunk_layers=2,
        decoder_layers=2,
        n_atom_layers=1,
    ).to(device)

    B, L = 1, 10

    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Create GT atoms and centroids
    gt_atoms = torch.randn(B, L, 4, 3, device=device)
    gt_centroids = gt_atoms.mean(dim=2)  # [B, L, 3]

    # Case 1: All residues anchored
    print("\nCase 1: All residues anchored (anchor_pos = gt_centroids)")
    anchor_pos_all = gt_centroids.clone()
    pred_all = model(aa_seq, chain_ids, res_idx, anchor_pos_all, mask)

    # The residual adds anchor_pos to offsets, so pred = offsets + gt_centroids
    # Expected: offsets should be small (~gt_atoms - gt_centroids)
    pred_centroids_all = pred_all.mean(dim=2)
    offset_from_anchor = (pred_all - anchor_pos_all.unsqueeze(2)).abs().mean()
    print(f"  Pred centroid vs GT centroid error: {(pred_centroids_all - gt_centroids).abs().mean():.4f}")
    print(f"  Mean offset from anchor: {offset_from_anchor:.4f}")
    print(f"  Pred atoms vs GT atoms error: {(pred_all - gt_atoms).abs().mean():.4f}")

    # Case 2: No residues anchored
    print("\nCase 2: No residues anchored (anchor_pos = zeros)")
    anchor_pos_none = torch.zeros_like(gt_centroids)
    pred_none = model(aa_seq, chain_ids, res_idx, anchor_pos_none, mask)

    # The residual adds 0, so pred = offsets directly (absolute positions)
    print(f"  Pred atoms mean: {pred_none.mean():.4f}, std: {pred_none.std():.4f}")
    print(f"  GT atoms mean: {gt_atoms.mean():.4f}, std: {gt_atoms.std():.4f}")
    print(f"  Pred atoms vs GT atoms error: {(pred_none - gt_atoms).abs().mean():.4f}")

    # Case 3: 50% anchored
    print("\nCase 3: 50% anchored")
    anchor_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    anchor_mask[:, :L//2] = True
    anchor_pos_half = gt_centroids * anchor_mask.unsqueeze(-1).float()
    pred_half = model(aa_seq, chain_ids, res_idx, anchor_pos_half, mask)

    # For anchored residues: pred = offsets + gt_centroids
    # For non-anchored: pred = offsets + 0 = offsets (absolute)
    anchored_error = (pred_half[:, :L//2] - gt_atoms[:, :L//2]).abs().mean()
    non_anchored_error = (pred_half[:, L//2:] - gt_atoms[:, L//2:]).abs().mean()
    print(f"  Anchored residues error: {anchored_error:.4f}")
    print(f"  Non-anchored residues error: {non_anchored_error:.4f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print("""
For ANCHORED residues:
  - anchor_pos = gt_centroid
  - pred = network_offset + gt_centroid
  - Network learns to predict SMALL OFFSETS from centroid
  - This is EASY - backbone atoms are ~1-2A from centroid

For NON-ANCHORED residues:
  - anchor_pos = 0
  - pred = network_offset + 0 = network_offset
  - Network must predict ABSOLUTE POSITIONS
  - This requires knowing the global structure!

PROBLEM: The network learns two different tasks:
  1. Small offsets for anchored (easy)
  2. Absolute positions for non-anchored (hard, requires global info)

The network has NO position information for non-anchored residues!
It only knows sequence and which residues are anchored.
""")


def test_gradient_flow():
    """Check gradient flow through the architecture."""
    print("\n" + "=" * 60)
    print("Testing gradient flow")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IterFold(
        c_token=64,
        trunk_layers=2,
        decoder_layers=2,
        n_atom_layers=1,
    ).to(device)

    B, L = 1, 10

    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    gt_atoms = torch.randn(B, L, 4, 3, device=device)
    gt_centroids = gt_atoms.mean(dim=2)

    # 50% anchored
    anchor_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    anchor_mask[:, :L//2] = True
    anchor_pos = gt_centroids * anchor_mask.unsqueeze(-1).float()

    model.zero_grad()
    pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)

    # Compute loss only on non-anchored
    non_anchor_mask = ~anchor_mask
    loss = ((pred[:, L//2:] - gt_atoms[:, L//2:]) ** 2).mean()
    loss.backward()

    # Check gradients
    print("\nGradients for loss on NON-ANCHORED residues only:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0.001:
                print(f"  {name}: grad_norm={grad_norm:.4f}")


def test_what_model_sees():
    """Visualize what information the model has for non-anchored residues."""
    print("\n" + "=" * 60)
    print("What does the model 'see' for non-anchored residues?")
    print("=" * 60)

    print("""
For residue i that is NOT anchored:

INPUT to decoder:
  - trunk_tokens[i]: embedding from sequence (aa, chain, position)
  - anchor_pos[i] = [0, 0, 0]
  - pos_embed(anchor_pos[i]) gets REPLACED with learned unknown_embed

So the decoder sees:
  - trunk_tokens[i] + unknown_embed  (same for ALL non-anchored residues!)

The ONLY spatial information comes from:
  - Cross-attention to OTHER residues (including anchored ones)
  - But anchored residues' features come from trunk_tokens + pos_embed(gt_centroid)

PROBLEM: If residue i is far from any anchored residue,
the cross-attention provides weak signal about where i should be!

For overfitting to work, the model needs to somehow memorize
the absolute position of EVERY residue from just the sequence.
This is nearly impossible without explicit position conditioning.
""")


def propose_fixes():
    print("\n" + "=" * 60)
    print("PROPOSED FIXES")
    print("=" * 60)
    print("""
Option 1: ALWAYS provide some position hint
  - Instead of anchor_pos=0 for unknown, use noisy GT
  - anchor_pos = gt_centroid + noise * (1 - is_anchored)
  - Model always has approximate position info

Option 2: Remove residual connection, predict absolute
  - pred = network_output (no + anchor_pos)
  - But then anchored residues don't get advantage

Option 3: Iterative training (multiple forward passes)
  - Start with some anchors
  - Forward pass -> get predictions
  - Use predictions as new anchor_pos for non-anchored
  - Forward again with more "pseudo-anchors"
  - This matches inference procedure

Option 4: Much higher anchor ratio during training
  - Train with 80-90% anchored
  - Model learns local geometry well
  - Relies on iterative inference for global structure

Option 5: Add explicit position conditioning for ALL residues
  - Like diffusion: always provide noisy x for all residues
  - anchor_pos_full = gt_centroid * anchor_mask + noisy_pred * (1-anchor_mask)
  - Where noisy_pred comes from previous iteration or random init

RECOMMENDED: Option 3 or 5 - iterative training that matches inference
""")


if __name__ == '__main__':
    test_residual_connection()
    test_gradient_flow()
    test_what_model_sees()
    propose_fixes()
