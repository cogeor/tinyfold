#!/usr/bin/env python
"""Minimal overfitting test for IterFold.

Test hypothesis: The original training has too much randomness for overfitting:
1. Random anchor mask every step
2. Rotation augmentation
3. K next selection based on changing predictions

This test uses:
- Fixed 100% anchor mask (all residues known) -> predict offsets from centroid
- No rotation augmentation
- Loss on ALL atoms

If model can't overfit with 100% anchors, there's a fundamental bug.
If it CAN overfit with 100% but not partial, the issue is the training strategy.
"""

import torch
import torch.nn as nn
import os
import sys

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import pyarrow.parquet as pq
from models.iterfold import IterFold

def load_single_sample(data_path, idx=6532):
    """Load a single sample for overfitting test."""
    table = pq.read_table(data_path)

    coords = torch.tensor(table['atom_coords'][idx].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][idx].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][idx].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][idx].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center and normalize
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()
    coords = coords / std

    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'coords_res': coords_res,  # [L, 4, 3]
        'centroids': centroids,    # [L, 3]
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_res': n_res,
    }


def test_100_percent_anchors():
    """Test with 100% anchors - simplest possible case."""
    print("=" * 60)
    print("TEST 1: 100% anchors (all residues known)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    sample = load_single_sample(data_path)

    L = sample['n_res']
    print(f"Sample: {L} residues")

    # Move to device and add batch dim
    coords_res = sample['coords_res'].unsqueeze(0).to(device)  # [1, L, 4, 3]
    centroids = sample['centroids'].unsqueeze(0).to(device)    # [1, L, 3]
    aa_seq = sample['aa_seq'].unsqueeze(0).to(device)
    chain_ids = sample['chain_ids'].unsqueeze(0).to(device)
    res_idx = sample['res_idx'].unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    # Create model
    model = IterFold(
        c_token=128,
        trunk_layers=4,
        decoder_layers=4,
        n_atom_layers=2,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 100% anchors = all centroids provided
    anchor_pos = centroids.clone()  # [1, L, 3]

    print("\nTraining with 100% anchors (loss on ALL atoms):")
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()

        pred_atoms = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)

        # Loss on ALL atoms
        loss = ((pred_atoms - coords_res) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            rmse = (loss.item() ** 0.5) * sample['std']
            print(f"  Step {step:3d}: loss={loss.item():.6f}, RMSE={rmse:.4f}A")

    # Final check
    model.eval()
    with torch.no_grad():
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        final_loss = ((pred - coords_res) ** 2).mean().item()
        final_rmse = (final_loss ** 0.5) * sample['std']

    print(f"\nFinal: loss={final_loss:.6f}, RMSE={final_rmse:.4f}A")

    if final_rmse < 0.5:
        print("SUCCESS: Model can overfit with 100% anchors")
        return True
    else:
        print("FAILURE: Model CANNOT overfit even with 100% anchors - BUG!")
        return False


def test_partial_anchors_fixed():
    """Test with fixed 30% anchors (same mask every step)."""
    print("\n" + "=" * 60)
    print("TEST 2: 30% anchors, FIXED mask (same every step)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    sample = load_single_sample(data_path)

    L = sample['n_res']
    print(f"Sample: {L} residues")

    coords_res = sample['coords_res'].unsqueeze(0).to(device)
    centroids = sample['centroids'].unsqueeze(0).to(device)
    aa_seq = sample['aa_seq'].unsqueeze(0).to(device)
    chain_ids = sample['chain_ids'].unsqueeze(0).to(device)
    res_idx = sample['res_idx'].unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    model = IterFold(
        c_token=128,
        trunk_layers=4,
        decoder_layers=4,
        n_atom_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # FIXED 30% anchor mask (same every step!)
    torch.manual_seed(42)
    anchor_mask = torch.rand(1, L, device=device) < 0.3
    n_anchored = anchor_mask.sum().item()
    print(f"Anchored: {n_anchored}/{L} = {100*n_anchored/L:.0f}%")

    anchor_pos = centroids * anchor_mask.unsqueeze(-1).float()

    print("\nTraining with FIXED 30% anchors (loss on ALL atoms):")
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()

        pred_atoms = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)

        # Loss on ALL atoms
        loss = ((pred_atoms - coords_res) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            rmse = (loss.item() ** 0.5) * sample['std']
            # Also compute separate RMSE for anchored vs non-anchored
            with torch.no_grad():
                pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
                anchored_err = ((pred[anchor_mask.unsqueeze(-1).expand(-1,-1,4).unsqueeze(-1).expand(-1,-1,-1,3)] -
                                coords_res[anchor_mask.unsqueeze(-1).expand(-1,-1,4).unsqueeze(-1).expand(-1,-1,-1,3)]) ** 2)
                # Simpler approach
                diff = (pred - coords_res) ** 2  # [1, L, 4, 3]
                diff_res = diff.sum(dim=-1).sum(dim=-1)  # [1, L]
                anchored_mse = diff_res[anchor_mask].mean().item()
                non_anchored_mse = diff_res[~anchor_mask].mean().item()
            print(f"  Step {step:3d}: RMSE={rmse:.4f}A, anchored={anchored_mse**.5 * sample['std']:.4f}A, non-anchored={non_anchored_mse**.5 * sample['std']:.4f}A")

    model.eval()
    with torch.no_grad():
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        final_loss = ((pred - coords_res) ** 2).mean().item()
        final_rmse = (final_loss ** 0.5) * sample['std']

    print(f"\nFinal: RMSE={final_rmse:.4f}A")

    if final_rmse < 1.0:
        print("SUCCESS: Model can overfit with fixed 30% anchors")
        return True
    else:
        print("FAILURE: Model cannot overfit with fixed 30% anchors")
        return False


def test_partial_anchors_random():
    """Test with RANDOM 30% anchors (different mask every step)."""
    print("\n" + "=" * 60)
    print("TEST 3: 30% anchors, RANDOM mask (different every step)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    sample = load_single_sample(data_path)

    L = sample['n_res']
    print(f"Sample: {L} residues")

    coords_res = sample['coords_res'].unsqueeze(0).to(device)
    centroids = sample['centroids'].unsqueeze(0).to(device)
    aa_seq = sample['aa_seq'].unsqueeze(0).to(device)
    chain_ids = sample['chain_ids'].unsqueeze(0).to(device)
    res_idx = sample['res_idx'].unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    model = IterFold(
        c_token=128,
        trunk_layers=4,
        decoder_layers=4,
        n_atom_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining with RANDOM 30% anchors each step (loss on ALL atoms):")
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()

        # RANDOM anchor mask each step
        anchor_mask = torch.rand(1, L, device=device) < 0.3
        anchor_pos = centroids * anchor_mask.unsqueeze(-1).float()

        pred_atoms = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        loss = ((pred_atoms - coords_res) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            rmse = (loss.item() ** 0.5) * sample['std']
            print(f"  Step {step:3d}: RMSE={rmse:.4f}A")

    # Eval with fixed mask
    model.eval()
    torch.manual_seed(99)
    anchor_mask = torch.rand(1, L, device=device) < 0.3
    anchor_pos = centroids * anchor_mask.unsqueeze(-1).float()
    with torch.no_grad():
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        final_loss = ((pred - coords_res) ** 2).mean().item()
        final_rmse = (final_loss ** 0.5) * sample['std']

    print(f"\nFinal (eval with random mask): RMSE={final_rmse:.4f}A")

    if final_rmse < 2.0:
        print("SUCCESS: Model can handle random anchors")
        return True
    else:
        print("NOTE: Random anchors are harder - this is expected")
        return False


def test_zero_anchors():
    """Test with 0% anchors (all unknown) - pure sequence-to-structure."""
    print("\n" + "=" * 60)
    print("TEST 4: 0% anchors (all unknown) - sequence only")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    sample = load_single_sample(data_path)

    L = sample['n_res']
    print(f"Sample: {L} residues")

    coords_res = sample['coords_res'].unsqueeze(0).to(device)
    aa_seq = sample['aa_seq'].unsqueeze(0).to(device)
    chain_ids = sample['chain_ids'].unsqueeze(0).to(device)
    res_idx = sample['res_idx'].unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    model = IterFold(
        c_token=128,
        trunk_layers=4,
        decoder_layers=4,
        n_atom_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 0% anchors = all zeros
    anchor_pos = torch.zeros(1, L, 3, device=device)

    print("\nTraining with 0% anchors (loss on ALL atoms):")
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()

        pred_atoms = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        loss = ((pred_atoms - coords_res) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            rmse = (loss.item() ** 0.5) * sample['std']
            print(f"  Step {step:3d}: RMSE={rmse:.4f}A")

    model.eval()
    with torch.no_grad():
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        final_loss = ((pred - coords_res) ** 2).mean().item()
        final_rmse = (final_loss ** 0.5) * sample['std']

    print(f"\nFinal: RMSE={final_rmse:.4f}A")
    print("(0% anchors is hardest - model must predict structure from sequence alone)")

    return final_rmse


if __name__ == '__main__':
    print("IterFold Overfitting Diagnosis")
    print("=" * 60)
    print()

    # Test 1: 100% anchors should definitely work
    test1_pass = test_100_percent_anchors()

    # Test 2: Fixed 30% anchors should also work for overfitting
    test2_pass = test_partial_anchors_fixed()

    # Test 3: Random anchors - harder but should work with enough steps
    test3_pass = test_partial_anchors_random()

    # Test 4: 0% anchors - hardest case
    test4_rmse = test_zero_anchors()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"100% anchors:        {'PASS' if test1_pass else 'FAIL'}")
    print(f"30% fixed anchors:   {'PASS' if test2_pass else 'FAIL'}")
    print(f"30% random anchors:  {'PASS' if test3_pass else 'harder (expected)'}")
    print(f"0% anchors:          Final RMSE = {test4_rmse:.2f}A")

    if not test1_pass:
        print("\nCRITICAL BUG: Model cannot overfit even with 100% anchors!")
        print("Check: residual connection, position embedding, gradient flow")
    elif not test2_pass:
        print("\nBUG: Model cannot overfit with fixed partial anchors")
        print("The issue is with how non-anchored residues are handled")
    else:
        print("\nThe original training script's randomness prevents overfitting.")
        print("For overfitting tests, use fixed anchor masks.")
