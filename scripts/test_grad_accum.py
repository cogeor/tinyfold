#!/usr/bin/env python
"""Test gradient accumulation with the fixed per-sample loss."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tinyfold.model.losses import compute_mse_loss

def test_fixed_grad_accum():
    print("=== Testing FIXED per-sample loss with gradient accumulation ===\n")

    torch.manual_seed(42)
    model = nn.Linear(10, 3, bias=False)

    # Create batch with variable-length sequences (padded)
    B, L = 4, 10
    x = torch.randn(B, L, 10)
    t = torch.randn(B, L, 3)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :5] = True   # 5 valid
    mask[1, :8] = True   # 8 valid
    mask[2, :3] = True   # 3 valid
    mask[3, :6] = True   # 6 valid

    # Method 1: Full batch with fixed loss
    model.zero_grad()
    out = model(x.view(-1, 10)).view(B, L, 3)
    loss_full = compute_mse_loss(out, t, mask, use_kabsch=False)
    loss_full.backward()
    grad_full = model.weight.grad.clone()
    print(f"Full batch (4 samples):")
    print(f"  Loss: {loss_full.item():.6f}")
    print(f"  Grad norm: {grad_full.norm().item():.6f}")

    # Method 2: Two half-batches with standard grad accum
    model.zero_grad()
    grad_accum = 2

    # First half
    out1 = model(x[:2].view(-1, 10)).view(2, L, 3)
    loss1 = compute_mse_loss(out1, t[:2], mask[:2], use_kabsch=False)
    (loss1 / grad_accum).backward()

    # Second half
    out2 = model(x[2:].view(-1, 10)).view(2, L, 3)
    loss2 = compute_mse_loss(out2, t[2:], mask[2:], use_kabsch=False)
    (loss2 / grad_accum).backward()

    grad_accum_result = model.weight.grad.clone()
    print(f"\nTwo half-batches (grad_accum=2):")
    print(f"  Loss 1: {loss1.item():.6f}")
    print(f"  Loss 2: {loss2.item():.6f}")
    print(f"  Avg loss: {(loss1.item() + loss2.item())/2:.6f}")
    print(f"  Grad norm: {grad_accum_result.norm().item():.6f}")
    print(f"  Grad diff from full: {(grad_full - grad_accum_result).abs().max().item():.6f}")

    # Check
    print(f"\n=== Result ===")
    diff = (grad_full - grad_accum_result).abs().max().item()
    if diff < 1e-5:
        print(f"SUCCESS: Gradient accumulation matches full batch (diff={diff:.2e})")
    else:
        print(f"FAILED: Gradient diff = {diff:.6f}")

if __name__ == "__main__":
    test_fixed_grad_accum()
