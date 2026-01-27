#!/usr/bin/env python
"""Quick test to verify E2E model imports and shapes."""

import torch
import sys

print("Testing E2E model imports...")

try:
    from models.atomrefine_multi_sample import AtomRefinerV2MultiSample
    from models.resfold_e2e import ResFoldE2E, sample_e2e
    print("  Imports OK")
except Exception as e:
    print(f"  Import error: {e}")
    sys.exit(1)

# Test AtomRefinerV2MultiSample
print("\nTesting AtomRefinerV2MultiSample...")
model = AtomRefinerV2MultiSample(c_token=256, n_layers=6, n_samples=5)
counts = model.count_parameters()
print(f"  Params: {counts['total']:,}")

# Test shapes
B, K, L = 2, 5, 50
trunk_tokens = torch.randn(B, L, 256)
centroids_samples = torch.randn(B, K, L, 3)
mask = torch.ones(B, L, dtype=torch.bool)

out = model(trunk_tokens, centroids_samples, mask)
print(f"  Input: trunk={trunk_tokens.shape}, centroids={centroids_samples.shape}")
print(f"  Output: {out.shape}")
assert out.shape == (B, L, 4, 3), f"Expected (2, 50, 4, 3), got {out.shape}"
print("  OK!")

# Test ResFoldE2E
print("\nTesting ResFoldE2E...")
e2e_model = ResFoldE2E(
    c_token=256,
    trunk_layers=4,
    denoiser_blocks=3,
    s2_layers=3,
    n_samples=5
)
counts = e2e_model.count_parameters()
print(f"  Total params: {counts['total']:,}")
print(f"  Stage 1: {counts['stage1']:,} ({counts['stage1_pct']:.1f}%)")
print(f"  Stage 2: {counts['stage2']:,} ({counts['stage2_pct']:.1f}%)")

# Test forward (without noiser for basic check)
aa_seq = torch.randint(0, 21, (B, L))
chain_ids = torch.zeros(B, L, dtype=torch.long)
chain_ids[:, L//2:] = 1
res_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
centroids = torch.randn(B, L, 3)
mask = torch.ones(B, L, dtype=torch.bool)

# Test trunk tokens
trunk_tokens = e2e_model.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)
print(f"  Trunk tokens shape: {trunk_tokens.shape}")
assert trunk_tokens.shape == (B, L, 256)

# Test Stage 2 forward with given samples
centroids_samples = torch.randn(B, 5, L, 3)
atoms = e2e_model.forward_e2e_with_given_samples(centroids_samples, aa_seq, chain_ids, res_idx, mask)
print(f"  Atoms shape: {atoms.shape}")
assert atoms.shape == (B, L, 4, 3)

print("  OK!")

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
