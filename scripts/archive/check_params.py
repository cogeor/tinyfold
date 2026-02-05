#!/usr/bin/env python
"""Quick script to check af3_style parameter counts."""
import sys
sys.path.insert(0, 'scripts')
from models.af3_style import AF3StyleDecoder

# Test with default train.py config (h_dim=128, n_layers=6)
model = AF3StyleDecoder(
    c_token=256,       # h_dim * 2 = 128 * 2
    c_atom=128,        # h_dim = 128
    trunk_layers=9,    # n_layers + 3 = 6 + 3
    denoiser_blocks=7, # n_layers + 1 = 6 + 1
    atom_attn_blocks=3,
    n_timesteps=50,
)

counts = model.count_parameters()
print(f"Default config (h_dim=128, n_layers=6):")
print(f"  Total params: {counts['total']:,}")
print(f"  Trunk params: {counts['trunk']:,} ({counts['trunk_pct']:.1f}%)")
print(f"  Denoiser params: {counts['denoiser']:,} ({counts['denoiser_pct']:.1f}%)")
print()

# Try various configs to find ~10M
configs = [
    {'c_token': 256, 'c_atom': 128, 'trunk_layers': 6, 'denoiser_blocks': 5, 'atom_attn_blocks': 3},
    {'c_token': 256, 'c_atom': 128, 'trunk_layers': 5, 'denoiser_blocks': 5, 'atom_attn_blocks': 3},
    {'c_token': 256, 'c_atom': 128, 'trunk_layers': 4, 'denoiser_blocks': 5, 'atom_attn_blocks': 3},
    {'c_token': 256, 'c_atom': 128, 'trunk_layers': 4, 'denoiser_blocks': 4, 'atom_attn_blocks': 3},
    {'c_token': 224, 'c_atom': 112, 'trunk_layers': 5, 'denoiser_blocks': 5, 'atom_attn_blocks': 3},
    {'c_token': 192, 'c_atom': 96, 'trunk_layers': 6, 'denoiser_blocks': 6, 'atom_attn_blocks': 3},
]

print("Testing configs to find ~10M params:")
for cfg in configs:
    m = AF3StyleDecoder(n_timesteps=50, **cfg)
    c = m.count_parameters()
    print(f"  c_token={cfg['c_token']}, c_atom={cfg['c_atom']}, trunk={cfg['trunk_layers']}, denoiser={cfg['denoiser_blocks']} -> {c['total']:,} params (trunk: {c['trunk']:,}, denoiser: {c['denoiser']:,})")
