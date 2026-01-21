"""Test omega loss on ground truth atom positions."""
import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold')

import torch
import numpy as np
from scripts.models.geometry_losses import omega_loss, dihedral_angle

# Load real proteins
import pyarrow.parquet as pq
table = pq.read_table("C:/Users/costa/src/tinyfold/data/processed/samples.parquet")

print("Testing omega loss on ground truth proteins:")
print("=" * 60)

# Get first few samples
for i in range(5):
    coords_flat = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    
    # Reshape to [L, 4, 3]
    coords = coords_flat.reshape(n_atoms, 3).view(n_res, 4, 3)
    
    # Normalize like training does
    centroid = coords.mean()
    coords = coords - coords.view(-1, 3).mean(0)
    coord_std = coords.std()
    coords_norm = coords / coord_std
    
    # Add batch dim: [1, L, 4, 3]
    coords_norm = coords_norm.unsqueeze(0)
    
    L = n_res
    
    # Compute omega loss
    loss = omega_loss(coords_norm)
    
    # Also compute raw omega angles to see distribution
    N = coords_norm[:, :, 0]   # [B, L, 3]
    CA = coords_norm[:, :, 1]  # [B, L, 3]
    C = coords_norm[:, :, 2]   # [B, L, 3]
    
    # Omega: CA(i) - C(i) - N(i+1) - CA(i+1)
    omega_angles = dihedral_angle(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])
    omega_deg = omega_angles * 180 / np.pi
    
    print(f"\nSample {i}: L={L}, coord_std={coord_std:.2f}")
    print(f"  Omega loss (bounded): {loss.item():.6f}")
    print(f"  Omega angles (deg): min={omega_deg.min().item():.1f}, max={omega_deg.max().item():.1f}")
    print(f"  Mean |omega|: {omega_deg.abs().mean().item():.1f} deg")
    print(f"  Trans (|omega|>150): {(omega_deg.abs() > 150).sum().item()}/{omega_deg.numel()}")
    print(f"  Cis (|omega|<30): {(omega_deg.abs() < 30).sum().item()}/{omega_deg.numel()}")
    
    # Show distribution of deviations from 180
    dev_from_trans = torch.abs(torch.abs(omega_deg) - 180)
    print(f"  Deviation from 180: mean={dev_from_trans.mean().item():.1f}, max={dev_from_trans.max().item():.1f}")
