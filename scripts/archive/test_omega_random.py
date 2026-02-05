"""Test omega loss for random vs ground truth."""
import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold')

import torch
import numpy as np
import math
from scripts.models.geometry_losses import omega_loss, dihedral_angle, bounded_loss, GeometryLoss

# Load a real protein
import pyarrow.parquet as pq
table = pq.read_table("C:/Users/costa/src/tinyfold/data/processed/samples.parquet")

# Load first sample
coords_flat = torch.tensor(table['atom_coords'][0].as_py(), dtype=torch.float32)
atom_types = torch.tensor(table['atom_type'][0].as_py(), dtype=torch.long)
n_atoms = len(atom_types)
n_res = n_atoms // 4

# Ground truth coords [1, L, 4, 3]
coords_gt = coords_flat.reshape(n_atoms, 3).view(n_res, 4, 3)
coords_gt = coords_gt - coords_gt.view(-1, 3).mean(0)
coord_std = coords_gt.std()
coords_gt_norm = (coords_gt / coord_std).unsqueeze(0)

print(f"Sample: L={n_res}, coord_std={coord_std:.2f}")
print("=" * 60)

# Test ground truth
gt_loss = omega_loss(coords_gt_norm)
print(f"Ground truth omega loss: {gt_loss.item():.6f}")

# Test with slightly perturbed coords (like early training)
for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
    coords_noisy = coords_gt_norm + noise_level * torch.randn_like(coords_gt_norm)
    noisy_loss = omega_loss(coords_noisy)
    print(f"Noise={noise_level:.2f} -> omega loss: {noisy_loss.item():.4f}")

print()
print("Testing raw vs bounded omega loss:")
# Compute raw omega angles for random coords
print("-" * 60)

# Random offsets from centroids (like untrained model)
centroids = coords_gt_norm.mean(dim=2)  # [1, L, 3]

for offset_scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    random_offsets = offset_scale * torch.randn(1, n_res, 4, 3)
    coords_random = centroids.unsqueeze(2) + random_offsets
    
    # Raw omega values
    CA = coords_random[..., 1, :]
    C = coords_random[..., 2, :]
    N_next = coords_random[:, 1:, 0, :]
    CA_next = coords_random[:, 1:, 1, :]
    
    omega = dihedral_angle(CA[:, :-1], C[:, :-1], N_next, CA_next)
    
    # Raw loss (before bounding)
    trans_dev = (torch.abs(omega) - math.pi)**2
    cis_dev = omega**2
    raw_loss = torch.min(trans_dev, cis_dev).mean()
    
    # Bounded loss  
    bounded = omega_loss(coords_random)
    
    omega_deg = omega * 180 / math.pi
    print(f"offset_scale={offset_scale:.2f} | raw={raw_loss.item():.4f} | bounded={bounded.item():.4f} | omega: {omega_deg.min().item():.0f} to {omega_deg.max().item():.0f} deg")

print()
print("Testing GeometryLoss class:")
print("-" * 60)
geom_loss_fn = GeometryLoss(weights={'bond_length': 1.0, 'bond_angle': 0.1, 'omega': 0.1})

# GT
losses_gt = geom_loss_fn(coords_gt_norm)
print(f"Ground truth: total={losses_gt['total'].item():.6f}, bond={losses_gt['bond_length'].item():.6f}, angle={losses_gt['bond_angle'].item():.6f}, omega={losses_gt['omega'].item():.6f}")

# Random
random_offsets = 0.2 * torch.randn(1, n_res, 4, 3)
coords_random = centroids.unsqueeze(2) + random_offsets
losses_random = geom_loss_fn(coords_random)
print(f"Random (0.2): total={losses_random['total'].item():.4f}, bond={losses_random['bond_length'].item():.4f}, angle={losses_random['bond_angle'].item():.4f}, omega={losses_random['omega'].item():.4f}")
