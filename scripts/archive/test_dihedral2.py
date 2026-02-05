"""Find valid peptide bonds and test omega."""
import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold')

import torch
import math
from scripts.models.geometry_losses import dihedral_angle
import pyarrow.parquet as pq

table = pq.read_table("C:/Users/costa/src/tinyfold/data/processed/samples.parquet")
coords_flat = torch.tensor(table['atom_coords'][0].as_py(), dtype=torch.float32)
n_atoms = len(table['atom_type'][0].as_py())
n_res = n_atoms // 4
coords = coords_flat.reshape(n_atoms, 3).view(n_res, 4, 3)

print(f"Protein: {n_res} residues")
print("=" * 60)

# Find C-N distances
C = coords[:, 2]   # [L, 3]
N = coords[:, 0]   # [L, 3]
C_prev = C[:-1]    # [L-1, 3]
N_next = N[1:]     # [L-1, 3]

c_n_dist = (C_prev - N_next).norm(dim=-1)
print(f"C-N distances: min={c_n_dist.min().item():.2f}, max={c_n_dist.max().item():.2f}, mean={c_n_dist.mean().item():.2f}")
print(f"Valid peptide bonds (d < 2A): {(c_n_dist < 2).sum().item()}/{len(c_n_dist)}")

# Check omega for valid peptide bonds
CA = coords[:, 1]
valid_mask = c_n_dist < 2.0

print("\nOmega angles for first 10 VALID peptide bonds:")
valid_indices = torch.where(valid_mask)[0]
for idx in valid_indices[:10]:
    i = idx.item()
    omega = dihedral_angle(
        CA[i:i+1], C[i:i+1], N[i+1:i+2], CA[i+1:i+2]
    )
    dist = c_n_dist[i].item()
    omega_deg = omega.item() * 180 / math.pi
    print(f"  Residue {i}->{i+1}: C-N={dist:.2f}A, omega={omega_deg:.1f} deg")

print("\nOmega angles for first 10 INVALID (gap) peptide bonds:")
invalid_indices = torch.where(~valid_mask)[0]
for idx in invalid_indices[:10]:
    i = idx.item()
    omega = dihedral_angle(
        CA[i:i+1], C[i:i+1], N[i+1:i+2], CA[i+1:i+2]
    )
    dist = c_n_dist[i].item()
    omega_deg = omega.item() * 180 / math.pi
    print(f"  Residue {i}->{i+1}: C-N={dist:.2f}A, omega={omega_deg:.1f} deg")
