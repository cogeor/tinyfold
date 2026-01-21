"""Verify dihedral angle computation."""
import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold')

import torch
import numpy as np
import math
from scripts.models.geometry_losses import dihedral_angle

# Test 1: Known dihedral - trans configuration (180 deg)
# Put 4 atoms in a trans configuration
p0 = torch.tensor([[[0.0, 0.0, 0.0]]])  # CA_i
p1 = torch.tensor([[[1.0, 0.0, 0.0]]])  # C_i  
p2 = torch.tensor([[[2.0, 0.0, 0.0]]])  # N_i+1
p3 = torch.tensor([[[3.0, 0.0, 0.0]]])  # CA_i+1

omega = dihedral_angle(p0, p1, p2, p3)
print(f"Colinear atoms: omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 2: Trans peptide (180 deg)
# Atoms in a plane, opposite sides
p0 = torch.tensor([[[0.0, 0.0, 0.0]]])   # CA_i
p1 = torch.tensor([[[1.5, 0.0, 0.0]]])   # C_i  
p2 = torch.tensor([[[2.0, 1.3, 0.0]]])   # N_i+1 (peptide bond)
p3 = torch.tensor([[[3.5, 1.3, 0.0]]])   # CA_i+1 (same side as CA_i relative to C-N)

omega = dihedral_angle(p0, p1, p2, p3)
print(f"Same plane, same side: omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 3: Trans peptide - opposite sides
p0 = torch.tensor([[[0.0, 0.0, 0.0]]])   # CA_i (below plane)
p1 = torch.tensor([[[1.5, 0.0, 0.0]]])   # C_i  
p2 = torch.tensor([[[2.0, 1.3, 0.0]]])   # N_i+1 
p3 = torch.tensor([[[1.5, 2.6, 0.0]]])   # CA_i+1 (opposite side)

omega = dihedral_angle(p0, p1, p2, p3)
print(f"Same plane, opposite side: omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 4: True trans - 180 deg
# Classic trans peptide geometry
p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i (at origin)
p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
p3 = torch.tensor([[[1.3, -1.0, 0.0]]])  # CA_i+1 (opposite side from CA_i)

omega = dihedral_angle(p0, p1, p2, p3)
print(f"Trans (180 deg expected): omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 5: Cis - 0 deg  
p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i
p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
p3 = torch.tensor([[[1.3, 1.0, 0.0]]])   # CA_i+1 (same side as CA_i)

omega = dihedral_angle(p0, p1, p2, p3)
print(f"Cis (0 deg expected): omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 6: 90 deg
p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i
p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
p3 = torch.tensor([[[1.3, 0.0, 1.0]]])   # CA_i+1 (perpendicular)

omega = dihedral_angle(p0, p1, p2, p3)
print(f"90 deg expected: omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Test 7: Real protein data
import pyarrow.parquet as pq
table = pq.read_table("C:/Users/costa/src/tinyfold/data/processed/samples.parquet")
coords_flat = torch.tensor(table['atom_coords'][0].as_py(), dtype=torch.float32)
n_atoms = len(table['atom_type'][0].as_py())
n_res = n_atoms // 4
coords = coords_flat.reshape(n_atoms, 3).view(n_res, 4, 3).unsqueeze(0)

# First peptide bond
CA_0 = coords[:, 0, 1:2, :]  # [1,1,3]
C_0 = coords[:, 0, 2:3, :]
N_1 = coords[:, 1, 0:1, :]
CA_1 = coords[:, 1, 1:2, :]

omega = dihedral_angle(CA_0.squeeze(1), C_0.squeeze(1), N_1.squeeze(1), CA_1.squeeze(1))
print(f"\nReal protein first peptide bond: omega = {omega.item():.4f} rad = {omega.item()*180/math.pi:.1f} deg")

# Check C-N distance
c_n_dist = (C_0 - N_1).norm().item()
print(f"C-N distance: {c_n_dist:.3f} A (should be ~1.33)")
