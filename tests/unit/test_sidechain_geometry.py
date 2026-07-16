"""B1: chi extraction from atom14 coords (variant-A torsion geometry).

Locks the dihedral convention, the [21,4,4]/[21,4] chi tables, and the masking
(residue-type presence AND atom resolution).
"""

import math

import numpy as np
import torch

from tinyfold.atom14 import chi_atom14_indices, restype_chi_mask
from tinyfold.constants import AA_CODES
from tinyfold.sidechain_geometry import dihedral, extract_chi


def _aa_index(one_letter: str) -> int:
    return AA_CODES.index(one_letter)


def test_dihedral_matches_known_angle():
    # Place four points with an analytically known torsion (IUPAC right-hand
    # convention gives -90 degrees for this configuration).
    p0 = torch.tensor([[1.0, 0.0, 1.0]])
    p1 = torch.tensor([[0.0, 0.0, 1.0]])
    p2 = torch.tensor([[0.0, 0.0, 0.0]])
    p3 = torch.tensor([[0.0, 1.0, 0.0]])
    ang = dihedral(p0, p1, p2, p3).item()
    assert math.isclose(ang, -math.pi / 2, abs_tol=1e-5)


def test_dihedral_sign_flips_with_mirrored_point():
    p0 = torch.tensor([[1.0, 0.0, 1.0]])
    p1 = torch.tensor([[0.0, 0.0, 1.0]])
    p2 = torch.tensor([[0.0, 0.0, 0.0]])
    up = dihedral(p0, p1, p2, torch.tensor([[0.0, 1.0, 0.0]])).item()
    dn = dihedral(p0, p1, p2, torch.tensor([[0.0, -1.0, 0.0]])).item()
    assert math.isclose(up, -dn, abs_tol=1e-5)


def test_extract_recovers_planted_chi1():
    # ARG: chi1 atoms are N, CA, CB, CG (slots 0,1,4,5). Plant a known dihedral
    # into those slots and check extract_chi returns it.
    arg = _aa_index("R")
    coords = torch.zeros(1, 1, 14, 3)
    coords[0, 0, 0] = torch.tensor([1.0, 0.0, 1.0])   # N
    coords[0, 0, 1] = torch.tensor([0.0, 0.0, 1.0])   # CA
    coords[0, 0, 4] = torch.tensor([0.0, 0.0, 0.0])   # CB
    coords[0, 0, 5] = torch.tensor([0.0, 1.0, 0.0])   # CG (dihedral -90)
    aatype = torch.tensor([[arg]])
    chi, mask = extract_chi(coords, aatype)
    # chi1 index in the table matches N,CA,CB,CG.
    assert chi_atom14_indices("ARG")[0].tolist() == [0, 1, 4, 5]
    assert math.isclose(chi[0, 0, 0].item(), -math.pi / 2, abs_tol=1e-5)
    assert mask[0, 0, 0].item() is True


def test_chi_mask_counts_per_restype():
    # GLY/ALA have 0 chis; ARG/LYS have 4; SER has 1.
    table = restype_chi_mask()
    assert table[_aa_index("G")].sum() == 0
    assert table[_aa_index("A")].sum() == 0
    assert table[_aa_index("R")].sum() == 4
    assert table[_aa_index("K")].sum() == 4
    assert table[_aa_index("S")].sum() == 1


def test_atom_mask_gates_unresolved_chi():
    arg = _aa_index("R")
    coords = torch.randn(1, 1, 14, 3)
    aatype = torch.tensor([[arg]])
    atom_mask = torch.ones(1, 1, 14, dtype=torch.bool)
    atom_mask[0, 0, 5] = False  # CG missing -> chi1 invalid (needs N,CA,CB,CG)
    chi, mask = extract_chi(coords, aatype, atom_mask)
    assert mask[0, 0, 0].item() is False
    assert chi[0, 0, 0].item() == 0.0  # zeroed when masked


def test_batched_shapes():
    rng = np.random.default_rng(0)
    B, L = 2, 5
    coords = torch.from_numpy(rng.standard_normal((B, L, 14, 3))).float()
    aatype = torch.from_numpy(rng.integers(0, 20, (B, L))).long()
    chi, mask = extract_chi(coords, aatype)
    assert chi.shape == (B, L, 4) and mask.shape == (B, L, 4)
    assert (chi.abs() <= math.pi + 1e-5).all()
