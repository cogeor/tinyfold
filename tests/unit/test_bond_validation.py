"""Bond-length validation must check every bond type against its own reference.

Regression guard for the stale 2-type validator: after the 4-type bond refactor
(N-CA=0, CA-C=1, C-O=2, PEPTIDE=3) the old code only checked types 0 and 1, so
C-O (type 2) and the real peptide bond (type 3) were never validated.
"""

import numpy as np

from tinyfold.constants import BOND_LENGTHS
from tinyfold.data.processing.atomization import build_bonds
from tinyfold.data.processing.filters import validate_bond_lengths

N_CA = BOND_LENGTHS["N-CA"]
CA_C = BOND_LENGTHS["CA-C"]
C_N = BOND_LENGTHS["C-N"]  # peptide
SPACING = N_CA + CA_C + C_N  # N(res) -> N(res+1) along x


def _ideal_chain(n_res: int):
    """Build a single chain of `n_res` residues with ideal backbone geometry.

    Atom order per residue is [N, CA, C, O] (types 0..3), matching atomize/build_bonds.
    """
    coords = np.zeros((n_res * 4, 3), dtype=np.float64)
    for r in range(n_res):
        base = r * SPACING
        coords[r * 4 + 0] = [base, 0.0, 0.0]                 # N
        coords[r * 4 + 1] = [base + N_CA, 0.0, 0.0]          # CA
        c_x = base + N_CA + CA_C
        coords[r * 4 + 2] = [c_x, 0.0, 0.0]                  # C
        coords[r * 4 + 3] = [c_x, BOND_LENGTHS["C-O"], 0.0]  # O (C-O along y)
    mask = np.ones(n_res * 4, dtype=bool)
    src, dst, btype = build_bonds(n_res, 0, mask)
    return coords, mask, src, dst, btype


def test_valid_backbone_passes():
    coords, mask, src, dst, btype = _ideal_chain(4)
    assert validate_bond_lengths(coords, src, dst, btype, mask).passed


def test_bad_peptide_bond_rejected():
    # Rigidly translate residue 1's whole block: internal bonds stay ideal,
    # only the peptide bond C(0)-N(1) is stretched. Old code never checked it.
    coords, mask, src, dst, btype = _ideal_chain(4)
    coords[4:8] += np.array([0.0, 0.0, 50.0])
    assert not validate_bond_lengths(coords, src, dst, btype, mask).passed


def test_bad_c_o_bond_rejected():
    # O participates only in the C-O bond (type 2), which the old code never checked.
    coords, mask, src, dst, btype = _ideal_chain(4)
    coords[3] += np.array([0.0, 0.0, 50.0])
    assert not validate_bond_lengths(coords, src, dst, btype, mask).passed
