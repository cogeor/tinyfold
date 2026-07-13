"""Golden-sample test on a real complex (CLAUDE.md testing philosophy).

`tests/fixtures/complex_2chain.pdb` is a compact backbone-only fixture derived
from the real PDB entry 1A2K (chains A and B). Values below are hardcoded and
checked every run, per the golden-sample requirement.
"""

from pathlib import Path

import numpy as np
import pytest

from tinyfold.constants import BOND_LENGTHS, BOND_TYPE_CA_C, BOND_TYPE_N_CA
from tinyfold.data.parsing.structure_io import (
    extract_chain,
    get_backbone_atoms,
    load_structure,
)
from tinyfold.data.processing.atomization import (
    atomize_chains,
    build_bonds,
    compute_bond_lengths,
)

FIXTURE = Path(__file__).parent / "fixtures" / "complex_2chain.pdb"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="golden fixture tests/fixtures/complex_2chain.pdb missing"
)

# Golden values (1A2K chains A/B, backbone).
GOLDEN_LA = 124
GOLDEN_LB = 124


def _chains():
    st = load_structure(str(FIXTURE))
    return get_backbone_atoms(extract_chain(st, "A")), get_backbone_atoms(extract_chain(st, "B"))


def test_golden_chain_lengths():
    a, b = _chains()
    assert len(a.sequence) == GOLDEN_LA
    assert len(b.sequence) == GOLDEN_LB


def test_atom_count_is_four_times_residues():
    a, b = _chains()
    coords, mask, _, _, chain_id = atomize_chains(a.coords, a.mask, b.coords, b.mask)
    assert coords.shape[0] == 4 * (GOLDEN_LA + GOLDEN_LB)
    assert not np.isnan(coords).any()
    # Chain A atoms first (id 0), chain B after (id 1).
    assert (chain_id[: GOLDEN_LA * 4] == 0).all()
    assert (chain_id[GOLDEN_LA * 4 :] == 1).all()


def test_bond_lengths_chemically_reasonable():
    a, b = _chains()
    coords, mask, _, _, _ = atomize_chains(a.coords, a.mask, b.coords, b.mask)
    src, dst, btype = build_bonds(GOLDEN_LA, GOLDEN_LB, mask)
    lengths = compute_bond_lengths(coords, src, dst, mask)
    n_ca = lengths[btype == BOND_TYPE_N_CA]
    ca_c = lengths[btype == BOND_TYPE_CA_C]
    assert abs(n_ca.mean() - BOND_LENGTHS["N-CA"]) < 0.1
    assert abs(ca_c.mean() - BOND_LENGTHS["CA-C"]) < 0.1
