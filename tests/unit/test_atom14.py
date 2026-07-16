"""atom14 layout, chi definitions, and symmetric-atom swaps.

These are lifted reference tables, so the tests assert STRUCTURAL INVARIANTS
(cross-checks the tables must satisfy against each other) rather than
re-stating the table -- a transcription slip shows up as a broken invariant.
"""

import numpy as np
import pytest

from tinyfold.atom14 import (
    AMBIGUOUS_ATOM_SWAPS,
    ATOM14_NAMES,
    CHI_ANGLES_ATOMS,
    CHI_PI_PERIODIC,
    NUM_ATOM14,
    alt_atom14_permutation,
    atom14_mask,
    atom14_names,
    n_chi,
    restype_alt_permutation,
    restype_atom14_mask,
)
from tinyfold.constants import AA3_TO_AA1, AA_CODES

ALL_AA3 = sorted(ATOM14_NAMES)


class TestLayoutInvariants:
    def test_all_twenty_standard_residues_present(self):
        assert len(ALL_AA3) == 20
        assert set(ALL_AA3) == set(AA3_TO_AA1)

    def test_every_residue_has_exactly_14_slots(self):
        for aa3 in ALL_AA3:
            assert len(ATOM14_NAMES[aa3]) == NUM_ATOM14, aa3

    def test_backbone_occupies_slots_0_to_3(self):
        # atom14[..., :4, :] must BE the existing coords_res, or every current
        # backbone path silently breaks.
        for aa3 in ALL_AA3:
            assert ATOM14_NAMES[aa3][:4] == ["N", "CA", "C", "O"], aa3

    def test_no_duplicate_atom_names_within_a_residue(self):
        for aa3 in ALL_AA3:
            used = [n for n in ATOM14_NAMES[aa3] if n]
            assert len(used) == len(set(used)), aa3

    def test_no_gaps_before_empty_slots(self):
        # Present atoms must be contiguous from slot 0 (mask packing assumption).
        for aa3 in ALL_AA3:
            names = ATOM14_NAMES[aa3]
            used = [bool(n) for n in names]
            assert used == sorted(used, reverse=True), aa3

    def test_trp_is_the_widest_at_14(self):
        # TRP is why the layout is 14 wide.
        assert all(n for n in ATOM14_NAMES["TRP"])
        assert atom14_mask("TRP").sum() == 14

    def test_glycine_is_backbone_only(self):
        assert atom14_mask("GLY").sum() == 4

    def test_alanine_adds_only_cb(self):
        assert atom14_mask("ALA").sum() == 5

    def test_no_residue_exceeds_14(self):
        for aa3 in ALL_AA3:
            assert atom14_mask(aa3).sum() <= NUM_ATOM14


class TestNameLookup:
    def test_accepts_one_and_three_letter_codes(self):
        assert atom14_names("W") == atom14_names("TRP")

    def test_unknown_falls_back_to_glycine(self):
        assert atom14_names("X") == atom14_names("GLY")

    def test_every_standard_aa1_resolves(self):
        for aa1 in AA_CODES:
            assert len(atom14_names(aa1)) == NUM_ATOM14


class TestChiDefinitions:
    def test_every_residue_has_chi_entry(self):
        assert set(CHI_ANGLES_ATOMS) == set(ATOM14_NAMES)

    def test_chi_atoms_exist_in_the_atom14_layout(self):
        # A chi defined on an atom the residue does not have is unplaceable.
        for aa3, chis in CHI_ANGLES_ATOMS.items():
            present = {n for n in ATOM14_NAMES[aa3] if n}
            for chi in chis:
                assert set(chi) <= present, (aa3, chi)

    def test_each_chi_is_four_atoms(self):
        for aa3, chis in CHI_ANGLES_ATOMS.items():
            for chi in chis:
                assert len(chi) == 4, (aa3, chi)

    def test_gly_and_ala_have_no_chi(self):
        # §9: "Mask GLY/ALA (0 sidechain torsions)."
        assert n_chi("GLY") == 0
        assert n_chi("ALA") == 0

    def test_at_most_four_chi(self):
        for aa3 in ALL_AA3:
            assert n_chi(aa3) <= 4

    def test_arg_and_lys_have_four(self):
        assert n_chi("ARG") == 4
        assert n_chi("LYS") == 4

    def test_chi1_is_always_n_ca_cb(self):
        for aa3, chis in CHI_ANGLES_ATOMS.items():
            if chis:
                assert chis[0][:3] == ["N", "CA", "CB"], aa3

    def test_pi_periodic_covers_every_residue(self):
        assert set(CHI_PI_PERIODIC) == set(ATOM14_NAMES)

    def test_pi_periodic_flags_only_within_existing_chi(self):
        for aa3, flags in CHI_PI_PERIODIC.items():
            for i, flag in enumerate(flags):
                if flag:
                    assert i < n_chi(aa3), (aa3, i)


class TestAmbiguousSwaps:
    def test_swapped_atoms_exist(self):
        for aa3, swaps in AMBIGUOUS_ATOM_SWAPS.items():
            present = {n for n in ATOM14_NAMES[aa3] if n}
            for a, b in swaps.items():
                assert a in present and b in present, (aa3, a, b)

    def test_the_five_known_ambiguous_residues(self):
        # ASP/GLU/PHE/TYR/ARG -- exactly the set §9 calls mandatory.
        assert set(AMBIGUOUS_ATOM_SWAPS) == {"ARG", "ASP", "GLU", "PHE", "TYR"}

    def test_arg_is_renaming_ambiguous_but_not_pi_periodic(self):
        # The two symmetry notions are NOT the same; conflating them is the bug
        # this test exists to prevent.
        assert "ARG" in AMBIGUOUS_ATOM_SWAPS
        assert not any(CHI_PI_PERIODIC["ARG"])


class TestAltPermutation:
    def test_identity_for_unambiguous_residues(self):
        for aa3 in ["GLY", "ALA", "TRP", "SER", "LYS"]:
            np.testing.assert_array_equal(
                alt_atom14_permutation(aa3), np.arange(NUM_ATOM14)
            )

    def test_asp_swaps_od1_and_od2(self):
        perm = alt_atom14_permutation("ASP")
        names = atom14_names("ASP")
        i1, i2 = names.index("OD1"), names.index("OD2")
        assert perm[i1] == i2 and perm[i2] == i1

    def test_permutation_is_an_involution(self):
        # Every swap is a transposition, so applying it twice is the identity.
        for aa3 in ALL_AA3:
            perm = alt_atom14_permutation(aa3)
            np.testing.assert_array_equal(perm[perm], np.arange(NUM_ATOM14))

    def test_permutation_is_a_valid_permutation(self):
        for aa3 in ALL_AA3:
            assert sorted(alt_atom14_permutation(aa3)) == list(range(NUM_ATOM14))

    def test_swap_preserves_the_presence_mask(self):
        # A swap must never move a present atom into an absent slot.
        for aa3 in ALL_AA3:
            mask = atom14_mask(aa3)
            perm = alt_atom14_permutation(aa3)
            np.testing.assert_array_equal(mask[perm], mask)

    def test_backbone_is_never_swapped(self):
        for aa3 in ALL_AA3:
            np.testing.assert_array_equal(alt_atom14_permutation(aa3)[:4], np.arange(4))

    def test_phe_swaps_both_ring_pairs(self):
        perm = alt_atom14_permutation("PHE")
        names = atom14_names("PHE")
        for a, b in [("CD1", "CD2"), ("CE1", "CE2")]:
            ia, ib = names.index(a), names.index(b)
            assert perm[ia] == ib and perm[ib] == ia


class TestBatchTables:
    def test_mask_table_shape_covers_x(self):
        assert restype_atom14_mask().shape == (21, NUM_ATOM14)

    def test_perm_table_shape_covers_x(self):
        assert restype_alt_permutation().shape == (21, NUM_ATOM14)

    def test_mask_table_matches_per_residue_lookup(self):
        table = restype_atom14_mask()
        for i, aa1 in enumerate(AA_CODES):
            np.testing.assert_array_equal(table[i], atom14_mask(aa1))

    def test_unknown_row_is_glycine_and_identity(self):
        assert restype_atom14_mask()[20].sum() == 4
        np.testing.assert_array_equal(restype_alt_permutation()[20], np.arange(NUM_ATOM14))

    def test_every_perm_row_is_a_permutation(self):
        for row in restype_alt_permutation():
            assert sorted(row) == list(range(NUM_ATOM14))


class TestSidechainSlots:
    def test_sidechain_atom_counts_are_sane(self):
        # A couple of hand-checks against chemistry, as a table sanity anchor.
        expected = {"GLY": 0, "ALA": 1, "SER": 2, "VAL": 3, "TRP": 10, "TYR": 8}
        for aa3, n_sc in expected.items():
            assert atom14_mask(aa3).sum() - 4 == n_sc, aa3

    @pytest.mark.parametrize("aa3", ["ASP", "GLU", "PHE", "TYR", "ARG"])
    def test_ambiguous_residues_have_the_swapped_atoms_in_sidechain(self, aa3):
        names = atom14_names(aa3)
        for a, b in AMBIGUOUS_ATOM_SWAPS[aa3].items():
            assert names.index(a) >= 4 and names.index(b) >= 4
