"""atom14 extraction from the DIPS atom-level DataFrame (S3).

The premise being tested is the one that unblocks Phase 3: the DIPS source
ALREADY carries sidechain atoms, and we throw them away at parse time. These
tests use a synthetic DataFrame with the same schema, so they run without the
14.6 GB raw re-download.

The critical invariant is agreement with the EXISTING backbone extractor: if the
two disagree about which residues exist or where the backbone is, the atom14
cache silently misaligns against samples.parquet.
"""

import numpy as np
import pandas as pd
import pytest

from tinyfold.atom14 import NUM_ATOM14, atom14_names
from tinyfold.data.parsing.dips_loader import (
    extract_atom14_from_dataframe,
    extract_backbone_from_dataframe,
)


def _rows(residue, resname, atoms):
    """atoms: {atom_name: (x, y, z)} -> DIPS-schema rows."""
    return [
        {"residue": residue, "resname": resname, "atom_name": name,
         "x": xyz[0], "y": xyz[1], "z": xyz[2]}
        for name, xyz in atoms.items()
    ]


def _full_residue(residue, resname, base=0.0):
    """Every atom14 slot of a residue type, at distinguishable coordinates."""
    names = [n for n in atom14_names(resname) if n]
    return _rows(residue, resname,
                 {n: (base + i, base + i + 0.5, base + i + 0.25) for i, n in enumerate(names)})


def df_of(*row_lists):
    return pd.DataFrame([r for rows in row_lists for r in rows])


class TestSidechainsSurvive:
    def test_tryptophan_keeps_all_14_atoms(self):
        # The whole point: the backbone extractor would drop 10 of these.
        out = extract_atom14_from_dataframe(df_of(_full_residue(1, "TRP")))
        assert out.mask.sum() == 14

    def test_backbone_extractor_drops_them(self):
        # Demonstrates the discard that created the Phase-3 blocker.
        df = df_of(_full_residue(1, "TRP"))
        assert extract_backbone_from_dataframe(df).mask.sum() == 4
        assert extract_atom14_from_dataframe(df).mask.sum() == 14

    def test_aspartate_keeps_both_carboxyl_oxygens(self):
        out = extract_atom14_from_dataframe(df_of(_full_residue(1, "ASP")))
        names = atom14_names("ASP")
        assert out.mask[0, names.index("OD1")]
        assert out.mask[0, names.index("OD2")]


class TestShapesAndLayout:
    def test_coords_shape_is_L_14_3(self):
        out = extract_atom14_from_dataframe(
            df_of(_full_residue(1, "ALA"), _full_residue(2, "TRP"))
        )
        assert out.coords.shape == (2, NUM_ATOM14, 3)
        assert out.mask.shape == (2, NUM_ATOM14)

    def test_backbone_occupies_the_first_four_slots(self):
        out = extract_atom14_from_dataframe(df_of(_full_residue(1, "LEU")))
        # Slots 0-3 must be exactly the N/CA/C/O we placed.
        names = atom14_names("LEU")
        for i, n in enumerate(["N", "CA", "C", "O"]):
            assert names[i] == n
            assert out.mask[0, i]

    def test_backbone_slots_match_the_backbone_extractor_exactly(self):
        # THE alignment invariant against samples.parquet.
        df = df_of(_full_residue(1, "TYR"), _full_residue(2, "GLY", base=10.0))
        bb = extract_backbone_from_dataframe(df)
        a14 = extract_atom14_from_dataframe(df)
        np.testing.assert_array_equal(a14.coords[:, :4], bb.coords)
        np.testing.assert_array_equal(a14.mask[:, :4], bb.mask)

    def test_sequence_matches_the_backbone_extractor(self):
        df = df_of(_full_residue(1, "TRP"), _full_residue(2, "ASP", base=5.0))
        assert (
            extract_atom14_from_dataframe(df).sequence
            == extract_backbone_from_dataframe(df).sequence
        )


class TestMissingAtoms:
    def test_absent_sidechain_atoms_are_masked_not_zeroed_silently(self):
        # A residue with only its backbone resolved.
        df = df_of(_rows(1, "TRP", {"N": (0, 0, 0), "CA": (1, 0, 0),
                                    "C": (2, 0, 0), "O": (3, 0, 0)}))
        out = extract_atom14_from_dataframe(df)
        assert out.mask[0, :4].all()
        assert not out.mask[0, 4:].any()

    def test_glycine_has_no_sidechain_slots_set(self):
        out = extract_atom14_from_dataframe(df_of(_full_residue(1, "GLY")))
        assert out.mask[0].sum() == 4

    def test_unknown_atom_names_are_ignored(self):
        # Hydrogens / OXT / altlocs must not land in a slot.
        df = df_of(_rows(1, "ALA", {"N": (0, 0, 0), "CA": (1, 0, 0), "C": (2, 0, 0),
                                    "O": (3, 0, 0), "CB": (4, 0, 0),
                                    "OXT": (9, 9, 9), "HA": (8, 8, 8)}))
        out = extract_atom14_from_dataframe(df)
        assert out.mask[0].sum() == 5
        assert not (out.coords[0] == 9).any()


class TestResidueFiltering:
    def test_waters_are_skipped(self):
        df = df_of(_full_residue(1, "ALA"), _rows(2, "HOH", {"O": (9, 9, 9)}))
        assert len(extract_atom14_from_dataframe(df).sequence) == 1

    def test_filtering_agrees_with_the_backbone_extractor(self):
        df = df_of(
            _full_residue(1, "ALA"),
            _rows(2, "HOH", {"O": (9, 9, 9)}),
            _full_residue(3, "MSE", base=4.0),  # modified residue -> MET
        )
        assert (
            extract_atom14_from_dataframe(df).sequence
            == extract_backbone_from_dataframe(df).sequence
        )

    def test_empty_dataframe(self):
        out = extract_atom14_from_dataframe(
            pd.DataFrame(columns=["residue", "resname", "atom_name", "x", "y", "z"])
        )
        assert out.coords.shape == (0, NUM_ATOM14, 3)
        assert out.sequence == []


class TestCacheSize:
    def test_atom14_is_cheap_to_store(self):
        # Sanity-anchors the plan's ~1.9 GB estimate for 22.03 M residues fp16:
        # 14 slots * 3 coords * 2 bytes = 84 B/residue.
        bytes_per_residue = NUM_ATOM14 * 3 * 2
        assert bytes_per_residue == 84
        assert 22.03e6 * bytes_per_residue / 1e9 == pytest.approx(1.85, abs=0.05)
