"""Cross-chain MSA pairing -- THE CRUX of the coevolution bet.

From notes/2026-07-14-msa-coevolution-pair-prior-SPEC.md §6:

    "MSA PAIRING -- the crux. Cross-chain coevolution needs ortholog pairing
     across chains (species / genome proximity). Get pairing wrong => zero
     docking signal (just two better monomer profiles)."

That failure is SILENT: a wrongly-paired MSA still produces a well-formed
[L, L, F] tensor and a model that trains happily to no benefit. So the tests
here assert the pairing SEMANTICS, not just shapes.
"""

import pytest

from tinyfold.msa.a3m import MsaRecord
from tinyfold.msa.pairing import pair_msas, pairing_stats


def rec(seq, taxid, header="h"):
    return MsaRecord(header=header, seq=seq, taxid=taxid)


# Query rows carry no taxid (they are our own chains, not database hits).
QA = rec("AAAA", None, "queryA")
QB = rec("BBBB", None, "queryB")


class TestQueryRow:
    """Row 0 must always be query_A + query_B -- everything indexes against it."""

    def test_query_pair_is_row_zero(self):
        a, b = pair_msas([QA], [QB])
        assert (a[0].seq, b[0].seq) == ("AAAA", "BBBB")

    def test_query_survives_even_with_no_shared_species(self):
        a, b = pair_msas([QA, rec("CCCC", 1)], [QB, rec("DDDD", 2)])
        assert len(a) == len(b) == 1
        assert a[0].seq == "AAAA"

    def test_empty_msas_still_require_a_query(self):
        with pytest.raises(ValueError):
            pair_msas([], [QB])


class TestSpeciesJoin:
    def test_shared_species_pairs(self):
        a, b = pair_msas([QA, rec("CCCC", 9606)], [QB, rec("DDDD", 9606)])
        assert len(a) == 2
        assert (a[1].seq, b[1].seq) == ("CCCC", "DDDD")

    def test_species_in_only_one_chain_is_dropped(self):
        # 10090 exists only on chain A -> contributes NO paired row.
        a, b = pair_msas(
            [QA, rec("CCCC", 9606), rec("EEEE", 10090)],
            [QB, rec("DDDD", 9606)],
        )
        assert len(a) == len(b) == 2
        assert "EEEE" not in [r.seq for r in a]

    def test_untaxonomised_hits_never_pair(self):
        # The BFD case: no taxonomy => unpairable, however deep the MSA.
        a, b = pair_msas([QA, rec("CCCC", None)], [QB, rec("DDDD", None)])
        assert len(a) == 1  # query only

    def test_paired_rows_always_share_a_species(self):
        a, b = pair_msas(
            [QA, rec("C1", 1), rec("C2", 2), rec("C3", 3)],
            [QB, rec("D2", 2), rec("D3", 3), rec("D9", 9)],
        )
        for ra, rb in zip(a[1:], b[1:]):
            assert ra.taxid == rb.taxid

    def test_rows_are_aligned_pairwise(self):
        a, b = pair_msas([QA, rec("C1", 1)], [QB, rec("D1", 1)])
        assert len(a) == len(b)


class TestRankPairingWithinSpecies:
    """AF-Multimer semantics: within a species, pair by rank (best-to-best)."""

    def test_multiple_hits_per_species_pair_by_rank(self):
        # Both chains have 2 hits in species 42; input order IS rank order.
        a, b = pair_msas(
            [QA, rec("A1", 42), rec("A2", 42)],
            [QB, rec("B1", 42), rec("B2", 42)],
        )
        pairs = [(x.seq, y.seq) for x, y in zip(a[1:], b[1:])]
        assert pairs == [("A1", "B1"), ("A2", "B2")]

    def test_unequal_hit_counts_truncate_to_the_shorter(self):
        # 3 hits vs 1 hit in the same species -> only 1 pair (no invention).
        a, b = pair_msas(
            [QA, rec("A1", 42), rec("A2", 42), rec("A3", 42)],
            [QB, rec("B1", 42)],
        )
        assert len(a) == 2
        assert (a[1].seq, b[1].seq) == ("A1", "B1")

    def test_max_per_species_caps_paralog_explosion(self):
        a, b = pair_msas(
            [QA, rec("A1", 42), rec("A2", 42), rec("A3", 42)],
            [QB, rec("B1", 42), rec("B2", 42), rec("B3", 42)],
            max_per_species=2,
        )
        assert len(a) == 3  # query + 2


class TestDeterminism:
    def test_species_order_is_deterministic(self):
        args = (
            [QA, rec("A9", 9), rec("A1", 1), rec("A5", 5)],
            [QB, rec("B9", 9), rec("B1", 1), rec("B5", 5)],
        )
        first = [r.seq for r in pair_msas(*args)[0]]
        for _ in range(3):
            assert [r.seq for r in pair_msas(*args)[0]] == first

    def test_species_sorted_by_taxid(self):
        a, _ = pair_msas(
            [QA, rec("A9", 9), rec("A1", 1)],
            [QB, rec("B9", 9), rec("B1", 1)],
        )
        assert [r.seq for r in a] == ["AAAA", "A1", "A9"]


class TestPairingStats:
    """Step 0 reads these numbers to decide whether to buy a database at all."""

    def test_reports_paired_depth(self):
        st = pairing_stats([QA, rec("C", 1)], [QB, rec("D", 1)])
        assert st["n_paired"] == 2  # query + 1

    def test_reports_shared_species(self):
        st = pairing_stats(
            [QA, rec("C1", 1), rec("C2", 2)],
            [QB, rec("D2", 2), rec("D3", 3)],
        )
        assert st["n_species_a"] == 2
        assert st["n_species_b"] == 2
        assert st["n_species_shared"] == 1

    def test_reports_unpairable_fraction(self):
        # 2 of 4 chain-A hits carry no taxonomy.
        st = pairing_stats(
            [QA, rec("C1", 1), rec("C2", None), rec("C3", None)],
            [QB, rec("D1", 1)],
        )
        assert st["frac_no_taxid_a"] == pytest.approx(2 / 3)

    def test_flags_the_shallow_case(self):
        # No shared species => the coevolution signal is nil for this complex.
        st = pairing_stats([QA, rec("C", 1)], [QB, rec("D", 2)])
        assert st["n_paired"] == 1
        assert st["n_species_shared"] == 0
