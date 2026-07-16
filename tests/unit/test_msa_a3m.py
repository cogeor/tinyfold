"""a3m parsing, taxonomy extraction, dedup, and depth (Neff).

These back Step 0 of notes/2026-07-14-msa-coevolution-pair-prior-SPEC.md: measure
paired-MSA depth BEFORE investing in a database. Everything here is pure and runs
on fixtures -- no MMseqs2, no UniRef download.
"""

import pytest

from tinyfold.msa.a3m import (
    dedup_by_sequence,
    neff,
    parse_a3m,
    parse_taxid,
    strip_insertions,
)

# Minimal a3m: query + 3 hits. Lowercase letters are insertions relative to the
# query and must be stripped to recover alignment columns.
A3M = """>query
MKVLA
>UniRef100_A n=1 Tax=Homo sapiens TaxID=9606 RepID=A_HUMAN
MKVLA
>UniRef100_B n=1 Tax=Mus musculus TaxID=10090 RepID=B_MOUSE
MK-LA
>UniRef100_C n=2 Tax=Escherichia coli TaxID=562 RepID=C_ECOLI
MKvILA
"""


class TestStripInsertions:
    def test_lowercase_removed(self):
        assert strip_insertions("MKvVLA") == "MKVLA"

    def test_uppercase_and_gaps_kept(self):
        assert strip_insertions("MK-LA") == "MK-LA"

    def test_all_rows_share_query_length(self):
        recs = parse_a3m(A3M)
        assert {len(r.seq) for r in recs} == {5}


class TestParseA3M:
    def test_query_is_first(self):
        recs = parse_a3m(A3M)
        assert recs[0].header == "query"
        assert recs[0].seq == "MKVLA"

    def test_all_records_parsed(self):
        assert len(parse_a3m(A3M)) == 4

    def test_taxids_attached(self):
        recs = parse_a3m(A3M)
        assert [r.taxid for r in recs] == [None, 9606, 10090, 562]

    def test_empty_input(self):
        assert parse_a3m("") == []

    def test_ignores_blank_lines(self):
        assert len(parse_a3m("\n\n>q\nMKVLA\n\n")) == 1

    def test_multiline_sequence_is_joined(self):
        recs = parse_a3m(">q\nMKV\nLA\n")
        assert recs[0].seq == "MKVLA"

    def test_raises_on_ragged_alignment(self):
        # A row that is not query-length after stripping insertions means the a3m
        # is malformed -- fail loudly rather than silently mis-index residues.
        with pytest.raises(ValueError, match="length"):
            parse_a3m(">q\nMKVLA\n>bad\nMK\n")


class TestParseTaxid:
    def test_uniref_taxid_field(self):
        assert parse_taxid("UniRef100_A n=1 Tax=Homo sapiens TaxID=9606 RepID=X") == 9606

    def test_uniprot_ox_field(self):
        # UniProt-style fasta headers use OX= instead of TaxID=.
        assert parse_taxid("sp|P1|N_HUMAN Desc OS=Homo sapiens OX=9606 GN=X") == 9606

    def test_missing_taxonomy_returns_none(self):
        # This is the BFD/DIPS-Plus case: no taxonomy => unpairable.
        assert parse_taxid("some_bfd_hit_without_taxonomy") is None

    def test_non_numeric_taxid_returns_none(self):
        assert parse_taxid("x TaxID=N/A y") is None

    def test_case_insensitive(self):
        assert parse_taxid("x taxid=9606 y") == 9606


class TestDedup:
    def test_identical_sequences_collapse(self):
        recs = parse_a3m(A3M)
        # query and hit A are both "MKVLA".
        assert len(dedup_by_sequence(recs)) == 3

    def test_query_row_always_survives_dedup(self):
        recs = dedup_by_sequence(parse_a3m(A3M))
        assert recs[0].header == "query"

    def test_order_is_otherwise_preserved(self):
        recs = dedup_by_sequence(parse_a3m(A3M))
        assert [r.taxid for r in recs] == [None, 10090, 562]


class TestNeff:
    def test_identical_sequences_give_neff_one(self):
        # N copies of one sequence carry the information of ~1 sequence.
        assert neff(["MKVLA"] * 10) == pytest.approx(1.0)

    def test_distinct_sequences_give_neff_n(self):
        # Fully dissimilar rows are each their own cluster.
        assert neff(["AAAAA", "CCCCC", "DDDDD"]) == pytest.approx(3.0)

    def test_neff_between_one_and_n(self):
        seqs = ["MKVLA", "MKVLA", "MKILA", "WWWWW"]
        assert 1.0 <= neff(seqs) <= len(seqs)

    def test_empty_msa_is_zero(self):
        assert neff([]) == 0.0

    def test_gaps_do_not_count_as_matches(self):
        # "-----" vs "MKVLA" share no residue identity.
        assert neff(["MKVLA", "-----"]) == pytest.approx(2.0)
