"""Complex -> per-chain MSA work-list.

The dedup asserted here is what makes Phase 2 affordable: 10,400 chain instances
in clean_le600 collapse to 4,750 unique sequences.
"""

import numpy as np
import pytest

from tinyfold.constants import AA_TO_IDX
from tinyfold.msa.dataset import (
    ComplexChains,
    chain_key,
    decode_sequence,
    split_complex,
    unique_chains,
    write_fasta,
)


def encode(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX[c] for c in seq], dtype=np.uint8)


class TestDecode:
    def test_roundtrip(self):
        assert decode_sequence(encode("MKVLA")) == "MKVLA"

    def test_unknown_index_becomes_x(self):
        assert decode_sequence(np.array([20])) == "X"


class TestChainKey:
    def test_same_sequence_same_key(self):
        assert chain_key("MKVLA") == chain_key("MKVLA")

    def test_different_sequence_different_key(self):
        assert chain_key("MKVLA") != chain_key("MKVLC")

    def test_key_is_filename_safe(self):
        assert chain_key("MKVLA").isalnum()


class TestSplitComplex:
    def test_splits_at_la(self):
        c = split_complex(encode("AAAKKK"), la=3, lb=3, sample_id="s")
        assert (c.seq_a, c.seq_b) == ("AAA", "KKK")

    def test_asymmetric_chains(self):
        c = split_complex(encode("AAKKKK"), la=2, lb=4, sample_id="s")
        assert (c.seq_a, c.seq_b) == ("AA", "KKKK")

    def test_rejects_inconsistent_lengths(self):
        with pytest.raises(ValueError, match="exceeds"):
            split_complex(encode("AAA"), la=3, lb=3, sample_id="bad")


class TestUniqueChains:
    def test_dedups_repeated_chain_across_complexes(self):
        # A homodimer plus a complex reusing chain A: 4 instances, 2 unique.
        cs = [
            ComplexChains("s1", "AAA", "AAA"),
            ComplexChains("s2", "AAA", "KKK"),
        ]
        assert len(unique_chains(cs)) == 2

    def test_homodimer_counts_once(self):
        assert len(unique_chains([ComplexChains("s", "AAA", "AAA")])) == 1

    def test_maps_key_to_sequence(self):
        chains = unique_chains([ComplexChains("s", "AAA", "KKK")])
        assert chains[chain_key("AAA")] == "AAA"

    def test_distinct_chains_all_kept(self):
        cs = [ComplexChains(f"s{i}", f"AA{c}", "KKK") for i, c in enumerate("CDEF")]
        assert len(unique_chains(cs)) == 5  # 4 distinct A + 1 shared B


class TestWriteFasta:
    def test_writes_one_record_per_unique_chain(self, tmp_path):
        p = tmp_path / "q.fasta"
        n = write_fasta({"k1": "AAA", "k2": "KKK"}, p)
        assert n == 2
        assert p.read_text().count(">") == 2

    def test_headers_are_chain_keys(self, tmp_path):
        p = tmp_path / "q.fasta"
        write_fasta({"abc123": "AAA"}, p)
        assert ">abc123" in p.read_text()

    def test_order_is_deterministic(self, tmp_path):
        chains = {"z": "AAA", "a": "KKK", "m": "CCC"}
        a, b = tmp_path / "a.fa", tmp_path / "b.fa"
        write_fasta(chains, a)
        write_fasta(dict(reversed(list(chains.items()))), b)
        assert a.read_text() == b.read_text()
