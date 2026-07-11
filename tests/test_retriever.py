"""C4 tests: monomer-fold retriever (exclusion, alignment, no-hit)."""

from tinyfold.retrieval.retriever import (
    ChainEntry,
    MonomerRetriever,
    needleman_wunsch,
)


def test_nw_identity_map():
    a = [1, 2, 3, 4, 5]
    assert needleman_wunsch(a, a) == [0, 1, 2, 3, 4]


def test_nw_with_gap():
    # template missing the middle residue -> query pos 2 should be a gap (-1).
    a = [1, 2, 3, 4]
    b = [1, 2, 4]
    mapping = needleman_wunsch(a, b)
    assert len(mapping) == 4
    assert mapping[0] == 0 and mapping[1] == 1
    assert mapping[3] == 2          # query 4 -> template idx 2 (value 4)
    assert mapping[2] == -1         # query 3 uncovered


def _entry(sid, chain, seq, cc, xc):
    return ChainEntry(sample_id=sid, chain=chain, seq=tuple(seq),
                      chain_cluster=cc, complex_cluster=xc)


def test_retrieves_homolog_from_other_sample():
    r = MonomerRetriever(k=3)
    seq = list(range(20)) * 3          # 60-res chain
    r.add_chain(_entry("Q", 0, seq, cc=5, xc=100))
    # identical-seq homolog in a different sample -> valid hit (docking hidden
    # via per-chain frames, so same complex-cluster is allowed by default).
    r.add_chain(_entry("H", 1, seq, cc=5, xc=100))
    tmpl = r.retrieve_sample("Q")[0]
    assert tmpl is not None
    assert tmpl.src_sample_id == "H"
    assert tmpl.query_to_src == list(range(len(seq)))   # identity (exact seq)


def test_strict_excludes_same_complex_cluster():
    r = MonomerRetriever(k=3, exclude_same_complex=True)
    seq = list(range(20)) * 3
    r.add_chain(_entry("Q", 0, seq, cc=5, xc=100))
    r.add_chain(_entry("S", 1, seq, cc=5, xc=100))   # same complex cluster
    assert r.retrieve_sample("Q")[0] is None
    # but a different complex-cluster homolog IS returned
    r.add_chain(_entry("H", 0, seq, cc=5, xc=200))
    assert r.retrieve_sample("Q")[0].src_sample_id == "H"


def test_never_templates_from_own_sample():
    r = MonomerRetriever(k=3)
    seq = list(range(20)) * 3
    r.add_chain(_entry("Q", 0, seq, cc=7, xc=100))   # only the query's own chain
    assert r.retrieve_sample("Q")[0] is None


def test_singleton_cluster_no_hit():
    r = MonomerRetriever(k=3)
    r.add_chain(_entry("Q", 0, list(range(30)), cc=9, xc=100))
    assert r.retrieve_sample("Q")[0] is None
