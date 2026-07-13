"""C4: monomer-fold retriever (pure-Python; deviation D1/D4).

Given a query complex, retrieve a homologous MONOMER fold for each of its two
chains from the library, to be supplied as a per-chain template (docking hidden
via per-chain frames). Honest-leakage protocol (see BUILD-REPORT D4):

  * The library is the full pool of chains (all complexes).
  * For a query chain, candidates are library chains in the SAME chain-cluster
    (fold homologs) that come from a DIFFERENT complex-cluster than the query.
    Excluding the query's own complex-cluster prevents handing back the query's
    own dimer; providing a homologous monomer fold (not the docking) is exactly
    what a template is. The model is trained with COMPLEX-cluster holdout, so a
    test complex's docking is never seen — the template supplies folds only.
  * Rank candidates by sequence similarity (k-mer Jaccard); the top hit's
    backbone is sequence-aligned onto the query residues (identity map when the
    sequence is identical, else Needleman-Wunsch). Uncovered query residues are
    masked.

The retriever returns, per query sample, a list of per-chain assignments
naming the SOURCE (sample_id, chain, residue map). Coordinates are copied from
those sources by the cache builder (prepare_templates.py, C6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .seq_clustering import jaccard, kmer_set


@dataclass
class ChainEntry:
    sample_id: str
    chain: int            # 0 (A) or 1 (B)
    seq: tuple[int, ...]
    chain_cluster: int
    complex_cluster: int


@dataclass
class ChainTemplate:
    """A retrieved template for one query chain.

    ``query_to_src[q]`` gives the source residue index aligned to query residue
    ``q`` (within this chain), or -1 if query residue q is uncovered.
    """
    src_sample_id: str
    src_chain: int
    query_to_src: list[int]
    identity: float       # k-mer Jaccard of the hit (diagnostic)


def needleman_wunsch(a: Sequence[int], b: Sequence[int],
                     match: int = 1, mismatch: int = -1, gap: int = -1) -> list[int]:
    """Global align b (template) onto a (query).

    Returns ``a_to_b`` of length len(a): for each query position, the aligned
    template index or -1 (gap). Simple O(len(a)*len(b)) DP; fine for chains of a
    few hundred residues.
    """
    na, nb = len(a), len(b)
    # DP score matrix.
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = i * gap
    for j in range(1, nb + 1):
        dp[0][j] = j * gap
    for i in range(1, na + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, nb + 1):
            s = match if ai == b[j - 1] else mismatch
            row[j] = max(prev[j - 1] + s, prev[j] + gap, row[j - 1] + gap)
    # Traceback.
    a_to_b = [-1] * na
    i, j = na, nb
    while i > 0 and j > 0:
        s = match if a[i - 1] == b[j - 1] else mismatch
        if dp[i][j] == dp[i - 1][j - 1] + s:
            a_to_b[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return a_to_b


class MonomerRetriever:
    """Build a chain-cluster index and retrieve per-chain monomer templates."""

    def __init__(self, k: int = 3, exclude_same_complex: bool = False):
        # exclude_same_complex=False: only the query's own sample is excluded.
        # Templates are supplied with per-chain frames, so the inter-chain
        # DOCKING never leaks regardless of the source complex — this is the
        # "fold-retrieve + learn-docking" decomposition. Set True for the
        # stricter different-complex-cluster protocol (much lower coverage).
        self.k = k
        self.exclude_same_complex = bool(exclude_same_complex)
        self.entries: list[ChainEntry] = []
        self.by_chain_cluster: dict[int, list[int]] = {}   # cluster -> entry indices
        self.by_sample: dict[str, list[int]] = {}          # sample_id -> [entryA, entryB]
        self._kmer_cache: dict[int, set] = {}

    def add_chain(self, entry: ChainEntry) -> None:
        idx = len(self.entries)
        self.entries.append(entry)
        self.by_chain_cluster.setdefault(entry.chain_cluster, []).append(idx)
        self.by_sample.setdefault(entry.sample_id, []).append(idx)

    def _kmers(self, idx: int) -> set:
        ks = self._kmer_cache.get(idx)
        if ks is None:
            ks = kmer_set(self.entries[idx].seq, self.k)
            self._kmer_cache[idx] = ks
        return ks

    def retrieve_chain(self, query_idx: int) -> ChainTemplate | None:
        """Best homologous monomer template for entry ``query_idx``, or None."""
        q = self.entries[query_idx]
        cands = self.by_chain_cluster.get(q.chain_cluster, [])
        q_km = self._kmers(query_idx)
        best_j = None
        best_sim = -1.0
        for j in cands:
            if j == query_idx:
                continue
            e = self.entries[j]
            if e.sample_id == q.sample_id:
                continue  # never template from the query's own complex
            if self.exclude_same_complex and e.complex_cluster == q.complex_cluster:
                continue  # stricter: different complex-cluster only
            sim = jaccard(q_km, self._kmers(j))
            # Prefer higher similarity; break ties toward exact-length match.
            if sim > best_sim or (sim == best_sim and best_j is not None
                                  and abs(len(e.seq) - len(q.seq))
                                  < abs(len(self.entries[best_j].seq) - len(q.seq))):
                best_sim = sim
                best_j = j
        if best_j is None:
            return None
        src = self.entries[best_j]
        if src.seq == q.seq:
            q_to_src = list(range(len(q.seq)))          # identity map
        else:
            q_to_src = needleman_wunsch(q.seq, src.seq)
        return ChainTemplate(
            src_sample_id=src.sample_id, src_chain=src.chain,
            query_to_src=q_to_src, identity=best_sim,
        )

    def retrieve_sample(self, sample_id: str) -> dict[int, ChainTemplate | None]:
        """Return {query_chain: ChainTemplate or None} for a query sample."""
        out: dict[int, ChainTemplate | None] = {}
        for idx in self.by_sample.get(sample_id, []):
            chain = self.entries[idx].chain
            out[chain] = self.retrieve_chain(idx)
        return out
