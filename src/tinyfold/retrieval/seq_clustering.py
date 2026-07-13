"""Pure-Python sequence-identity clustering (MinHash + LSH union-find).

Deviation D1 (see BUILD-REPORT): the SPEC calls for MMseqs2 + Foldseek. Neither
is installed and Foldseek has no clean Windows build, so we cluster by SEQUENCE
identity in pure Python — the dominant leakage axis (PINDER/DIPS both cluster on
sequence first). Structural 3Di clustering is deferred.

Method
------
1. Represent each chain by its set of k-mers over the integer AA sequence.
2. MinHash each chain (num_perm hashes) -> compact signature.
3. LSH-band the signatures so only plausibly-similar chains become candidate
   pairs (avoids the O(N^2) all-pairs comparison over ~10^4 chains).
4. Union-Find: merge a candidate pair iff its EXACT k-mer Jaccard >= threshold.
5. A complex's cluster is the canonical unordered pair of its two chain
   clusters, so two dimers leak into each other only if BOTH chains are
   homologous.

Everything is deterministic (fixed hash seeds).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MERSENNE = (1 << 61) - 1  # large prime for universal hashing


def kmer_set(seq: Sequence[int], k: int) -> set:
    """Set of k-mers (as python ints) over an integer sequence.

    Each length-k window is packed into a single int in base (max_aa+1). For
    sequences shorter than k the whole sequence is used as one token so tiny
    fragments still hash to something.
    """
    n = len(seq)
    if n < k:
        return {tuple(seq)} if n else set()
    base = 32  # > 20 AA types + specials; fixed so packing is stable
    out = set()
    # rolling pack
    val = 0
    for i in range(k):
        val = val * base + int(seq[i])
    out.add(val)
    top = base ** (k - 1)
    for i in range(k, n):
        val = (val - int(seq[i - k]) * top) * base + int(seq[i])
        out.add(val)
    return out


def _hash_coeffs(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE, size=num_perm, dtype=np.int64)
    b = rng.randint(0, _MERSENNE, size=num_perm, dtype=np.int64)
    return a, b


def minhash_signatures(
    kmer_sets: list[set], num_perm: int, seed: int = 1234
) -> np.ndarray:
    """[N, num_perm] int64 MinHash signatures (max value for empty sets)."""
    a, b = _hash_coeffs(num_perm, seed)
    N = len(kmer_sets)
    sig = np.full((N, num_perm), np.iinfo(np.int64).max, dtype=np.int64)
    for i, ks in enumerate(kmer_sets):
        if not ks:
            continue
        km = np.fromiter(ks, dtype=np.int64, count=len(ks))
        # (a * km + b) mod prime for each perm -> [num_perm, len(km)], min over kmers
        hashed = (np.outer(a, km) + b[:, None]) % _MERSENNE
        sig[i] = hashed.min(axis=1)
    return sig


def lsh_candidate_pairs(sig: np.ndarray, bands: int, rows: int) -> set:
    """Candidate (i, j) index pairs that collide in >=1 LSH band."""
    N, num_perm = sig.shape
    assert bands * rows <= num_perm, f"bands*rows={bands*rows} > num_perm={num_perm}"
    pairs = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        chunk = sig[:, band * rows:(band + 1) * rows]
        for i in range(N):
            key = chunk[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def cluster_sequences(
    seqs: list[Sequence[int]],
    k: int = 3,
    threshold: float = 0.3,
    num_perm: int = 128,
    bands: int = 32,
    rows: int = 4,
    seed: int = 1234,
) -> list[int]:
    """Cluster integer sequences by k-mer Jaccard >= threshold.

    Returns a list of cluster labels (contiguous ints from 0), one per input
    sequence. Identical sequences always land in the same cluster; homologs
    merge when their exact Jaccard clears ``threshold``.
    """
    kmer_sets = [kmer_set(s, k) for s in seqs]
    sig = minhash_signatures(kmer_sets, num_perm, seed)
    candidates = lsh_candidate_pairs(sig, bands, rows)
    uf = _UnionFind(len(seqs))
    for i, j in candidates:
        if jaccard(kmer_sets[i], kmer_sets[j]) >= threshold:
            uf.union(i, j)
    # Relabel roots to contiguous ids (deterministic by first appearance).
    label_of: dict[int, int] = {}
    labels = []
    for i in range(len(seqs)):
        r = uf.find(i)
        if r not in label_of:
            label_of[r] = len(label_of)
        labels.append(label_of[r])
    return labels
