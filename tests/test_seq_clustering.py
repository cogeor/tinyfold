"""C1 tests: pure-Python sequence clustering (MinHash + LSH union-find)."""

import random

from tinyfold.retrieval.seq_clustering import (
    cluster_sequences,
    jaccard,
    kmer_set,
)


def test_identical_sequences_same_cluster():
    s = [random.randint(0, 19) for _ in range(120)]
    seqs = [list(s), list(s), list(s)]
    labels = cluster_sequences(seqs, k=3, threshold=0.3)
    assert labels[0] == labels[1] == labels[2]


def test_unrelated_sequences_separate():
    rng = random.Random(0)
    a = [rng.randint(0, 19) for _ in range(150)]
    b = [rng.randint(0, 19) for _ in range(150)]
    labels = cluster_sequences([a, b], k=4, threshold=0.3)
    # Two random 20-letter sequences share almost no 4-mers.
    assert labels[0] != labels[1]


def test_near_duplicate_merges():
    rng = random.Random(1)
    a = [rng.randint(0, 19) for _ in range(200)]
    b = list(a)
    # Mutate 10% of positions -> still high identity homolog.
    for i in range(0, 200, 10):
        b[i] = (b[i] + 1) % 20
    labels = cluster_sequences([a, b], k=3, threshold=0.3)
    assert labels[0] == labels[1]


def test_deterministic():
    rng = random.Random(2)
    seqs = [[rng.randint(0, 19) for _ in range(rng.randint(60, 180))] for _ in range(40)]
    l1 = cluster_sequences(seqs, seed=7)
    l2 = cluster_sequences(seqs, seed=7)
    assert l1 == l2


def test_kmer_and_jaccard_basic():
    assert kmer_set([1, 2, 3], k=3) == {1 * 32 * 32 + 2 * 32 + 3}
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4
    assert jaccard(set(), set()) == 1.0
