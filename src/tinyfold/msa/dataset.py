"""Map the complex dataset onto per-CHAIN MSA work.

MSAs are per unique chain SEQUENCE, not per complex. The dedup is large and is
what makes Phase 2 tractable at all (measured 2026-07-16):

    clean_le600 : 5,200 complexes -> 10,400 chain instances -> 4,750 unique
    full        : 41,883 complexes -> 83,766 chain instances -> 22,293 unique

So the search runs ~2.2x (le600) to 3.8x (full) fewer queries than a naive
per-complex-per-chain pipeline, and the a3m cache is keyed by sequence hash --
two complexes sharing a chain share its MSA.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from tinyfold.constants import IDX_TO_AA


def decode_sequence(seq_ids: np.ndarray) -> str:
    """Parquet stores sequences as amino-acid indices; render them as letters."""
    return "".join(IDX_TO_AA.get(int(i), "X") for i in np.asarray(seq_ids))


def chain_key(seq: str) -> str:
    """Stable content-addressed id for a chain sequence (the a3m cache key)."""
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ComplexChains:
    """The two chains of one complex, by sequence and by cache key."""

    sample_id: str
    seq_a: str
    seq_b: str

    @property
    def key_a(self) -> str:
        return chain_key(self.seq_a)

    @property
    def key_b(self) -> str:
        return chain_key(self.seq_b)


def split_complex(seq_ids: np.ndarray, la: int, lb: int, sample_id: str) -> ComplexChains:
    """Split a concatenated complex sequence into its two chains.

    The dataset stores ``seq`` as chain A followed by chain B, with lengths LA
    and LB -- the same layout the coevolution features assume.
    """
    seq = decode_sequence(seq_ids)
    if la + lb > len(seq):
        raise ValueError(
            f"{sample_id}: LA+LB={la + lb} exceeds sequence length {len(seq)}"
        )
    return ComplexChains(sample_id=sample_id, seq_a=seq[:la], seq_b=seq[la : la + lb])


def unique_chains(complexes: list[ComplexChains]) -> dict[str, str]:
    """``{chain_key: sequence}`` over both chains of every complex.

    This is the actual MSA work-list: one search per entry, not per chain
    instance.
    """
    out: dict[str, str] = {}
    for c in complexes:
        out.setdefault(c.key_a, c.seq_a)
        out.setdefault(c.key_b, c.seq_b)
    return out


def write_fasta(chains: dict[str, str], path) -> int:
    """Write the deduped work-list as FASTA (the MMseqs2 query db input)."""
    from pathlib import Path

    lines = []
    for key, seq in sorted(chains.items()):  # sorted => deterministic db order
        lines.append(f">{key}\n{seq}\n")
    Path(path).write_text("".join(lines))
    return len(chains)
