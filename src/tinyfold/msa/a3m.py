"""a3m parsing, taxonomy extraction, dedup, and depth.

a3m is the alignment format MMseqs2/ColabFold/HHblits emit. Two properties matter
here:

* **Lowercase letters are INSERTIONS** relative to the query and are not
  alignment columns. Stripping them makes every row query-length, which is what
  lets us index a row by our residue index.
* **Taxonomy lives in the header**, and taxonomy is the ONLY thing that lets us
  pair orthologs across chains. ``TaxID=`` (UniRef) and ``OX=`` (UniProt) are
  both supported.

  Headers with NO taxonomy are the BFD case -- which is exactly why DIPS-Plus's
  free 11 GB HHblits-vs-BFD MSA tarball cannot give cross-chain coevolution, and
  why the database has to have taxonomy (see notes/2026-07-16-three-phase-plan.md
  §B4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# TaxID=9606 (UniRef) or OX=9606 (UniProt). Case-insensitive; value must be an
# integer -- 'TaxID=N/A' and friends are treated as missing.
_TAXID_RE = re.compile(r"\b(?:taxid|ox)=(\d+)\b", re.IGNORECASE)

# Insertion columns are lowercase; '.' is also an insertion placeholder in a3m.
_INSERTION_RE = re.compile(r"[a-z.]")


@dataclass(frozen=True)
class MsaRecord:
    """One aligned row: header, query-length sequence, and species (if known)."""

    header: str
    seq: str
    taxid: int | None = None


def strip_insertions(seq: str) -> str:
    """Drop insertion columns (lowercase / '.'), leaving alignment columns only."""
    return _INSERTION_RE.sub("", seq)


def parse_taxid(header: str) -> int | None:
    """Extract an NCBI taxon id from an a3m header, or None if absent.

    None means the row is UNPAIRABLE -- it can still deepen a per-chain profile,
    but it contributes nothing to cross-chain coevolution.
    """
    m = _TAXID_RE.search(header)
    return int(m.group(1)) if m else None


def parse_a3m(text: str) -> list[MsaRecord]:
    """Parse a3m text into query-length records. Row 0 is the query.

    Raises:
        ValueError: if a row is not query-length after stripping insertions
            (a malformed a3m would silently mis-index residues otherwise).
    """
    records: list[MsaRecord] = []
    header: str | None = None
    chunks: list[str] = []
    query_len: int | None = None

    def flush() -> None:
        nonlocal header, chunks, query_len
        if header is None:
            return
        seq = strip_insertions("".join(chunks))
        if query_len is None:
            query_len = len(seq)
        elif len(seq) != query_len:
            raise ValueError(
                f"a3m row {header!r} has length {len(seq)} after stripping "
                f"insertions, expected query length {query_len}"
            )
        records.append(MsaRecord(header=header, seq=seq, taxid=parse_taxid(header)))
        header, chunks = None, []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line)
    flush()
    return records


def read_a3m(path: str | Path) -> list[MsaRecord]:
    """Parse an a3m file (plain or gzipped by extension)."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as fh:
            return parse_a3m(fh.read())
    return parse_a3m(path.read_text())


def dedup_by_sequence(records: list[MsaRecord]) -> list[MsaRecord]:
    """Drop exact duplicate sequences, keeping first occurrence.

    The query (row 0) always survives: it is the anchor every downstream step
    indexes against.
    """
    seen: set[str] = set()
    out: list[MsaRecord] = []
    for rec in records:
        if rec.seq in seen:
            continue
        seen.add(rec.seq)
        out.append(rec)
    return out


def _encode(seqs: list[str]) -> np.ndarray:
    """[N, L] uint8 of raw byte codes."""
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)


def neff(seqs: list[str], identity_threshold: float = 0.8) -> float:
    """Effective sequence count: sum_i 1 / |{j : seqid(i, j) > threshold}|.

    The standard MSA-depth statistic. A deep-but-redundant alignment (many near
    copies) has a LOW Neff and carries little coevolution signal -- which is
    exactly what Step 0 needs to measure.

    Gaps never count as a match, so an all-gap row is dissimilar to everything.

    O(N^2 L). Fine for the Step-0 sample; cap N before calling on deep MSAs.
    """
    if not seqs:
        return 0.0
    arr = _encode(seqs)
    n, length = arr.shape
    if length == 0:
        return float(n)

    gap = ord("-")
    is_res = arr != gap
    # same[i, j] = # positions where i and j carry the SAME non-gap residue.
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        same = (arr[i] == arr) & is_res[i] & is_res
        ident = same.sum(axis=1) / length
        # A row is always its own neighbour, even when it shares no residue
        # identity with anything (e.g. an all-gap row, whose self-identity is 0
        # because gaps never match). Without the floor such a row would divide
        # by zero instead of forming its own cluster.
        weights[i] = 1.0 / max(int(np.count_nonzero(ident > identity_threshold)), 1)
    return float(weights.sum())
