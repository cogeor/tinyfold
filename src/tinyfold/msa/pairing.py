"""Pair two chains' MSAs by species -- the cross-chain coevolution join.

THE CRUX (spec §6). Coevolution between chain A residue i and chain B residue j
is only meaningful if row n of the paired alignment holds sequences from the SAME
organism: we are asking "when A_i mutated in this species, did B_j co-mutate?".
Pair the rows wrong and every downstream tensor still has the right shape, the
model still trains, and the answer is just noise -- a silent failure.

Semantics (AF-Multimer / ColabFold):
* Row 0 is always ``query_A + query_B``.
* A species present in only ONE chain contributes nothing.
* A hit with no taxonomy is unpairable (the BFD case -- see a3m.parse_taxid).
* Within a species, hits are paired BY RANK (best-to-best, 2nd-to-2nd, ...).
  Input order is taken to BE rank order: MMseqs2/HHblits emit hits best-first, so
  the caller must not re-sort before pairing.
* Unequal hit counts truncate to the shorter side -- we never invent a partner.

``pairing_stats`` is the Step-0 instrument: it reports paired depth and shared
species per complex WITHOUT building any features, so the "is there signal on
DIPS at all?" question can be answered before committing to a database.
"""

from __future__ import annotations

from collections import defaultdict

from tinyfold.msa.a3m import MsaRecord


def _by_species(records: list[MsaRecord]) -> dict[int, list[MsaRecord]]:
    """Group hits (rows 1..N) by taxid, preserving rank order within a species."""
    groups: dict[int, list[MsaRecord]] = defaultdict(list)
    for rec in records[1:]:
        if rec.taxid is not None:
            groups[rec.taxid].append(rec)
    return groups


def pair_msas(
    msa_a: list[MsaRecord],
    msa_b: list[MsaRecord],
    max_per_species: int | None = None,
) -> tuple[list[MsaRecord], list[MsaRecord]]:
    """Join two chains' MSAs by species into two row-aligned alignments.

    Args:
        msa_a, msa_b: per-chain MSAs, row 0 = query, hits best-first.
        max_per_species: cap on paired rows per species (bounds paralog
            explosion). None = uncapped.

    Returns:
        ``(paired_a, paired_b)`` of equal length. Row 0 is the query pair; each
        later row i has ``paired_a[i].taxid == paired_b[i].taxid``.

    Raises:
        ValueError: if either MSA is empty (no query to anchor against).
    """
    if not msa_a or not msa_b:
        raise ValueError("both MSAs must contain at least a query row")

    out_a: list[MsaRecord] = [msa_a[0]]
    out_b: list[MsaRecord] = [msa_b[0]]

    groups_a = _by_species(msa_a)
    groups_b = _by_species(msa_b)

    # Sorted for determinism: the cache must not depend on dict iteration order.
    for taxid in sorted(set(groups_a) & set(groups_b)):
        hits_a, hits_b = groups_a[taxid], groups_b[taxid]
        n = min(len(hits_a), len(hits_b))
        if max_per_species is not None:
            n = min(n, max_per_species)
        out_a.extend(hits_a[:n])
        out_b.extend(hits_b[:n])

    return out_a, out_b


def pairing_stats(msa_a: list[MsaRecord], msa_b: list[MsaRecord]) -> dict:
    """Depth/pairability report for ONE complex (the Step-0 instrument).

    Returns a dict with:
        n_a, n_b:            raw per-chain depth.
        n_species_a/b:       distinct taxids per chain.
        n_species_shared:    taxids present in BOTH -- the pairing substrate.
        n_paired:            paired rows (including the query row).
        frac_no_taxid_a/b:   fraction of HITS lacking taxonomy (unpairable).

    A complex with ``n_paired`` in the single digits has effectively no
    cross-chain coevolution signal, no matter how deep its per-chain MSAs are.
    """
    groups_a = _by_species(msa_a)
    groups_b = _by_species(msa_b)
    hits_a, hits_b = msa_a[1:], msa_b[1:]
    paired_a, _ = pair_msas(msa_a, msa_b)

    def frac_no_taxid(hits: list[MsaRecord]) -> float:
        if not hits:
            return 0.0
        return sum(1 for r in hits if r.taxid is None) / len(hits)

    return {
        "n_a": len(msa_a),
        "n_b": len(msa_b),
        "n_species_a": len(groups_a),
        "n_species_b": len(groups_b),
        "n_species_shared": len(set(groups_a) & set(groups_b)),
        "n_paired": len(paired_a),
        "frac_no_taxid_a": frac_no_taxid(hits_a),
        "frac_no_taxid_b": frac_no_taxid(hits_b),
    }
