"""MSA / cross-chain coevolution as a pair-track prior.

Design: notes/2026-07-14-msa-coevolution-pair-prior-SPEC.md
Plan:   notes/2026-07-16-three-phase-plan.md

The bet in one line: ESM already internalises PER-CHAIN evolutionary statistics,
so the one thing it cannot give us is the docking-relevant part -- CROSS-CHAIN
coevolution. Paired-MSA covariation IS that prior, and it rides the same
zero-init pair-channel bus the retrieved template already uses.

Layout:
* ``a3m``     -- parse a3m, extract taxonomy, dedup, measure depth (Neff).
* ``pairing`` -- join two chains' MSAs by species. THE CRUX: get this wrong and
                 you get two better monomer profiles and ZERO docking signal.
* ``features``-- paired MSA -> APC-corrected coupling pair features [L, L, F].

Everything here is pure/offline: no database is needed to import or test it.
"""

from tinyfold.msa.a3m import (
    MsaRecord,
    dedup_by_sequence,
    neff,
    parse_a3m,
    parse_taxid,
    read_a3m,
    strip_insertions,
)
from tinyfold.msa.pairing import pair_msas, pairing_stats

__all__ = [
    "MsaRecord",
    "dedup_by_sequence",
    "neff",
    "pair_msas",
    "pairing_stats",
    "parse_a3m",
    "parse_taxid",
    "read_a3m",
    "strip_insertions",
]
