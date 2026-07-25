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

STATUS — OFF, and not wired into the training entry point (removed in D1).
The bet did not pay off in our regime. The M0/M1 A/B at matched steps was null:
M1 - M0 = +0.0006 DockQ at 3k steps and +0.0055 at 6k -- inside noise. Two
reasons it fails here: (1) the Phase-2 greenlight sample had median paired depth
928, but the actual training population has median 109 with 21% of complexes
below 10 pairs, so the pairing crux above degrades to near-monomer profiles on
most targets; (2) at 6.7M params with ESM2-35M we have neither the capacity to
exploit paired MSAs nor a pLM weak enough to need them (Chai-1 shows a strong pLM
track substitutes for MSAs outright). C9's larger frozen pLM is the cheaper bet
on the same axis.

This library and the PairTrack ``msa_cond`` plumbing (types.py / data.py /
cropping.py / prepare_msa_* scripts) are KEPT as a verified artifact -- 4,750
chains searched, the depth instrument reproducible to ratio-median 0.998 -- so
the result is re-derivable without redoing the search. It is simply no longer
reachable from scripts/train_resfold.py. Do not re-run it expecting a win; read
the deltas above first.
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
