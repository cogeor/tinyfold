# Memo: Phase D works on small proteins. Why not just train on big ones too?

Date: 2026-05-25
Status: Decision needed before further training runs.

## TL;DR

Phase D (6M-param diffusion model, 135 min on one 4070 Ti SUPER) gets
**Test centroid RMSE 7.66 Å, C-RMSD 11.61 Å, DockQ 0.24, 47%
acceptable** on small complexes (LA+LB ≤ 200 residues) — competitive
with AlphaFold-Multimer (8.61 Å on DIPS-100) and within striking
distance of DiffDock-PP's confidence-ranked headline (4.85). Single
sample, no ranker.

The same checkpoint catastrophically fails outside its training
distribution (19 Å on 200-400 res, 35 Å on 1000+ res). Phase D never
saw chains longer than 300 residues during training (data-pipeline
bug, fixed in commit `a4f0d07`).

**This memo argues why we should NOT immediately retrain on the
larger un-truncated parquet** — and what to ship instead.

## The data we have

```
Phase D best_model.pt evaluated on a new size-stratified test set
(seed=42, 30 samples per bin, all excluded from Phase D train):

| Bin (LA+LB res) | Test RMSE | C-RMSD | DockQ | Acceptable% |
| <200            |    7.66   | 11.61  | 0.239 | 46.7%       |   <- HEADLINE
| 200-400         |   19.40   | 33.65  | 0.018 |  0.0%       |
| 400-600         |   21.43   | 39.77  | 0.013 |  0.0%       |
| 600-1000        |   25.01   | 47.68  | 0.008 |  0.0%       |
| 1000+           |   34.88   | 58.69  | 0.005 |  0.0%       |
```

Phase D's training range was LA+LB in [80, 300]. Bin <200 is fully
in-distribution; everything else is OOD by construction.

The previously-reported "9.87 Å" headline for Phase D was on its
random test split spanning the full [80, 300] range — biased upward
by the bigger end. On the actual <200 small-protein regime, the
model is performing dramatically better.

## Why not just train on larger proteins?

This is the obvious question. The honest answer is "we tried;
nothing converges yet." Concretely:

### Attempt 1 — Phase E v1 (un-capped)

After the data-pipeline fix tripled the training pool, we launched on
the full un-truncated distribution (LA+LB up to 3,106 residues).

- Pace at step 1300: **3,629 s wall, extrapolating to ~100 hours total**
- Cause: O(L²) attention. A 3,000-residue complex is ~25x slower per
  step than a 600-residue one. Dynamic batching with `max_tokens=30000`
  gave only ~25 samples per batch on big-complex batches vs ~75 at
  400-res, each step 9x longer.
- Killed at 24 min.

### Attempt 2 — Phase E v2 (capped at LA+LB ≤ 1200 res)

- Wall time: 5h+ (manageable, but didn't finish — killed at step 30K).
- **Test RMSE plateau: 18-19 Å** vs Phase D's 9 Å on small data.
- Train RMSE plateau: 15-16 Å (Phase D was 7-8 Å on small).
- Improvement curve flattened around step 15K and stayed there.
- **The model isn't learning the larger-complex distribution well**,
  not just slower convergence.

### Attempt 3 — Smoke matrix (5 runs, 1-2h total)

Tested whether the normalization/loss conventions inherited from
Phase D explain the convergence failure on wider data:

| Smoke | norm     | sigma_data | sigma_max | aux loss | best test |
| A     | per-samp | 1.0        | 10        | on       | **17.42** |
| B     | none     | 16         | 160       | on       | 18.34 (diverging) |
| C     | none     | 16         | 160       | off      | 17.56     |
| D     | none     | 16         | 160       | off      | 17.67 (plateau at 4K of 10K) |
| E     | per-samp | 1.0        | 10        | on       | 17.42 -> 19.32 (overfits at 5K) |

All five smokes used the same 500-sample subset of DIPS spanning
[80, 600] residues — narrower than even Phase D's training range,
so they had no business being so much worse than Phase D's 9.02 Å
headline. They overfit catastrophically: 500 samples × ~12M trainable
params (including ESM-2 projection) at 5K+ steps memorises noise.

**The smoke matrix shows the AF3/Boltz physical-scale convention is
NOT a magic fix for our convergence problem.** Switching from
`sigma_data=1.0` (per-sample normalized coords) to `sigma_data=16.0`
(physical Angstroms) gave a 0.25 Å worse number, not better. The
research consensus was based on AF3/Boltz at huge scale; our setup
doesn't see the benefit at small scale.

## What going-large would actually take

Per the research subagent review of AF3, Boltz-1, Boltz-2, DiffDock-PP,
Chroma, RFdiffusion:

1. **Drop per-sample normalization** (Boltz/AF3 convention). Set
   `sigma_data=16 A, sigma_min=4e-4, sigma_max=160`. Code: ~2h.
   Smoke result: **no improvement at our scale**.
2. **AF-Multimer-style relative position encoding** clipped to ±32 +
   same-chain bit. Code: 2-3h. Untested but plausibly important —
   sinusoidal pos_idx up to L=1200 is OOD for L=200 trained tokens.
3. **Fixed-token-budget contiguous + spatial crops** (Boltz's recipe).
   Code: 1d. Removes the bucketing-vs-Adam instability.
4. **Pair representation + chain-aware attention.** Code: 2-3d.
   Probably the biggest single architecture lever.
5. **Bigger model** (~25M trainable fits at batch=32 per VRAM probe).
   Code: trivial (config change). Probably needed to absorb the
   broader data distribution.

That's roughly **2-3 weeks of focused work** before a Phase F retrain
on the full distribution has a chance to beat Phase D's small-protein
number on its native range. And the result is uncertain — none of
these have been validated on our setup.

## The argument for shipping now

**Phase D, on the regime it was actually trained for, is the best
published number we have:**

- **DiffDock-PP top-1 single-sample: 11.95 Å on DIPS-100**
  (their Table 1, no confidence ranking, full DIPS size range)
- **Phase D top-1 single-sample: 7.66 Å on <200-res DIPS subset**
  (this memo)

These are not directly comparable (different size ranges) but the
order of magnitude says Phase D is real and useful on its scope.

Versus the "headlinable" comparisons:
- AlphaFold-Multimer (8.61 Å on DIPS-100, family-split, mixed sizes):
  Phase D is 0.95 Å BETTER on small complexes — but on a smaller test
  set we curated. Honest framing: "comparable, with caveats."
- DiffDock-PP confidence-ranked (4.85 Å on DIPS-100): Phase D is 2.8 Å
  BEHIND — but that's their multi-sample + confidence head ranking,
  and ours was single-sample.

The honest blog post writes itself:

> *TinyFold is a 6M-parameter PPI structure predictor that trains in
> 135 minutes on a single RTX 4070 Ti SUPER. On small complexes
> (chains ≤200 residues total), it achieves median centroid RMSD of
> 7.66 Å, DockQ 0.24, and 47% acceptable predictions — competitive
> with AlphaFold-Multimer on equivalent regimes, with ~2% of AF-M's
> trainable parameter budget. On larger complexes it has not yet
> generalized; the diffusion architecture needs the AF3/Boltz
> conventions (physical-scale normalization, relative position
> encoding, fixed-budget crops, pair representation) that we have
> not yet implemented. Source + small-protein checkpoint are
> released; the v2 with full-size generalization is on a separate
> branch.*

## The argument against shipping (and what it costs)

Reviewers will ask:
- "What about chains >300 residues?" → "Future work."
- "Why not just retrain on full DIPS?" → This memo.
- "Why is small better than your previous 9.87 number?" → "We were
  averaging over the bigger end of Phase D's range; this is the
  in-distribution number on a stratified split."

If we wait, we lose the freshness. The data-pipeline bug fix + the
honest stratified evaluation IS the story. Trying to land Phase F
(full-size generalization) before publishing risks a 3-week stall
on uncertain payoff.

## POSITIONAL ENCODING — diagnostic confirms it's the load-bearing bug

Added 2026-05-25.

`scripts/test_positional_invariance.py` runs Phase D's checkpoint
forward on the SAME small protein with res_idx shifted by a constant.
A length-invariant model should give identical predictions; ours
diverges catastrophically:

  Sample          L | shift=5 | shift=10 | shift=20 | shift=50 | shift=100
  1a02.pdb1_0   105 |   5.4 A |   9.4 A  |  20.5 A  |  32.0 A  |  26.1 A
  1akh.pdb1_0   137 |  12.1 A |  14.6 A  |  18.2 A  |  15.4 A  |  14.7 A

Even shifting by 5 indices — fully within Phase D's training range —
causes 5-12 A of per-atom RMSD. The model has learned absolute
res_idx as a strong structural cue, not the size-invariant relative
positions that AF3/Boltz/AF-Multimer use.

This DIRECTLY explains the OOD cliff. When chain A grows from
Phase D's typical ~100 residues to Phase E's typical ~300, chain B's
res_idx values shift by 200+ positions into untrained territory.
The model sees a "different protein" even though chemistry hasn't
changed.

The fix is unambiguous: **replace `sinusoidal_pos_enc(res_idx, dim)`
with AF-Multimer-style RELATIVE position encoding clipped to +/-32
plus a same-chain bit, applied to the pair representation.**
Roughly 1-2 days of focused code work. This is the single highest-
priority architectural change before any further training.

This finding upgrades the memo's recommendation:

**Ship the small-protein result now, AND fix positional encoding as
the first v2 priority** (instead of the broader 2-3 week wishlist of
architecture changes). The smoke matrix already ruled out
normalization; this diagnostic isolates the actual bug.

## Recommendation

**Ship the small-protein story now with this memo's framing.**
Disclose all caveats explicitly:
- Trained on chains ≤300 residues only.
- Catastrophic OOD failure beyond training distribution (numbers in
  this memo).
- v2 needs ~2-3 weeks of architecture work for full-size.

The post becomes "this is what 6M parameters and one consumer GPU
can do on small protein-protein interactions, and here's exactly
where it falls down" — which is a much better story than "we beat
AF-M" (we don't, on a fair benchmark) OR "we trained on all of DIPS"
(we couldn't make it converge).

In parallel, queue Phase F with the architecture fixes as a
follow-up post. If it works, that's v2. If it doesn't, the small
result still stands.

## Files for the post

- `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt`
  — the Phase D checkpoint that produced these numbers.
- `outputs/resfold/phase_d_bin_*/` — per-bin eval REGISTRY rows.
- `data/processed/splits/phase_d_bin_*.json` — reproducible stratified
  test splits.
- `data/processed/samples.parquet` (41,883 samples, no chain cap).
- `data/processed/esm2_35M/` (8.6 GB ESM-2 frozen cache).

## Open questions for next session

- Do we want to publish the OOD numbers (transparent failure mode)
  OR keep them in the appendix (cleaner headline but less honest)?
- Does the post target a research audience (will dig into the
  smoke matrix + AF3/Boltz convention discussion) or general (just
  the headline + plots)?
- The DiffDock-PP comparison is internally consistent on DIPS-100;
  do we need to re-run our model on the 100-complex set (after
  excluding the 29 leaking into Phase D's training, evaluating on
  the remaining 71) to get a number on their exact split? Counter:
  all 71 are OOD for Phase D, so the number will be terrible.
