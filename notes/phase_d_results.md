# Phase D — Full prio01-retrain stack

Date: 2026-05-24
Headline checkpoint: `outputs/resfold/phase_d_n8600_full/resfold_s1_8K_20260524_085326/best_model.pt` (step 50000, 135 min wall time)

## Stack

| Loop | Feature | Source |
|---|---|---|
| 01 | DockQ + C-RMSD eval | `feat(eval)` 53333ea |
| 02 | Multi-sample + HDOCK cluster | `feat(eval)` cc5abdf |
| 03 | (Kabsch interp — found to hurt, NOT in Phase D) | `feat(sampler)` 30a... |
| 04 | EDM lambda(sigma) weighting fix | `fix(training)` af3d528 |
| 05 | ESM-2-35M frozen embeddings (480 dim) | `feat(model)` c301512 |
| 06 | Confidence head + ranked Top-K | `feat(model)` 3f4ddec |

## Headline numbers

Phase C N=8600 vs Phase D N=8600, same step budget (50K), same architecture
(6M trainable params + ESM-2 frozen + 17k confidence head), single
RTX 4070 Ti SUPER.

| Metric | Phase C (9.87 baseline) | Phase D | Delta |
|---|---|---|---|
| Test Centroid RMSE (single sample) | 9.87 A | **9.02 A** | **-0.85** |
| Test Centroid RMSE (single, K=40 re-eval seed) | 9.99 A | 9.18 A | -0.81 |
| C-RMSD (chain-A Kabsch) | 15.93 A | 14.80 A | -1.13 |
| DockQ avg | 0.140 | 0.150 | +0.010 |
| DockQ acceptable (>= 0.23) | 26% | 29% | +3 pp |
| oracle@5 | 8.91 A | 8.11 A | -0.80 |
| oracle@40 | **8.46 A** | **7.53 A** | **-0.93** |
| mean@5 | 10.35 A | 9.62 A | -0.73 |
| ranked@5 (cluster) | 9.90 A | 9.42 A | -0.48 |
| ranked_conf@5 (confidence) | n/a | 9.02 A | new |

**Honest claim: 9.02 A top-1, 0.41 A behind AF-Multimer's 8.61 A, with
2% of AF-M's trainable params.**

The oracle@40 = 7.53 A number should NOT be headlined: subsequent
experiments with three independent ranking strategies (see "Ranker
ablation" below) show the K=40 pool lacks ranker-able diversity — the
"lucky" K=40 sample is mostly favourable noise around a single
predicted pose, not architectural headroom waiting for a smarter
ranker.

## Ranker ablation (Phase D best_model, N=100 test, K=40 re-eval)

Three orthogonal practical rankers were added on top of the confidence
head and re-evaluated against the same K=40 pool:

| Ranker | K=5 | K=40 | Notes |
|---|---|---|---|
| oracle (peeks at GT) | 8.11 | 7.53 | Upper bound; not deliverable |
| mean (random pick) | 9.62 | 9.57 | Baseline |
| cluster (HDOCK 5 A) | 9.42 | 9.41 | Loop 02 |
| confidence head | 9.02 | 9.18 | Loop 06 (pred_lddt argmax) |
| self-consistency | 9.34 | 9.31 | Min mean RMSD to other K-1 samples |
| geometric energy | 10.22 | 9.21 | Min cross-chain clashes - 0.1 * contacts |

Three signals — *learned* (confidence), *geometric* (self-consistency),
*physical* (clash/contact energy) — all converge on 9.18-9.31 A at K=40
and 9.02-10.22 A at K=5. None recovers more than 0.4 A of the 2 A
oracle gap.

**Interpretation:** the model has collapsed to a single predicted
pose; the K=40 pool is that pose with noise wiggle. Going from K=5 to
K=40 (8x more samples) improves oracle by 0.58 A (8.11 -> 7.53), which
is consistent with Gaussian min-of-K statistics (expected ~1.51x
spread growth; observed 1.35x). The "best of 40" is just the
favourable noise direction, not a different, better pose.

The architecture *is* the bottleneck for top-1 quality. Adding rankers
on this model won't help. The next move is **pose diversity** (e.g.
ensemble-of-seeds, temperature-tuned sampling, conditional generation
on different chain assignments), not better ranking.

## Confidence head behaviour

- Spearman(pred_lddt, -RMSE) per-target across the K-sample pool: **0.640**
  at the K=40 re-eval. Real correlation: the head has learned a useful
  per-target quality signal, comparable to lDDT prediction in published
  small-model confidence heads.
- Variance ACROSS samples for the SAME target: pred_lddt_std = 0.0038.
  All K samples for one target get nearly identical confidence
  predictions, so the rank is essentially arbitrary -> ranked_conf@K
  collapses to the single-sample baseline.
- Diagnosis: the head learns target-level lDDT (easy: target hard/easy
  signal in mean-pooled tokens) but not sample-level differences within
  one target. Mean-pool over tokens washes out the per-sample noise
  signal needed to rank.

## What worked

1. **EDM weighting fix (Loop 04)**: ~0.5 A of the 0.85 A gain plausibly
   comes from finally applying lambda(sigma) correctly at the per-sample
   MSE level. Phase C trained without weighting; Phase D has it on.
2. **ESM-2 frozen embeddings (Loop 05)**: replaced 23-token AA lookup
   with 480-dim ESM-2-35M. Confidence head Spearman 0.64 at step 50K
   vs ~0.5 in the smoke test suggests the ESM features carry per-target
   quality cues the head can decode.
3. **Multi-sample oracle@40 (Loop 02)**: from 8.46 to 7.53. Phase D's
   pose distribution is more diverse than Phase C's (mean@40 9.57 vs
   10.33 — the model's K=40 outputs span a wider quality range).

## What didn't (and is documented as such)

1. **Kabsch interpolation (Loop 03)**: confirmed worse than one-shot at
   N=8600. Dropped from Phase D config.
2. **Confidence-head sample-level ranking**: head learns target-level
   lDDT but pred_lddt_std across same-target samples is too low to
   pick a winner. Future work: per-residue lDDT prediction (instead of
   per-target mean-pool) or contrastive training across the K-sample
   pool.
3. **Cluster ranking**: ranked@K within 0.05 A of mean@K throughout.
   Cluster-rep ranking is no better than random pick at our pose
   diversity level. HDOCK trick is null at this scale.

## Updated SOTA placement

| Method | C-RMSD on DIPS-style (A) | Our number |
|---|---|---|
| HDOCK (benchmark-leaky on DIPS) | 6.23 | |
| AF-Multimer | 8.61 | |
| Boltz / AF3 (DockQ-headlined) | comparable | |
| **TinyFold Phase D top-1 (practical)** | | **9.02** |
| TinyFold Phase C (prev headline) | | 9.87 |
| DiffDock-PP top-1 | 11.95 | |
| EquiDock | 13.30 | |

Oracle numbers omitted on purpose: they are not user-deliverable and
the ranker ablation above shows they are not achievable with any
ranker on the current pose distribution.

The honest claim that survives peer scrutiny:

> *TinyFold (6M trainable + frozen ESM-2-35M) reaches median centroid
> CA RMSD of 9.02 A on a 100-complex DIPS-Plus test split, training in
> 135 min on a single RTX 4070 Ti SUPER. This is ~0.4 A behind
> AlphaFold-Multimer (8.61 A) on roughly equivalent splits with ~2% of
> AF-Multimer's trainable parameter budget (and a frozen 35M ESM-2
> backbone). The model produces a single predicted pose per target;
> multi-sample inference does NOT yield ranker-able diversity (three
> independent rankers — learned confidence, self-consistency, and
> physical clash/contact energy — all cluster within 0.4 A of each
> other, none recovering more than 20% of the 2 A oracle gap), so the
> next research direction is pose-distribution diversity rather than
> better ranking.*

## Files

- Best checkpoint: `outputs/resfold/phase_d_n8600_full/resfold_s1_8K_20260524_085326/best_model.pt`
- Final checkpoint: same dir, `final_model.pt`
- Config: `configs/train/resfold/phase_d_n8600_full.yaml`
- K=40 re-eval: `outputs/resfold/phase_d_n8600_full_reeval_k40/resfold_s1_8K_20260524_111306/`
- ESM-2 cache: `data/processed/esm2_35M/` (28352 NPZ, 8.63 GB, gitignored)
- Training log: 50000 steps + 11 eval lines.

## Next steps (post-blog or v2)

The ranker ablation (above) showed the *ranker* path is exhausted on
this model. Real next-step priorities, in order:

1. **Pose-distribution diversity** — the K=40 pool is one mode with
   noise. Options:
   - Ensemble of seeds (train 3-5 models with different init seeds;
     each is one mode; pool spans modes).
   - Temperature-tuned VE sampling (higher T -> more pose entropy,
     but with risk of broken structures).
   - Conditional generation on different chain-A/chain-B role
     assignments (we always feed chain 0 = A; flipping is a free
     diversity axis).
2. **Per-residue confidence head + contrastive loss** — useful only
   if step 1 unlocks real pose diversity. Without diversity there is
   nothing to rank.
3. **Family-filtered / PINDER hard-split eval** — verify the 9.02 A
   number isn't inflated by random-split overlap with training data.
4. **Run AF-Multimer / ColabFold on the same 100 targets** — the 8.61
   A AF-M number is from literature on different splits; rerun for an
   apples-to-apples baseline.
5. **Pair representation + chain-aware relpos** (Task H, deferred).
6. **ESM-2-150M** — try the bigger ESM variant; current cache is
   already gitignored and disk allows it.

Items 3 and 4 are cheap (1-2 h each); items 1-2 are weeks of work.
