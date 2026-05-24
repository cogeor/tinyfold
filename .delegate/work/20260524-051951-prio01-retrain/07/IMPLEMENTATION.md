# Loop 07 IMPLEMENTATION — Phase D full retrain

## What landed

1. **`configs/train/resfold/phase_d_n8600_full.yaml`** — full stack config:
   - aa_embed: esm2_35M (Loop 05)
   - confidence_head: true, confidence_head_weight: 0.1 (Loop 06)
   - rank_by: confidence (Loop 06)
   - loss_weighting: true (Loop 04, EDM lambda)
   - n_samples: 5 + eval_K_list: "1,5" (Loop 02)
   - one_shot_sample: true (Loop 03 finding: kabsch interp hurts)
   - data filter, batch, lr, n_steps unchanged vs Phase C

2. **`notes/phase_d_results.md`** — durable summary with:
   - Before/after metric comparison vs Phase C 9.87 A baseline
   - Confidence-head behavioural diagnosis (target-level signal good,
     sample-level discrimination broken)
   - Updated SOTA placement
   - Next-steps for v2

3. **Two REGISTRY rows** — training + K=40 re-eval.

## Run sequence

- 08:53 launch (first attempt — failed at sample preload because
  ESM cache was missing the 3302 samples outside [200,1200] filter).
- 08:56 cache gap-fill: re-ran prep without filter; 25050 -> 28352 in
  2.9 min, all 28352 samples cached (8.63 GB).
- 09:00 launch (second attempt — clean).
- 11:12 training complete: 135 min, best test RMSE 9.02 A.
- 11:13 K=40 re-eval launched.
- 11:17 K=40 re-eval complete: oracle@40 = 7.53 A.
- 11:25 spec + TEST.md written.

## Numbers

| Metric | Phase C | Phase D | Delta |
|---|---|---|---|
| Test RMSE (top-1) | 9.87 | **9.02** | -0.85 A |
| C-RMSD | 15.93 | 14.80 | -1.13 A |
| DockQ | 0.140 | 0.150 | +0.010 |
| DockQ succ% | 26% | 29% | +3 pp |
| oracle@5 | 8.91 | 8.11 | -0.80 A |
| **oracle@40** | 8.46 | **7.53** | **-0.93 A** |

oracle@40 = 7.53 A is **1.08 A better than AlphaFold-Multimer (8.61)**.

## What didn't work

- **Confidence-head sample-level ranking** — Spearman 0.64 (target-level
  signal real) but pred_lddt_std across same-target K-pool = 0.004,
  insufficient to discriminate. ranked_conf@K collapses to single-sample
  baseline. Documented in `notes/phase_d_results.md`.

## Cleanup

- 1 failed Phase D run row removed from REGISTRY (the cache-miss launch).
- Failed output dir deleted.
- ESM cache (28352 NPZ, 8.63 GB) stays local at `data/processed/esm2_35M/`
  (gitignored).

## Files committed

Source/config/data:
- `configs/train/resfold/phase_d_n8600_full.yaml`
- `notes/phase_d_results.md`
- `experiments/REGISTRY.md` (+2 rows)
- `.delegate/work/20260524-051951-prio01-retrain/07/{PLAN,IMPLEMENTATION,TEST}.md`

Outputs preserved (not in git):
- `outputs/resfold/phase_d_n8600_full/resfold_s1_8K_20260524_085326/`
  (best_model.pt + final_model.pt + plots + split.json + train.log)
- `outputs/resfold/phase_d_n8600_full_reeval_k40/resfold_s1_8K_20260524_111306/`