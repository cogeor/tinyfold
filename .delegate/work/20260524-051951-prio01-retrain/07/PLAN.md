# Loop 07 PLAN — Phase D full retrain + REGISTRY + spec update

## Goal

Stack Loops 01-06 in a single Phase D config, train on the same N=8600
filter the Phase C 9.87 A baseline used, re-eval at K=5 and K=40, and
write the headline number into `notes/phase_d_results.md`.

## Files touched

- CREATE `configs/train/resfold/phase_d_n8600_full.yaml` — the stack
  config.
- CREATE `notes/phase_d_results.md` — durable summary of the run +
  comparison table.
- APPEND `experiments/REGISTRY.md` — 2 new rows (training run +
  K=40 re-eval).

Loop 03 finding (Kabsch interp HURTS) was already documented in its
commit and is preserved by Phase D leaving `one_shot_sample: true`.

## Tasks

### Task 1: Phase D config

Single new YAML mirroring `phase_c_n8600.yaml`'s data/architecture
fields and adding:

```yaml
aa_embed: esm2_35M
confidence_head: true
confidence_head_weight: 0.1
rank_by: confidence
loss_weighting: true
n_samples: 5
eval_K_list: "1,5"
cluster_radius: 5.0
one_shot_sample: true
```

### Task 2: Launch retrain

`python scripts/train_resfold.py --config configs/train/resfold/phase_d_n8600_full.yaml`
- Expected wall time: ~135 min (4070 Ti SUPER).
- Auto-appends a REGISTRY row on completion via the existing finally:
  block.

### Task 3: K=40 re-eval

`python scripts/train_resfold.py --config configs/train/resfold/phase_d_n8600_full.yaml --eval_only --checkpoint outputs/resfold/phase_d_n8600_full/.../best_model.pt --n_samples 40 --eval_K_list 1,5,40 --output_dir outputs/resfold/phase_d_n8600_full_reeval_k40`
- Expected wall time: ~5 min.
- Adds the second REGISTRY row with oracle@40 / mean@40 / ranked@40.

### Task 4: Spec update

Write `notes/phase_d_results.md` with the comparison table vs Phase C
9.87 baseline, what worked, what didn't, updated SOTA placement, and
next steps.

### Task 5: Cleanup + commit

- Delete the failed earlier launch dir (cache wasn't fully populated).
- Strip any stray REGISTRY rows from pytest tmpdir runs.
- Commit.

## Acceptance

- Phase D Test RMSE < 9.87 A (Phase C baseline). **Achieved: 9.02 A**.
- Phase D oracle@40 < Phase C oracle@40 = 8.46. **Achieved: 7.53 A.**
- All Loop 01-06 features present in the Phase D log
  (DockQ, C-RMSD, oracle@K, mean@K, ranked@K, ranked_conf@K, Spearman).
- Spec file written with honest summary of gains AND limitations.

## Run timeline (executed)

- 2026-05-24 08:53: Phase D launched (after one false start when ESM cache
  was incomplete; cache resumed to 28352 files, run restarted clean).
- 2026-05-24 11:12: Phase D training complete (135 min wall time).
- 2026-05-24 11:13: K=40 re-eval launched.
- 2026-05-24 11:17: K=40 re-eval complete (4 min).
- 2026-05-24 11:25: Spec written + this PLAN finalised.
