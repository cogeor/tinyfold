# Loop 04 TEST — EDM lambda(sigma) loss weighting fix

## Commit gate
- ready: yes
- reason: Three defects fixed; regression + new tests pass; smoke runs do not crash and stay in same OOM as unweighted baseline.

## Verification log

### 1. New unit tests
Command: `.venv/Scripts/python.exe -m pytest tests/test_edm_loss_weight.py -v`
Result: 6 passed.
Tests cover:
- Reference values at (sigma, sigma_data) = (1.0, 1.0), (0.5, 0.5), (0.1, 0.5), (1.0, 0.5), (10.0, 0.5).
- Zero-sigma input does NOT raise (clamp(min=1e-8) guard).
- Shape preservation ([B] -> [B], [B, 1] -> [B, 1]).
- Deprecation alias `af3_loss_weight` emits `DeprecationWarning` and returns identical values.

### 2. Regression suite (prior loops 01-03)
Command: `.venv/Scripts/python.exe -m pytest tests/test_edm_loss_weight.py tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py tests/unit/test_kabsch_rigid.py tests/test_pose_clustering.py tests/test_multisample_eval.py tests/test_kabsch_interp_sampler.py -v`
Result: 30 passed, 1 skipped (opt-in slow integration). Zero regressions from the Kabsch refactor or earlier loops.

### 3. Smoke training with --loss_weighting
Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --config configs/train/resfold/phase_c_n8600.yaml \
    --n_train 64 --n_test 8 --n_eval_train 8 \
    --n_steps 50 --eval_every 50 --batch_size 4 \
    --loss_weighting \
    --output_dir outputs/resfold/loop04_smoke_weighted_v2
```
Result: exit 0, total time 8s, test centroid RMSE 16.68 A at step 50. Did NOT crash on step 1 (defect 2 confirmed fixed).

### 4. Smoke training without --loss_weighting (baseline)
Same command minus the flag.
Result: exit 0, total time 7s, test centroid RMSE 14.86 A at step 50.

### 5. Same OOM check
Weighted/unweighted RMSE ratio = 16.68/14.86 = 1.12. Well within the 3x acceptance band. (50 steps is far below convergence; this is noise around an unconverged loss surface, not signal about EDM weighting quality. The signal will come from Loop 07 Phase D.)

### 6. REGISTRY hygiene
Both smoke rows reverted from `experiments/REGISTRY.md`. Smoke output dirs deleted.

### 7. Phase C checkpoint integrity
Not exercised by this loop. No `--checkpoint` arg used.

## Open concerns
- The EDM weighting bug existed since the flag was added but was never triggered. Anyone reading old commits should NOT assume `--loss_weighting` did anything meaningful before this commit.
- Phase D (Loop 07) is the first run with corrected EDM weighting; its delta vs the unweighted 9.87 A baseline will be the real test of the weighting's value.
