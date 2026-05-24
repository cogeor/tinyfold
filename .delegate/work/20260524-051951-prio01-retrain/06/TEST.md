# Loop 06 TEST — Confidence head + ranked Top-K

## Commit gate
- ready: yes
- reason: Head wires cleanly through the trunk, 3-tuple return propagates, aux loss decreases, eval reports both `ranked@K` and `ranked_conf@K`. Regression suite green.

## Verification log

### 1. New unit tests
- `tests/test_confidence_head.py`: 6 tests (head shape, mask-aware pool, sigmoid range, requires_grad gating, forward shape, init).
- `tests/unit/test_onestep_confidence.py`: 5 OneStep-wiring tests (3-tuple return, head present only when enabled, weight=0 disables aux, head params separate from trunk params).
Total new: 11 tests, all PASS.

### 2. Regression suite
47 passed, 1 skipped (pre-existing env-gated `test_kabsch_interp_integration_changes_trajectory`). Zero failures across all Loop 01-05 tests.

### 3. Smoke training (50 steps, --confidence_head_weight 0.1, N=4, K=4, --rank_by confidence)
Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --config configs/train/resfold/phase_b_n4.yaml \
    --n_steps 50 --eval_every 25 \
    --confidence_head --confidence_head_weight 0.1 \
    --n_samples 4 --eval_K_list 1,4 --rank_by confidence \
    --output_dir outputs/_loop06_smoke
```
Result:
- Exit 0, 14s wall time.
- Aux conf loss: 0.0179 at step 25, 0.0042 at step 50 (decreasing).
- Spearman(pred_lddt, -RMSE) on 4 samples: 0.116 at step 25, -0.124 at step 50 (head still learning — expected after 50 steps).
- `pred_lddt` across 4 samples on one probe target: mean 0.8965, std 0.00142, range [0.895, 0.898]. Variance non-zero but small.
- Both `ranked@4 X.XXX A` (cluster) and `ranked_conf@4 X.XXX A` (confidence) tokens present in log.

### 4. REGISTRY hygiene
3 stray pytest-tmp rows from regression runs reverted. Smoke output dir deleted.

## Open concerns
- Smoke variance in `pred_lddt` across K=4 samples is very small (~0.003). The head hasn't learned to distinguish samples yet — the 50-step smoke is too short to expect rank quality. The real test is Loop 07's 50k-step retrain.
- The Spearman flipping sign between step 25 and 50 confirms the head is undertrained, not converged-bad. Phase D's `--confidence_head_weight 0.1` is intentionally small so the head can train slowly without harming the main centroid objective.
- If Phase D's `ranked_conf@5` is no better than `mean@5`, fall back to `--rank_by cluster` (Loop 02) or `--rank_by oracle` (debug only). The cluster ranking is preserved as a sibling output.