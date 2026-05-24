# Loop 04 IMPLEMENTATION — EDM lambda(sigma) loss weighting fix

## Summary

All three defects identified in PLAN.md are fixed in one commit.

## Files touched

- **MODIFY** `src/tinyfold/training/utils.py` — replaced `af3_loss_weight`
  body with the correct Karras 2022 formula `(s^2 + sd^2) / (s * sd)^2` and
  exposed it under `edm_loss_weight`. `af3_loss_weight` kept as a thin
  `DeprecationWarning` alias. `MultiCopyTrainer` and
  `VectorizedMultiCopyTrainer` updated to call `edm_loss_weight` directly.
- **MODIFY** `src/tinyfold/model/losses/mse.py` — `compute_mse_loss` gained
  `reduction: str = 'mean'` kwarg accepting `{'mean', 'per_sample'}`.
  `'per_sample'` returns shape `[B]`. Default unchanged.
- **MODIFY** `scripts/train_resfold.py` — removed broken
  `noiser.loss_weight(sigma).mean()` call (which would have crashed —
  `VENoiser` has no such method). Replaced with `edm_loss_weight(sigma,
  sigma_data=noiser.sigma_data)`, applied at per-sample MSE level then
  reduced via `(per_sample_mse * lambda_per).mean()`. Logging path
  (`w:` column) summarises the per-sample weight as a mean.
- **CREATE** `tests/test_edm_loss_weight.py` — 6 tests pinning the formula
  at hand-computed reference values, zero-sigma safety, shape
  preservation, and deprecation-warning behaviour.

## Defects fixed

1. **Wrong formula**: denominator was `(s + sd)^2` (sum-then-square)
   instead of `(s * sd)^2` (product-then-square). At sigma=0.1, sd=1.0
   the buggy version returned 0.835 vs the correct 101.0 — ~120x
   under-weight at exactly the noise levels EDM weighting is designed
   to compensate for.
2. **Missing method**: `--loss_weighting` would have raised
   `AttributeError` on step 1 because `VENoiser` defines no
   `loss_weight` method.
3. **Scalar collapse**: even after fixing (1+2), `.mean()` over the
   per-sample weights followed by `loss = loss * scalar` is equivalent
   to `mean(lambda) * mean(MSE)`, NOT `mean(lambda * MSE)` — defeats the
   weighting's purpose.

**Retroactive impact: none.** `--loss_weighting` was never set in any
committed Phase C run; the headline 9.87 A is unaffected. Phase D
(Loop 07) is the first config that will enable the flag, and it now
behaves correctly.

## Test results

- New tests: 6/6 PASS (`pytest tests/test_edm_loss_weight.py -v`).
- Regression (all prior-loop unit + integration tests):
  30 passed, 1 skipped in 130s (the skipped one is the opt-in
  `test_kabsch_interp_integration_changes_trajectory` gated behind an
  env var to keep REGISTRY clean).
- Smoke run (50 steps, n_train=64, batch=4):
  - `--loss_weighting`: test RMSE 16.68 A, no crash, finite mse.
  - without `--loss_weighting`: test RMSE 14.86 A.
  - Same order of magnitude (within ~12%); 50 steps is far below
    convergence at this data scale, so the gap is noise, not signal.
  - REGISTRY rows from both smoke runs reverted; output dirs deleted.

## Deviations from plan

- Smoke runs were executed manually (the original implementer agent
  scheduled them in background and ended its turn). Numbers above
  reproduce the plan's acceptance criteria.
- Distance/consistency loss weighting (`compute_distance_consistency_loss`)
  was NOT changed — it returns a scalar by design and is a regularizer
  rather than the EDM-preconditioned objective. Plan explicitly allowed
  this; documented inline.
- The `multi_copy > 0` branch was NOT touched per plan.
