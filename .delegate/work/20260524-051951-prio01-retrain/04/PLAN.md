# Loop 04: EDM lambda(sigma) loss weighting audit + fix (Task E)

## Overview

Audit and fix the `--loss_weighting` path in `scripts/train_resfold.py` so it implements the canonical EDM/Karras 2022 weighting

```
lambda(sigma) = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
```

multiplied at the **per-sample** MSE level. Phase D (loop 07) turns this flag on, so the path must be correct before retrain.

## Current state of the code (verified in this audit)

Three concrete defects exist today. All three are independently blocking and must be fixed in one commit so the loss surface only shifts once.

### Defect 1 — WRONG formula

`src/tinyfold/training/utils.py:46`

```python
return (sigma**2 + sigma_data**2) / (sigma + sigma_data + 1e-8)**2
```

The denominator is `(sigma + sigma_data)**2` (sum-then-square). The Karras 2022 formula has `(sigma * sigma_data)**2` (product-then-square). The docstring at line 36 even quotes the wrong formula ("(σ + σ_data)²"), so this is a definitional bug, not a typo.

Numerical impact (sigma_data=1.0):
- sigma=0.1 -> current 0.835, correct 101.0   (~120x under-weight)
- sigma=1.0 -> current 0.5,   correct 2.0     (~4x under-weight)
- sigma=10  -> current 0.835, correct 1.01    (~1.2x under-weight)
- Bonus: the current formula is bounded in [0.5, 1.0] (docstring brags about this), which is exactly the wrong property — Karras's whole point is that lambda blows up at small sigma to compensate for the c_out**2 = (sigma*sigma_data)**2 / (sigma**2 + sigma_data**2) factor that suppresses MSE there. The current weighting does almost nothing.

### Defect 2 — AttributeError: noiser has no `loss_weight` method

`scripts/train_resfold.py:1313`

```python
loss_weight = noiser.loss_weight(sigma).mean() if args.loss_weighting else 1.0
```

`VENoiser` (`src/tinyfold/model/diffusion/noise.py:364`) defines no `loss_weight` method. `--loss_weighting` would crash on the first step. Confirmed by `grep "def loss_weight" src/`: zero matches. The correct helper `af3_loss_weight` lives in `src/tinyfold/training/utils.py` and is not imported by `train_resfold.py`. This means `--loss_weighting` has never been exercised in any committed Phase C run, and the Phase C 9.87 A baseline (`configs/train/resfold/phase_c_n8600.yaml`) does NOT set the flag — no retroactive impact, but Phase D would have crashed at step 1.

### Defect 3 — `.mean()` collapses per-sample weighting to a scalar

Same line (`loss_weight = noiser.loss_weight(sigma).mean()`). Even after Defect 2 is fixed, calling `.mean()` over the per-sample weights and multiplying the already-reduced scalar `loss` by it (line 1432: `loss = loss * loss_weight`) is mathematically equivalent to `mean(lambda) * mean(MSE)`, NOT `mean(lambda * MSE)`. Different noise levels do NOT contribute equally to gradient signal in this regime — defeating the entire purpose of EDM weighting.

To weight per-sample we need per-sample MSE. `compute_mse_loss` (`src/tinyfold/model/losses/mse.py:58`) currently returns a scalar (line 105: `loss = per_sample_loss.mean()`). It already computes `per_sample_loss: [B]` internally; we just need an optional `reduction='per_sample'` knob to expose it.

### Notes on what was already CORRECT

- `sigma_data = 1.0` is consistent across the model (`onestep.py:134`, `denoiser.py:295`), the noiser (`noise.py:386`), and every config (`grep sigma_data configs/`). The model's `_edm_coefficients` (`onestep.py:171-187`) uses the same `sd = self.sigma_data = 1.0`. No drift to fix here — just thread `sigma_data` through to the weighting fn so it stays coupled.
- The flag's intent in `LOOPS.yaml` matches Karras 2022; only the implementation drifted.

## Tasks

### Task 1: Fix `af3_loss_weight` formula + add canonical alias

**Goal:** Replace the broken formula with the Karras 2022 closed form, attach a citation + derivation comment, and expose it under a name that matches the math (`edm_loss_weight`). Keep the old name as a thin alias so other callers (if any) don't break — but make it emit a `DeprecationWarning` so we catch stragglers.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/training/utils.py` |

**Steps:**
1. Replace the body of `af3_loss_weight` with the correct formula. Final implementation:
   ```python
   def edm_loss_weight(sigma: Tensor, sigma_data: float = 1.0) -> Tensor:
       """EDM/Karras 2022 per-sample loss weighting.

       Reference: Karras, Aittala, Aila, Laine (2022),
         "Elucidating the Design Space of Diffusion-Based Generative Models",
         NeurIPS 2022, Eq. 7 (the "effective weight" lambda(sigma)).

       Closed form:
           lambda(sigma) = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2

       Derivation: in EDM preconditioning, the model's output is scaled by
       c_out(sigma) = sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)
       (see ResFoldOneStep._edm_coefficients). The training MSE on the raw
       network output F is therefore implicitly multiplied by c_out**2 when
       measured in data space. To make every noise level contribute equally
       to gradient signal we multiply the data-space MSE by 1 / c_out**2,
       which is exactly the lambda above.

       Asymptotics (sigma_data = 1):
           sigma -> 0   : lambda -> 1 / sigma**2   (blows up; rescues small-sigma
                                                    samples that c_out squashes)
           sigma -> inf : lambda -> 1               (high-sigma samples already
                                                    have full gradient magnitude)
           sigma = 1    : lambda = 2

       Args:
           sigma:      [B] per-sample noise level (sigma, not log-sigma).
           sigma_data: float, must match the model's sigma_data (1.0 throughout
                       tinyfold; see onestep.py:134, denoiser.py:295).

       Returns:
           weight: [B] per-sample loss weights. MUST be multiplied at the
                   per-sample MSE level, NOT after reducing across the batch.
       """
       # Tiny epsilon on sigma only (sigma_data is a known positive constant);
       # protects against the schedule occasionally returning sigma == 0.
       sigma_safe = sigma.clamp(min=1e-8)
       return (sigma_safe ** 2 + sigma_data ** 2) / (sigma_safe * sigma_data) ** 2
   ```
2. Add a backwards-compatible alias:
   ```python
   def af3_loss_weight(sigma: Tensor, sigma_data: float = 1.0) -> Tensor:
       """Deprecated alias for edm_loss_weight (kept for old call sites)."""
       import warnings
       warnings.warn(
           "af3_loss_weight is deprecated; use edm_loss_weight (same formula, "
           "corrected from the buggy (sigma+sigma_data)**2 denominator).",
           DeprecationWarning,
           stacklevel=2,
       )
       return edm_loss_weight(sigma, sigma_data)
   ```
3. `MultiCopyTrainer.train_step` (line 162) and `VectorizedMultiCopyTrainer.train_step` (line 287) both call `af3_loss_weight(sigma)`. Update both to call `edm_loss_weight(sigma)` directly so the alias never fires in production code (the alias only catches external/notebook callers).

**Verify:** `python -c "from tinyfold.training.utils import edm_loss_weight; import torch; print(edm_loss_weight(torch.tensor([0.1, 1.0, 10.0])))"` prints approximately `[101.0, 2.0, 1.01]`.

### Task 2: Expose per-sample MSE and wire correct per-sample weighting into the trainer

**Goal:** Make `compute_mse_loss` return per-sample loss when requested, then replace the broken `noiser.loss_weight(sigma).mean()` call in `train_resfold.py` with `mean(lambda(sigma) * per_sample_MSE)`.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/model/losses/mse.py` |
| MODIFY | `scripts/train_resfold.py` |

**Steps:**
1. In `compute_mse_loss` (`src/tinyfold/model/losses/mse.py:58`):
   - Add a `reduction: str = 'mean'` kwarg accepting `{'mean', 'per_sample'}`.
   - When `reduction == 'per_sample'`, return the existing `per_sample_loss` tensor (shape `[B]`) instead of `per_sample_loss.mean()`. For the no-mask branch, average over `(N, 3)` per sample first (`sq_diff.mean(dim=1)`) and return `[B]`.
   - Default stays `'mean'` so every existing call site is unchanged.
2. In `scripts/train_resfold.py`:
   - Add a module-level import: `from tinyfold.training.utils import edm_loss_weight`.
   - Delete line 1313 (`loss_weight = noiser.loss_weight(sigma).mean() if args.loss_weighting else 1.0`). Replace the continuous-sigma branch (lines 1291-1313) so `loss_weight` becomes a `[B]` tensor (or `None` when the flag is off):
     ```python
     # EDM/Karras 2022 per-sample loss weighting (Eq. 7).
     # See tinyfold/training/utils.py::edm_loss_weight for derivation.
     # MUST be applied at per-sample MSE level, not after batch reduction.
     loss_weight = (
         edm_loss_weight(sigma, sigma_data=noiser.sigma_data)
         if args.loss_weighting else None
     )
     ```
   - At the loss-computation site for `continuous_sigma + not multi_copy` (around line 1425):
     - Call `compute_mse_loss(..., reduction='per_sample' if loss_weight is not None else 'mean')`.
     - When `loss_weight is not None`: `loss_mse = (loss_mse_per_sample * loss_weight).mean()`.
     - Same treatment for `compute_distance_consistency_loss` IFF it also returns per-sample (check; if it only returns a scalar, leave it unweighted and add a one-line comment explaining why — dist loss is a regularizer, not the EDM-preconditioned objective, so unweighted is defensible).
     - Replace line 1432 (`loss = loss * loss_weight`) entirely; weighting is now folded into `loss_mse` directly.
   - Logging at line 1487 currently does `loss_weight if isinstance(loss_weight, float) else loss_weight.item()`. Replace with `loss_weight.mean().item() if loss_weight is not None else 1.0` so the printed `w:` column is still a scalar summary.
   - Logging at line 1647 (`weight_str` formatting) needs no change beyond reading from the same dict key.
3. **Do NOT touch the multi_copy branch** (lines 1327-1383) — it doesn't use `--loss_weighting` today and Phase D doesn't enable `multi_copy` either (`phase_c_n8600.yaml:38 multi_copy: 0`, will carry through to phase_d). Leave a TODO comment if a future loop wants to add it; out of scope here.

**Verify:** `python -c "from tinyfold.model.losses.mse import compute_mse_loss; import torch; p=torch.randn(4,8,3); t=torch.randn(4,8,3); print(compute_mse_loss(p,t,reduction='per_sample').shape)"` prints `torch.Size([4])`.

### Task 3: Pin the formula with a unit test

**Goal:** Hand-computed reference values from the closed form, asserted to 1e-6. Catches both the Defect-1 regression (denominator sum-vs-product) and any future drift.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `tests/test_edm_loss_weight.py` |

**Steps:**
1. Test cases (use `pytest.approx(rel=1e-6)` or `torch.testing.assert_close`):
   - `edm_loss_weight(tensor([1.0]), sigma_data=1.0)` -> `2.0`.
   - `edm_loss_weight(tensor([0.5]), sigma_data=0.5)` -> `(0.25 + 0.25) / (0.5 * 0.5)**2 = 0.5 / 0.0625 = 8.0`.
   - Three-point reference at `sigma=[0.1, 1.0, 10.0]`, `sigma_data=0.5`:
     - 0.1: `(0.01 + 0.25) / (0.05)**2 = 0.26 / 0.0025 = 104.0`
     - 1.0: `(1.0 + 0.25) / (0.5)**2 = 1.25 / 0.25 = 5.0`
     - 10.0: `(100.0 + 0.25) / (5.0)**2 = 100.25 / 25.0 = 4.01`
   - Edge case: `edm_loss_weight(tensor([0.0]), sigma_data=1.0)` must NOT raise (the `clamp(min=1e-8)` guard kicks in). Assert the returned value is finite and large.
   - Shape preservation: input `[B]` returns `[B]`; input `[B, 1]` returns `[B, 1]`.
   - Deprecation alias: `af3_loss_weight(tensor([1.0]))` emits a `DeprecationWarning` (use `pytest.warns(DeprecationWarning)`) and returns the same value as `edm_loss_weight`.
2. The test should `from tinyfold.training.utils import edm_loss_weight, af3_loss_weight`.

**Verify:** `pytest tests/test_edm_loss_weight.py -v` — six tests pass.

### Task 4: 50-step training smoke run

**Goal:** Confirm `--loss_weighting` no longer crashes (Defect 2) and that the loss magnitude is in the same order as without weighting (no 100x regression that would indicate we accidentally inverted lambda or something).

**Files:** none — runtime check only.

**Steps:**
1. Run, capture last-step loss:
   ```powershell
   python scripts/train_resfold.py `
     --config configs/train/resfold/phase_c_n8600.yaml `
     --n_train 64 --n_test 8 --n_eval_train 8 `
     --n_steps 50 --eval_every 50 --batch_size 4 `
     --loss_weighting `
     --output_dir outputs/resfold/loop04_smoke_weighted
   ```
2. Run the same command without `--loss_weighting`, output dir `loop04_smoke_unweighted`.
3. Compare final-step `mse` field (NOT total `loss` — total includes the lambda factor by construction). Acceptance: `mse_weighted` is within ~3x of `mse_unweighted` in either direction. Order-of-magnitude is what we care about; exact equality is not expected because the optimizer is taking different steps once lambda is applied.
4. Verify the printed `w:` column in the weighted run shows values varying step-to-step in a plausible range (mean weight per batch should typically be O(1)-O(100) given the AF3 sigma sampling distribution — `sigma = exp(-1.2 + 1.5 * N(0,1))`).
5. If `mse_weighted` is >100x `mse_unweighted` or NaN: STOP, do not commit; revisit Task 2 wiring. Likely cause: weights applied additionally on top of pre-reduced loss (double counting), or per-sample dim mismatch causing broadcasting into `[B, B]`.

**Verify:** Smoke complete, no crash, mse ratio within tolerance, REGISTRY untouched (this is a throwaway run; delete `outputs/resfold/loop04_smoke_*` after verification).

## Acceptance Criteria

- [ ] `edm_loss_weight` exists in `src/tinyfold/training/utils.py` with the correct Karras 2022 formula and a docstring containing the citation + derivation comment.
- [ ] `af3_loss_weight` is preserved as a thin alias emitting `DeprecationWarning`, returns identical values to `edm_loss_weight`.
- [ ] `scripts/train_resfold.py` imports `edm_loss_weight` directly; the broken `noiser.loss_weight(sigma).mean()` call is gone.
- [ ] `compute_mse_loss` accepts `reduction='per_sample'` returning shape `[B]`; default `'mean'` behavior unchanged.
- [ ] Per-sample weighting is applied at per-sample MSE level, then reduced (`(per_sample_mse * weight).mean()`), NOT scalar-multiplied after reduction.
- [ ] `tests/test_edm_loss_weight.py` passes with the six cases listed in Task 3.
- [ ] 50-step smoke run with `--loss_weighting` completes without crash; `mse` field stays within ~3x of the unweighted baseline on the same data subset.
- [ ] Commit message documents the formula bug AND the missing-method crash, and notes that no committed Phase C run was affected (the flag was never enabled in Phase C configs).

## Out of scope

- The sigma sampling distribution itself (`sample_sigma_af3`, `sample_sigma_stratified` in `noise.py`) — not part of Task E.
- Weighting the dist/contact/atom auxiliary losses by lambda(sigma) — these are regularizers, not the EDM-preconditioned objective; weighting them changes the loss landscape in ways outside this audit's scope.
- The `multi_copy > 0` branch in the trainer (lines 1327-1383). Phase D config does not enable it; revisit if a future loop does.
- Any change to `ResFoldOneStep._edm_coefficients` — `c_skip`, `c_out`, `c_in`, `c_noise` are correct against Karras 2022.
