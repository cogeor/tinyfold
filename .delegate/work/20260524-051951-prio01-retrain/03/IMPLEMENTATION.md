# Implementation Log — Loop 03 (Boltz Kabsch-interpolation sampler)

Tasks 1-4 (CODE) + Task 5 prep (multistep config) complete. Tasks 5 (re-eval
commands) and 6 (notes update) deferred to the tester per loop assignment.

---

## Task 1: Factor `kabsch_rigid` primitive

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/geometry/__init__.py` (CREATE) — exports `kabsch_rigid`.
- `src/tinyfold/model/geometry/kabsch.py` (CREATE) — single shared primitive
  `kabsch_rigid(src, tgt, mask=None) -> (R, t, aligned)`. Implements the
  standard Kabsch SVD recipe with proper-rotation guard (sign-flip-D trick)
  and mask-aware centroids. The returned `aligned` is in target's translated
  frame (callers that want the origin-centred convention subtract `tgt_mean`
  themselves).
- `src/tinyfold/model/losses/mse.py` (MODIFY) — `kabsch_align` now delegates
  to `kabsch_rigid` and subtracts `target_mean` from the helper's output to
  preserve the historical `(pred_aligned, target_centered)` contract.
  `compute_c_rmsd` left unchanged (explicitly needs split `R` and `t`; PLAN
  D1 says leave-as-is is fine).
- `src/tinyfold/model/diffusion/utils.py` (MODIFY) —
  `kabsch_align_to_target` is now a one-line adapter over `kabsch_rigid`.

### Verification

- [x] `pytest tests/unit/test_losses.py tests/unit/test_c_rmsd.py
      tests/unit/test_geometry.py -v` — Kabsch-related tests all PASS.
      Two pre-existing failures (`test_all_geometry_imports` —
      `BOND_LENGTHS` re-export missing; `test_random_vs_gt_comparison` —
      flaky numerical comparison) confirmed pre-existing via `git stash`
      bisect; both unrelated to the Kabsch refactor.
- [x] `kabsch_align`'s `(pred_aligned, target_centered)` contract preserved:
      `TestKabschAlign::test_identity_alignment`, `test_rotation_invariance`,
      `test_with_mask` all PASS.
- [x] `compute_rmse` (which calls `kabsch_align`) still produces sane values:
      `TestRMSE::test_rmse_with_random_noise` PASS.
- [x] `compute_c_rmsd` still PASSES — its 3-test suite is green.

---

## Task 2: `kabsch_interp` in `sample_centroids_ve`

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py` (MODIFY) — added
  `kabsch_interp: bool = False` kwarg to `sample_centroids_ve` (after
  `recenter`, before `self_cond`). Inside the Euler loop:
  - Snapshot `x_prev = x.clone()` at the top of each iteration (only when
    `kabsch_interp=True` — saves the clone on the default path).
  - After `x = x + d * dt` and BEFORE the optional `recenter` block, call
    `_, _, x = kabsch_rigid(x, x_prev, mask)` to rigid-align the freshly-
    stepped `x` onto the pre-step frame.
- Added the local import `from tinyfold.model.geometry import kabsch_rigid`
  to the top-of-file import block (alongside the existing
  `kabsch_align_to_target` import).
- Added a comment near the post-loop atom-head call (PLAN D5) documenting
  that `x` is the kabsch-aligned terminal value and atoms automatically
  follow the centroid frame via the head's `[L, 4, 3]` per-residue offsets
  — no extra Kabsch on atoms needed.
- Added `kabsch_interp: bool = False` kwarg to `sample_k_centroids` and
  forwarded it through to `sample_centroids_ve` on the multi-step VE
  path. (The `is_onestep and one_shot` fast path doesn't use the Euler
  loop, so the kwarg has no effect there — documented in the docstring.)

### Verification

- [x] Default-off byte-identical behaviour: with `kabsch_interp=False`,
      `x_prev` is None and the kabsch_interp branch never runs — the loop
      is identical to pre-refactor.
- [x] Unit tests (Task 4) confirm:
  - Identity-denoiser invariance: `kabsch_interp=True` matches
    `kabsch_interp=False` to 1e-3 when the denoiser is a fixed point.
  - Rotation-denoiser divergence: `kabsch_interp=True` produces a
    measurably different trajectory than `kabsch_interp=False` (flag is
    actually reaching the sampler).
- [x] Atom head audit (PLAN D5): confirmed by reading
      `scripts/train_resfold.py:273-282` that the `if is_onestep:` block
      uses the post-loop `x` directly. With kabsch_interp, that `x` is
      already aligned. Atom head consumes the centroid prediction and
      emits per-residue offsets in the centroid frame — atoms follow
      automatically. Comment added in source for next reader.

---

## Task 3: Plumb `--kabsch_interp` through the eval CLI

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py` (MODIFY): added `--kabsch_interp` flag in
  `parse_args` next to `--align_per_step` / `--recenter`. Threaded through
  ALL four `sample_centroids_ve` call sites:
  1. `_run_test_eval` multi-sample path (via `sample_k_centroids`).
  2. `_run_test_eval` single-sample path.
  3. In-training eval body.
  4. Plot-path eval.
- Added a one-line log at the top of `_run_training` so the train.log
  records `kabsch_interp: True (Boltz trajectory-frame alignment)`
  when the flag is set.

### Verification

- [x] `python scripts/train_resfold.py --help | grep kabsch_interp` shows
      the new flag with help text.
- [x] CLI flag overrides YAML default (argparse `set_defaults(**filtered)`
      runs FIRST, so a `--kabsch_interp` on the command line wins). The
      `phase_c_n8600.yaml` config doesn't set `kabsch_interp` at all, so
      the default is `False` and `--kabsch_interp` flips it on cleanly.
- [x] All four call sites verified by grepping `kabsch_interp=args` in
      `scripts/train_resfold.py` — four hits, one per site.

---

## Task 4: Unit + integration tests

Completed: 2026-05-24

### Changes

- `tests/unit/test_kabsch_rigid.py` (CREATE) — 5 unit tests for the shared
  primitive:
  - `test_proper_rotation_recovered` — `R` is orthonormal with det=+1,
    `aligned` reproduces `tgt`.
  - `test_identity_when_src_equals_tgt` — `R = I`, `t = 0`, `aligned = src`.
  - `test_mask_ones_matches_nomask` — passing `mask=ones` is equivalent to
    `mask=None`.
  - `test_partial_mask_aligns_unmasked_subset` — masked positions are
    correctly excluded from the SVD.
  - `test_batch_independence` — independent `(R, t)` per batch element.
- `tests/test_kabsch_interp_sampler.py` (CREATE) — 2 unit + 1 integration:
  - `test_kabsch_interp_identity_denoiser_invariance` — with a fixed-point
    stub denoiser, `kabsch_interp=True` matches `=False` to 1e-3 (no
    spurious drift). Uses a small `sigma_max=1.0` schedule so the inner
    `clamp(x0_pred, -3, 3)` doesn't kick in.
  - `test_kabsch_interp_rotation_denoiser_changes_trajectory` — with a
    centroid-preserving rotation stub, `kabsch_interp=True` produces a
    measurably different terminal `x` than `=False`. Validates the flag
    actually reaches the sampler.
  - `test_kabsch_interp_integration_changes_trajectory`
    (`@pytest.mark.slow`, opt-in via
    `TINYFOLD_RUN_KABSCH_INTEGRATION=1`) — runs the full
    `train_resfold.py --eval_only --kabsch_interp` subprocess against the
    Phase C checkpoint on 4 targets vs without, asserts RMSEs differ.
    Verified manually (passed: no-flag = 7.4165 A, with-flag = 7.5048 A on
    4 targets) then opt-in-gated to avoid REGISTRY.md pollution during
    routine `pytest tests/` runs. Skips cleanly if the Phase C checkpoint
    or the multistep config is missing.

### Verification

- [x] `pytest tests/unit/test_kabsch_rigid.py
      tests/test_kabsch_interp_sampler.py -v` — **7 passed, 1 skipped**
      (the slow integration test is opt-in via env var).
- [x] Integration test (executed manually with checkpoint present):
      no-flag RMSE = 7.4165 A, with-flag RMSE = 7.5048 A on 4 Phase C test
      targets. Trajectory measurably differs (|delta| = 0.088 A >> 1e-3).
- [x] Full `pytest tests/ -v -m "not slow"` — **168 passed, 13 failed,
      2 deselected** (the 13 failures all pre-exist this loop; confirmed
      via `git stash` bisect). Net +7 new passing tests, zero regressions.

---

## Task 5 prep (config only — re-eval commands deferred to tester)

Completed: 2026-05-24

### Changes

- `configs/train/resfold/phase_c_n8600_multistep.yaml` (CREATE) — copy of
  `phase_c_n8600.yaml` with `one_shot_sample: false` flipped. Required
  because the Phase C config's `one_shot_sample: true` short-circuits
  `sample_centroids_ve` entirely (PLAN Task 5 "Notes for the runner"); the
  multistep override is the only YAML diff needed to make `--kabsch_interp`
  observable.
- `output_dir` set to `outputs/resfold/phase_c_n8600_multistep` so it
  doesn't collide with the source checkpoint directory.

### Verification

- [x] Config parses cleanly and the integration test (which uses this
      config) PASSES end-to-end when run manually.
- [x] Source checkpoint at
      `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt`
      remains untouched (multistep config writes to its own output_dir).

---

## Tester handoff: what's left

The TESTER for this loop must run:

1. **Row A** (HEADLINE — does Boltz fix multi-step?):
   ```powershell
   python scripts/train_resfold.py `
       --config configs/train/resfold/phase_c_n8600_multistep.yaml `
       --eval_only `
       --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
       --kabsch_interp `
       --output_dir outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k1
   ```
2. **Row B** (K=5 cluster-then-rank):
   ```powershell
   python scripts/train_resfold.py `
       --config configs/train/resfold/phase_c_n8600_multistep.yaml `
       --eval_only `
       --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
       --kabsch_interp `
       --n_samples 5 `
       --eval_K_list 1,5 `
       --output_dir outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k5
   ```
3. Task 6: append the Row A outcome to `notes/post_compact_spec.md` C.3
   ablation block + the appropriate "what worked" / "what didn't work"
   section per PLAN D7's three-case decision tree.

Both commands use `phase_c_n8600_multistep.yaml` (created in this loop)
to force the multi-step VE sampler. Both write to fresh timestamped
subdirs and do NOT touch the Phase C source checkpoint dir.

---

## Smoke training run (loop-level gate)

Ran `python scripts/train_resfold.py --config configs/train/resfold/phase_b_n4.yaml --n_steps 10 --eval_every 5 --output_dir outputs/_loop03_smoke`:

```
Training for 10 steps...
  >>> Train Centroid RMSE (4): 13.6778 A | Test Centroid RMSE (14): 14.3330 A | DockQ: 0.0831 (succ 14.3%) | Atom RMSE: 16.0628 | C-RMSD: 20.4016 A
  >>> Train Centroid RMSE (4): 12.6303 A | Test Centroid RMSE (14): 14.1151 A | DockQ: 0.0634 (succ 7.1%) | Atom RMSE: 15.5144 | C-RMSD: 18.9999 A
Training complete
  Total time: 2s (0.0 min)
  Best test RMSE: 14.1151 A
```

End-to-end training + eval + plot + registry-write pipeline survives the
refactor. (Smoke output dir + registry pollution cleaned up post-run.)

---

## Files touched (absolute paths)

CREATE:
- `C:\Users\costa\src\tinyfold\src\tinyfold\model\geometry\__init__.py`
- `C:\Users\costa\src\tinyfold\src\tinyfold\model\geometry\kabsch.py`
- `C:\Users\costa\src\tinyfold\tests\unit\test_kabsch_rigid.py`
- `C:\Users\costa\src\tinyfold\tests\test_kabsch_interp_sampler.py`
- `C:\Users\costa\src\tinyfold\configs\train\resfold\phase_c_n8600_multistep.yaml`

MODIFY:
- `C:\Users\costa\src\tinyfold\src\tinyfold\model\losses\mse.py`
- `C:\Users\costa\src\tinyfold\src\tinyfold\model\diffusion\utils.py`
- `C:\Users\costa\src\tinyfold\scripts\train_resfold.py`
