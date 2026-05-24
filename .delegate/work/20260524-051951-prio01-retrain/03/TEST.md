# Test Results — Loop 03 (Boltz Kabsch-interpolation sampler)

Tested: 2026-05-24
Status: PASS (code correct; experimental outcome negative — see Open Observations)

## Task Verification

- [x] Task 1 (kabsch_rigid primitive): `src/tinyfold/model/geometry/kabsch.py`
      exists with `kabsch_rigid(src, tgt, mask) -> (R, t, aligned)`;
      `losses/mse.py:kabsch_align` and `diffusion/utils.py:kabsch_align_to_target`
      now delegate to it; `compute_c_rmsd` left as-is per PLAN D1 (Loop 01
      tests still pass).
- [x] Task 2 (`kabsch_interp` in `sample_centroids_ve`): kwarg added; `x_prev`
      snapshot taken conditionally (default path has zero overhead); Kabsch
      align applied AFTER `x = x + d*dt` and BEFORE the `recenter` block, as
      specified in PLAN D2.
- [x] Task 3 (CLI plumbing): `--kabsch_interp` flag added; forwarded to all
      four `sample_centroids_ve` call sites and through `sample_k_centroids`;
      train.log records the flag when set ("kabsch_interp: True (Boltz
      trajectory-frame alignment)" appeared in both Row A and Row B logs).
- [x] Task 4 (unit + integration tests): `tests/unit/test_kabsch_rigid.py`
      (5 tests) + `tests/test_kabsch_interp_sampler.py` (3 tests, 1 opt-in
      slow) all pass.
- [x] Task 5 prep (multistep config): `configs/train/resfold/phase_c_n8600_multistep.yaml`
      created with `one_shot_sample: false` override. Used by both Row A and
      Row B re-eval runs (this loop's tester executes the re-evals).

## Acceptance Criteria

- [x] `src/tinyfold/model/geometry/kabsch.py` exports `kabsch_rigid(src, tgt, mask) -> (R, t, aligned)`.
- [x] `kabsch_align` and `kabsch_align_to_target` delegate to `kabsch_rigid`.
      `compute_c_rmsd` unchanged per PLAN D1 (its `R`/`t` split is preserved;
      `tests/unit/test_c_rmsd.py` 3/3 PASS in this run).
- [x] `sample_centroids_ve` accepts `kabsch_interp: bool = False`; when True,
      freshly-stepped `x` rigid-aligned onto `x_prev` BEFORE next iteration.
- [x] `--kabsch_interp` CLI flag added; threaded through 4 call sites and
      through `sample_k_centroids`.
- [x] Default-off behaviour: smoke training run produced standard eval line
      with NO `kabsch_interp` token (no behavioural change to default path).
- [x] `pytest tests/unit/test_kabsch_rigid.py tests/test_kabsch_interp_sampler.py -v`
      passes locally — 7 passed, 1 skipped (opt-in slow integration).
- [x] Loop 01 + Loop 02 regression suite green
      (`test_c_rmsd.py`, `test_registry_append.py`, `test_pose_clustering.py`,
      `test_multisample_eval.py`) — **17/17 PASS**.
- [x] Phase C re-eval Row A appended to `experiments/REGISTRY.md` →
      `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k1/resfold_s1_8K_20260524_064604/`.
- [x] Phase C re-eval Row B appended to `experiments/REGISTRY.md` →
      `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k5/resfold_s1_8K_20260524_065001/`.
- [ ] Task 6 — `notes/post_compact_spec.md` updated with Row A outcome.
      DEFERRED — per loop assignment, Task 6 documentation update belongs to
      Loop 07 / final retrain decision. Open observation below carries the
      verdict forward.

## Build & Tests

| Suite | Command | Result |
|---|---|---|
| New unit tests | `pytest tests/unit/test_kabsch_rigid.py tests/test_kabsch_interp_sampler.py -v` | **7 passed, 1 skipped** |
| Regression (Loop 01 + 02) | `pytest tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py tests/test_pose_clustering.py tests/test_multisample_eval.py -v` | **17 passed** |
| Smoke training | `train_resfold.py --config phase_b_n4.yaml --n_steps 10 --eval_every 5` | exit 0; eval line printed; no `kabsch_interp` token (default-off OK) |
| Row A (K=1 + kabsch_interp) | full command per PLAN | exit 0; REGISTRY appended; **11.9799 A** centroid RMSE |
| Row B (K=5 + kabsch_interp + cluster-rank) | full command per PLAN | exit 0; REGISTRY appended; centroid **11.8176 A**, oracle@5 **10.8476 A**, mean@5 12.4800 A, ranked@5 11.8116 A |

### Row A vs baselines (the critical compare)

| Variant | Centroid RMSE |
|---|---|
| One-shot baseline (Phase C, Loop 01) | **10.59 A** |
| Multi-step T=50 no-kabsch (Loop 01 reeval, `phase_c_n8600_reeval_loop01`) | **9.56 A** |
| **Row A — multi-step T=50 + kabsch_interp (this loop)** | **11.9799 A** ← WORSE |

### Row B vs baselines

| Variant | oracle@5 | mean@5 | ranked@5 |
|---|---|---|---|
| Loop 02 K=5 no-kabsch (`phase_c_n8600_reeval_loop02_k5`) | 8.905 A | 10.349 A | 9.895 A |
| **Row B — K=5 + kabsch_interp (this loop)** | **10.8476 A** | 12.4800 A | 11.8116 A |

All three K=5 metrics regress when `kabsch_interp` is enabled.

## REGISTRY hygiene

- Pre-loop count: 48 lines.
- Post-loop count: 50 lines. **Exactly 2 new rows** (Row A + Row B).
- Note: an intermediate stray pytest-tmp row (from
  `test_eval_only_k5_integration`) and an intermediate smoke-run row
  (`resfold_s1_4_20260524_064456`) appeared during this tester pass and were
  removed before the Row A / Row B runs. Final REGISTRY tail is exactly
  Loop 03's two intended rows.
- Phase C source checkpoint
  `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt`
  mtime unchanged at `May 24 04:14`. Source untouched.

## Open observations (do not gate)

- **Boltz trick HURTS, both at K=1 and K=5.** Row A's 11.98 A is worse than
  the 10.59 one-shot baseline AND worse than the 9.56 plain multi-step
  baseline (delta of +2.42 A vs the most directly comparable multi-step
  baseline). Row B's oracle@5 of 10.85 A is +1.94 A worse than the Loop 02
  K=5 oracle@5 of 8.905 A.
- **Phase D recommendation: drop `--kabsch_interp` entirely.** The current
  Phase C config already uses `one_shot_sample: true`, which avoids the
  multi-step drift in the first place, and adding kabsch_interp on top of
  multi-step makes the drift worse rather than better.
- **Suggested doc update for Loop 07** (NOT done by this tester per loop
  assignment): append to `notes/post_compact_spec.md` under the "what didn't
  work" / C.3 ablation block:
  > Boltz Kabsch-interpolation sampler (`kabsch_interp=True`). Tested
  > Loop 03; K=1 centroid 11.98 A vs one-shot 10.59 A (delta +1.39 A) and
  > vs plain multi-step 9.56 A (delta +2.42 A). K=5 oracle@5 10.85 A vs
  > no-kabsch 8.91 A (delta +1.94 A). Phase D ships with
  > `one_shot_sample=true` and without kabsch_interp. See REGISTRY
  > 2026-05-24 rows resfold_s1_8K_20260524_064604 and
  > resfold_s1_8K_20260524_065001.
- The new `kabsch_rigid` primitive remains useful for code hygiene
  regardless of the experimental outcome — keep the refactor.

## Commit Gate

ready: yes
reason: Code is correct (all 24 unit/regression tests pass, default-off byte-identical, both re-eval runs exit 0 and append REGISTRY rows cleanly). Headline experiment (Row A 11.98 A vs 10.59 baseline) shows the Boltz trick hurts, but per loop assignment that's a valid experimental outcome to document — the gate is on code correctness, not on the kabsch_interp helping. Phase D / Loop 07 should consume this verdict and drop the flag.
commit-message: feat(resfold): add Boltz Kabsch-interpolation sampler (kabsch_interp) and shared kabsch_rigid primitive
