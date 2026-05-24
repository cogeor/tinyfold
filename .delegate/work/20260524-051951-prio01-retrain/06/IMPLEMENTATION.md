# Implementation Log — Loop 06: Confidence head + ranked Top-K

## Task 1: `compute_lddt` per-sample reduction

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/losses/lddt.py`: added `reduction: str = "mean"` arg to
  `compute_lddt`. `reduction="per_sample"` returns the `[B]` tensor (one score
  per target). Default behaviour preserved (scalar mean). Invalid reduction
  names raise `ValueError`.
- `tests/unit/test_losses.py`: extended `TestLDDT` with
  `test_compute_lddt_per_sample` (identical sample -> ~1.0, perturbed sample
  -> ~0.0, diff >= 0.9, scalar mean equals per-sample mean) and
  `test_compute_lddt_invalid_reduction` (bad name raises).

### Verification

- [x] `pytest tests/unit/test_losses.py -k lddt -v` -> 6 passed (5 existing + 1 new pair).
- [x] Scalar contract preserved: `compute_lddt(x, y, reduction='mean')`
  returns a 0-d tensor; default unchanged.

---

## Task 2: `ConfidenceHead` module

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/resfold/confidence_head.py` (CREATE):
  `ConfidenceHead(c_token, hidden=None, dropout=0.0)` —
  `Linear(c_token -> hidden) -> SiLU -> Dropout -> Linear(hidden -> 1) -> sigmoid`.
  Mask-aware mean pool. Output bias initialised to 0 -> head predicts 0.5 at init.
- `src/tinyfold/model/resfold/__init__.py`: re-export `ConfidenceHead`.
- `tests/test_confidence_head.py` (CREATE): six tests — shape, mask=None vs
  all-True mask, mask invariance under garbage-padded tokens (atol=1e-6),
  gradient flow into MLP weights + input tokens, midpoint init sanity, and
  param-count tiny check (under `2 * c_token^2`).

### Verification

- [x] `pytest tests/test_confidence_head.py -v` -> 6 passed.
- [x] Param count at `c_token=128`: 16,641 (`= 128^2 + 2*128 + 1`).
- [x] Output bounded in `[0, 1]`; mask invariance holds at 1e-6.

---

## Task 3: Wire `ConfidenceHead` into `ResFoldOneStep`

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/resfold/onestep.py`:
  - `__init__`: new `confidence_head: bool = False` arg. When True,
    instantiates `self.confidence_head = ConfidenceHead(c_token, dropout=dropout)`.
    Otherwise `self.confidence_head = None`. Import is local.
  - New helper `_predict_confidence(denoiser_tokens, mask)` returns `None`
    when the head is disabled, else `[B]` predicted lDDT.
  - `forward_sigma` and `forward_sigma_with_trunk` now return a 3-tuple
    `(centroid_pred, atoms_pred, pred_lddt_or_None)`. Discrete-timestep
    `forward()` left as a 2-tuple (legacy / unused in active eval).
  - `count_parameters()` adds `confidence_head` bucket (0 when disabled) and
    `confidence_head_pct`.

### Caller migration

Grep for `forward_sigma` (excluding archive/pipeline-only sites) -> updated
sites in `scripts/train_resfold.py`:

- `sample_centroids_one_shot` (line ~180): drops the new third slot, returns
  legacy `(centroid, atoms)` for backwards compatibility.
- `sample_centroids_ve` (line ~307): the sigma_min "atoms reader" forward now
  index-accesses `[1]` instead of unpacking 2 values.
- `sample_k_centroids` (line ~383): unpacks the 3-tuple; if the OneStep model
  has a confidence head, collects `pred_lddt` per sample. On the slow VE path
  (no head-output during the trajectory), runs one extra forward at
  `sigma_min` on the final centroid to recover the head's score.
- Train loop self-conditioning (line ~1494): `sc_out[0]` works for both
  legacy 2-tuple and new 3-tuple — no change needed there.
- Train loop forward (line ~1503): explicit `centroids_pred, atoms_pred, pred_lddt = fwd_out` on the onestep branch.

`ResidueDenoiser.forward_sigma` (the non-onestep path used by
`train_resfold_stage2.py`, `eval_two_stage.py`, `e2e.py`, and the mock
denoisers in `test_kabsch_interp_sampler.py`) is UNCHANGED — they keep their
single-tensor return.

### Verification

- [x] `pytest tests/unit/test_onestep_confidence.py -v` -> 5 passed: 3-tuple
  return, `pred_lddt is None` when head disabled, `forward_sigma_with_trunk`
  matches contract, gradient flows from pred_lddt -> head AND -> denoiser
  (through the pool), `count_parameters` reports the bucket.
- [x] `pytest tests/test_multisample_eval.py -v` -> 4 passed (after updating
  the three call sites to unpack the new 3-tuple).

---

## Task 4: Confidence loss in the trainer

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py`:
  - CLI: `--confidence_head` (store_true) and `--confidence_head_weight`
    (default 0.0). Added right after `--atom_warmup_steps`.
  - Validation in the OneStep model construction block: `--confidence_head_weight > 0`
    requires `--confidence_head`; `--rank_by confidence` requires `--confidence_head`.
  - Model construction: forwards `confidence_head=args.confidence_head` to
    `ResFoldOneStep`; param-count log line appears only when the bucket is non-zero.
  - Self-conditioning unpacking: now uses `sc_out[0]` (works for both 2- and
    3-tuple returns). Forward unpacking on the onestep branch: explicit
    `centroids_pred, atoms_pred, pred_lddt = fwd_out`. `pred_lddt = None`
    initialised in the non-conf branch.
  - Aux loss block inserted right after the atom-MSE + geometry-loss block
    and before contact-loss: `compute_lddt(centroids_pred.detach(),
    centroids_target, mask, coord_scale=1.0, reduction="per_sample")` under
    `no_grad`, then `smooth_l1(pred_lddt, gt_lddt) * args.confidence_head_weight`
    added to `loss`. Records `gt_lddt_mean` and `pred_lddt_mean` for the log.
  - `loss_components` (onestep branch) gains `conf`, `gt_lddt_mean`,
    `pred_lddt_mean` keys.
  - Per-step logger: now fires on `step % 100 == 0 OR step % eval_every == 0`
    (so short runs surface the per-step line). When
    `args.confidence_head` is True, appends `conf: <val> (pred=<m> gt=<m>)` to
    the stage1_only log line.
  - Import: added `compute_lddt` alongside the existing `compute_lddt_metrics`.

### Coord-scale convention

Trainer coords are normalised (per-sample std ~1.0 absorbed at eval via
`s['std']`). Passing `coord_scale=1.0` to `compute_lddt` keeps the lDDT
distance thresholds comparable in the normalised space. Smoke check at step 50
shows `gt_lddt mean ~ 0.887` and `pred_lddt mean ~ 0.878` — both in a sane
band, not saturating at 0 or 1.

### Detach contract

The TARGET `gt_lddt` is computed under `torch.no_grad()` so gradients NEVER
flow back through it. The HEAD output `pred_lddt` is the live forward — so
gradients flow into both the confidence head AND through the pool back into
the denoiser tokens (verified by `test_confidence_loss_gradient_reaches_denoiser`).

### Verification

- [x] CLI flags parse; smoke run with `--confidence_head --confidence_head_weight 0.1`
  emits `conf: 0.0179 -> 0.0042` at steps 25 and 50 with finite values.
- [x] `gt_lddt mean` between 0.77 and 0.89 across smoke steps — not collapsed,
  not saturated.
- [x] With `--confidence_head_weight 0` and the head ON, manual check confirms
  no grad reaches `confidence_head.parameters()` (dormant) — see acceptance
  criterion at bottom of PLAN.

---

## Task 5: `--rank_by confidence` in multi-sample eval

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py`:
  - CLI: `--rank_by` with choices `["oracle", "cluster", "confidence"]`,
    default `cluster` (Loop 02 default preserved). Lives near `--n_samples`.
  - `sample_k_centroids` returns a 3-tuple
    `(centroids[K,B,L,3], atoms_or_None[K,B,L,4,3], pred_lddts_or_None[K,B])`.
    `pred_lddts` is populated on both the fast `one_shot` path (cheap — head
    output is part of the forward) and the slow VE path (one extra
    `forward_sigma` at sigma_min on the final centroid).
  - `_run_test_eval`:
    - New `per_k_ranked_conf` accumulator alongside `per_k_oracle`,
      `per_k_mean`, `per_k_ranked`. Populated only when the head fires.
    - Per-K cluster/conf/oracle rep indices recorded; downstream metrics
      (DockQ, atom RMSE, C-RMSD) pick the sample chosen by `--rank_by` at the
      FULL K so the headline ranker and downstream numbers stay consistent.
    - Log line appends `ranked_conf@{k}: <r> A` (when populated) and the new
      Spearman / Pearson correlation token.
    - Smoke instrumentation: `pred_lddt_std(per-target)` reports the mean of
      per-target stddevs — a near-zero value means the head collapsed.
  - `extra_tokens` (REGISTRY columns) gains `ranked_conf@K`, `spearman` (or
    `pearson` fallback), and `pred_lddt_std` tokens.

### Spearman fallback

Tries `scipy.stats.spearmanr`. On ImportError or any failure, falls back to
`torch.corrcoef` for Pearson (skipped if either side has zero std). The
smoke run completed with the scipy path (Spearman = 0.116 at step 25, -0.124
at step 50 — head still learning, expected on 50 steps).

### Verification

- [x] Smoke log at step 25 contains both
  `ranked@4: 13.2671 A | ranked_conf@4: 13.2226 A` and
  `Spearman(pred_lddt,-RMSE): 0.116 | pred_lddt_std(per-target): 0.0006`.
- [x] Per-sample `pred_lddt` values across K=4 samples for one target:
  `[0.896, 0.897, 0.895, 0.898]` — mean 0.8965, std 0.00142, non-zero variance
  confirmed (head responding to different denoiser inputs at 50 steps; would
  spread further with full training).

---

## Task 6: Unit + integration tests

Completed: 2026-05-24

### Changes

- `tests/test_confidence_head.py` (CREATE, 6 tests) — head-only contracts.
- `tests/unit/test_onestep_confidence.py` (CREATE, 5 tests) — OneStep wiring:
  3-tuple return on/off, gradient flow through pool into denoiser,
  `count_parameters` bucket.
- `tests/test_multisample_eval.py` (MODIFY): updated three existing tests to
  unpack the new 3-tuple from `sample_k_centroids`. Added `pred_lddts is None`
  assertions for the head-off model.

### Verification

- [x] `pytest tests/test_confidence_head.py tests/unit/test_onestep_confidence.py -v` -> 11 passed.
- [x] Full required regression set passes:
  `pytest tests/test_confidence_head.py tests/test_esm2_cache.py tests/test_edm_loss_weight.py tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py tests/unit/test_kabsch_rigid.py tests/test_pose_clustering.py tests/test_multisample_eval.py tests/test_kabsch_interp_sampler.py tests/unit/test_onestep_confidence.py -v`
  -> 47 passed, 1 skipped (`test_kabsch_interp_integration_changes_trajectory`,
  unrelated environment-gated integration test).

---

## Task 7: 50-step smoke run

Completed: 2026-05-24

### Command

```
.venv/Scripts/python.exe scripts/train_resfold.py \
  --config configs/train/resfold/phase_b_n4.yaml \
  --n_steps 50 --eval_every 25 \
  --confidence_head --confidence_head_weight 0.1 \
  --n_samples 4 --eval_K_list 1,4 --rank_by confidence \
  --output_dir outputs/_loop06_smoke
```

### Result

- Exit 0; total wall time 14s on RTX 4070 Ti SUPER (CUDA).
- Param count log: `Confidence-head params: 16,641 (0.8%)` — exactly the
  expected `c_token^2 + 2*c_token + 1` for `c_token=128`.
- Step 25: `loss 1.277 | mse 1.117 | dst 1.014 | conf 0.0179 (pred=0.932 gt=0.769)`
- Step 50: `loss 0.688 | mse 0.584 | dst 0.376 | conf 0.0042 (pred=0.878 gt=0.887)`
- Eval at step 25: `ranked@4: 13.2671 A | ranked_conf@4: 13.2226 A | Spearman: 0.116`.
- Eval at step 50: `ranked@4: 14.3518 A | ranked_conf@4: 14.3206 A | Spearman: -0.124`.
- Across K=4 samples on a probe target, `pred_lddt` values are NOT identical
  (std = 0.00142, range 0.895 to 0.898). The head is responding to different
  denoiser tokens — not collapsed at init even though 50 steps is far too few
  for any meaningful regression signal (full training in Loop 07 will reveal
  the real Spearman).
- Mean of per-target stddevs reported in eval: 0.0006 — tiny but nonzero.
  Expected at 50 steps; this is the "collapse-to-mean" metric Loop 07 will
  watch on `phase_d_n8600_full.yaml`.

### Cleanup

- Output dir `outputs/_loop06_smoke/` removed.
- REGISTRY.md row containing `_loop06_smoke` removed (1 line scrubbed).

---

## Acceptance Criteria

- [x] `compute_lddt` supports `reduction="per_sample"` returning `[B]`;
  existing scalar behaviour preserved.
- [x] `ConfidenceHead` module exists, masked-mean over residues,
  sigmoid-bounded, passes all unit tests.
- [x] `ResFoldOneStep(confidence_head=True)` returns `pred_lddt` from both
  `forward_sigma` and `forward_sigma_with_trunk`.
- [x] Trainer accepts `--confidence_head` and `--confidence_head_weight`;
  aux loss appears in `loss_components` and the per-step log line.
- [x] `--rank_by confidence` wired into `_run_test_eval`, emits `ranked_conf@K`.
- [x] Unit tests in `tests/test_confidence_head.py` and
  `tests/unit/test_onestep_confidence.py` all pass.
- [x] Smoke run completes 50 steps + 4-target eval; `pred_lddt` is non-constant
  across K samples (std 0.00142, not zero).
- [x] `--confidence_head_weight 0` keeps the head dormant: no grad reaches
  `confidence_head.parameters()` when the loss term is zero (verified by
  hand). The aux-loss block guards on `args.confidence_head_weight > 0`.

---

## Files changed

| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/model/losses/lddt.py` |
| CREATE | `src/tinyfold/model/resfold/confidence_head.py` |
| MODIFY | `src/tinyfold/model/resfold/__init__.py` |
| MODIFY | `src/tinyfold/model/resfold/onestep.py` |
| MODIFY | `scripts/train_resfold.py` |
| CREATE | `tests/test_confidence_head.py` |
| CREATE | `tests/unit/test_onestep_confidence.py` |
| MODIFY | `tests/unit/test_losses.py` |
| MODIFY | `tests/test_multisample_eval.py` |

File count: 9 (3 created, 6 modified).

Smoke results: exit 0, conf-aux finite (0.0179 -> 0.0042), `ranked@4` and
`ranked_conf@4` both in log, pred_lddt mean/std across K=4 = 0.8965 / 0.00142
(non-zero variance).

Regression pass count: 47 passed, 1 skipped (pre-existing env-gated skip),
0 failures across the required regression set.

Blockers: NONE. Loop 06 ready for Loop 07 to wire
`phase_d_n8600_full.yaml` (confidence_head + weight 0.1 + rank_by confidence).
