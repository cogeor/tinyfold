# Test Results

Tested: 2026-05-24 06:02
Status: PASS

## Step 1: New + regression unit tests

Command:
```
.venv/Scripts/python.exe -m pytest tests/test_pose_clustering.py tests/test_multisample_eval.py tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py -v
```

Result: **17 passed in 42.87s** (0 failed, 0 skipped).

Breakdown:
- `tests/test_pose_clustering.py`: 7/7 (interface mask, pairwise RMSD,
  two-mode clustering, singletons, empty-mask raise).
- `tests/test_multisample_eval.py`: 4/4 (one-shot distinct + reproducible,
  VE distinct, target_idx seed-mix, Phase C K=5 end-to-end integration).
- `tests/unit/test_c_rmsd.py`: 3/3 (Loop 01 regression — chain B offset
  nontrivial, whole-complex rigid transform zero, too-few-residues raise).
- `tests/unit/test_registry_append.py`: 3/3 (Loop 01 regression —
  dockq+C-RMSD, dockq without success rate, back-compat with `None`).

No Loop 01 tests broke. Verdict: **PASS**.

## Step 2: Smoke training run (back-compat)

Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --config configs/train/resfold/phase_b_n4.yaml \
    --n_steps 10 --eval_every 5 \
    --output_dir outputs/_loop02_smoke
```

Result:
- Exit code 0, wall time ~28 s.
- Eval lines at steps 5 and 10 both contain
  `Centroid RMSE`, `DockQ`, `Atom RMSE`, `C-RMSD`.
  Example: `Test Centroid RMSE (14): 14.1151 A | DockQ: 0.0634 (succ
  7.1%) | Atom RMSE: 15.5144 | C-RMSD: 18.9999 A`.
- REGISTRY row (last in file at that point):
  `... test RMSE 14.1151 A — converged; C-RMSD 18.9999 A; DockQ 0.063
  succ 7.1% | converged | [outputs/_loop02_smoke/...]`.
- **No `oracle@`, `mean@`, or `ranked@` tokens** in the smoke row
  (n_samples defaulted to 1). Back-compat confirmed.

Verdict: **PASS**.

## Step 3: K=5 re-eval on Phase C best_model.pt

Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --config configs/train/resfold/phase_c_n8600.yaml \
    --eval_only \
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt \
    --n_samples 5 --eval_K_list 1,5 \
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop02_k5
```

Result:
- Exit code 0, wall time **~45 s** (well under 8 min ceiling).
- Eval line:
  `Test Centroid RMSE (100): 9.9956 A | DockQ: 0.1345 (succ 25.0%) |
  Atom RMSE: 10.1499 | C-RMSD: 16.8854 A | oracle@5: 8.9047 A |
  mean@5: 10.3493 A | ranked@5: 9.8945 A`.
- All three tokens `oracle@5`, `mean@5`, `ranked@5` present with finite
  values.
- REGISTRY last row:
  `... test RMSE 9.9956 A — converged; C-RMSD 16.8854 A; DockQ 0.134
  succ 25.0%; oracle@5 8.905 A; mean@5 10.349 A; ranked@5 9.895 A | ... |
  [outputs/resfold/phase_c_n8600_reeval_loop02_k5/...]`.
- Sanity: `oracle@5 (8.9047) <= mean@5 (10.3493)` — PASS (oracle is min).

Verdict: **PASS**.

## Step 4: K=40 re-eval on Phase C best_model.pt

Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --config configs/train/resfold/phase_c_n8600.yaml \
    --eval_only \
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt \
    --n_samples 40 --eval_K_list 1,5,40 \
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop02_k40
```

Result:
- Exit code 0, wall time **~68 s** (well under 20 min ceiling — beat
  the plan's 8-12 min estimate substantially).
- Eval line:
  `Test Centroid RMSE (100): 9.9956 A | DockQ: 0.1345 (succ 25.0%) |
  Atom RMSE: 10.1499 | C-RMSD: 16.8854 A | oracle@5: 8.9047 A |
  mean@5: 10.3493 A | ranked@5: 9.8945 A | oracle@40: 8.4597 A |
  mean@40: 10.3287 A | ranked@40: 9.8741 A`.
- All six K>1 tokens present with finite values.
- REGISTRY last row contains all six tokens plus DockQ + C-RMSD:
  `... test RMSE 9.9956 A — converged; C-RMSD 16.8854 A; DockQ 0.134
  succ 25.0%; oracle@5 8.905 A; mean@5 10.349 A; ranked@5 9.895 A;
  oracle@40 8.460 A; mean@40 10.329 A; ranked@40 9.874 A | ... |
  [outputs/resfold/phase_c_n8600_reeval_loop02_k40/...]`.

Sanity checks (all pass):
- `oracle@40 (8.4597) <= oracle@5 (8.9047)` — larger pool finds a better
  min. PASS.
- `oracle@40 (8.4597) <= mean@40 (10.3287)` — oracle is a min within
  the same pool. PASS.
- `|mean@40 - mean@5| = |10.3287 - 10.3493| = 0.0206 A` — well within
  0.5 A; the 5-sample mean is a reasonable estimator of the 40-sample
  mean. PASS.

Verdict: **PASS**.

## Step 5: REGISTRY hygiene

Final `git diff experiments/REGISTRY.md` shows **four** added rows beyond
HEAD:

| # | run_id                          | source                                |
|---|---------------------------------|---------------------------------------|
| 1 | `resfold_s1_8K_20260524_055828` | pytest integration test (tmp dir)     |
| 2 | `resfold_s1_4_20260524_055928`  | Step 2 smoke training                 |
| 3 | `resfold_s1_8K_20260524_060008` | Step 3 Phase C K=5 re-eval (headline) |
| 4 | `resfold_s1_8K_20260524_060106` | Step 4 Phase C K=40 re-eval (headline)|

- Rows **3** and **4** are the two intended Loop 02 headline rows.
- Row **2** is from the back-compat smoke; harmless but optional to
  retain in the commit (orchestrator decision).
- Row **1** is a **stray row** from the pytest integration test
  (`tests/test_multisample_eval.py::test_eval_only_k5_integration`).
  Its directory link points at
  `C:/Users/costa/AppData/Local/Temp/pytest-of-costa/pytest-1361/...`
  which won't exist after pytest cleanup. **Flag**: this row should
  be removed from the diff before commit (and arguably the integration
  test should pass `output_dir` outside REGISTRY's reach or use a
  `dryrun` mode that suppresses the append). Not a hard fail of Loop
  02's plan — the K=5/K=40 headline rows are clean.

Cross-row consistency check (same seed should give same first-5 pool):
- K=5 row `oracle@5 = 8.905 A`.
- K=40 row `oracle@5 = 8.905 A` (computed from the first 5 samples of
  the 40-sample pool).
- Difference: 0.000 A (well within the 0.01 A tolerance). PASS — seed
  plumbing is correctly deterministic.

Verdict: **PASS** (with stray-row flag — to be pruned at commit time,
not gating).

## Step 6: Phase C checkpoint integrity

Command:
```
Get-Item outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt
```

Result:
- `Name`: `best_model.pt`
- `Length`: 46,580,577 bytes
- `LastWriteTime`: **5/24/2026 4:14:00 AM** — matches expected baseline
  of 2026-05-24 04:14:00.

Phase C `best_model.pt` was **not overwritten**. The two `--eval_only`
runs wrote into fresh `outputs/.../_reeval_loop02_k5/` and `_k40/`
subdirs as the plan specified.

Verdict: **PASS**.

## Step 7: Cleanup outline

New files / dirs created by this tester run, for orchestrator
to triage before commit:

**Intentional (Loop 02 deliverables, should be staged):**
- `src/tinyfold/model/metrics/cluster.py` (CREATE — implementer)
- `src/tinyfold/model/metrics/__init__.py` (MODIFY — implementer)
- `src/tinyfold/training/registry_append.py` (MODIFY — implementer)
- `scripts/train_resfold.py` (MODIFY — implementer)
- `tests/test_pose_clustering.py` (CREATE — implementer)
- `tests/test_multisample_eval.py` (CREATE — implementer)
- `experiments/REGISTRY.md` (MODIFY — append K=5 + K=40 rows, optionally
  the smoke row; remove the pytest-tmp row)

**Output dirs the tester created (artefacts; usually NOT committed):**
- `outputs/_loop02_smoke/resfold_s1_4_20260524_055928/` — Step 2 smoke
  (~5 MB; final_model.pt + plots + train.log).
- `outputs/resfold/phase_c_n8600_reeval_loop02_k5/resfold_s1_8K_20260524_060008/`
  — Step 3 headline run output (split.json + train.log).
- `outputs/resfold/phase_c_n8600_reeval_loop02_k40/resfold_s1_8K_20260524_060106/`
  — Step 4 headline run output (split.json + train.log).

**Unchanged (verified intact):**
- `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt`
  — checkpoint mtime unchanged at 2026-05-24 04:14:00.

**Pre-existing repo modifications NOT introduced by Loop 02** (visible
in initial `git status` from the session start — orchestrator should
NOT bundle these into the Loop 02 commit):
- `doc/data_pipeline.md`, `doc/frontend.md` (modified)
- `src/tinyfold/model/registry.py` (modified)
- `src/tinyfold/model/archive/` (untracked)
- Deletions: `diffusion_analysis_report.md`, `epsilon_rollout_plan.md`,
  `plan_readme.md`.

## Acceptance Criteria (from PLAN.md)

- [x] `src/tinyfold/model/metrics/cluster.py` exists with three public
      functions and is exported from `tinyfold.model.metrics`.
- [x] `sample_centroids_one_shot` and `sample_centroids_ve` accept
      `generator: Optional[torch.Generator] = None`; default behaviour
      preserved (Step 2 smoke).
- [x] `--n_samples`, `--cluster_radius`, `--eval_K_list` CLI flags added.
- [x] `_run_test_eval` returns `extra_tokens` and prints `oracle@K
      mean@K ranked@K` for every K > 1 (Steps 3 & 4 confirm).
- [x] `append_registry_row(..., extra_tokens=...)` extends outcome cell
      without breaking C-RMSD / DockQ tokens (Step 4 row has all 8
      tokens in the right order).
- [x] `--n_samples 1` produces a row structurally identical to Loop 01
      (Step 2 smoke row has no `oracle@*` tokens).
- [x] `pytest tests/test_pose_clustering.py tests/test_multisample_eval.py
      -v` passes (Step 1 — 11/11 of those two files).
- [x] Phase C K=5 re-eval row appended.
- [x] Phase C K=40 re-eval row appended.
- [x] `oracle@40 (8.460) <= mean@40 (10.329)` sanity on K=40 row.

## Open observations (informational; not gating)

1. **`ranked@K ≈ mean@K`** — both K=5 and K=40 show only a ~0.45 A gap
   between the cluster-rep representative and the pool mean
   (ranked@5=9.895 vs mean@5=10.349; ranked@40=9.874 vs mean@40=10.329).
   This means the largest-cluster-rep ranking is barely better than
   random pick. **Expected per plan**; Loop 06 (confidence head) is
   designed to replace this ranker. Flag for tracking.
2. **`oracle@40 (8.460) << mean@40 (10.329)`** — 1.87 A oracle-vs-mean
   gap. Good news: there's meaningful diversity in the 40-sample pool,
   and a real ranker (Loop 06) should be able to recover a chunk of
   that ~2 A delta. Flag for prioritisation.
3. **Pytest integration test pollutes REGISTRY.** The
   `test_eval_only_k5_integration` smoke run appends a real REGISTRY
   row pointing at a pytest tmp dir. Recommendation for a future loop:
   either give the integration test a `--no_registry` flag, or have
   it write to a tmp REGISTRY copy. Not a Loop 02 gate.

## Build & Tests

- Build: N/A (no native build; Python package via `pip install -e .`).
- Tests targeted in Step 1: **17 / 17 PASS** in 42.87s.
- Smoke training (Step 2): exit 0, registry row clean.
- Headline eval runs (Steps 3 & 4): both exit 0, all tokens present,
  all sanity checks PASS.

## Commit Gate

ready: yes
reason: All 17 targeted tests pass, both headline re-evals (K=5 and K=40) produced finite metrics with all required tokens in stdout and REGISTRY rows, every numerical sanity check (oracle<=mean within pool, oracle@40<=oracle@5, mean@40~mean@5, seed-deterministic oracle@5 across runs) holds, the Phase C checkpoint mtime is unchanged at 2026-05-24 04:14:00, and the back-compat smoke run produces a Loop-01-shaped row. Orchestrator should prune the stray pytest-tmp REGISTRY row (`resfold_s1_8K_20260524_055828`) before committing.
commit-message: feat(eval): multi-sample inference with oracle@K, mean@K, ranked@K for HDOCK-style cluster-then-rank
