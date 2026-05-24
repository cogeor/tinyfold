# Implementation Log

## Task 1: Pose-cluster helper module

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/metrics/cluster.py`: new module with three public
  functions — `interface_mask_from_gt`, `pairwise_interface_rmsd`,
  `cluster_poses`. CPU-only implementation per plan note (torch.linalg
  on K<=40 small tensors is dominated by CUDA sync overhead; the helper
  expects `.cpu()` tensors from the caller).
- `src/tinyfold/model/metrics/__init__.py`: re-exports the three helpers
  alongside the existing `compute_dockq`.

### Verification

- [x] Three public functions exist and are exported.
- [x] Unit tests in `tests/test_pose_clustering.py` cover: interface mask
  picks close residues, excludes far ones; pairwise RMSD is symmetric
  with zero diagonal; two-mode pose set produces two clusters with
  correct membership; singletons; representative is always a member;
  empty interface mask raises `ValueError`.

---

## Task 2: Seeded multi-sample sampler

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py`:
  - `sample_centroids_one_shot` gained `generator: Optional[torch.Generator] = None`
    kwarg threaded into the `torch.randn` init at the noise-sigma_init line.
  - `sample_centroids_ve` gained the same `generator` kwarg threaded into
    its `torch.randn(B, L, 3, ...)` init.
  - New `sample_k_centroids` helper (~line 282) runs the trunk once (via
    `model.get_trunk_tokens(...)` then `forward_sigma_with_trunk(...)` K
    times) when `is_onestep and one_shot`, otherwise calls
    `sample_centroids_ve`/`sample_centroids_one_shot` K times with a
    per-sample `torch.Generator` seeded
    `base_seed * 100003 + target_idx * 1009 + sample_idx`.
  - Returns `(centroids [K, B, L, 3], atoms_or_None [K, B, L, 4, 3])`.

### Verification

- [x] `test_sample_k_centroids_one_shot_returns_distinct_samples` asserts
  K=4 samples are pairwise distinct (norm > 1e-3) and bit-identical on a
  second call with the same seeds.
- [x] `test_sample_k_centroids_ve_path_distinct` covers the multi-step
  VE path.
- [x] `test_different_target_idx_gives_independent_samples` confirms
  `target_idx` participates in the seed mix.
- [x] Default `generator=None` preserves prior behaviour: the 10-step
  smoke training run (`--n_samples 1` implicit default) produced a
  REGISTRY row with no `oracle@*` tokens, structurally identical to
  Loop 01 rows.

---

## Task 3: Wire K-sample into `_run_test_eval`

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py`:
  - Three new CLI flags added next to the existing sampling flags:
    `--n_samples` (default 1), `--cluster_radius` (default 5.0),
    `--eval_K_list` (default `"1,5,40"`).
  - `_run_test_eval` parses `eval_K_list` once at entry. When
    `--n_samples > 1`, `k_list = sorted({...})` and the assert
    `max(k_list) <= args.n_samples` guards the contract; otherwise
    `k_list = [1]` and the multi-sample machinery is silent.
  - Per-K accumulators (`per_k_oracle`, `per_k_mean`, `per_k_ranked`)
    seeded alongside the existing `test_rmses` / `test_c_rmsds` lists.
  - In the per-target loop, `args.continuous_sigma and args.n_samples > 1`
    routes through `sample_k_centroids`, computes per-sample RMSEs vs GT
    (Kabsch via `compute_rmse`), builds the GT-only interface mask (with
    an all-True fallback if zero contacts), then for each `k in k_list`
    appends `(min, mean, ranked_rep_rmse)` from the first-`k` sub-pool.
  - Downstream metrics (DockQ, atom RMSE, C-RMSD) keep using sample 0
    only (existing Loop 01 metrics — no K-of-them needed). `rmse` for
    the existing `test_rmses` accumulator is also sample-0 to preserve
    the published "test RMSE" cell.
  - After the loop, `extra_tokens` is built for every K > 1 in `k_list`
    (K=1 skipped — `oracle@1 == mean@1 == ranked@1 == test RMSE`).
  - Return tuple extended to 5 values: `(test_avg, dockq_avg,
    dockq_success_pct, c_rmsd_avg, extra_tokens)`.
  - Both call sites updated to unpack five values: the in-loop call
    around the current line 1666 and the `--eval_only` call at
    current line 1194.
  - `--eval_only` branch stashes `progress["extra_tokens"] = extra_tokens`
    when non-empty.
  - In-loop best-eval stash also writes `progress["extra_tokens"]` so
    `main()`'s finally-block reads the freshest Loop 02 tokens.
  - `main()`'s `append_registry_row(...)` call now passes
    `extra_tokens=progress.get("extra_tokens")`.

### Deviation note

The plan specified updating the in-loop call at "line 1483" and the
`--eval_only` call at "line 1013". After the Task 2 insertions of
`sample_k_centroids`, the line numbers shifted to 1666 and 1194
respectively. Both call sites were audited and updated to unpack five
values; this is exactly the situation the plan tester memo flagged.

### Verification

- [x] Three new CLI flags parsed and visible via `--help`.
- [x] In-loop and `--eval_only` call sites both unpack the 5-tuple.
- [x] Smoke training run (`--n_samples 1` implicit) produced the same
  Outcome cell shape as Loop 01 — no `oracle@*` tokens added.
- [x] Integration test (`test_eval_only_k5_integration`) ran the script
  with `--n_samples 5 --eval_K_list 1,5 --n_test 4` against the Phase C
  checkpoint and confirmed `oracle@5 <= mean@5` plus all three tokens
  in the log.

---

## Task 4: REGISTRY plumbing + extra_tokens

Completed: 2026-05-24

### Changes

- `src/tinyfold/training/registry_append.py`:
  - New kwarg `extra_tokens: Optional[List[str]] = None`.
  - Inserted `if extra_tokens: extras.extend(extra_tokens)` after the
    existing `c_rmsd` / `dockq_avg` token blocks so ordering is:
    C-RMSD, DockQ, then extras.
  - Docstring `Args:` block describes the new param and steers new
    callers toward it ("preferred for new metrics; the named kwargs are
    kept for Loop 01 back-compat only").
  - `typing.List` added to the import.

### Verification

- [x] All three Loop 01 tests in `tests/unit/test_registry_append.py`
  still pass (back-compat preserved):
  - `test_appends_dockq_and_c_rmsd_tokens`
  - `test_appends_dockq_without_success_rate`
  - `test_back_compat_when_none`
- [x] Integration test row in REGISTRY (Loop 02 smoke) shows
  `C-RMSD 6.6048 A; DockQ 0.294 succ 50.0%; oracle@5 5.275 A; mean@5
  5.528 A; ranked@5 5.522 A` — extras appear AFTER C-RMSD / DockQ in
  the cell as specified.

---

## Task 5: Unit + integration tests

Completed: 2026-05-24

### Changes

- `tests/test_pose_clustering.py`: 7 unit tests covering interface mask,
  pairwise RMSD, cluster_poses (two modes, all singletons, single pose,
  empty mask raises).
- `tests/test_multisample_eval.py`: 4 tests:
  1. `test_sample_k_centroids_one_shot_returns_distinct_samples` — tiny
     CPU `ResFoldOneStep(c_token=32, trunk_layers=1, denoiser_blocks=1)`,
     K=4 distinct + reproducible.
  2. `test_sample_k_centroids_ve_path_distinct` — VE multi-step path.
  3. `test_different_target_idx_gives_independent_samples` — seed-mix
     dependency check.
  4. `test_eval_only_k5_integration` — `@pytest.mark.slow` +
     `@pytest.mark.skipif(not PHASE_C_CKPT.exists(), ...)`. Spawns
     `python scripts/train_resfold.py --eval_only --n_samples 5
     --eval_K_list 1,5 --n_test 4 --checkpoint <phase_c>`, parses the
     train.log, asserts `oracle@5 <= mean@5` and all three tokens
     present.

### Verification

- [x] `pytest tests/test_pose_clustering.py tests/test_multisample_eval.py -v`:
  11 passed in 50.93s.
- [x] Pre-existing repo failures in `tests/unit/test_geometry.py` and
  `tests/unit/test_losses.py::TestImports::test_all_geometry_imports`
  confirmed to be unrelated to Loop 02 (BOND_LENGTHS export bug; random
  seed flakiness in geometry-loss integration test). Not introduced by
  this loop.

---

## Task 6: Phase C K=5 and K=40 re-eval

Deferred to tester per implementer instructions. Two commands documented in
PLAN.md Task 6 are ready to run; output dirs
`outputs/resfold/phase_c_n8600_reeval_loop02_k5` and `_k40` will be
created by the tester. Phase C checkpoint at
`outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt`
remains intact (read-only path).

---

## Files touched

| Action | Path |
|--------|------|
| CREATE | `src/tinyfold/model/metrics/cluster.py` |
| MODIFY | `src/tinyfold/model/metrics/__init__.py` |
| MODIFY | `src/tinyfold/training/registry_append.py` |
| MODIFY | `scripts/train_resfold.py` |
| CREATE | `tests/test_pose_clustering.py` |
| CREATE | `tests/test_multisample_eval.py` |

---

## Test results

- `pytest tests/test_pose_clustering.py tests/test_multisample_eval.py -v`:
  **11 passed, 0 failed in 50.93s**.
- `pytest tests/unit/test_registry_append.py -v` (Loop 01 regression):
  **3 passed, 0 failed**.
- `pytest tests/unit/ -v`: 42 passed, 2 failed — both failures
  pre-existing and unrelated to Loop 02 (geometry loss imports +
  random-seed flakiness).
- 10-step smoke training run (`--config phase_b_n4.yaml --n_steps 10
  --eval_every 5`): completed in 2 s; in-loop `_run_test_eval` was
  called twice (steps 5 and 10) and the 5-tuple unpack worked both
  times. Registry row written with the legacy outcome-cell shape
  (no `oracle@*` tokens since `--n_samples` defaulted to 1).

---

## Acceptance criteria

- [x] `src/tinyfold/model/metrics/cluster.py` exists with the three
  public functions, exported from `tinyfold.model.metrics.__init__`.
- [x] `sample_centroids_one_shot` and `sample_centroids_ve` accept
  `generator: Optional[torch.Generator]` (default `None`).
- [x] `--n_samples`, `--cluster_radius`, `--eval_K_list` CLI flags
  added.
- [x] `_run_test_eval` returns `extra_tokens` (5-tuple) and prints
  `oracle@K mean@K ranked@K` tokens for every K > 1.
- [x] `append_registry_row(..., extra_tokens=...)` extends the outcome
  cell without breaking Loop 01's tokens.
- [x] `--n_samples 1` produces a row whose outcome cell is structurally
  identical to Loop 01 (verified via smoke run).
- [x] `pytest tests/test_pose_clustering.py tests/test_multisample_eval.py -v`
  passes locally (11/11 green).
- [ ] Phase C K=5 re-eval row (tester to run).
- [ ] Phase C K=40 re-eval row (tester to run).
- [ ] `oracle@40 <= mean@40` sanity (tester to verify on K=40 row).
