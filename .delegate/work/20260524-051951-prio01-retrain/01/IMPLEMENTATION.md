# Implementation Log

## Task 1: add `compute_c_rmsd` to `losses/mse.py`

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/losses/mse.py`: appended `compute_c_rmsd` immediately after `compute_rmse`. Inlines the same SVD recipe as `kabsch_align` (lines 50-58 of the same file) so we can keep `R` and `t` explicit and then apply them to the FULL prediction (both chains). The implementation matches the plan's code listing verbatim except for stylistic comment trims.

### Verification

- [x] Function is defined and callable: imported it via `from tinyfold.model.losses import compute_c_rmsd` in the unit test, which passes.
- [x] Whole-complex rigid-transform sanity check: when both chains are rotated/translated by the same `R, t`, the chain-A Kabsch recovers the exact transform and C-RMSD is ~0 (test `test_whole_complex_rigid_transform_is_zero` PASS).
- [x] Chain-A-only-correct sanity check: when chain B is moved independently, the residual equals `sqrt(n_b / L) * rmsd_chain_b` (test `test_chain_b_offset_gives_nontrivial_rmsd` PASS).
- [x] Assert guard: fewer than 3 chain-A residues raises `AssertionError` (test `test_too_few_chain_a_residues_raises` PASS).

---

## Task 2: re-export `compute_c_rmsd` from `losses/__init__.py`

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/losses/__init__.py`: added `compute_c_rmsd` to both the `from .mse import (...)` block (between `compute_rmse` and `compute_relative_distance_loss`) and the `__all__` list (in the matching position).

### Verification

- [x] Train script imports it via `from tinyfold.model.losses import compute_c_rmsd` (added in Task 3). `python scripts/train_resfold.py --help` parses successfully.
- [x] Unit test imports it via the same path and runs.

---

## Task 3: wire DockQ + C-RMSD into the stage1_only onestep eval branch

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py` (imports): added `compute_c_rmsd` to the `from tinyfold.model.losses import (...)` block.
- `scripts/train_resfold.py` (test-eval body): added `test_c_rmsds = []` accumulator next to `test_atom_rmses`. Inside the `if args.mode == "stage1_only":` branch (after the `atom_rmse` block), added:
  - Unconditional `compute_c_rmsd(...)` call on `centroids_pred[:, :n_res]` vs `batch['centroids'][:, :n_res]`, denormalized by `s['std']`.
  - Conditional DockQ block (`if is_onestep and atoms_pred_onestep is not None:`) calling `compute_dockq` on `atoms_pred_onestep[0, :n_res]` per the plan.
- `scripts/train_resfold.py` (summary log): extended `log_msg` to include `DockQ X.XXXX (succ Y.Y%)` and `C-RMSD X.XXXX A` substrings. Promoted the previous DockQ block (avg only) to the version that also reports the >=0.23 success rate — kept it as ONE block (no double-print), and exposed `dockq_avg`, `dockq_success_pct`, `c_rmsd_avg` as locals so the surrounding save-best logic can stash them into `progress`.

### Decisions / deviations

- The plan listed line numbers 1255-1379 for the eval body; the actual file had drifted slightly. I followed the plan's INTENT and located the corresponding code sections by structure (the `test_*` accumulator list, the stage1_only branch, the summary log_msg).
- For the summary line wording I kept the existing prefix (`         >>> ...`) and inserted the new tokens in the same `|`-separated style. C-RMSD uses 4 decimals to mirror the centroid-RMSE precision; DockQ keeps 4 decimals for the avg (existing convention) but uses 1 decimal for the success rate (matches the plan's `succ Y.Y%` format).

### Verification

- [x] Script parses (`python scripts/train_resfold.py --help` succeeds).
- [x] No double-print of DockQ (only one `if test_dockq_scores:` block in the summary).
- [ ] End-to-end smoke (re-eval against the Phase C checkpoint) — out of scope for the implementer; the tester runs this.

---

## Task 4: surface new metrics in REGISTRY outcome cell

Completed: 2026-05-24

### Changes

- `src/tinyfold/training/registry_append.py`:
  - Extended `append_registry_row` signature with three optional kwargs: `dockq_avg`, `dockq_success_pct`, `c_rmsd`. All default to `None` (back-compat: existing call sites unchanged).
  - Updated the docstring with descriptions of the three new kwargs.
  - After the existing `outcome_cell` build, appended a `" | C-RMSD X.XXXX A"` and/or `" | DockQ X.XXX [succ Y.Y%]"` token sequence ONLY when the corresponding values are not None. Order: C-RMSD first, then DockQ. The success-rate suffix is omitted if `dockq_success_pct` is None.
- `scripts/train_resfold.py` (best-model save): inside the `if test_avg < best_rmse:` block, stash the latest `dockq_avg`, `dockq_success_pct`, `c_rmsd_avg` from the current eval into the `progress` dict.
- `scripts/train_resfold.py` (`finally:` block): threaded `progress.get("dockq_avg")`, `progress.get("dockq_success_pct")`, `progress.get("c_rmsd")` into the `append_registry_row` call.

### Decisions / deviations

- The plan suggested capturing metrics at the best-eval step. I followed that exactly: only when `test_avg < best_rmse` do we overwrite the progress entries. This matches the existing `best_rmse` semantics (the REGISTRY row reports the best, not the most recent). The `--eval_only` path writes them unconditionally (only one eval pass, so "best" == "only").

### Verification

- [x] `test_appends_dockq_and_c_rmsd_tokens` PASS: all three kwargs surface as `C-RMSD 8.7000 A | DockQ 0.420 succ 35.0%` in the outcome cell.
- [x] `test_appends_dockq_without_success_rate` PASS: `dockq_avg` without `dockq_success_pct` -> `DockQ 0.150` with no `succ` token and no `%`.
- [x] `test_back_compat_when_none` PASS: legacy callers (all three kwargs at default None) produce a row identical to the historical format — no `C-RMSD` or `DockQ` substring.

---

## Task 5: add `--eval_only` / factor out `_run_test_eval`

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py` (args): added `--eval_only` (action="store_true") immediately after the existing `--checkpoint` argument (so both eval-only flags live together).
- `scripts/train_resfold.py` (new helper `_run_test_eval`): factored the entire test-set eval body (the test loop over `test_indices`, including ALL three mode branches `stage1_only` / `stage2_only` cached / fallback, plus the summary `log_msg` construction) into a module-level function `_run_test_eval(model, test_samples, test_indices, noiser, eval_sampler, device, args, is_onestep, logger, train_avg=None, n_eval=None)`. The function:
  - Wraps its body in `model.eval(); with torch.no_grad():` (redundant when called from the in-loop site but required for the eval-only path).
  - Logs the summary line, omitting the `Train X` prefix when `train_avg=None` (the eval-only path has no train-side numbers).
  - Returns `(test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg)`. Does NOT touch `progress` — the caller stores whatever it wants.
- `scripts/train_resfold.py` (in-loop call site): replaced the now-extracted test-eval body with a single call to `_run_test_eval(...)` passing `train_avg=train_avg, n_eval=n_eval`. The save-best logic below it is unchanged and consumes the returned `test_avg`, `dockq_avg`, `dockq_success_pct`, `c_rmsd_avg`.
- `scripts/train_resfold.py` (`_run_training` body): right after the contact-loss-setup block (before the optimizer is built), added the `if args.eval_only:` branch. It:
  - Asserts `args.checkpoint is not None` (the plan-required pre-condition).
  - Logs a banner.
  - Calls `_run_test_eval(...)` with no `train_avg` / `n_eval`.
  - Writes the returned metrics into `progress` (`best_rmse`, `dockq_avg`, `dockq_success_pct`, `c_rmsd`) so the unchanged outer `finally:` block writes a correct REGISTRY row.
  - Returns early — never builds the optimizer or enters the training step loop, and never overwrites `best_model.pt` (it lands in the new `--output_dir` only because the script always creates one, but no `torch.save` is called from this branch).

### Decisions / deviations

- The plan suggested the helper could update `progress` directly. I chose the cleaner pure-function shape (returns tuple, caller stores). Rationale: the in-loop call still wants to track `best_rmse` as the MIN over evals — having the helper unconditionally overwrite `progress["best_rmse"]` would corrupt that comparison. By returning metrics and letting the caller decide, both call sites stay correct without conditional logic inside the helper.
- The checkpoint load happens via the existing `load_model_checkpoint(model, args.checkpoint, args.mode, device, logger)` call that's already in place at lines ~727 / ~752 of `_run_training` (BEFORE the new `if args.eval_only:` branch). So when `--eval_only --checkpoint X` is passed, the model is already loaded by the time we hit the branch — no extra load needed. The plan's hint that the checkpoint save uses `torch.save(model.state_dict(), ...)` is slightly off: the actual save wraps it in a dict with key `model_state_dict`, but `load_model_checkpoint` handles that key, so nothing changes.
- I placed the `--eval_only` branch after the contact-loss setup so that the model + dataloaders + noiser + eval_sampler are all fully constructed (matching how the in-loop eval expects them).
- Plot generation is skipped in `--eval_only`. The plan didn't ask for it and the existing plotting code lives inside `if step % args.eval_every == 0:` — keeping it in-loop is the smallest surface change.

### Verification

- [x] Script parses (`python -c "import ast; ast.parse(...)"` returns OK).
- [x] `python scripts/train_resfold.py --help` lists both `--checkpoint` and `--eval_only`.
- [x] In-loop eval path unchanged in behavior (the only change is a function call boundary; the same code runs with the same inputs).
- [ ] End-to-end re-eval against Phase C `best_model.pt` — out of scope for the implementer; the tester runs the command listed in PLAN.md Section "Re-eval command".

---

## Test Results

Ran `pytest tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py -v` (using `.venv/Scripts/python.exe`).

```
tests/unit/test_c_rmsd.py::test_chain_b_offset_gives_nontrivial_rmsd PASSED [ 16%]
tests/unit/test_c_rmsd.py::test_whole_complex_rigid_transform_is_zero PASSED [ 33%]
tests/unit/test_c_rmsd.py::test_too_few_chain_a_residues_raises PASSED   [ 50%]
tests/unit/test_registry_append.py::test_appends_dockq_and_c_rmsd_tokens PASSED [ 66%]
tests/unit/test_registry_append.py::test_appends_dockq_without_success_rate PASSED [ 83%]
tests/unit/test_registry_append.py::test_back_compat_when_none PASSED    [100%]

============================== 6 passed in 1.72s ==============================
```

All 6 new tests pass.

Sanity-ran the pre-existing `tests/unit/test_losses.py` to check no regressions: 18/19 pass. The 1 failure (`TestImports::test_all_geometry_imports`) is a pre-existing issue unrelated to this loop — it expects `BOND_LENGTHS` from `tinyfold.model.losses` but the symbol was renamed to `BOND_LENGTHS_ANGSTROM` in a prior loop. Not in scope.

## Acceptance Criteria (from PLAN.md)

- [x] `compute_c_rmsd` exists in `losses/mse.py` and is re-exported from `losses/__init__.py`.
- [x] `scripts/train_resfold.py` test-eval block (stage1_only onestep) emits DockQ + C-RMSD.
- [x] Summary log line includes Centroid RMSE, DockQ (avg + succ%), and C-RMSD substrings.
- [x] `append_registry_row` accepts the three new optional kwargs and surfaces them in the Outcome cell, with back-compat when omitted.
- [x] `--eval_only` flag exists and gates a no-train re-eval path that drives `_run_test_eval` and writes to REGISTRY via the unchanged `finally:` block.
- [x] Unit tests pass (`test_c_rmsd.py`, `test_registry_append.py`).
- [ ] Smoke re-eval against Phase C `best_model.pt` (deferred to tester).
