# Loop 01 TEST

## Commit gate
- ready: yes
- reason: All 6 unit tests pass; in-loop eval smoke emits Centroid RMSE / DockQ / C-RMSD at both eval points and writes a REGISTRY row with the new tokens; Phase C re-eval loads the frozen checkpoint cleanly, reports Centroid RMSE 9.5607 A (inside the 9.0-11.0 A green band), DockQ 0.140 (succ 26.0%), C-RMSD 15.9309 A, and the new REGISTRY row contains all three new tokens. No files outside the work folder and the two expected output dirs were touched; the protected Phase C best_model.pt is unchanged (mtime 2026-05-24 04:14:00, pre-dates the re-eval).
- commit-message: feat(eval): add DockQ + C-RMSD to stage1_only eval and --eval_only re-eval path

## Verification log

### 1. Unit tests
Command:
```
.venv/Scripts/python.exe -m pytest tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py -v
```

Output excerpt:
```
tests/unit/test_c_rmsd.py::test_chain_b_offset_gives_nontrivial_rmsd PASSED [ 16%]
tests/unit/test_c_rmsd.py::test_whole_complex_rigid_transform_is_zero PASSED [ 33%]
tests/unit/test_c_rmsd.py::test_too_few_chain_a_residues_raises PASSED   [ 50%]
tests/unit/test_registry_append.py::test_appends_dockq_and_c_rmsd_tokens PASSED [ 66%]
tests/unit/test_registry_append.py::test_appends_dockq_without_success_rate PASSED [ 83%]
tests/unit/test_registry_append.py::test_back_compat_when_none PASSED    [100%]

============================== 6 passed in 1.71s ==============================
```

PASS (6/6 expected, 6/6 observed)

### 2. In-loop eval smoke
Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py --config configs/train/resfold/phase_b_n4.yaml --n_steps 10 --eval_every 5 --output_dir outputs/_loop01_smoke
```

Output excerpt (eval lines at step 5 and step 10):
```
         >>> Train Centroid RMSE (4): 13.6778 A | Test Centroid RMSE (14): 14.3330 A | DockQ: 0.0831 (succ 14.3%) | Atom RMSE: 16.0628 | C-RMSD: 20.4016 A
         >>> Saved plot: outputs/_loop01_smoke\resfold_s1_4_20260524_053818\plots\step_000005.png
         >>> New best test RMSE! Saved.
         >>> Train Centroid RMSE (4): 12.6303 A | Test Centroid RMSE (14): 14.1151 A | DockQ: 0.0634 (succ 7.1%) | Atom RMSE: 15.5144 | C-RMSD: 18.9999 A
         >>> Saved plot: outputs/_loop01_smoke\resfold_s1_4_20260524_053818\plots\step_000010.png
         >>> New best test RMSE! Saved.
======================================================================
Training complete
  Total time: 2s (0.0 min)
  Best test RMSE: 14.1151 A
[registry] appended row to C:\Users\costa\src\tinyfold\experiments\REGISTRY.md
```

REGISTRY last row after smoke:
```
| 2026-05-24 | resfold_s1_4_20260524_053818 | resfold | config: configs/train/resfold/phase_b_n4.yaml | test RMSE 14.1151 A — converged | C-RMSD 18.9999 A | DockQ 0.063 succ 7.1% | converged | [outputs/_loop01_smoke/resfold_s1_4_20260524_053818/](../outputs/_loop01_smoke/resfold_s1_4_20260524_053818/) |
```

PASS:
- exit code 0
- Both eval lines (step 5 + step 10) contain `Centroid RMSE`, `DockQ`, `C-RMSD` substrings
- REGISTRY last row contains `C-RMSD 18.9999 A` and `DockQ 0.063 succ 7.1%`
- In-loop call site to the new `_run_test_eval` helper works (factoring did not break the live training path)

### 3. Re-eval Phase C best_model.pt
Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py --config configs/train/resfold/phase_c_n8600.yaml --eval_only --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt --output_dir outputs/resfold/phase_c_n8600_reeval_loop01
```

Output excerpt:
```
Loading checkpoint: outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt
  Loaded from step 45000

Model: ResFoldOneStep (stage1_only)
  Trunk params:     4,892,288 (42.1%)
  Denoiser params:  5,672,707 (48.8%)
  Atom-head params: 1,057,804 (9.1%)
  Total params:     11,622,799

======================================================================
Eval-only: scoring outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt on test split (N=100)
======================================================================
         >>> Test Centroid RMSE (100): 9.5607 A | DockQ: 0.1404 (succ 26.0%) | Atom RMSE: 9.7328 | C-RMSD: 15.9309 A
Finished: 2026-05-24 05:39:45
[registry] appended row to C:\Users\costa\src\tinyfold\experiments\REGISTRY.md
```

REGISTRY last row after re-eval:
```
| 2026-05-24 | resfold_s1_8K_20260524_053904 | resfold | config: configs/train/resfold/phase_c_n8600.yaml | test RMSE 9.5607 A — converged | C-RMSD 15.9309 A | DockQ 0.140 succ 26.0% | converged | [outputs/resfold/phase_c_n8600_reeval_loop01/resfold_s1_8K_20260524_053904/](../outputs/resfold/phase_c_n8600_reeval_loop01/resfold_s1_8K_20260524_053904/) |
```

Wall time: ~41 s (well under the 10-minute budget).

PASS:
- exit code 0
- Checkpoint loaded cleanly from step 45000 — no state-dict mismatch, no missing keys
- Eval line has `Centroid RMSE 9.5607 A` (inside 9.0-11.0 A green band; ~0.3 A below the 9.8693 A 3-seed reference, well within single-seed noise)
- Eval line has `DockQ: 0.1404 (succ 26.0%)` and `C-RMSD: 15.9309 A` (finite values)
- REGISTRY last row exists for `resfold_s1_8K_20260524_053904` (the reeval_loop01 run) and contains `C-RMSD 15.9309 A` and `DockQ 0.140 succ 26.0%`

### 4. Cleanup

`git status --short` after all runs:
```
 M experiments/REGISTRY.md       <- two new rows appended (smoke + re-eval), expected
 M scripts/train_resfold.py      <- loop source change
 M src/tinyfold/model/losses/__init__.py   <- loop source change
 M src/tinyfold/model/losses/mse.py        <- loop source change
 M src/tinyfold/training/registry_append.py <- loop source change
?? tests/unit/test_c_rmsd.py     <- loop new test
?? tests/unit/test_registry_append.py     <- loop new test
```

New output directories created (both expected per command spec):
- `outputs/_loop01_smoke/resfold_s1_4_20260524_053818/` (smoke artifacts: plots, split.json, train.log, best/final/best_train .pt)
- `outputs/resfold/phase_c_n8600_reeval_loop01/resfold_s1_8K_20260524_053904/` (re-eval artifacts: split.json, train.log)

Protected file check: `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt` has mtime `2026-05-24 04:14:00`, pre-dating the re-eval (05:39). Checkpoint was read-only (the `--eval_only` branch returns early before any `torch.save`).

No files outside the work folder, the two expected output dirs, the loop's source files, and `experiments/REGISTRY.md` were modified.

## Open concerns
- Re-eval Centroid RMSE came in at 9.5607 A versus the 9.8693 A reference from the 3-seed sweep — better than expected, but plausibly inside single-seed noise (the plan explicitly green-bands 9.0-11.0 A for single-seed runs). Worth noting in the loop summary but not a blocker.
- DockQ 0.140 (succ 26.0%) and C-RMSD 15.93 A on Phase C: these are the first ever recorded values for this checkpoint on the new metrics — there is no historical baseline to cross-check against. The unit-test `compute_c_rmsd` sanity checks (whole-complex rigid is ~0, chain-B-only-moved is in 4-7 A) give some confidence the function is mathematically sound, but a second independent implementation or a known PDB pair with a published DockQ would be the gold standard. Out of scope for this loop.
- Pre-existing test failure `tests/unit/test_losses.py::TestImports::test_all_geometry_imports` (noted in IMPLEMENTATION.md) was NOT re-run as part of this verification; it is unrelated to the loop's scope and the implementer correctly flagged it as a separate pre-existing issue.
- `experiments/REGISTRY.md` is now modified with two new rows from this loop's testing. The smoke-run row (resfold_s1_4_20260524_053818) is essentially a side-effect of running the test plan. If the committer wants a clean REGISTRY diff for the loop commit, they can either (a) accept both rows or (b) revert only the smoke row before committing. The re-eval row is the deliverable.
