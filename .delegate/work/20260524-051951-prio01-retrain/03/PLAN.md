# Loop 03: Boltz Kabsch-interpolation sampler (Task D)

## Overview

Phase C N=8600 multi-step Euler drifts ~2 A above one-shot
(`notes/post_compact_spec.md:91`, 12.59 A vs 10.59 A). We've already tried
"per-step Kabsch alignment of x0_pred to x" (`align_per_step=True` via
`diffusion.utils.kabsch_align_to_target` at `scripts/train_resfold.py:249`) —
that variant is in the C.3 ablation matrix and got 12.75 A, **worse**, not
better. Phase A also tried it and got worse (`notes/phase_a_findings.md:52`).

The Boltz-1 trick we have NOT tried: after each Euler step, rigid-align
`x_new` onto `x_prev` (the trajectory's previous state), not `x0_pred` onto
`x`. The frame of reference is anchored by the trajectory itself, not by the
predicted clean structure. This is the "Kabsch interpolation" variant
flagged at `notes/post_compact_spec.md:106` and gap-item #1 at
`notes/post_compact_spec.md:124`.

This loop:

1. Factors the Kabsch primitive into a stable `kabsch_rigid(src, tgt, mask)
   -> (R, t, aligned)` helper at `src/tinyfold/model/geometry/kabsch.py`.
   Both `losses/mse.py:kabsch_align` and `diffusion/utils.py:kabsch_align_to_target`
   currently re-implement the same SVD by hand — we collapse those into one
   primitive and refactor the existing two callers to use it. No behaviour
   change to those callers (regression-gated by the existing unit tests in
   `tests/unit/test_losses.py`).
2. Adds a `kabsch_interp: bool = False` arg to `sample_centroids_ve`
   (`scripts/train_resfold.py:189`). When True, AFTER the Euler step that
   computes `x = x + d * dt` (line 258), align that new `x` onto the
   pre-step `x_prev` and overwrite `x`. Applied BEFORE the optional
   `recenter` block and BEFORE the next loop iteration.
3. Exposes `--kabsch_interp` on the eval CLI; threads it through both eval
   sampling sites (in-loop eval at `scripts/train_resfold.py:1631` and the
   per-target loop in `_run_test_eval`).
4. Re-evals Phase C `best_model.pt` on N=8600 test split across four
   sampler configs (the headline matrix below). Two new REGISTRY rows
   land in this loop.

Headline question: does multi-step + kabsch_interp beat the one-shot
10.59 A? Success target = **<10.5 A** on N=8600 multi-step K=1.

Out of scope (explicit):
- Anything from Loops 04-07.
- ESM-2, confidence head, EDM weighting audit.
- Atom-head architecture changes; we only ensure the atom head sees the
  aligned centroids correctly via the existing `sample_centroids_ve` exit
  path (`scripts/train_resfold.py:270-277`).

## Design decisions (locked here, do not relitigate)

### D1. Where the Kabsch primitive lives

CREATE `src/tinyfold/model/geometry/kabsch.py` exporting one function:

```python
def kabsch_rigid(
    src: Tensor,                       # [B, N, 3]
    tgt: Tensor,                       # [B, N, 3]
    mask: Optional[Tensor] = None,     # [B, N] bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Find (R, t) minimising ||R @ src + t - tgt||^2 per batch element.

    Returns:
        R:       [B, 3, 3] proper rotation matrices (det=+1).
        t:       [B, 3]    translations.
        aligned: [B, N, 3] equal to torch.bmm(src, R.transpose(1,2)) + t[:, None, :].
    """
```

Implementation = exactly the SVD recipe used today in three places
(`losses/mse.py:50-58`, `diffusion/utils.py:54-66`, `losses/mse.py:201-211`).
Returning `(R, t, aligned)` lets call sites pick whichever piece they need:

- `losses/mse.py:kabsch_align` (used at training-time and by `compute_rmse`
  + `compute_mse_loss`): wants only the aligned tensor — adapter line.
- `diffusion/utils.py:kabsch_align_to_target` (used by existing
  `align_per_step` path): wants only the aligned tensor — adapter line.
- The new sampler path (this loop): wants `aligned` (and ignores R, t).
- `losses/mse.py:compute_c_rmsd` (Loop 01) explicitly needs `R` and `t` for
  the chain-A-then-apply-to-both-chains protocol — already split out. We
  optionally refactor it onto `kabsch_rigid` for consistency; if the diff
  risks the Loop 01 unit tests (`tests/unit/test_c_rmsd.py`), leave it as-is
  and document.

Existing tests that must still pass after the refactor (gate, not new
work):
- `tests/unit/test_losses.py`
- `tests/unit/test_c_rmsd.py`
- `tests/unit/test_geometry.py`
- `tests/test_pose_clustering.py` and `tests/test_multisample_eval.py`
  (Loop 02 didn't touch Kabsch, but the run-the-whole-test-suite gate is
  cheap).

### D2. What "align x_new to x_prev" means concretely

For batch element `b`, snapshot `x_prev = x.clone()` BEFORE the Euler
update (so `x_prev` is the input to that step's denoiser). After the line
`x = x + d * dt` (`scripts/train_resfold.py:258`) but BEFORE the optional
`recenter` block (line 261):

```python
if kabsch_interp:
    R, t, _ = kabsch_rigid(x, x_prev, mask)        # solve R @ x + t ~ x_prev
    x = torch.bmm(x, R.transpose(1, 2)) + t[:, None, :]
```

We align the FRESHLY-STEPPED `x` onto the PRE-STEP `x_prev`. The frame
of reference for the next denoiser call is therefore the same frame as the
current iteration's input — the denoiser sees a self-consistent rigid
frame across consecutive steps. Drift between steps is suppressed by
construction; the denoiser is free to refine internal structure within
that frame.

This is the inverse of the existing `align_per_step`: that one moves
`x0_pred` to `x`'s frame and is applied at line 249. Both can be set
together — they touch different lines and are conceptually orthogonal —
but the headline experiment is `kabsch_interp=True, align_per_step=False`
to isolate the new variable.

### D3. Recenter interaction

`recenter=True` currently subtracts the centroid of `x` each step
(`scripts/train_resfold.py:261-268`). The Kabsch alignment in D2 already
matches the centroid of `x_prev` (the rigid `(R, t)` includes translation),
so `recenter` becomes a no-op on the kabsch_interp branch and we can leave
`recenter` plumbing untouched. We do NOT re-test all combinations of
`recenter x kabsch_interp`; the headline matrix below fixes `recenter` to
match the Loop 01 / Loop 02 baseline (which used the Phase C config defaults
— `recenter` is False in `configs/train/resfold/phase_c_n8600.yaml`).

### D4. CLI surface

Add ONE new flag to `parse_args`:

```python
parser.add_argument("--kabsch_interp", action="store_true",
    help="Boltz Kabsch-interpolation sampler: after each Euler step, "
         "rigid-align x_new onto the previous step's x. Targets the "
         "+2 A multi-step drift documented in notes/phase_a_findings.md.")
```

Default `False` preserves byte-identical behaviour vs Loop 02's
`cc5abdf`. Plumbed into the three call sites where `sample_centroids_ve`
is invoked:

- `scripts/train_resfold.py:769` (eval body, single-sample path)
- `scripts/train_resfold.py:363` (`sample_k_centroids` multi-sample path)
- `scripts/train_resfold.py:1631` (in-training eval) AND
  `scripts/train_resfold.py:1684` (plot-path eval)

Each call adds `kabsch_interp=args.kabsch_interp` alongside the existing
`align_per_step=args.align_per_step, recenter=args.recenter`. The
`sample_k_centroids` signature gains a `kabsch_interp: bool = False`
kwarg forwarded to `sample_centroids_ve`.

### D5. Atom head: does kabsch_interp also need to apply to atoms?

`sample_centroids_ve` only diffuses centroids. For onestep models, the
atom head runs once at the END of the loop (`scripts/train_resfold.py:270-277`)
on the final `x` (the centroid). The atom head reads its input centroid
and produces per-residue `[L, 4, 3]` offsets in the centroid's frame — atoms
move with the centroid by construction.

Concretely: there is NO mid-loop atom prediction to align. The final atom
head call sees the post-final-step `x` (which is the kabsch-aligned
centroid trajectory's terminal state) and writes atoms in THAT frame. No
extra Kabsch needed on the atoms; the existing line at 273-276 is correct
as-is once `x` is the aligned centroid.

ONE thing to verify by reading (zero code change expected): the final
forward call uses `x` (line 273), which is the same `x` we mutated inside
the loop. Confirmed at `scripts/train_resfold.py:273-276` — yes, `x` is
shadowed inside the loop and the post-loop read picks up the aligned final
value. Document this in a comment near the new kabsch_interp block so the
next reader doesn't re-ask the question.

### D6. Self-conditioning interaction

`self_cond=True` passes `x0_prev` (the previous step's predicted x0) into
the next forward. With kabsch_interp, the `x0_prev` is detached BEFORE
the Kabsch alignment of x (line 252: `x0_prev = x0_pred.detach()`), so
self-cond still sees x0 in the pre-alignment frame. That's intentional:
self-cond's purpose is to give the denoiser a hint about clean structure,
not about frame consistency. We do not touch the self-cond plumbing.

### D7. Headline re-eval matrix

Four cells. The two `kabsch_interp=True` rows are the new REGISTRY entries
(the `False` rows already exist from Loop 01 and Loop 02):

| K | Sampler                                  | kabsch_interp | Source     |
|---|------------------------------------------|---------------|------------|
| 1 | `sample_centroids_ve` (T=50, default)    | False         | EXISTING — Loop 01 `53333ea` row, 9.56 A |
| 1 | `sample_centroids_ve` (T=50, default)    | **True**      | NEW Row A (this loop) |
| 5 | `sample_centroids_ve` + cluster-then-rank | False         | EXISTING — Loop 02 `cc5abdf` row, K=5 |
| 5 | `sample_centroids_ve` + cluster-then-rank | **True**      | NEW Row B (this loop) |

Success interpretation:
- Row A centroid RMSE **< 10.5 A** = Boltz trick beats one-shot 10.59 A;
  Phase D (Loop 07) ships with `kabsch_interp=true`.
- Row A centroid RMSE **>= 10.5 A but < 9.56 A** = improvement vs Loop 01
  multi-step baseline; still ship in Phase D for the multi-sample story.
- Row A **>= 9.56 A** = Boltz trick doesn't fix the drift; Phase D config
  falls back to `one_shot_sample: true` (the path Phase C already uses,
  per `configs/train/resfold/phase_c_n8600.yaml:48`). Document under
  "what didn't work" in `notes/post_compact_spec.md` per the TASK
  acceptance criterion (line 42).
- Row B vs Row A tells us whether kabsch_interp also helps the
  cluster-then-rank ranked@5 metric (independent of headline RMSE).

We pick K=5 (not K=40) for Row B because K=40 already exists in Loop 02
and the marginal cost of repeating it here is high relative to the signal
— ranked@5 vs ranked@40 in Loop 02 only moved from 9.895 to 9.874 A.

## Tasks

### Task 1: Factor `kabsch_rigid` primitive

**Goal:** Single `(R, t, aligned)` helper used by all three current Kabsch
sites. No behaviour change for existing callers.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `src/tinyfold/model/geometry/__init__.py` |
| CREATE | `src/tinyfold/model/geometry/kabsch.py` |
| MODIFY | `src/tinyfold/model/losses/mse.py` (`kabsch_align` at line 12; do NOT touch `compute_c_rmsd` unless its unit tests still pass — see D1) |
| MODIFY | `src/tinyfold/model/diffusion/utils.py` (`kabsch_align_to_target` at line 7) |

**Steps:**
1. Write `kabsch_rigid(src, tgt, mask=None)` per D1. Body = the SVD code
   currently at `losses/mse.py:33-58` generalised to take an explicit
   `tgt` (rather than `pred`/`target` directional naming). Return
   `(R, t, aligned)` where:
   - `R: [B, 3, 3]` solves `R @ src_centered ~ tgt_centered` (proper
     rotation, det enforced via the standard sign-flip-D matrix trick).
   - `t: [B, 3]` = `tgt_mean - R @ src_mean`.
   - `aligned: [B, N, 3]` = `torch.bmm(src, R.transpose(1,2)) + t[:, None, :]`.
   - Mask handling matches existing code: zero out masked entries before
     SVD, divide centroids by `n_valid.clamp(min=1)`.
2. Create `src/tinyfold/model/geometry/__init__.py` exporting
   `kabsch_rigid`.
3. Refactor `losses/mse.py:kabsch_align` to delegate:
   ```python
   def kabsch_align(pred, target, mask=None):
       _, _, aligned = kabsch_rigid(pred, target, mask)
       # Centred target for back-compat — old API returned (aligned, target_c).
       target_mean = _masked_mean(target, mask)
       target_c = (target - target_mean)
       if mask is not None:
           target_c = target_c * mask.unsqueeze(-1).float()
       # Old kabsch_align ALSO subtracted pred_mean (returned aligned in
       # zero-centred frame, NOT in target's translated frame). Preserve
       # exactly: subtract target_mean from `aligned` before returning.
       return aligned - target_mean, target_c
   ```
   IMPORTANT: read `losses/mse.py:42-61` again before writing this — the
   old function returns `pred_aligned, target_centered`, both centred at
   the origin. `kabsch_rigid`'s `aligned` is in `tgt`'s **translated**
   frame. The adapter must subtract `target_mean` to restore the centred
   convention. If this gets fiddly, accept the alternative of keeping
   `kabsch_align`'s body untouched and ONLY using `kabsch_rigid` in the
   new sampler path — the refactor is a nice-to-have, the sampler change
   is the deliverable.
4. Refactor `diffusion/utils.py:kabsch_align_to_target` to delegate:
   ```python
   def kabsch_align_to_target(pred, target, mask=None):
       _, _, aligned = kabsch_rigid(pred, target, mask)
       return aligned   # already in target's translated frame, matches old behaviour
   ```
5. Re-run `pytest tests/unit/test_losses.py tests/unit/test_c_rmsd.py
   tests/unit/test_geometry.py -v` — must be green.

**Verify:** the three existing call sites import the new primitive (grep
the diff for `kabsch_rigid`); both Loop 01 and Loop 02 tests still pass.

---

### Task 2: `kabsch_interp` in `sample_centroids_ve`

**Goal:** Add the Boltz trick. Default-off; opt-in via kwarg.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` (`sample_centroids_ve` at line 188-279; `sample_k_centroids` at line 282-377) |

**Steps:**
1. Add `kabsch_interp: bool = False` kwarg to `sample_centroids_ve`
   (insert after `recenter`, before `self_cond`). Update the docstring
   `Args` block with one sentence pointing at this PLAN.
2. Inside the Euler loop body (`scripts/train_resfold.py:232-268`):
   - Snapshot `x_prev = x.clone()` at the TOP of the loop, BEFORE the
     denoiser forward (`scripts/train_resfold.py:240`). The clone is
     necessary because `x` is rebound at line 258.
   - After line 258 (`x = x + d * dt`) and BEFORE the optional `recenter`
     block at line 261, insert:
     ```python
     if kabsch_interp:
         # Boltz Kabsch-interpolation: align the freshly-stepped x onto
         # x_prev so consecutive denoiser inputs share a rigid frame.
         # See .delegate/work/20260524-051951-prio01-retrain/03/PLAN.md D2.
         from tinyfold.model.geometry import kabsch_rigid
         _, _, x = kabsch_rigid(x, x_prev, mask)
     ```
     (Import is local to avoid changing the top-of-file imports; or hoist
     it to the import block at line 58-69 — picker's choice; consistent
     with existing imports there.)
3. Confirm the `is_onestep` post-loop atom-head call (line 273-276) reads
   the now-aligned `x` — no code change needed but add a one-line comment
   near it pointing at PLAN D5.
4. Forward the new kwarg through `sample_k_centroids` (line 282-377):
   - Add `kabsch_interp: bool = False` kwarg to its signature (insert
     after `align_per_step`/`recenter`).
   - Pass `kabsch_interp=kabsch_interp` into the
     `sample_centroids_ve(...)` call at line 363-367.

**Verify:** unit test below (Task 4) passes; running with `kabsch_interp=False`
must produce byte-identical centroids vs Loop 02's `cc5abdf` for a fixed
seed (regression-gated by Task 4's identity-denoiser smoke test).

---

### Task 3: Plumb `--kabsch_interp` through the eval CLI

**Goal:** One CLI flag → all four `sample_centroids_ve` callers.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` (`parse_args` around line 618-621; four call sites listed below) |

**Steps:**
1. Add the flag in `parse_args` next to `--align_per_step` (after line
   621):
   ```python
   parser.add_argument("--kabsch_interp", action="store_true",
       help="Boltz Kabsch-interpolation sampler: after each Euler step, "
            "rigid-align x_new onto the previous step's x.")
   ```
2. Plumb into the four call sites — at each site, add
   `kabsch_interp=args.kabsch_interp` to the kwargs:
   - `_run_test_eval` single-sample path: `scripts/train_resfold.py:769`
   - `_run_test_eval` multi-sample path: pass through `sample_k_centroids`
     at `scripts/train_resfold.py:725-730` (add the kwarg there) which
     forwards to `sample_centroids_ve` inside `sample_k_centroids`.
   - In-training eval body: `scripts/train_resfold.py:1631-1636`
   - Plot-path eval: `scripts/train_resfold.py:1684-1689`
3. Add a one-line log at the top of `_run_training` so the train.log
   records which sampler variant was used, near line 1156-1158:
   ```python
   if args.kabsch_interp:
       logger.log(f"  kabsch_interp: True (Boltz trajectory-frame alignment)")
   ```

**Verify:** `python scripts/train_resfold.py --help | grep kabsch_interp`
shows the new flag; a no-flag run produces the exact same REGISTRY row as
before this loop.

---

### Task 4: Unit + integration tests

**Goal:** Three behavioural claims, three small tests.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `tests/unit/test_kabsch_rigid.py` |
| CREATE | `tests/test_kabsch_interp_sampler.py` |

**Tests:**

1. **Unit — `kabsch_rigid` correctness**
   (`tests/unit/test_kabsch_rigid.py`):
   ```
   - Build src = torch.randn(2, 50, 3); pick random R (via the existing
     random_rotation_matrix util in tinyfold.training.utils), random t in
     [-5, 5]. tgt = (R @ src.T).T + t.
   - R_hat, t_hat, aligned = kabsch_rigid(src, tgt).
   - assert torch.allclose(R_hat @ R_hat.T, eye(3), atol=1e-5)   # orthonormal
   - assert torch.allclose(torch.det(R_hat), torch.tensor(1.0), atol=1e-5)
   - assert torch.allclose(aligned, tgt, atol=1e-5)
   - With mask=ones, must match the no-mask path.
   - With a partial mask (half True / half False) and src/tgt that AGREE
     on masked positions but disagree on unmasked ones, the recovered R
     should still align the unmasked subset to ~1e-5.
   ```
2. **Unit — kabsch_interp identity-denoiser invariance**
   (`tests/test_kabsch_interp_sampler.py`):
   ```
   - Stub denoiser: a torch.nn.Module whose forward_sigma returns x0_pred
     = x (so each Euler step has direction d = (x - x) / sigma = 0 and
     dt has no effect; x stays put across all steps).
   - With kabsch_interp=True, snapshot x BEFORE the first iteration and
     AFTER every iteration; assert each post-iter x is within 1e-4 of
     the pre-iter x (no drift introduced by the Kabsch step itself when
     the trajectory is already at a fixed point).
   - With kabsch_interp=True and a denoiser that returns x0_pred = ROT @ x
     for a fixed small rotation ROT (~5 deg), assert that after kabsch
     alignment ||x_new - x_prev|| stays bounded (i.e. the alignment is
     fighting the rotation, not amplifying it).
   ```
3. **Integration — 4-target K=1 dry run with --kabsch_interp**
   (`tests/test_kabsch_interp_sampler.py`, `@pytest.mark.slow`,
   skip-if-checkpoint-missing):
   ```
   - Invoke the script as subprocess against the Phase C checkpoint and a
     4-target test slice:
       python scripts/train_resfold.py \
           --config configs/train/resfold/phase_c_n8600.yaml \
           --eval_only \
           --checkpoint outputs/resfold/phase_c_n8600/.../best_model.pt \
           --n_test 4 \
           --kabsch_interp \
           --output_dir <tmp>/kabsch_interp_smoke
   - Parse the REGISTRY row; assert centroid RMSE is finite and < 20 A
     (sanity, not an accuracy gate).
   - ALSO invoke WITHOUT --kabsch_interp on the same 4 targets and assert
     the two RMSEs DIFFER by > 1e-3 A — the flag is actually changing the
     trajectory.
   - Skip cleanly if the checkpoint isn't on disk (CI-safe).
   ```

**Verify:** `pytest tests/unit/test_kabsch_rigid.py
tests/test_kabsch_interp_sampler.py -v` green locally; full
`pytest tests/ -v` still green (no regressions).

---

### Task 5: Headline re-eval (Row A + Row B)

**Goal:** Two new REGISTRY rows; the answer to "does Boltz fix multi-step?"

**Files:**
| Action | Path |
|--------|------|
| READ-ONLY | `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt` |
| WRITE | `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k1/...` |
| WRITE | `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k5/...` |
| APPEND | `experiments/REGISTRY.md` |

**Commands (run both; the K=1 one is the headline):**

```powershell
# Row A: K=1 multi-step + kabsch_interp (HEADLINE)
python scripts/train_resfold.py `
    --config configs/train/resfold/phase_c_n8600.yaml `
    --eval_only `
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
    --kabsch_interp `
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k1

# Row B: K=5 multi-step + kabsch_interp + cluster-then-rank
python scripts/train_resfold.py `
    --config configs/train/resfold/phase_c_n8600.yaml `
    --eval_only `
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
    --kabsch_interp `
    --n_samples 5 `
    --eval_K_list 1,5 `
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k5
```

Notes for the runner:
- The Phase C config has `one_shot_sample: true`. We OVERRIDE it
  implicitly by not setting `--one_shot_sample` from CLI: argparse
  config-then-CLI ordering... wait, actually the config sets the default
  via `parser.set_defaults(**filtered)` at `scripts/train_resfold.py:661`,
  so the YAML default WINS unless the CLI overrides. Add an explicit
  override flag — there is no `--no-one_shot_sample`; the cleanest fix is
  to write a one-off override config:
  `configs/train/resfold/phase_c_n8600_multistep.yaml` that copies
  `phase_c_n8600.yaml` and sets `one_shot_sample: false`. Use that
  config in both commands. Loop 02 hit this same issue — check its
  TEST.md before assuming.
- Phase C checkpoint MUST stay intact (TASK constraint line 33). Both
  commands write to fresh timestamped subdirs.
- The four-cell matrix wants Row A to compare cleanly against the Loop
  01 multi-step row (9.56 A from `phase_c_n8600_reeval_loop01` —
  REGISTRY row dated 2026-05-24). Confirm in your eyeball-diff that the
  Loop 01 row also disabled `one_shot_sample`; if not, the apples-to-apples
  baseline is Row A vs Loop 01's row, with both running multi-step VE.

**Verify:** two new REGISTRY rows; Row A's outcome cell contains the
single number to compare against 10.59 A (one-shot) and 9.56 A (Loop 01
multi-step). Capture the number in IMPLEMENTATION.md and forward to the
spec update task in Loop 07 / final retrain.

---

### Task 6: Document outcome in notes/

**Goal:** Whichever way Row A lands, leave the next reader a one-line
answer.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `notes/post_compact_spec.md` (the C.3 ablation table at line 83-99 and the "Things NOT to retry" section at line 216-) |

**Steps:**
1. After Row A lands, append ONE new row to the C.3 ablation block:
   ```
   multi-step Euler T=50 + Kabsch-interp  centroid=<RowA_value> A  ← Loop 03
   ```
2. If Row A >= 10.59 A (Boltz trick didn't help): append to the
   "Things NOT to retry" section:
   ```
   - Boltz Kabsch-interpolation sampler (kabsch_interp=True). Tested in
     Loop 03; RMSE <RowA_value> A vs one-shot 10.59 A. Phase D ships
     with one_shot_sample=true. See REGISTRY 2026-05-24 row
     resfold_s1_8K_phase_c_n8600_reeval_loop03_kabsch_k1.
   ```
3. If Row A < 10.59 A: append to the "What worked / next steps" section
   instead (the Phase A findings doc — `notes/phase_a_findings.md:56`
   open-questions block has the right home), and mark the Phase D config
   to use `--kabsch_interp`.

**Verify:** the spec doc reflects the experimental answer; Loop 07 can
read it to decide its sampler config (`notes/post_compact_spec.md`'s
"Final retrain" section gates on this).

## Acceptance Criteria

- [ ] `src/tinyfold/model/geometry/kabsch.py` exists exporting
      `kabsch_rigid(src, tgt, mask) -> (R, t, aligned)`.
- [ ] `kabsch_align` (losses) and `kabsch_align_to_target` (diffusion)
      delegate to `kabsch_rigid` OR remain unchanged with a comment
      pointing to the new primitive (Task 1 step 3 decision).
- [ ] `sample_centroids_ve` accepts `kabsch_interp: bool = False`; when
      True, the freshly-stepped `x` is rigid-aligned onto `x_prev`
      BEFORE the next iteration.
- [ ] `--kabsch_interp` CLI flag added to `scripts/train_resfold.py`;
      threaded through all four `sample_centroids_ve` call sites and
      through `sample_k_centroids`.
- [ ] Default-off behaviour: with `--kabsch_interp` NOT set, the script
      produces a REGISTRY row whose outcome cell is structurally
      identical to Loop 02's row (no behavioural change to the existing
      sampler path).
- [ ] `pytest tests/unit/test_kabsch_rigid.py
      tests/test_kabsch_interp_sampler.py -v` passes locally.
- [ ] Full `pytest tests/ -v` is green (Loop 01 + Loop 02 tests still
      pass after the Kabsch refactor).
- [ ] Phase C re-eval Row A appended to `experiments/REGISTRY.md`
      (K=1 multi-step + kabsch_interp); writes to
      `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k1/` and
      does NOT touch the source checkpoint dir.
- [ ] Phase C re-eval Row B appended to `experiments/REGISTRY.md`
      (K=5 multi-step + kabsch_interp + cluster-then-rank); writes to
      `outputs/resfold/phase_c_n8600_reeval_loop03_kabsch_k5/`.
- [ ] `notes/post_compact_spec.md` updated with the Row A number in the
      C.3 ablation block and the verdict logged in either the
      "what didn't work" or the "what worked" section per D7.
