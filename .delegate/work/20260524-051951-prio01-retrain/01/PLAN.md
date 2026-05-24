# Loop 01 PLAN — DockQ + C-RMSD eval wiring

## Goal

Wire `compute_dockq` and a new `compute_c_rmsd` (Kabsch-align chain A only, then
score all CA) into the `stage1_only` onestep eval branch of
`scripts/train_resfold.py`, surface both numbers in the registry outcome string,
and re-eval Phase C `best_model.pt` (N=8600) so a new REGISTRY row reports the
two new metrics alongside the existing 9.8693 A centroid RMSE.

## Files touched

- `src/tinyfold/model/losses/mse.py` (MODIFY) — add `compute_c_rmsd` next to the
  existing `kabsch_align` / `compute_rmse` helpers it reuses.
- `src/tinyfold/model/losses/__init__.py` (MODIFY) — re-export `compute_c_rmsd`
  so it can be imported via `tinyfold.model.losses` (consistent with
  `compute_rmse`, `kabsch_align`).
- `scripts/train_resfold.py` (MODIFY) — (1) add DockQ + C-RMSD accumulation to
  the `stage1_only` onestep test-eval branch (currently only the
  `stage2_only` / `end_to_end` branches log DockQ); (2) extend the printed
  summary line; (3) thread the new metrics into the REGISTRY outcome string;
  (4) add `--eval_only` and `--checkpoint` re-eval flags so we can score the
  frozen Phase C ckpt without retraining.
- `src/tinyfold/training/registry_append.py` (MODIFY) — extend
  `append_registry_row` with optional `dockq_avg`, `dockq_success_pct`,
  `c_rmsd` kwargs; when provided, append `" | DockQ {x:.3f} (succ {y:.1f}%) | C-RMSD {z:.2f} A"`
  to the existing outcome cell so old rows stay readable (no schema change to
  the markdown header).
- `tests/unit/test_c_rmsd.py` (CREATE) — synthetic 2-chain CA tensor with
  chain B rigidly offset by a known rotation+translation; assert
  `compute_c_rmsd` returns < 1e-4 (chain-A-aligned, transform applied to B).
- `tests/unit/test_registry_append.py` (CREATE) — round-trip the new optional
  kwargs through `append_registry_row` against a tmp_path REGISTRY.md and
  assert the appended row contains the DockQ + C-RMSD substrings.

We are NOT touching `src/tinyfold/model/registry.py` (the model factory the
LOOPS.yaml accidentally names) — the actual registry writer is
`src/tinyfold/training/registry_append.py`. The plan corrects that pointer.

## Tasks (in execution order)

### Task 1: add `compute_c_rmsd` to `losses/mse.py`

**File:** `src/tinyfold/model/losses/mse.py` (append after line 145, right after
`compute_rmse`)

**Change:**
```python
def compute_c_rmsd(
    pred_ca: Tensor,        # [B, L, 3] predicted CA (or centroid) coords
    gt_ca: Tensor,          # [B, L, 3] ground-truth CA coords
    chain_ids: Tensor,      # [B, L] long, values in {0, 1}; 0 == chain A
    mask: Optional[Tensor] = None,  # [B, L] bool, valid residues
) -> Tensor:
    """Complex-RMSD: Kabsch-align pred chain A onto GT chain A, apply that
    SINGLE rigid transform to the full prediction (both chains), then return
    RMSD over all valid CA.

    This measures inter-chain placement: identical to per-chain RMSD if
    chain B sits where GT says it does after chain A is aligned, and large
    when the predicted complex has the right chains but the wrong relative
    pose.

    Implementation notes:
        - Reuses `kabsch_align` for the chain-A SVD; we need R and t
          explicitly so we recompute them inline from the same H = pred_c^T @ gt_c
          decomposition (see kabsch_align, lines 50-58).
        - Requires at least 3 valid chain-A CA per batch element; otherwise
          falls back to plain Kabsch over the whole complex (logged once).
    """
    assert pred_ca.shape == gt_ca.shape, \
        f"shape mismatch {pred_ca.shape} vs {gt_ca.shape}"
    assert chain_ids.shape == pred_ca.shape[:2], \
        f"chain_ids shape {chain_ids.shape} != {pred_ca.shape[:2]}"

    B, L, _ = pred_ca.shape
    device = pred_ca.device

    # Chain A mask (chain id == 0), AND'd with validity mask
    chain_a = (chain_ids == 0)
    if mask is not None:
        chain_a = chain_a & mask.bool()
        full_mask = mask.bool()
    else:
        full_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Guard: need >=3 chain-A residues per batch element for a stable Kabsch.
    n_a = chain_a.sum(dim=1)
    assert (n_a >= 3).all(), \
        f"compute_c_rmsd needs >=3 chain-A residues per sample, got {n_a.tolist()}"

    # --- Per-batch chain-A Kabsch fit (R, t) ---
    # Centroids over chain-A only.
    chain_a_f = chain_a.unsqueeze(-1).float()
    n_a_exp = chain_a_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1,1]
    pred_a_mean = (pred_ca * chain_a_f).sum(dim=1, keepdim=True) / n_a_exp
    gt_a_mean   = (gt_ca   * chain_a_f).sum(dim=1, keepdim=True) / n_a_exp

    pred_a_c = (pred_ca - pred_a_mean) * chain_a_f
    gt_a_c   = (gt_ca   - gt_a_mean)   * chain_a_f

    # H = pred_a_c^T @ gt_a_c  -> SVD -> R that maps pred onto gt.
    H = torch.bmm(pred_a_c.transpose(1, 2), gt_a_c)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d
    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # [B,3,3]
    # Translation so that R @ pred_a_mean + t = gt_a_mean.
    t = gt_a_mean.squeeze(1) - torch.bmm(pred_a_mean, R.transpose(1, 2)).squeeze(1)  # [B,3]

    # --- Apply (R, t) to FULL prediction (both chains) ---
    pred_aligned = torch.bmm(pred_ca, R.transpose(1, 2)) + t.unsqueeze(1)  # [B,L,3]

    # --- RMSD over all valid CA (both chains) ---
    sq_diff = ((pred_aligned - gt_ca) ** 2).sum(dim=-1)  # [B,L]
    full_mask_f = full_mask.float()
    n_valid = full_mask_f.sum().clamp(min=1.0)
    rmsd = torch.sqrt((sq_diff * full_mask_f).sum() / n_valid)
    return rmsd
```

**Rationale:** We cannot just call `kabsch_align(pred, gt, mask=chain_a)`
because that function only returns the *aligned predicted slice* — it discards
R and t. C-RMSD needs the rigid transform applied to the *other* chain too, so
we inline the same SVD recipe (mirroring lines 50-58 of `kabsch_align`) and
keep `R, t` explicit. Reusing `kabsch_align` would otherwise force a second
pass with a workaround mask, which is more code than copying 8 lines of SVD.

### Task 2: re-export from losses package

**File:** `src/tinyfold/model/losses/mse.py` — already covered by Task 1.

**File:** `src/tinyfold/model/losses/__init__.py:11-17` (the `from .mse import (...)` block)

**Change:**
```python
from .mse import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_c_rmsd,                  # <-- new
    compute_relative_distance_loss,
    compute_distance_consistency_loss,
)
```
And add `"compute_c_rmsd"` to `__all__` (line 47, after `"compute_rmse"`).

**Rationale:** Mirrors the existing `compute_rmse` export so the train script
can `from tinyfold.model.losses import compute_c_rmsd` in line with the other
loss-symbol imports at `scripts/train_resfold.py:72-80`.

### Task 3: wire DockQ + C-RMSD into the stage1_only onestep eval branch

**File:** `scripts/train_resfold.py:72-80` (add `compute_c_rmsd` to the losses
import block)

```python
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_c_rmsd,                  # <-- new
    compute_distance_consistency_loss,
    ...
)
```

**File:** `scripts/train_resfold.py:1255-1299` (the `test_*_scores` setup +
`stage1_only` branch)

**Change:** add `test_c_rmsds = []` next to `test_atom_rmses = []` (line 1259).
Inside the `if args.mode == "stage1_only":` branch, after the existing
`atom_rmse` block (line 1299), append:

```python
# --- NEW: DockQ + C-RMSD for stage1_only onestep ---
# C-RMSD needs centroids only (works regardless of is_onestep).
n_res = s['n_res']
c_rmsd = compute_c_rmsd(
    pred_ca=centroids_pred[:, :n_res],
    gt_ca=batch['centroids'][:, :n_res],
    chain_ids=batch['chain_ids'][:, :n_res],
    mask=batch['mask_res'][:, :n_res],
).item() * s['std']
test_c_rmsds.append(c_rmsd)

# DockQ only when we have atom-level predictions (i.e. onestep).
if is_onestep and atoms_pred_onestep is not None:
    pred_coords_res = atoms_pred_onestep[0, :n_res]   # [L, 4, 3]
    gt_coords_res   = batch['coords_res'][0, :n_res]
    dockq_result = compute_dockq(
        pred_coords_res, gt_coords_res,
        batch['aa_seq'][0, :n_res], batch['chain_ids'][0, :n_res],
        std=s['std'],
    )
    if dockq_result['dockq'] is not None:
        test_dockq_scores.append(dockq_result['dockq'])
```

**File:** `scripts/train_resfold.py:1365-1379` (the summary `log_msg` block)

After the existing `Atom RMSE` append, add:

```python
dockq_success_pct = None
if test_dockq_scores:
    dockq_avg = sum(test_dockq_scores) / len(test_dockq_scores)
    dockq_success_pct = 100.0 * sum(1 for d in test_dockq_scores if d >= 0.23) / len(test_dockq_scores)
    log_msg += f" | DockQ: {dockq_avg:.4f} (succ {dockq_success_pct:.1f}%)"
if test_c_rmsds:
    c_rmsd_avg = sum(test_c_rmsds) / len(test_c_rmsds)
    log_msg += f" | C-RMSD: {c_rmsd_avg:.4f} A"
```

Note: the existing DockQ line at 1369 only logged the average; we replace it
with the version that also reports the >=0.23 success rate. Be careful not to
double-print DockQ — re-use the *one* `if test_dockq_scores:` block, do not
add a second one.

**Rationale:** The Phase C config (`configs/train/resfold/phase_c_n8600.yaml`)
runs `mode: stage1_only` with `model_kind: onestep`, so the onestep path
already emits `atoms_pred_onestep` at line 1279 — DockQ has the inputs it
needs, it just wasn't wired. C-RMSD only needs centroids and chain IDs which
are unconditionally available, so we compute it for every test sample.

### Task 4: surface the new metrics in the REGISTRY outcome cell

**File:** `src/tinyfold/training/registry_append.py:19-100`

**Change:** extend the function signature and the outcome-cell formatting:

```python
def append_registry_row(
    run_name: str,
    model: str,
    config_path: Optional[str],
    final_metric: Optional[float],
    outcome: str,
    registry_path: Optional[Path] = None,
    output_dir: Optional[str] = None,
    dockq_avg: Optional[float] = None,            # <-- new
    dockq_success_pct: Optional[float] = None,    # <-- new
    c_rmsd: Optional[float] = None,               # <-- new
) -> Path:
    ...
    if final_metric is not None:
        outcome_cell = f"test RMSE {final_metric:.4f} A — {outcome_clean}"
    else:
        outcome_cell = outcome_clean

    # NEW: append the loop-01 metrics if provided. Old callers pass None and
    # the cell looks identical to today's rows (back-compat).
    extras = []
    if c_rmsd is not None:
        extras.append(f"C-RMSD {c_rmsd:.4f} A")
    if dockq_avg is not None:
        succ = f" succ {dockq_success_pct:.1f}%" if dockq_success_pct is not None else ""
        extras.append(f"DockQ {dockq_avg:.3f}{succ}")
    if extras:
        outcome_cell = outcome_cell + " | " + " | ".join(extras)
```

**File:** `scripts/train_resfold.py:1445-1471` — capture `dockq_avg`,
`dockq_success_pct`, `c_rmsd_avg` into `progress` so the outer `finally:`
block can read them:

```python
if test_avg < best_rmse:
    best_rmse = test_avg
    progress["best_rmse"] = best_rmse
    # NEW: stash the latest values from the best-eval step.
    if test_dockq_scores:
        progress["dockq_avg"] = sum(test_dockq_scores) / len(test_dockq_scores)
        progress["dockq_success_pct"] = 100.0 * sum(
            1 for d in test_dockq_scores if d >= 0.23
        ) / len(test_dockq_scores)
    if test_c_rmsds:
        progress["c_rmsd"] = sum(test_c_rmsds) / len(test_c_rmsds)
    ...
```

**File:** `scripts/train_resfold.py:1505-1527` (the `finally:` block)

```python
progress = {"best_rmse": float('inf')}  # leave as-is
...
finally:
    best = progress.get("best_rmse", float('inf'))
    final_metric = best if best != float('inf') else None
    try:
        registry_path = append_registry_row(
            run_name=run_name,
            model="resfold",
            config_path=getattr(args, "config", None),
            final_metric=final_metric,
            outcome=outcome,
            output_dir=getattr(args, "output_dir", None),
            dockq_avg=progress.get("dockq_avg"),                 # <-- new
            dockq_success_pct=progress.get("dockq_success_pct"), # <-- new
            c_rmsd=progress.get("c_rmsd"),                       # <-- new
        )
```

**Rationale:** We extend the *outcome cell* instead of adding markdown columns
because the existing header at `experiments/REGISTRY.md:7` (`| Date | Run |
Model | What was tried | Outcome | Why stopped | Files |`) is shared with 36
historical rows. Adding new columns would visually break every row before
2026-05-24. Keyword-style metric tokens (`| DockQ 0.42 succ 35.0%`) in the
existing "Outcome" cell stay grep-friendly and don't widen the table.

### Task 5: add `--eval_only` / `--checkpoint` re-eval mode

**File:** `scripts/train_resfold.py` — args block (after line 538) and the
training-loop top (`_run_training`, around line 810).

**Change (args):**
```python
parser.add_argument("--eval_only", action="store_true",
                    help="Skip training; load --checkpoint and run one eval pass on the "
                         "test split, then write a REGISTRY row with the metrics.")
```
(`--checkpoint` already exists at line 504.)

**Change (training loop):** at the very top of `_run_training` after the model
and dataloaders are built but before the optimizer step loop starts, branch
on `args.eval_only`. The cleanest seam is right before the `for step in range
(1, args.n_steps + 1):` loop (skim around lines 810-830 for the exact line):

```python
if args.eval_only:
    assert args.checkpoint is not None, "--eval_only requires --checkpoint"
    # Force the eval block to run once: short-circuit by setting step
    # to args.eval_every and breaking after the first eval. Simpler:
    # extract the test-eval body of `if step % args.eval_every == 0:`
    # (lines 1254-1379) into a helper `_run_test_eval(...)` and call it
    # here directly, populating `progress` exactly as the in-loop eval does.
    _run_test_eval(model, test_samples, test_indices, noiser, eval_sampler,
                   device, args, is_onestep, progress, logger)
    return  # falls through to main()'s finally: -> REGISTRY append
```

The implementer should pull the eval body (lines ~1255-1379) into
`_run_test_eval(...)` and call it from both the in-loop check and the new
`eval_only` branch. The function MUST update `progress` with `best_rmse`,
`dockq_avg`, `dockq_success_pct`, `c_rmsd` so the unchanged `finally:` block
writes the right REGISTRY row.

**Rationale:** Re-eval is required by the TASK ("Re-eval Phase C N=8600
best_model.pt"). Without a flag the only way to score the existing ckpt is to
re-run 50k training steps, which would (a) take ~2.5 h and (b) overwrite the
checkpoint that TASK.md explicitly protects ("Keep best_model.pt from
outputs/resfold/phase_c_n8600/... intact"). Factoring the eval body into a
helper is the smallest surface area that satisfies the constraint.

## Tests

- `pytest tests/unit/test_c_rmsd.py -v`
  - Construct `gt_ca`: chain A = 10 random points, chain B = 10 random points.
  - Construct `pred_ca`: chain A copied verbatim; chain B = chain B rotated
    by 30° around z + translated by [5, -2, 3].
  - Assert `compute_c_rmsd(pred, gt, chain_ids)` returns < 1e-4 *only when*
    chain B follows the same rigid transform applied to chain A → in this
    test, the predicted complex is *wrong* (chain B moved), so assert the
    returned RMSD is approximately `sqrt(mean(||B_moved - B_gt||^2))` over
    chain B (specifically > 4 A and < 7 A). Then build a second `pred`
    where chain A AND chain B are jointly rotated/translated by the same R,t
    (i.e. the whole complex is moved as one rigid body) and assert RMSD < 1e-4.
  - Add an assertion test: passing chain_ids with fewer than 3 zeros raises
    AssertionError.

- `pytest tests/unit/test_registry_append.py -v`
  - Create a tmp REGISTRY.md with the header from `experiments/REGISTRY.md:7-8`.
  - Call `append_registry_row(..., dockq_avg=0.42, dockq_success_pct=35.0,
    c_rmsd=8.7)`; read the file; assert the last row contains
    `"C-RMSD 8.7000 A"` and `"DockQ 0.420 succ 35.0%"`.
  - Call again with all three set to `None`; assert the appended row does
    NOT contain `"C-RMSD"` or `"DockQ"` (back-compat).

- Smoke-eval: run the re-eval command (below) and confirm stdout's eval line
  contains `Centroid RMSE`, `C-RMSD`, and `DockQ` substrings, AND that the
  appended REGISTRY row contains the same numbers.

## Out of scope for this loop

- We are NOT modifying the training loop body (forward, loss, optimizer).
  Only the test-eval block and arg parsing change.
- We are NOT adding multi-sample inference (`--n_samples K`) — that's Loop 02.
- We are NOT changing the existing markdown REGISTRY header / column count —
  metrics piggyback on the existing "Outcome" cell so loop 02-07 can keep
  appending tokens (`oracle@5 ...`, etc.) without another schema change.
- We are NOT touching `src/tinyfold/model/registry.py` (model factory). The
  LOOPS.yaml entry mis-named it; the real target is
  `src/tinyfold/training/registry_append.py`.
- We are NOT retraining anything. The single new REGISTRY row comes from a
  pure inference pass over the existing Phase C `best_model.pt`.

## Re-eval command

Run from the repo root after Tasks 1-5 land. This loads the frozen Phase C
checkpoint, runs one eval pass over the N=100 Phase C test split, and the
`finally:` block appends one new row to `experiments/REGISTRY.md` with
centroid RMSE, C-RMSD, and DockQ + success%.

```powershell
python scripts/train_resfold.py `
    --config configs/train/resfold/phase_c_n8600.yaml `
    --eval_only `
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop01
```

After the command finishes, verify with:

```powershell
Select-String -Path experiments\REGISTRY.md -Pattern "C-RMSD|DockQ" | Select-Object -Last 3
```

The newest row should match the centroid RMSE printed at the end of the eval
pass (expected ~9.87 A, identical to the 2026-05-24 row 45) and include
`C-RMSD X.XX A | DockQ 0.YY succ ZZ.Z%`.
