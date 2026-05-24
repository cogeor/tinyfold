# Loop 06: Confidence head for multi-sample ranking (Task G)

## Overview

Add a tiny per-target confidence MLP on top of pooled denoiser tokens that regresses predicted lDDT against GT. Train it jointly with a low auxiliary weight; at multi-sample eval, use it to rank K samples and replace Loop 02's oracle ranking. The goal is to claw back the 1.87 A gap Loop 02 surfaced (oracle@40 = 8.46 A vs mean@40 = 10.33 A) without needing GT at inference.

Risk is HIGH (per LOOPS.yaml): lDDT-as-regression on N=8600 may collapse to the dataset mean. Mitigations: low weight (0.1), Phase D fallback to oracle/cluster ranking, head exists only when `--confidence_head` is set.

### Pre-flight notes from required reading

- `src/tinyfold/model/losses/lddt.py` ALREADY has `compute_lddt(pred_ca, gt_ca, mask, coord_scale)` — but it returns a SCALAR (`lddt.mean()` across the batch). We need per-sample `[B]`. Solution: add an optional `reduction: str = "mean"` arg with `reduction="per_sample"` returning `[B]`. No new file needed.
- `src/tinyfold/model/resfold/onestep.py` `_heads_edm` is the natural pooling point — `denoiser_tokens` (shape `[B, L, c_token]`) is the head's input. The atom head lives there; the confidence head sits next to it.
- `scripts/train_resfold.py` line ~1465: `forward_sigma` returns `(centroids_pred, atoms_pred)`. Loop 06 extends the tuple to `(centroids_pred, atoms_pred, pred_lddt_or_None)`. Auxiliary loss is added next to the existing `alpha_atom * atom_mse_tensor` branch (~line 1531).
- Multi-sample eval lives in `_run_test_eval` between lines 778-813. The new `--rank_by=confidence` path collects `pred_lddt` per sample, picks `argmax`, and reports `ranked_conf@K`. The existing `ranked@K` (cluster-rep) stays.

## Tasks

### Task 1: Add `compute_lddt` per-sample reduction

**Goal:** Allow `compute_lddt` to return `[B]` (one score per target) so the trainer can build per-target regression labels and the eval can correlate per-sample.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/model/losses/lddt.py` |
| MODIFY | `tests/unit/test_losses.py` (extend, do not break existing) |

**Steps:**
1. Add `reduction: str = "mean"` arg to `compute_lddt`. Accepted: `{"mean", "per_sample"}`. Default keeps current behaviour (scalar return).
2. When `reduction == "per_sample"`, return the `[B]` tensor (currently the value of `lddt` right before `lddt.mean()`).
3. Docstring: clarify shape contract for both reductions.
4. Test: `test_compute_lddt_per_sample` — build a `[2, L, 3]` pair where sample 0 is `pred == gt` (expect lDDT==1.0) and sample 1 is `pred = gt + 100A` shift (expect ~0.0). Assert `reduction="per_sample"` returns shape `[2]` and the two values differ by at least 0.9.

**Verify:** `pytest tests/unit/test_losses.py -k lddt -v` passes; existing scalar tests still green.

---

### Task 2: Add `ConfidenceHead` module

**Goal:** A 2-layer MLP that maps pooled (masked-mean) denoiser tokens to a scalar predicted lDDT in `[0, 1]`. Tiny: ~`2 * c_token^2` params.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `src/tinyfold/model/resfold/confidence_head.py` |
| MODIFY | `src/tinyfold/model/resfold/__init__.py` (export `ConfidenceHead`) |
| CREATE | `tests/unit/test_confidence_head.py` |

**Steps:**
1. Create `ConfidenceHead(nn.Module)` with:
   - ctor args: `c_token: int = 128`, `hidden: Optional[int] = None` (default to `c_token`), `dropout: float = 0.0`.
   - layers: `nn.Linear(c_token, hidden)` -> `nn.SiLU()` -> `nn.Dropout(dropout)` -> `nn.Linear(hidden, 1)`. Output passed through `torch.sigmoid` so the prediction is in `[0, 1]`.
   - `forward(tokens: Tensor, mask: Optional[Tensor]) -> Tensor`:
     - `tokens`: `[B, L, c_token]`. `mask`: `[B, L]` bool (True = real residue).
     - Masked mean: `pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)`. Cast mask to tokens.dtype.
     - Returns `pred_lddt`: `[B]` (after `.squeeze(-1)`).
   - Init: output bias = `inverse_sigmoid(0.5) = 0` so the head starts predicting 0.5 (the dataset prior mid-point) — keeps gradients live.
2. Export from `src/tinyfold/model/resfold/__init__.py`.
3. Tests in `tests/unit/test_confidence_head.py`:
   - `test_shape`: feed `[3, 17, 128]` tokens + a `[3, 17]` mask with 5 padded residues -> assert output shape `[3]` and all values in `[0, 1]`.
   - `test_mask_invariance`: padding extra zero tokens with mask=False should NOT change the prediction (assert close to a non-padded run, atol=1e-6).
   - `test_grad_flow`: backprop a `smooth_l1` loss vs a target; assert MLP weights' `.grad` is non-zero.

**Verify:** `pytest tests/unit/test_confidence_head.py -v` passes.

---

### Task 3: Wire `ConfidenceHead` into `ResFoldOneStep`

**Goal:** Make the head an opt-in part of the one-step model. Threading: `forward_sigma` and `forward_sigma_with_trunk` return an extra optional `pred_lddt` slot.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/model/resfold/onestep.py` |

**Steps:**
1. Ctor: add `confidence_head: bool = False`. When True, instantiate `self.confidence_head = ConfidenceHead(c_token=c_token, dropout=dropout)`. Otherwise set to `None`. Import is local (`from .confidence_head import ConfidenceHead`).
2. Refactor `_heads_edm` return: keep the existing 2-tuple return; add a sibling helper `_predict_confidence(denoiser_tokens, mask) -> Optional[Tensor]` that returns `None` when `self.confidence_head is None`.
3. Change `forward_sigma` signature to return `Tuple[Tensor, Tensor, Optional[Tensor]]` = `(centroid_pred, atoms_pred, pred_lddt)`. Same for `forward_sigma_with_trunk`.
4. Leave the legacy `forward()` 2-tuple unchanged (deprecated path, no callers in active eval).
5. Update `count_parameters()` to include `confidence_head` bucket (0 when None).

**Critical compatibility:** every caller that unpacks `forward_sigma` as `(centroid, atoms) = ...` must be updated. Grep for `forward_sigma(` in `scripts/train_resfold.py` and `src/tinyfold/model/resfold/` and switch to 3-tuple unpacking (`pred_lddt` may be `None`). Sites identified from pre-flight reading: train loop (~line 1465, 1456), `sample_k_centroids` (~line 383), `sample_centroids_one_shot` / `sample_centroids_ve` paths in same script.

**Verify:** Existing one-step unit tests (`pytest tests/ -k onestep`) still pass; new test `test_onestep_confidence_head_optional` confirms `confidence_head=False` returns `pred_lddt is None` and `confidence_head=True` returns `[B]` tensor.

---

### Task 4: Wire confidence loss into the trainer

**Goal:** When `--confidence_head_weight > 0`, compute `gt_lddt` from `centroids_pred` (the EDM-blended one-step x0) vs `centroids_target`, no grad on the target, and add `smooth_l1(pred_lddt, gt_lddt) * weight` to the training loss.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` |

**Steps:**
1. CLI:
   - `--confidence_head` (store_true) — instantiate head on the OneStep model.
   - `--confidence_head_weight` (float, default 0.0) — aux loss coefficient. Asserted: if `> 0`, requires `--confidence_head`.
   - Both go in the OneStep block alongside `--atom_weight` (~line 595).
2. Model construction (~line 1190): pass `confidence_head=args.confidence_head` to `ResFoldOneStep(...)`. Update the param-count log block to print `Confidence-head params: {pc['confidence_head']:,}` when nonzero.
3. Forward unpacking (~line 1465): change to
   ```
   centroids_pred, atoms_pred, pred_lddt = fwd_out
   ```
   The non-onestep branch returns 1-tuple as before; gate on `is_onestep`.
4. Aux loss block, inserted right after the atom-MSE block (~line 1531, BEFORE the geometry loss):
   ```python
   loss_conf = 0.0
   if (
       is_onestep
       and pred_lddt is not None
       and args.confidence_head_weight > 0
   ):
       with torch.no_grad():
           # Per-sample lDDT vs GT centroids (CA-only, since centroids are CA).
           gt_lddt = compute_lddt(
               centroids_pred.detach(),  # detach: head sees features, not the centroid path
               centroids_target,
               mask=batch['mask_res'],
               coord_scale=batch_std,  # or 1.0 if coords are already unnormalized
               reduction="per_sample",
           )
       conf_loss_tensor = torch.nn.functional.smooth_l1_loss(pred_lddt, gt_lddt)
       loss = loss + args.confidence_head_weight * conf_loss_tensor
       loss_conf = conf_loss_tensor.item()
   ```
   Note on `coord_scale`: `compute_lddt` scales coords by `coord_scale` before computing distances. Trainer uses normalized coords (std~1.0 per sample); pass `coord_scale=1.0 / args.target_std` or pull the per-sample `std` from the batch if available. Implementer: inspect `compute_mse_loss` call site to mirror the same scaling convention used for centroids RMSE — the eval path multiplies by `s['std']` after the fact, suggesting trainer coords are normalized. Safest: pass `coord_scale=1.0` (compares in normalized units, scale-invariant for L1) and verify gt_lddt distribution looks sane in smoke.
5. Loss components dict (~line 1567): add `'conf': loss_conf` to the OneStep `loss_components.update({...})` block. The logger that consumes this dict needs the new key — extend the log format string in `_log_step` (or wherever `loss_components` is rendered).
6. Detach contract: `pred_lddt` is fully connected to the denoiser MLP — gradients flow into the confidence head AND through the pool back into the denoiser tokens. This is intentional (head can shape features). The TARGET `gt_lddt` is in `no_grad`. Verify in unit test (Task 6).

**Verify:** A 50-step smoke run with `--confidence_head --confidence_head_weight 0.1` on N=4 finishes without NaN; `loss_conf` appears in the log line each step.

---

### Task 5: Wire `--rank_by confidence` into multi-sample eval

**Goal:** Add a third ranking strategy alongside oracle and cluster-rep. When chosen, sort K samples by predicted lDDT (descending) and report `ranked_conf@K` = RMSE of the top-ranked sample.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` |

**Steps:**
1. CLI: `--rank_by`, `choices=["oracle", "cluster", "confidence"]`, default `"cluster"`. Lives near `--n_samples`.
2. Extend `sample_k_centroids` (~line 314): when `is_onestep` and the model has `confidence_head`, collect per-sample `pred_lddt` and return a 3-tuple `(centroids, atoms, pred_lddts_or_None)` of shape `(K, B, L, 3)`, `(K, B, L, 4, 3)`, `(K, B)`. Update both branches (fast `one_shot` and VE).
3. `_run_test_eval` multi-sample block (~lines 778-813):
   - Collect `per_k_ranked_conf = {k: [] for k in k_list}` next to the existing `per_k_ranked` dict.
   - After `per_sample_rmses` is built, if `pred_lddts` is not None:
     ```python
     for k in k_list:
         conf_scores = pred_lddts[:k].squeeze(1).cpu().tolist()  # [k]
         best = int(max(range(k), key=lambda i: conf_scores[i]))
         per_k_ranked_conf[k].append(per_sample_rmses[best])
     ```
   - Honour `args.rank_by` only for the SELECTION of which downstream RMSE goes into `centroids_pred` / `atoms_pred_onestep` for the single-sample downstream metrics (DockQ/atom/C-RMSD): default cluster-rep keeps current behaviour; `confidence` picks `argmax(conf_scores)`; `oracle` picks `argmin(sub_rmses)`. This makes the downstream metrics consistent with the headline ranker.
4. Logging extension (~line 985): when `pred_lddts` exists, append `ranked_conf@{k}: {rc:.4f} A` to `log_msg` and `extra_tokens.append(f"ranked_conf@{k} {rc:.3f} A")`.
5. Optional correlation metric: at the end of the eval loop, if `scipy.stats` is importable, compute Spearman across the flattened `[targets * K]` pred_lddt vs `[targets * K]` -RMSE (negate RMSE so higher = better). Log `"Spearman(pred_lddt, -RMSE): {rho:.3f}"`. Fall back to a single Pearson via `torch.corrcoef` if scipy is absent. Skip entirely if `pred_lddts` is None.

**Verify:** Manual eval on a 2-target / K=4 slice with `--rank_by confidence`; assert `ranked_conf@4` shows up in the log and lies between `oracle@4` and `mean@4` (or worse, but finite).

---

### Task 6: Unit tests (head + integration smoke)

**Goal:** Lock in the contracts before training. Two test files cover the head alone, the loss gating, and an end-to-end forward.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `tests/unit/test_confidence_head.py` (extends Task 2) |
| CREATE | `tests/unit/test_onestep_confidence.py` |

**Steps:**
1. `test_onestep_confidence.py`:
   - `test_onestep_returns_pred_lddt`: build a `ResFoldOneStep(c_token=32, trunk_layers=1, denoiser_blocks=1, confidence_head=True)`, run `forward_sigma` on a `[2, 8, 3]` batch, assert returned tuple has length 3 and `pred_lddt.shape == (2,)` and values are in `[0, 1]`.
   - `test_onestep_no_head`: same model with `confidence_head=False`; assert `pred_lddt is None`.
   - `test_confidence_loss_gradient`: with the head on, build a mock `gt_lddt` tensor and run `smooth_l1(pred_lddt, gt_lddt).backward()`; assert `model.confidence_head.parameters()` have non-zero `.grad`, AND the denoiser params ALSO have non-zero `.grad` (gradients flow through the pool).
2. Reuse `pytest -m "not gpu"` markers (or whatever the project uses for CPU-only fast tests). Mark these as CPU.

**Verify:** `pytest tests/unit/test_confidence_head.py tests/unit/test_onestep_confidence.py -v` all green.

---

### Task 7: Smoke run

**Goal:** End-to-end 50-step train + 4-target eval with the head enabled, confirming no NaN and that `pred_lddt` differs across K samples for the same target.

**Command:**
```powershell
python scripts/train_resfold.py `
  --model_kind onestep `
  --mode stage1_only `
  --n_train 4 --n_test 4 `
  --epochs 1 --max_steps 50 `
  --continuous_sigma `
  --confidence_head --confidence_head_weight 0.1 `
  --n_samples 4 --rank_by confidence `
  --eval_K_list 1,4 `
  --eval_every 50 `
  --output_dir outputs/resfold/loop06_smoke `
  --seed 42
```

**Verify:**
- Training log shows `conf: <float>` in the per-step loss components and the value is finite for all 50 steps.
- Eval log line at step 50 contains `ranked_conf@4: ... A` and a `Spearman(pred_lddt, -RMSE)` token (or Pearson fallback).
- Across the 4 samples for a single target, `pred_lddt` values are NOT all identical (i.e., the head responds to different denoiser tokens — not collapsed). Add a temporary print or assert in `_run_test_eval` for the smoke run only; do not commit the print.

---

## Acceptance Criteria

- [ ] `compute_lddt` supports `reduction="per_sample"` returning `[B]`; existing scalar behaviour preserved (Task 1).
- [ ] `ConfidenceHead` module exists, is masked-mean over residues, sigmoid-bounded, passes all three unit tests (Task 2).
- [ ] `ResFoldOneStep(confidence_head=True)` returns `pred_lddt` from both `forward_sigma` and `forward_sigma_with_trunk` (Task 3).
- [ ] Trainer accepts `--confidence_head` and `--confidence_head_weight`; aux loss appears in `loss_components` (Task 4).
- [ ] `--rank_by confidence` is wired into `_run_test_eval` and emits `ranked_conf@K` tokens for K > 1 (Task 5).
- [ ] Unit tests in `tests/unit/test_confidence_head.py` and `tests/unit/test_onestep_confidence.py` all pass (Task 6).
- [ ] Smoke run completes 50 steps + 4-target eval; `pred_lddt` is non-constant across K samples per target (Task 7).
- [ ] `--confidence_head_weight 0` keeps the head dormant (no aux loss term, no gradient into head); covered by an additional unit assertion or by simply verifying the gating branch.

## Out of Scope

- Per-residue lDDT prediction. This loop's head is per-target scalar only.
- DockQ regression target. lDDT chosen because it is continuous; DockQ is bucketed.
- Cluster-confidence hybrid ranking (e.g., re-rank inside a cluster by confidence). Future work; not needed for Phase D.
- Confidence head on the legacy `ResFoldPipeline` (`resfold` model_kind). One-step only — pipeline is being phased out.

## Risk + Mitigations

- **Head collapses to dataset mean.** Spearman tracked in eval; if `< 0.1` on val after Phase D smoke, Loop 07's `phase_d_n8600_full.yaml` falls back to `--rank_by cluster` (already the default) and reports both `ranked@K` and `ranked_conf@K` so the regression is visible without blocking the loop.
- **Gradient through pool destabilises trunk.** Weight starts at 0.1 (10x below atom weight); if loss diverges, drop to 0.01 or set `--confidence_head_weight 0` for Phase D and revisit post-retrain.
- **Coordinate-scale mismatch in `compute_lddt`.** Trainer uses normalized coords; pass `coord_scale=1.0` and rely on lDDT's distance-difference (not absolute-distance) thresholds to be scale-comparable in the normalized space. Smoke check: print `gt_lddt.mean()` for a few steps and confirm it lives in a sane range like `[0.2, 0.9]`, not `0.99+` (which would indicate scale-suppressed errors).
- **3-tuple return breaks unseen callers.** Pre-emptively grep `forward_sigma(` and `forward_sigma_with_trunk(` across `src/` and `scripts/` in Task 3 implementation; the small project size makes this exhaustive.

## Phase D Hook (Loop 07 will wire this up)

Loop 07's `configs/train/resfold/phase_d_n8600_full.yaml` should set:
```yaml
confidence_head: true
confidence_head_weight: 0.1
n_samples: 40
eval_K_list: "1,5,40"
rank_by: confidence  # falls back to cluster if Spearman is poor on val
```
This loop just ensures the flags work; Loop 07 owns the final config.
