# Loop 02: Multi-sample inference with cluster-then-rank (Task C)

## Overview

Add HDOCK-style multi-sample inference on top of the Loop 01 eval factor.
Loop 01 landed `_run_test_eval` (`scripts/train_resfold.py:555`) and the
`--eval_only` path (`scripts/train_resfold.py:1008-1027`). This loop:

1. Adds a per-target K-sample loop *inside* `_run_test_eval`, reusing the
   trunk once per target and re-running only the denoiser K times via
   `forward_sigma_with_trunk` (`onestep.py:254`).
2. Adds a pose-clustering helper for greedy interface-RMSD clustering.
3. Reports `oracle@K`, `mean@K`, `ranked@K` (cluster-rep) for each K in
   `--eval_K_list`. Default K-list = `1,5,40`.
4. Generalises `append_registry_row` to take an `extra_tokens: list[str]`
   so we stop growing one kwarg per metric.
5. Re-evals the Phase C N=8600 `best_model.pt` (kept intact per TASK
   constraint) twice: K=5 and K=40. Headline of this loop.

The trunk runs ONCE per target; the denoiser runs K times. With L<=300,
c_token=128 and a 4-block denoiser, a single denoiser forward on the
4070 Ti SUPER is roughly 30 ms — K=40 fits comfortably under 1 s/target
on the test split. K=40 x 100 targets x 1 s = ~2 min wall, not memory.

Out of scope (explicit, from instructions):
- Confidence-head ranking (Loop 06 replaces cluster-rep ranking).
- Phase D retrain (Loop 07).
- Changes to trunk / denoiser internals (Loops 05/06).

## Design decisions (locked here, do not relitigate)

### D1. Memory plan for K=40

Per-target serial processing. The current `_run_test_eval`
(`scripts/train_resfold.py:591-616`) already loops `for idx in
test_indices: batch = collate_batch([s], device)` one target at a time.
Inside that body we run K forward passes serially, never stacking K
samples into a batch. Peak VRAM stays exactly the same as K=1.

Cost model for the budget check, not a hard requirement:
`K x N_test x t_denoiser ~ 40 x 100 x 0.03 s ~ 2 min`.

### D2. Per-sample noise seeding

A single `torch.Generator(device=device).manual_seed(args.seed + sample_idx)`
is created per (target, sample). The seed is `args.seed * 100003 +
target_idx * 1009 + sample_idx`, picked so the same `--seed` reproduces
the run bit-for-bit and the K samples within a target are independent
across targets.

The generator is THREADED into a new keyword arg `generator` on
`sample_centroids_one_shot` (`scripts/train_resfold.py:154`) and
`sample_centroids_ve` (`scripts/train_resfold.py:186`), used by every
`torch.randn(..., generator=gen, ...)` site inside those helpers
(at minimum the `x = sigma * torch.randn(...)` init at line 173 for
one-shot and line 220 for VE). Default `generator=None` preserves
current behaviour (uses global torch RNG).

### D3. Cluster radius and interface definition

`--cluster_radius` defaults to **5.0 A** (HDOCK convention).

"Interface residues" for a target = the union of:
  (a) all chain-A CA within 8.0 A of any chain-B CA in the **GT**, and
  (b) all chain-B CA within 8.0 A of any chain-A CA in the GT.

Computed ONCE per target from `batch['centroids'][:, :n_res]` and
`batch['chain_ids']`. The 8 A cutoff is a standard contact threshold;
we hard-code it rather than expose a CLI flag (one knob is enough).

Pairwise pose-to-pose distance = RMSD over interface residues only,
*without* Kabsch alignment between samples (all samples are already in
the GT frame because `sample_centroids_ve` recenters every step and
`compute_c_rmsd` reports global-frame numbers — sample-to-sample drift
in the same frame is what we want to cluster on).

If the GT has zero interface residues (edge case: chains too far
apart in GT), fall back to ALL residues — this is a sanity guard, not
a real research case, and only fires if data prep let through a
non-complex pair.

### D4. Oracle ranking (`oracle@K`)

For each target, compute per-sample centroid RMSE vs GT (after the
existing Kabsch step that `compute_rmse` does), pick the minimum.
Average across targets.

### D5. Cluster-then-rank (`ranked@K`)

Greedy NN clustering:
1. Sort samples by some stable order (sample index).
2. For each unassigned sample, open a new cluster centred on it; sweep
   the remaining unassigned and absorb any within `cluster_radius` of
   the centre.
3. Cluster representative = the sample with the smallest mean
   interface-RMSD to other cluster members (closest-to-centroid). For
   a singleton cluster the representative is the sample itself.
4. Sort clusters by size descending; ties broken by smallest mean
   intra-cluster RMSD (more compact cluster wins).
5. `ranked@K` = centroid RMSE of the rank-1 cluster's representative
   vs GT, averaged across targets.

This is the plain HDOCK recipe, no confidence head. Loop 06 replaces
step 4's ranking criterion with predicted lDDT.

### D6. Output metric matrix

For each `K` in `args.eval_K_list`:
- `oracle_K = avg over targets of min_i rmse(sample_i, GT)`
- `mean_K   = avg over targets of mean_i rmse(sample_i, GT)`
- `ranked_K = avg over targets of rmse(rep_of_largest_cluster, GT)`

For K=1, `oracle_1 == mean_1 == ranked_1` — we still compute the same
codepath so the K=1 row is consistent with the K=5 and K=40 rows. This
also means with `--n_samples 1` (the default) the script produces the
same single number it did before Loop 02.

### D7. REGISTRY plumbing

Refactor `append_registry_row` in
`src/tinyfold/training/registry_append.py:19` to take an optional
`extra_tokens: Optional[list[str]] = None` argument, appended to the
outcome cell with `; ` joiner, AFTER the existing C-RMSD / DockQ
tokens. Keep `dockq_avg`, `dockq_success_pct`, `c_rmsd` kwargs as-is
for back-compat with Loop 01. Loop 02 builds:

```python
extras = [
    f"oracle@5 {oracle_5:.3f} A",
    f"mean@5 {mean_5:.3f} A",
    f"ranked@5 {ranked_5:.3f} A",
    f"oracle@40 {oracle_40:.3f} A",
    f"mean@40 {mean_40:.3f} A",
    f"ranked@40 {ranked_40:.3f} A",
]
```

(K=1 not emitted to extras — it equals the existing test RMSE column.)

## Tasks

### Task 1: Pose-cluster helper module

**Goal:** Pure-function clusterer with no torch / CUDA dependencies in the
hot path. Receives a list of pose tensors and the interface mask.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `src/tinyfold/model/metrics/cluster.py` |
| MODIFY | `src/tinyfold/model/metrics/__init__.py` (export `cluster_poses`, `interface_mask_from_gt`) |

**API:**
```python
def interface_mask_from_gt(
    gt_ca: Tensor,            # [L, 3]
    chain_ids: Tensor,        # [L] in {0, 1}
    valid: Tensor,            # [L] bool
    contact_cutoff: float = 8.0,
) -> Tensor:                  # [L] bool, True for interface residues

def pairwise_interface_rmsd(
    poses: Tensor,            # [K, L, 3]
    interface_mask: Tensor,   # [L] bool
) -> Tensor:                  # [K, K] symmetric, diag=0, sqrt(mean sq dist)

def cluster_poses(
    poses: Tensor,            # [K, L, 3]
    interface_mask: Tensor,   # [L] bool
    radius: float,
) -> list[dict]:
    """Return list of cluster dicts ordered by size desc, then compactness asc.

    Each dict: {
        'members': list[int],      # sample indices into poses
        'representative': int,     # index of closest-to-centroid member
        'mean_intra_rmsd': float,  # 0.0 for singleton
    }
    """
```

**Steps:**
1. Implement `interface_mask_from_gt`: build pairwise cross-chain
   distance matrix from chain-A vs chain-B CA, take the min per
   residue, threshold at `contact_cutoff`. Union of A-side and B-side
   residues. Mask out invalid residues last.
2. Implement `pairwise_interface_rmsd`: gather interface residues,
   compute pairwise sq-dist over the K poses (vectorised), reduce mean
   then sqrt. O(K^2 * L_iface) — fine for K=40.
3. Implement `cluster_poses` per D5 (greedy NN, rep = closest to
   centroid, sort by size then compactness).
4. Edge cases:
   - K=1 -> single cluster of one element.
   - `interface_mask.sum() == 0` -> caller should pass a fallback mask
     (all-True); raise `ValueError` if mask is empty (defensive).

**Verify:** unit test below (Task 4) passes; no CUDA tensors created
inside the helper (assert everything is `.cpu()` on entry or accept
both — choose one and document).

---

### Task 2: Seeded multi-sample sampler

**Goal:** Make the existing samplers accept a `torch.Generator` so the
caller can produce K reproducible, distinct samples per target. Add a
thin wrapper that runs the trunk once and the denoiser K times.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` (`sample_centroids_one_shot` line 154; `sample_centroids_ve` line 186) |
| MODIFY | `scripts/train_resfold.py` (add `sample_k_centroids` helper near the samplers, ~line 280) |

**Steps:**
1. Add `generator: Optional[torch.Generator] = None` kwarg to both
   `sample_centroids_one_shot` and `sample_centroids_ve`. Thread it into
   every `torch.randn(...)` call in those functions (one-shot: line 173;
   VE: line 220). Default `None` preserves current global-RNG behaviour.
2. Add a new helper:

```python
@torch.no_grad()
def sample_k_centroids(
    model, batch, noiser, device, K: int, base_seed: int, target_idx: int,
    is_onestep: bool, one_shot: bool, align_per_step: bool, recenter: bool,
) -> tuple[Tensor, Optional[Tensor]]:
    """Return (centroids [K, L, 3], atoms_or_None [K, L, 4, 3]).

    Runs the trunk ONCE (forward_sigma_with_trunk path when is_onestep
    and one_shot), then K denoiser passes with per-sample generators
    seeded `base_seed * 100003 + target_idx * 1009 + i`.
    """
```

   - When `is_onestep and one_shot`: call
     `model.get_trunk_tokens(...)` ONCE
     (`onestep.py:297`), then loop i in range(K) calling
     `model.forward_sigma_with_trunk(...)` directly with a fresh
     `torch.randn(..., generator=gen_i)` init at `sigma_max`.
   - When `is_onestep and not one_shot` (VE path): we cannot easily
     bypass the trunk inside `sample_centroids_ve` without refactoring
     the whole step loop. Acceptable for Loop 02: just call
     `sample_centroids_ve(model, batch, noiser, device,
     generator=gen_i, ...)` K times; the trunk runs K times, which is
     a 4-block transformer over L<=300 tokens — measured cost is
     trivial compared to the denoiser's `len(sigmas)-1` steps. Document
     this in a comment; Loop 06 can revisit.
3. Stack K results: `torch.stack(centroid_list, dim=0)` ->
   `[K, B=1, L, 3]`, squeeze B -> `[K, L, 3]`. Same for atoms when
   `is_onestep`.

**Verify:** unit test below (Task 4) asserts K samples are pairwise
distinct (`!= each other` on > 0.0 norm).

---

### Task 3: Wire K-sample into `_run_test_eval`

**Goal:** Inside the existing per-target loop in `_run_test_eval`
(`scripts/train_resfold.py:591`), branch on `args.n_samples > 1` and
collect the per-K metrics. K=1 fast-path leaves current behaviour
unchanged.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` (argparse around line 511; `_run_test_eval` body 555-751; `main()` finally-block 1632) |

**Steps:**
1. Add CLI flags near the other sampling flags (~line 515):
   ```python
   parser.add_argument("--n_samples", type=int, default=1,
       help="K samples per target for multi-sample eval (default 1 = old behaviour)")
   parser.add_argument("--cluster_radius", type=float, default=5.0,
       help="Interface-CA RMSD radius (Angstroms) for HDOCK-style clustering")
   parser.add_argument("--eval_K_list", type=str, default="1,5,40",
       help="Comma-separated K values to report; only used when --n_samples > 1. "
            "Max(eval_K_list) must be <= --n_samples.")
   ```
2. Parse `eval_K_list` once at the top of `_run_test_eval`:
   ```python
   k_list = sorted({int(x) for x in args.eval_K_list.split(",")}) \
            if args.n_samples > 1 else [1]
   assert max(k_list) <= args.n_samples, "eval_K_list exceeds --n_samples"
   ```
3. New per-K accumulators alongside `test_rmses` / `test_c_rmsds`:
   ```python
   per_k_oracle = {k: [] for k in k_list}
   per_k_mean   = {k: [] for k in k_list}
   per_k_ranked = {k: [] for k in k_list}
   ```
4. Inside the per-target loop, in the `stage1_only` branch, when
   `args.n_samples > 1`:
   - Call `sample_k_centroids(...)` with `K = args.n_samples`.
   - Trim to `[:K, :n_res]`.
   - Compute the interface mask via `interface_mask_from_gt(
     batch['centroids'][0, :n_res], batch['chain_ids'][0, :n_res],
     batch['mask_res'][0, :n_res])`. Cache it for the cluster call.
   - For the **largest** k in `k_list`, take the first-k slice for
     each k value (so K=5 metrics use samples 0..4 of the 40-sample
     pool — deterministic subsets).
   - For each k:
     - Per-sample RMSE vs GT using the existing `compute_rmse`
       (Kabsch-aligned, scaled by `s['std']`).
     - `oracle_k = min(rmses)`, `mean_k = mean(rmses)`.
     - `clusters = cluster_poses(samples[:k], interface_mask,
       args.cluster_radius)`; `ranked_k = rmses[clusters[0]
       ['representative']]`.
     - Append to `per_k_*` accumulators.
   - For the rest of the per-target metrics (DockQ, atom RMSE,
     C-RMSD) keep using sample 0 only — those are existing Loop 01
     metrics and we don't need K-of-them.
   - Set `centroids_pred = samples[0:1]` and
     `atoms_pred_onestep = atoms[0:1]` so the rest of the
     function (which reads these) works unchanged.
5. After the per-target loop, build `extras: list[str]`:
   ```python
   extras = []
   for k in k_list:
       if k == 1:
           continue  # K=1 oracle/mean/ranked all equal the printed test RMSE
       extras.append(f"oracle@{k} {avg(per_k_oracle[k]):.3f} A")
       extras.append(f"mean@{k} {avg(per_k_mean[k]):.3f} A")
       extras.append(f"ranked@{k} {avg(per_k_ranked[k]):.3f} A")
   ```
   Stuff them into the log summary and into the returned tuple.
6. Extend `_run_test_eval`'s return signature from `(test_avg,
   dockq_avg, dockq_success_pct, c_rmsd_avg)` to ALSO return
   `extra_tokens` (default `[]`). Update both callers — the in-loop
   call at line 1483 and the `--eval_only` call at line 1013.
7. In the `--eval_only` branch, stash `progress["extra_tokens"] =
   extra_tokens`. In `main()`'s finally-block (line 1636), pull
   `progress.get("extra_tokens")` and pass as the new
   `extra_tokens=` kwarg to `append_registry_row`.

**Verify:** running with no new flags (`--n_samples` defaults to 1)
produces a REGISTRY row byte-identical (mod the date stamp) to the
Loop 01 row. Run the K=5 dry run below.

---

### Task 4: REGISTRY plumbing + extra_tokens

**Goal:** Lift the per-metric kwarg pattern out of
`append_registry_row` so adding Loop 02's six tokens (and Loop
06/07's future tokens) is a one-line caller change.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/training/registry_append.py:19` |

**Steps:**
1. Add `extra_tokens: Optional[list[str]] = None` kwarg.
2. After the existing `extras = []; if c_rmsd ...; if dockq_avg ...`
   block (line 88-93), append:
   ```python
   if extra_tokens:
       extras.extend(extra_tokens)
   ```
3. Update the docstring (the Args block) to describe the new param
   and note "preferred for new metrics; the named kwargs are kept for
   Loop 01 back-compat only."

**Verify:** the existing Loop 01 unit/smoke around C-RMSD + DockQ still
prints in the same order; the new tokens appear AFTER them in the cell.

---

### Task 5: Unit + integration tests

**Goal:** Cover the three behavioural claims this loop makes.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `tests/test_pose_clustering.py` |
| CREATE | `tests/test_multisample_eval.py` |

**Tests:**

1. **Unit — `cluster.py`** (`tests/test_pose_clustering.py`):
   - Build 6 poses in a fake 6-residue 2-chain target:
     - poses 0..2 = small jitter (< 1 A) around pose A.
     - poses 3..5 = small jitter around pose B.
     - pose A and pose B differ by translating chain B by +20 A.
   - GT = pose A (just for the interface-mask computation).
   - Assert `len(cluster_poses(poses, mask, radius=5.0)) == 2`.
   - Assert the larger (or equal-sized but more compact) cluster's
     members are exactly `{0, 1, 2}` (or some permutation thereof).
   - Assert each cluster's representative is one of its own members.

2. **Unit — seed plumbing** (`tests/test_multisample_eval.py`):
   - Spin up a tiny `ResFoldOneStep(c_token=32, trunk_layers=1,
     denoiser_blocks=1)` on CPU with random init.
   - Build a 1-target batch with L=10.
   - Call `sample_k_centroids(..., K=4, base_seed=42, target_idx=0)`.
   - Assert `samples.shape == (4, 1, 10, 3)`.
   - Assert pairwise `(samples[i] - samples[j]).norm() > 1e-3` for all
     i != j (K distinct).
   - Call AGAIN with same seeds; assert byte-identical (reproducible).

3. **Integration — 4-target K=5 dry run**
   (`tests/test_multisample_eval.py`):
   - Use a fixture parquet (existing tests/fixtures or the
     prepared cache — re-use the same one Loop 01's smoke test used).
   - Invoke the script as a subprocess:
     ```
     python scripts/train_resfold.py \
         --config configs/train/resfold/phase_c_n8600.yaml \
         --eval_only \
         --checkpoint outputs/resfold/phase_c_n8600/.../best_model.pt \
         --n_test 4 \
         --n_samples 5 \
         --eval_K_list 1,5 \
         --output_dir <tmp>/multisample_smoke
     ```
   - Parse stdout/REGISTRY row; assert `oracle@5 <= mean@5` (oracle is
     a min, mean is a mean) AND `ranked@5` token present.
   - Marked `@pytest.mark.slow` and skipped when the Phase C
     checkpoint is not present locally (CI-safe).

**Verify:** `pytest tests/test_pose_clustering.py
tests/test_multisample_eval.py -v` is green.

---

### Task 6: Phase C K=5 and K=40 re-eval

**Goal:** Produce the headline number for this loop. Two REGISTRY rows
added, no checkpoints overwritten.

**Files:**
| Action | Path |
|--------|------|
| RUN | `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt` (read-only) |
| WRITE | `outputs/resfold/phase_c_n8600_reeval_loop02_k5/...` |
| WRITE | `outputs/resfold/phase_c_n8600_reeval_loop02_k40/...` |
| APPEND | `experiments/REGISTRY.md` |

**Commands:**

```powershell
python scripts/train_resfold.py `
    --config configs/train/resfold/phase_c_n8600.yaml `
    --eval_only `
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
    --n_samples 5 `
    --eval_K_list 1,5 `
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop02_k5

python scripts/train_resfold.py `
    --config configs/train/resfold/phase_c_n8600.yaml `
    --eval_only `
    --checkpoint outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt `
    --n_samples 40 `
    --eval_K_list 1,5,40 `
    --output_dir outputs/resfold/phase_c_n8600_reeval_loop02_k40
```

Notes for the runner:
- `--one_shot_sample` is NOT set; the Loop 01 re-eval used the default
  multi-step VE path (`sample_centroids_ve`) so we match it. If that
  call OOMs (it shouldn't — same K=1 per-target memory), fall back to
  `--one_shot_sample`.
- Both runs use `--seed 42` (default) so the K=5 sub-pool of the K=40
  run is bit-identical to the K=5 standalone run — sanity check.
- Phase C checkpoint MUST stay intact (TASK constraint, line 33). The
  `--output_dir` arg points at a fresh timestamped subdir; we never
  touch `outputs/resfold/phase_c_n8600/.../`.

**Verify:** two new lines appended to `experiments/REGISTRY.md`, each
with `oracle@K`, `mean@K`, `ranked@K` tokens in the outcome cell, and
each linking to its own `outputs/.../` directory.

## Acceptance Criteria

- [ ] `src/tinyfold/model/metrics/cluster.py` exists with the three public
      functions in Task 1 and is exported from
      `tinyfold.model.metrics.__init__`.
- [ ] `sample_centroids_one_shot` and `sample_centroids_ve` accept a
      `generator: Optional[torch.Generator]` kwarg; default `None`
      preserves byte-identical behaviour vs `main` at `bb4cff6`.
- [ ] `--n_samples`, `--cluster_radius`, `--eval_K_list` CLI flags
      added to `scripts/train_resfold.py`.
- [ ] `_run_test_eval` returns `extra_tokens` and prints
      `oracle@K mean@K ranked@K` tokens for every K > 1 in the
      configured list.
- [ ] `append_registry_row(..., extra_tokens=...)` extends the outcome
      cell without breaking Loop 01's `c_rmsd` / `dockq_avg` tokens.
- [ ] Running with no new flags (i.e. `--n_samples 1`) produces a row
      whose outcome cell is structurally identical to Loop 01's row
      (same C-RMSD / DockQ tokens, no `oracle@*` tokens added).
- [ ] `pytest tests/test_pose_clustering.py
      tests/test_multisample_eval.py -v` passes locally.
- [ ] Phase C re-eval with K=5 appended to `experiments/REGISTRY.md`.
- [ ] Phase C re-eval with K=40 appended to `experiments/REGISTRY.md`.
- [ ] `oracle@40 <= mean@40` in the K=40 re-eval row (sanity, oracle is
      a min over the same pool the mean is computed on).
