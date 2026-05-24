# TinyFold — Diffusion Demo Plan (revised 2026-05-23)

Goal: one honest blog post and a turn-key repo. A single-GPU AlphaFold3-style PPI predictor that produces a defensible held-out number — sub-SOTA is fine; non-zero on DockQ/centroid-RMSE is the bar. The headline is **"hard problem, first that works on a small setup."**

This plan supersedes the earlier version. It reflects:
- The `pipeline.py:49` import bug is fixed (`bb4cff6`).
- `ResidueDenoiser` already absorbed the fixes from `diffusion_analysis_report.md` (continuous σ, self-conditioning, sequence-only trunk, per-step Kabsch).
- `stage1_10M` is the only meaningful prior run: train centroid RMSE ~6.5–7.5 Å, test ~10.5–11.7 Å at step 28K.
- Stage 2 has never been trained to completion.
- The codebase has shadow model trees, 20+ dead output dirs, and duplicate training entry points.

Constraint: one consumer GPU (RTX 4070 Ti). Time available, compute scarce. Plan for sequential runs, not sweeps.

---

## 0. Pre-flight: triage and "first to run" ranking

### 0.1 Network triage

| Rank | Network | Where | Diffusion? | Status | Pick rationale |
|---|---|---|---|---|---|
| **1** | `ResidueDenoiser` (Stage 1, standalone) | `src/tinyfold/model/resfold/denoiser.py` | yes (residue centroids) | partial run @ 28K steps in `stage1_10M`; 7Å train / 11Å test | Only diffusion model in the repo with a real partial signal. 4× fewer tokens than atom-level. Trunk runs once → fast sampling. The train/test gap is the *interesting* problem for the blog — "what was hard" is more honest than picking something easier. |
| 2 | `AtomRefinerV2` (Stage 2, standalone) | `src/tinyfold/model/resfold/refiner.py` | no (deterministic) | never trained to completion | Independent of Stage 1, runs in parallel on idle GPU time. From GT centroids it should overfit trivially. If it doesn't, the data pipeline is the bug — informative either way. |
| 3 | `ResFoldPipeline` (S1 frozen + S2) | `src/tinyfold/model/resfold/pipeline.py` | yes + no | import bug fixed in `bb4cff6`; never run end-to-end | The actual headline — what the README claims, what the assets PNG shows. Only worth running after 1 and 2 pass alone. |
| 4 | `ResFoldE2E` | `src/tinyfold/model/resfold/e2e.py` | yes + multi-sample S2 | untrained, more complex | Defer. |
| 5 | `IterFold` / `FrameDecoder` | `src/tinyfold/model/iterfold/` | no (masked anchor) | trained on 20 samples → DockQ 0.044 (noise) | Fallback only. |

**Headline pick: `ResidueDenoiser`.** Not because it's easiest — because the existing 7Å/11Å gives a non-zero starting point we can debug, and abandoning it loses ~5h of useful GPU history.

### 0.2 What "success" means at each stage

Per run we log:
1. **Train RMSE** (one-step denoising, Kabsch-aligned, Å).
2. **Sample RMSE** (full T=50 rollout, Kabsch-aligned).
3. **Sample RMSE on held-out** (N ≥ 16 only).
4. **Sample-vs-train gap** = sample_rmse − train_rmse (sampling-quality signal).
5. **Loss curve by σ bucket** (flat = healthy; U-shape = schedule/capacity mismatch).
6. **Per-residue RMSE binned by interface vs core** (8 Å cutoff to the other chain). This goes in *every* Phase B/C run — added early so the headline can't be misleading.

A run passes the sanity bar if `Sample RMSE ≤ 1.3 × Train RMSE` on the data it was trained on.

---

## 1. Phase 0 — Cleanup (no GPU, ~half a day)

The first thing the user emphasized. Without it the blog post can't credibly say "clone and reproduce." All work below is local, reversible only via `git`.

### 1.1 Repo hygiene

- [ ] **Commit and keep all model code, including failed/dead branches.** `src/tinyfold/model/archive/` stays committed and registry-importable. `scripts/models/` gets deleted only after we've confirmed every model file is mirrored in `src/tinyfold/model/archive/`. The point: nothing about a failed approach should require git archaeology to recover. The registry already names all of them (`attention_v2`, `hierarchical`, `pairformer`, `af3_style`, `iterfold`, etc.) so they remain instantiable from config — this prevents accidentally reinventing a dead architecture.
- [ ] Resolve git status:
  - `M doc/data_pipeline.md`, `M doc/frontend.md`, `M src/tinyfold/model/registry.py` — commit if intentional, otherwise revert.
  - Deleted `diffusion_analysis_report.md`, `epsilon_rollout_plan.md`, `plan_readme.md` at project root — these still exist in `notes/`; the deletions are correct, just need to be committed.
- [ ] Delete `scripts/dev/` and any remaining `scripts/test_*.py` shadows. Tests live in `tests/`.
- [ ] Consolidate the `doc/` mess: there are 5+ "architecture" docs (`architecture.md`, `model.md`, `decoder_redesign.md`, `resfold_design.md`, `diffusion.md`). For now, mark all of them stale at the top and let Phase E rewrite one canonical doc.

### 1.2 Preserve failed-experiment provenance (the "don't repeat yourself" guardrail)

The 20+ dead runs in `outputs/` total ~1.2 GB, mostly checkpoint binaries. We split them into two tiers:

- **Committed forever (small, human-readable):** for every past run, keep `train.log`, `split.json`, `plots/` (if any), and `config.yaml` (reconstruct from log if missing). Move these to `experiments/_archive/<run-name>/` and commit. They are small (≤ a few MB each), they're the *evidence* of what failed.
- **Local-only (large, recoverable):** checkpoints (`best_model.pt`, `final_model.pt`) stay in `outputs/_archive/<run-name>/` and that path is `.gitignored`. If a future experiment ever needs to resume from one of these, the training config is committed so it can be retrained — and at that point the failure is worth re-examining anyway.

Special case: `stage2_full_*` (4 KB each, no checkpoint) — these crashed at init due to the now-fixed `pipeline.py:49` import bug. Keep their logs explicitly labeled as "crashed at init, do not interpret as Stage 2 failure."

### 1.3 The failed-experiment registry

Create `experiments/REGISTRY.md`, committed. One row per past run with:

| Run | Model (registry name) | What was tried | Outcome | Why we stopped | Key files |
|---|---|---|---|---|---|

Seed it with the existing 21 runs based on log inspection. Each row should be one sentence in each column — not a writeup, just enough that *before starting any new experiment, we can scan this table and confirm we're not redoing a dead branch.*

Every future Phase A/B/C/D run appends a row when it ends (pass *or* fail). The registry is the structural reason we won't rerun the same idea.

### 1.4 Reproducibility plumbing

- [ ] One config per experiment phase under `configs/train/resfold/`: `phase_a_overfit.yaml`, `phase_b_n4.yaml`, `phase_b_n16.yaml`, `phase_b_n64.yaml`, `phase_b_n256.yaml`, `phase_c_data.yaml`. No CLI-flag soup.
- [ ] Each training run writes: `config.yaml` (exact), `split.json`, `best_model.pt`, `train.log`, `metrics.json`, `plots/`. This is the contract for reproducibility.
- [ ] On run completion (pass or fail), the training script appends a row to `experiments/REGISTRY.md`. Make this automatic — a manual step will be skipped under pressure.
- [ ] `tests/` audit — refactor.md notes these were written against `scripts/models/*`. Run `pytest tests/` and fix anything that broke when models moved to `src/tinyfold/model/`.

### 1.5 Turn-key install smoke test

- [ ] From a clean clone (use `git worktree`), walk through README: `pip install -e .` → `python scripts/data/prepare_data.py` → `python scripts/train_resfold.py --config configs/train/resfold/phase_a_overfit.yaml`. Document every undocumented flag, fix it, retest.
- [ ] Confirm `predict.py` runs on a checkpoint and produces a valid PDB.
- [ ] Confirm `web-light/` static viewer opens.

**Acceptance for Phase 0:** `git status` clean. `pytest tests/` passes. `ls outputs/` shows only active runs. `experiments/_archive/` and `experiments/REGISTRY.md` exist and cover every past run. All model files in `src/tinyfold/model/archive/` instantiable via the registry. The README's commands all work from a clean clone.

---

## 2. Phase A — Single-sample overfit (~2h GPU)

**Goal:** prove the forward+reverse loop is correct. If the model can't reconstruct one protein it has seen 10,000 times, nothing else matters.

### 2.1 Config

- Data: 1 sample from `data/processed/samples.parquet`. Prefer small (L ≈ 80, both chains ≤ 40).
- Model: `ResidueDenoiser`, `c_token=128, trunk_layers=4, denoiser_blocks=4` (~1.5–2M params).
- Training: 5,000 steps, `batch_size=1` (just repeat the sample), `lr=3e-4`, `continuous_sigma=True`, no augmentation, no self-conditioning yet.

### 2.2 Pass criteria

- Train one-step MSE → < 0.005 on normalized coords.
- Sample RMSE (T=50, per-step Kabsch) → < 0.5 Å on that one sample.

### 2.3 Probe checklist (do not skip; do before moving on)

- [ ] **Loss-by-σ plot** (10 bins of log σ). Flat or slowly rising = healthy; U-shape or end-spikes = schedule/data-scale mismatch.
- [ ] **Trajectory plot** at t = T, T/2, T/4, 0. Should see noise → blob → fold. Jumping to the answer at t=T = schedule too easy; never folding = conditioning ignored.
- [ ] **Self-consistency** across 5 seeds. Spread > 2 Å = sampling unstable.
- [ ] **Frame-drift check**: compare `align_per_step=False` vs `True`. Gap > 1 Å = rotation-equivariance leak; will need rotation augmentation in Phase B.

### 2.4 Failure modes (if Phase A fails)

| Symptom | Likely cause | First fix |
|---|---|---|
| Loss plateaus at ~1.0 | Conditioning disconnected | check `trunk_tokens.std() > 0.1` |
| Train great, sample huge | Schedule mismatch / sampler drift | turn on per-step Kabsch and recenter; verify `sigma_data` matches coord std |
| Loss → 0 in 200 steps, sample collapses to mean | Output proj init too small | `output_proj.weight.std() ≥ 0.02` |
| NaN | Adam at σ extremes | clip σ to `[0.002, 10.0]`, add `lr_warmup=200` |
| Phase A flat-out fails | Data pipeline bug | run Phase D.1 (deterministic Stage 2) to isolate |

---

## 3. Phase B — Tiny-N sweep (~6–8h GPU)

**Goal:** find the N where memorization breaks and measure the train/sample gap.

For N ∈ {1, 4, 16, 64, 256}, same Phase A architecture. Each run stops as soon as the criterion below is met.

### 3.1 Knobs (held constant unless noted)

- Model size: Phase A config until N=256 — then optionally bump c_token to 256.
- `n_steps ≈ 4000 * sqrt(N)`.
- `batch_size = min(N, 32)`, no grad accumulation.
- Hold-out: N=1, 4 → none (overfit only). N ≥ 16 → 4 samples held out.

### 3.2 Pass criteria

| N | Train one-step MSE | Sample RMSE (train) | Sample RMSE (held-out) |
|---|---|---|---|
| 1 | < 0.005 | < 0.5 Å | — |
| 4 | < 0.01 | < 0.8 Å | — |
| 16 | < 0.02 | < 1.2 Å | report |
| 64 | < 0.03 | < 1.5 Å | report |
| 256 | < 0.05 | < 2.0 Å | **< 6 Å** (first real signal) |

### 3.3 What to log every run

- All six metrics from §0.2 (note: **interface-vs-core binning is required from N=16 up**, not deferred to Phase C).
- Wall-clock / 1000 steps (for the blog).
- `best_model.pt` + `split.json` so any run is reproducible standalone.
- One sample-trajectory PNG.

### 3.4 Decision gates

- **Fails at N=4 after N=1 passes** → architecture issue. Don't grow N. Investigate conditioning / capacity.
- **Fails at N=16 after N=4 passes** → conditioning too weak. Bump `trunk_layers` to 6 before continuing.
- **N=256 passes train, fails held-out** → expected generalization wall. This is where the "real" experiment starts.

---

## 4. Phase C — Single scaling sweep + cheap ablations (~1–2 days GPU)

Only enter if Phase B N=256 passes train-set sample RMSE. The earlier plan proposed three sweeps; for blog scope this is compressed to **one data sweep + one capacity comparison + only the cheap ablations**.

### 4.1 Data sweep (the blog figure)

- N ∈ {256, 1K, 4K, 8.6K} (8.6K = full filtered DIPS-Plus, matches the existing `stage1_10M` regime).
- Same model size as best Phase B config.
- Fixed 100-sample held-out test set, committed to `split.json`.
- Plot: Train RMSE & Sample RMSE (held-out) vs N. This is *the* figure in the blog post.

Interpretation: train flat + held-out drops with N → bottleneck is data, keep going. Held-out plateaus → bottleneck is capacity, run §4.2.

### 4.2 Capacity comparison (only if data plateaus)

At the best N, run **one** bigger model: `trunk_layers=9, denoiser_blocks=6, c_token=256`. Just to answer "would more capacity help." Not a sweep — one yes/no.

### 4.3 Cheap ablations (do all; each is one run at best Phase C config)

- [ ] **No self-conditioning** vs `self_cond_prob=0.5`. Expected: self-cond ~0.5 Å better.
- [ ] **No per-step Kabsch** vs on. Expected: large degradation without it (validates the AF3/Boltz-2 choice).

### 4.4 Skipped for v1 (justify "future work" in blog)

- Rotation augmentation on/off, discrete vs continuous σ, distance-consistency weight ablation. Each is interesting but not headline material.

### 4.5 Failure mode instrumentation (already in every run from N=16)

- **Long-protein degradation**: bin held-out by L ∈ [40, 100, 200, 300].
- **Interface vs core RMSE** (8 Å cutoff to other chain) — the PPI-specific story.
- **Intra-chain vs inter-chain distance preservation** — separate the dist-consistency loss into the two components.
- **DockQ on held-out** once Sample RMSE crosses ~5 Å. Below 5 Å backbone is roughly where DockQ becomes meaningfully non-zero.

---

## 5. Phase D — Stage 2 sanity (parallel to A/B/C)

Independent of the diffusion track. Stage 2 is deterministic — a single forward pass — so it should overfit trivially. Run during idle GPU time. **If it fails, the data pipeline has an atom-order bug and everything downstream is suspect.**

### 5.1 D.1 — Standalone sanity (3K steps, ~30 min)

- Train `AtomRefinerV2` standalone on N=64 with **GT centroids as input**.
- `c_token=256, n_layers=6` (smaller than the prod 18 — overfit first, scale second).
- 3,000 steps, `lr=3e-4`, geometry losses on.
- Pass: atom RMSE (Kabsch-aligned, backbone) on train < 0.4 Å; bond lengths within 0.05 Å of ideal on val samples.

### 5.2 D.2 — Noise-robustness curve

- Add Gaussian noise σ ∈ {0.5, 1.0, 2.0} Å to GT centroids before refinement.
- Plot atom RMSE vs centroid noise. This curve directly answers "if Stage 1 gets within X Å, is the whole pipeline usable?"

### 5.3 D.3 — Two-stage end-to-end (only after A/B/C and D.1/D.2)

- Plug best Stage 1 into best Stage 2.
- Run `eval_two_stage.py` on the 100-sample held-out.
- Report DockQ, lDDT, atom RMSE. Save config + checkpoints + metrics JSON to `outputs/two_stage_demo/`.
- **This is the artifact the README and the blog claim. It must reproduce from `outputs/two_stage_demo/config.yaml` alone.**

---

## 6. Phase E — Freeze and write (1–2 days, no GPU)

Only after D.3 produces real numbers.

- [ ] One `RESULTS.md`: held-out numbers, exact training command, exact eval command, link to committed `split.json`. One paragraph of prose, no marketing.
- [ ] One architecture doc replacing the 5+ overlapping ones in `doc/`. Describe the actual ResFold two-stage architecture as run in Phase C/D.
- [ ] Update `scripts/README.md` to the actually-current scripts.
- [ ] Tag the repo `v0.1-demo`.
- [ ] *Then* hand off to `specdriven/handoff/tinyfold.md` Priority 1 (the blog post).

---

## 7. Before starting any new experiment

Read `experiments/REGISTRY.md` end-to-end. If the new idea matches a row already there, the default is **don't run it again** — instead, read the linked `train.log` and `plots/`, write down what would be different this time, and only then run. This rule is the structural reason we won't burn a week re-discovering that the `attention_v2` decoder doesn't converge on N=8K, or that `frame_decoder` plateaus at DockQ 0.04. The registry is cheap to scan; a redundant 8-hour training run is not.

## 8. Decision gates — where to stop, not power through

- **Phase A fails** → don't keep tuning the diffusion model; run Phase D.1 to isolate (data pipeline vs Stage 1 architecture).
- **Phase B N=64 fails on train** → architecture issue. One capacity bump, then if still failing, fall back to IterFold (rank 5) and write up that the diffusion approach didn't fit on a single GPU.
- **Phase C held-out plateaus > 8 Å** → that's the honest blog number. Stop training. Write it up.
- **Phase D.3 DockQ ≈ 0** → kill the two-stage story for v1; blog post is "Stage 1 centroid RMSE on a 4070 Ti" + scaling figure. Still defensible.

---

## 9. Out of scope for this round

These exist in the codebase or notes but **stay parked** until E ships:
- `ResFoldE2E` multi-sample diffusion conditioning.
- All `IterFold` variants (`AnchorDecoder`, `FrameDecoder`, `AtomAnchorDecoder`).
- Energy-based aux losses (LJ, electrostatics).
- DNA / small-molecule extension.
- MSA features.
- Hard-split benchmarking.
- Promotion / arXiv writeup (per `specdriven/handoff/tinyfold.md`).

The point is one defensible, reproducible diffusion result first.

---

## 10. Concrete first session

In order:

1. **Phase 0.1–0.3**: commit the `model/archive/` move (all failed models stay importable), build `experiments/_archive/<run>/` + seed `experiments/REGISTRY.md` from the 21 existing runs, resolve git status, audit/fix `tests/`.
2. **Phase 0.4–0.5**: build the `configs/train/resfold/phase_*.yaml` files, wire up auto-append to the registry, walk through the README from a clean worktree, fix any rough edges.
3. **Phase A.1**: single-sample overfit. ≤ 2h GPU.
4. **Phase A.2**: run all four probes. Write down what was healthy vs broken.
5. If A.1 passed cleanly, kick off Phase B N=4, N=16, N=64 in sequence overnight. Each ~30–90 min on the 4070 Ti SUPER.
6. **Stop.** Look at the data. Decide whether to continue to N=256 or fall back.

**No new model code in the first session.** The plan is to extract signal from the architecture we already have.
