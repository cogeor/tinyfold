# Phase A — what worked, what didn't, what to remember

**Date:** 2026-05-23
**Commit:** `5da01f5` (model+sampler), `4541b2f` (initial onestep build)
**Run:** `outputs/resfold/phase_a_overfit_onestep/resfold_s1_1_20260523_203428/`

## TL;DR

`ResFoldOneStep` (2.07M params, centroid diffusion + parallel atom head) overfits a single PPI sample to centroid RMSE **0.13 Å** / atom RMSE **0.68 Å**, well under the < 0.5 Å Phase A bar. Two non-obvious things were load-bearing: EDM preconditioning and one-shot inference. Multi-step Euler sampling does NOT recover the memorized structure even at zero one-step MSE.

## What worked

### 1. EDM (Karras 2022) preconditioning

```python
c_skip(σ) = σ_data² / (σ² + σ_data²)
c_out(σ)  = σ · σ_data / √(σ² + σ_data²)
c_in(σ)   = 1 / √(σ² + σ_data²)
output    = c_skip(σ) · x_t + c_out(σ) · F(c_in(σ) · x_t, c_noise(σ))
```

with `σ_data = 1.0` (matches our normalized coordinates).

**Why it matters:** without EDM, the model at low σ has to learn the *identity function* (output = input minus tiny noise). With EDM, the identity comes for free via `c_skip · x_t`; the network `F` learns the residual correction. Without this split, the network ends up overfit to the narrow `GT + small_noise` training distribution at low σ and can't generalize to anything else at inference. This is the standard AF3/Boltz formulation; we missed it on the first pass and the failure mode was textbook (1.2 Å sampler plateau even with one-step MSE ≈ 0).

### 2. One-shot inference at σ_max

For overfit verification, a SINGLE forward pass at `sigma = sigma_max` (=10) from pure noise gives **0.15 Å centroid / 0.68 Å atom** on the trained sample (5-seed avg). The model has memorized "this sequence → this structure," and at high σ the EDM output is dominated by the learned prior `F`, with `c_skip · x_t` contributing only ~1% of the output magnitude.

Wired as `--one_shot_sample` flag in `train_resfold.py` and as `one_shot_sample: true` in `phase_a_overfit_onestep.yaml`.

**This is appropriate for overfit (N=1) only.** For larger N where the model can't memorize, the multi-step trajectory should add value. Keep this flag off for Phase B/C.

### 3. Two parallel heads, both off the same denoiser tokens

`centroid_proj(denoiser_tokens) -> [L, 3]` and `atom_head(denoiser_tokens) -> [L, 4, 3]` offsets. Atoms = centroid + offset.

- Centroid is the diffusion target.
- Atom head sees the same conditioning signal — gradients from atom-MSE + geometry losses flow back through both heads, so the trunk learns features that respect atomic geometry without atom-level diffusion.
- Atom-head proj uses small init (`std=0.02`) so offsets start near 0 and geometry losses (bond length / bond angle / omega / O-chirality) bootstrap valid backbone shape.
- Atom-MSE loss has linear warmup (α = min(1, step/500) · 0.5) so the noisy early atom head doesn't poison the trunk.

Cost: +267K params over centroid-only (12.9% of total).

### 4. `final_model.pt` checkpoint, not just `best_model.pt`

`best_model.pt` is keyed on test RMSE. For N=1 the test signal is noise — best_model freezes at step 500 even though the model continues to improve through step 5000. The final-step save is the source of truth for "what the model actually learned." Added unconditionally; doesn't break larger-N runs.

## What didn't work (catalog so we don't retry)

- **Multi-step Euler sampling on overfit.** Even with EDM, T=50 Euler drifts to 4.8 Å. The model's predictions degrade after the first step because the trajectory state at intermediate σ doesn't match the training distribution `GT + σ · noise`.
- **Per-step Kabsch alignment** during sampling: *worse* (1.6 → 2.3 Å). The model has learned to predict in a specific frame; rotating disrupts that.
- **Heun 2nd-order sampler, T=200 Euler, Karras churn sampler**: all converge to the same ~4.6 Å. Sampler choice is not the bottleneck — the model's intermediate-σ behavior on off-distribution inputs is.
- **EDM "fix" without one-shot.** EDM helps the model behave well at training inputs across all σ, but doesn't itself rescue the sampler trajectory. The combination of EDM + one-shot is what passes Phase A.

## Open questions / for Phase B/C

- Does the multi-step drift disappear with diverse training data (N≥4)? The hypothesis: when the model can't memorize, the trajectory state at intermediate σ stays closer to the training manifold because the model has been trained on a wider input distribution. To test: run Phase B with multi-step sampling (drop `--one_shot_sample`) and compare against one-shot.
- If multi-step still drifts at N=64+, the diffusion process is doing no work over single-shot conditional regression. That would be a real finding for the blog — and would simplify the demo.
- For the blog: the σ ∈ [0.002, 10] training distribution underweights high σ (90th percentile is ~2). The model's "memorization at high σ" works because there's little data there to fit a distribution to, and the only signal available is sequence-conditional. Worth noting.

## Concrete artifacts

- Code: `src/tinyfold/model/resfold/onestep.py`, `scripts/train_resfold.py` (search `is_onestep`, `EDM`, `one_shot_sample`).
- Config: `configs/train/resfold/phase_a_overfit_onestep.yaml`.
- Run: `outputs/resfold/phase_a_overfit_onestep/resfold_s1_1_20260523_203428/{final_model.pt, plots/, split.json, train.log}`.
- Registry entry: `experiments/REGISTRY.md` (last row).
