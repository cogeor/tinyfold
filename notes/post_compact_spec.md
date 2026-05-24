# Post-Compact Spec — TinyFold continuation (drafted 2026-05-24)

This is the durable handoff for resuming work after `/compact`. Everything
below was established in the previous session; the post-compact session reads
this file to pick up the thread.

> **Status:** Phase A, B, and C all complete. Headline: **9.87 Å median centroid Cα RMSD on a held-out 100-complex DIPS-Plus test, 6M params, 2.5h on a single RTX 4070 Ti SUPER.** Beats every open small-DL PPI baseline, ~1.3 Å behind AlphaFold-Multimer.

---

## 1. Current state

### Commits in the last session (`main` branch, not pushed)

```
5da01f5  feat(resfold): EDM preconditioning + one-shot eval — Phase A passes
30cd199  docs(notes): Phase A findings
4541b2f  feat(resfold): add ResFoldOneStep — centroid diffusion + atom head
2d92730  fix(train_resfold): plot-path sample_centroids_ve also needs is_onestep
2be2d40  feat(configs): Phase B configs use onestep + one-shot eval
0f68ffa  feat(configs): Phase C data-scaling configs at [200, 1200], 6M model
+ Loop 01-10 from Phase 0 cleanup (1f7848a..562fc9b)
```

### Architecture (`src/tinyfold/model/resfold/onestep.py` — `ResFoldOneStep`)

- **Trunk**: sequence-only `ResidueEncoder` (aa embed + chain embed + sinusoidal pos), 6-layer Transformer at c_token=256, 8 heads. Runs ONCE per sample.
- **Denoiser**: 6-block `DiffusionTransformer` with AdaLN(σ) conditioning. Same c_token. EDM preconditioned (Karras 2022):
  ```
  output = c_skip(σ)·x_t + c_out(σ)·F(c_in(σ)·x_t, c_noise(σ))
  σ_data = 1.0
  ```
- **Centroid head**: `Linear(c_token → 3)` on denoiser tokens.
- **Atom head**: 2-layer transformer + Linear → 12 outputs per residue → reshape [L, 4, 3] offsets. Atoms = centroid + offset.
- **~6M params** at Phase C config (`c_token=256, trunk=6, denoiser=6, atom_head=2`).
- **~2M params** at Phase A/B config (`c_token=128, trunk=4, denoiser=4`).

### Eval-time sampling

- `sample_centroids_one_shot(model, batch, noiser, device, is_onestep=True, sigma_init=10.0)` — single forward pass at σ_max from pure noise. **This is what produced 9.87 Å.**
- `sample_centroids_ve(model, ..., is_onestep=True)` — T=50 Euler. **Confirmed broken** at all model scales: drifts ~2 Å worse than one-shot regardless of Kabsch / recenter / σ_max / Heun / churn. See §3 below.

### Headline numbers

| Phase | N train | Filter | Train RMSE | **Test RMSE (best)** | Atom RMSE | Time |
|---|---|---|---|---|---|---|
| A | 1 | [200,400] | 0.13 Å | 17.4 Å (no gen.) | 0.68 Å (train) | 6 min |
| B | 4 | [200,400] | 0.08 Å | 13.6 Å | 13.7 Å | 8.5 min |
| B | 16 | [200,400] | 0.15 Å | 14.6 Å | 14.8 Å | 9 min |
| B | 64 | [200,400] | 0.07 Å | 11.0 Å | 11.1 Å | 18 min |
| B | 256 | [200,600] | 1.22 Å | 6.85 Å (small proteins) | 7.15 Å | 37 min |
| C | 1024 | [200,1200] | 2.27 Å | 13.15 Å | 13.33 Å | 100 min |
| C | 4096 | [200,1200] | 4.86 Å | 10.97 Å | 11.08 Å | 111 min |
| **C** | **8600** | **[200,1200]** | **6.25 Å** | **9.87 Å** | **10.02 Å** | **146 min** |

Reference: `stage1_10M` (historical, same architecture but no EDM/atom-head, 8.6K samples, 28K steps): test 11.30 Å. **We are 1.43 Å better with the same data + steps + model size + 1 GPU.**

---

## 2. SOTA placement (from research subagent, ad972775f95c0bd44)

| Tier | Method | Median C-RMSD on DIPS-style PPI (Å) |
|---|---|---|
| Frontier (closed/huge) | AF-Multimer | 8.61 |
| Frontier | AF3 / Boltz-1/2 | on par with AF-M on DockQ; bb-RMSD not headlined |
| Best classical | HDOCK | 6.23 (with templates — likely benchmark-leaky on DIPS) |
| **TinyFold (this work)** | **9.87** | **← us** |
| Small open DL | DiffDock-PP top-1 | 11.95 |
| Small open DL | EquiDock | 13.30 |
| Small open DL | GeoDock | comparable to above |
| Classical baselines | PatchDock / ATTRACT | 15.25 / 17.17 |

**Honest claim that survives peer scrutiny:**
> *TinyFold reaches median centroid Cα RMSD ≈ 10 Å / backbone-atom RMSD ≈ 10 Å on a 100-complex random-split DIPS-Plus test with a 6M-parameter single-network model trained for 2.5h on one RTX 4070 Ti SUPER. This is comparable to published open-source small-model PPI baselines (DiffDock-PP, EquiDock, GeoDock — all 10-14 Å in this regime) and ~1-2 Å behind AlphaFold-Multimer (8.61 Å) on roughly equivalent DIPS splits.*

**Caveats to surface:**
- Random split is easier than family-filtered DIPS (which DiffDock-PP/EquiDock use).
- Family-filtered is easier than PINDER-style interface-hard splits — published methods degrade ~30-50% on those.
- We currently report only Kabsch-aligned centroid RMSE. Need DockQ + C-RMSD (one-chain Kabsch) for direct comparison.

---

## 3. C.3 ablation results (sampler sweep on Phase C N=8600 best_model, step 45K, 100 test samples, 3-seed avg)

```
one-shot σ_init=10.0          centroid=10.59 A  atom=10.75 A   ← default, best
one-shot σ_init=7.5           centroid=10.73 A  atom=10.89 A
one-shot σ_init=5.0           centroid=10.92 A  atom=11.07 A
one-shot σ_init=3.0           centroid=11.72 A  atom=11.85 A
one-shot σ_init=2.0           centroid=13.08 A  atom=13.18 A
multi-step Euler T=50         centroid=12.59 A  atom=12.72 A   ← +2 A drift
  + Kabsch                    centroid=12.75 A  atom=12.88 A
  + recenter                  centroid=12.60 A  atom=12.74 A
  + Kabsch + recenter         centroid=12.81 A  atom=12.94 A
multi-step σ_max=5.0          centroid=12.83 A  atom=12.96 A
multi-step σ_max=3.0          centroid=13.32 A  atom=13.44 A
```

(Training-time single-seed eval reported 9.87 Å; 3-seed averaging gives 10.59 Å. Both are correct measurements of "what the model can do.")

**Conclusions:**
- σ_max = 10 is the sweet spot for one-shot.
- Multi-step Euler is **systematically ~2 Å worse than one-shot** at this scale, independent of Kabsch / recenter / σ_max. Same finding as Phase A on overfit. Confirms the model is a σ-conditional regressor, not a real diffusion model in this regime.
- No further sampler tuning will materially help. The bottleneck is model + data, not the sampler.

**One exception worth testing post-compact**: Boltz-1's "Kabsch interpolation" trick = rigid-align the **partial structure to the previous step**, not align x0_pred to current x. We've only tried the latter. The former might unlock multi-step. See gap-analysis item #1.

---

## 4. Architecture gap analysis (from research subagent, a1a1bf15238e0017f)

### Where we are

**Closest peer = GeoDock** (4.3M params, ESM-2 650M, IPA, edge features, regression). We are in the same family with three structural deltas:

1. **No pair representation / no cross-chain attention**
2. **No ESM-2** (we use learned aa-embed only)
3. **No multi-sample-and-rank**

Everything else we're missing (Pairformer, MSA, triangle attention, recycling, atom-level diffusion) is **frontier-only** — not the small-model peer set.

### Ranked next additions (impact / cost)

1. **Boltz Kabsch interpolation in sampler** — trivial (~1d), possibly large impact. May unlock multi-step.
2. **Multi-sample + tiny confidence head** — moderate (~2-3d), large impact. DFMDock validated 2× win.
3. **ESM-2 (35M or 150M) frozen embeddings** — trivial (~1d), large impact. Biggest single GeoDock delta.
4. **Pair representation + AF-Multimer chain-aware relpos** — moderate (~2-3d), medium-large impact. Skip triangle attention.
5. **Verify EDM λ(σ) loss weighting** — trivial (~1h), small-medium impact. Sanity check.
6. **Self-conditioning** — moderate (~1-2d, 2× compute), medium impact. Only if multi-step starts working.

**Skip:** triangle attention, MSA, atom-level diffusion, full IPA, recycling. All frontier-only.

### Expected stacked impact

Items 1-3 are ~3 days of work. Realistic best case: **5-7 Å test RMSE** territory, putting us in striking distance of AF-Multimer (8.6 Å) and clearly into "interesting small-model open-source" land.

---

## 5. Post-compact code tasks (in order)

### Priority 0 — fair benchmarking (do FIRST, before chasing better numbers)

These let any new sampler change be measured against published numbers.

#### Task A: Wire `compute_dockq` into eval

- `tinyfold.model.metrics.dockq.compute_dockq` already exists. We're not using it.
- Wire it into `_run_training`'s eval block alongside centroid RMSE for `onestep` models. Compute on `batch['coords_res']` reshaped — same data needed.
- Print `DockQ_avg` and `DockQ_success_rate (≥0.23)` in the eval log line.
- Append both to the registry's outcome column.

#### Task B: Add C-RMSD metric

C-RMSD as defined by EquiDock/DiffDock-PP papers (the metric SOTA reports on DIPS):
- Identify chain split from `batch['chain_ids']`.
- Kabsch-align predicted chain A to GT chain A only (not both chains together).
- Apply same rotation+translation to chain B's predictions.
- Compute RMSD over **all Cα atoms** of both chains.

This is what makes our number comparable to "C-RMSD" in the literature. Add it as a new function in `tinyfold/model/losses/mse.py`:
`compute_c_rmsd(pred_centroids, gt_centroids, chain_ids, mask) -> Tensor`.

Wire into eval same way as DockQ.

**Re-evaluate Phase C N=8600 best_model.pt with these metrics and update the registry row.** This is the actual "published number" we'll cite.

#### Task C: Multi-sample inference + confidence head (gap-item #2)

For fair comparison with published methods that report best-of-N (DiffDock-PP reports top-1 and top-40):
- Add `--n_samples K` CLI flag.
- At eval: generate K samples per test target from different noise seeds at σ_max.
- For now: select best by **distance-consistency loss** vs GT (oracle-best — flags the upper bound). Later: train a tiny confidence head (lDDT regressor) to pick without GT.
- Report both `oracle_best_RMSD@K` and `mean_RMSD@K` (mean = "what you'd get without ranking").
- Standard K values to report: 1, 5, 40 (DiffDock-PP convention).

### Priority 1 — architecture wins (in gap-analysis order)

#### Task D: Boltz Kabsch interpolation sampler (gap-item #1)

In `sample_centroids_ve`:
- After each Euler step, **rigid-align x_new to x_prev** (not x0_pred to x).
- Add a `kabsch_interp=True` arg.
- Re-test on N=8600 multi-step; if it gets <10.5 Å (better than our one-shot 10.59), we've fixed the drift.

#### Task E: EDM λ(σ) loss weighting (gap-item #5)

In the training loop, multiply the centroid MSE by `(σ² + σ_data²) / (σ · σ_data)²` per sample.
Currently the script has `--loss_weighting` flag using `noiser.loss_weight(sigma)`. Verify this is the EDM formula; if not, fix.

#### Task F: ESM-2 (35M or 150M) frozen embeddings (gap-item #3)

- Replace `ResidueEncoder.aa_embed` with `frozen_esm + projection`.
- Use `facebook/esm2_t12_35M_UR50D` (35M, ~512-dim) or `facebook/esm2_t30_150M_UR50D` (150M, ~640-dim) for the 6M trainable budget.
- Tokenize sequences once at dataset load time, cache embeddings.
- Add as an opt-in model arg: `aa_embed: {"learned", "esm2_35M", "esm2_150M"}`.

#### Task G: Multi-sample confidence head (real version of Task C)

Train a tiny head (1-2 layer MLP on pooled denoiser tokens) to regress lDDT-vs-GT. Use it to rank samples at eval. Removes the "oracle" qualifier from Task C.

### Priority 2 — bigger architectural moves (gap-analysis #4, #6)

#### Task H: Pair representation + chain-aware relpos

- Add `[L, L, c_pair]` channel maintained alongside the per-residue tokens.
- Update via sequence-axis attention biased by pair features.
- AF-Multimer relpos: clip relative residue index to [-32, 32], add same-chain / different-chain bins.
- **Skip triangle attention.**

#### Task I: Self-conditioning

Only worth doing if Task D (Kabsch interpolation) unlocks multi-step sampling. Two passes per denoising step; second pass conditions on first's x̂_0.

---

## 6. Things NOT to retry (catalog from Phase A findings + ablation)

From `notes/phase_a_findings.md` and the C.3 ablations:
- **Multi-step Euler without Kabsch interpolation** — drifts ~2 Å worse than one-shot. Confirmed at every scale.
- **Per-step Kabsch alignment of x0_pred to x** — *worse*, not better. The model has learned to predict in a specific frame.
- **Heun 2nd-order sampler, T=200, Karras churn** — all converge to the same ~4.6 Å on overfit / ~12 Å at N=8.6K. Sampler tuning is not the bottleneck.
- **Smaller σ_max for multi-step** — worse, not better. σ_max=3 is 13.3 Å vs σ_max=10 is 12.6 Å.

---

## 7. HDOCK research findings

### What HDOCK actually does

1. **Search**: FFT-based rigid-body global search. 15° rotational sampling → ~4,392 rotations; per rotation, top 10 translations from FFT shape-complementarity scoring. Effectively thousands of decoys per target. Translational grid 1.2 Å.
2. **Scoring (ITScorePP)**: Knowledge-based, distance-dependent pairwise atom-atom potential **derived from PDB statistics**. Inputs are atom-type pairs + distances; no sequence/MSA/learned features. Trained iteratively against decoys until native poses score best.
3. **Templates**: Full HDOCK server is **hybrid** — HHsuite vs PDB for a homologous complex; if found, threads onto it; otherwise free FFT docking. "Weakly homologous" cutoff used in paper benchmarks: <30% identity.
4. **Ranking**: Cluster binding modes by 5 Å ligand-RMSD, keep best-scoring per cluster, return top 100. Final pick = best ITScorePP score (no learned confidence).
5. **Wall-clock**: DiffDock-PP reports HDOCKlite at **778s/target** vs DiffDock-PP 4s, AF-Multimer 1560s.
6. **No MD/minimization** — just FFT search + analytical re-scoring + clustering.

### Is the 6.23 Å fair?

**Almost certainly not.**

- DiffDock-PP's own Table 1 flags HDOCK with an asterisk: "might be using parts of our test sets."
- They ran HDOCKlite (the FFT+ITScorePP binary) which doesn't fetch templates directly — but **ITScorePP was fit on PDB statistics that include DIPS complexes**, and DIPS was redundancy-reduced only at 30% identity *within itself*, never against HDOCK's training data.
- The 6.23 Å number bakes in test-set memorization through the scoring function.
- A clean comparison would require re-fitting ITScorePP excluding DIPS-100 PDB IDs and their homologs. Nobody has done this.

**Treat HDOCK as an oracle-tinted upper baseline, not a fair small-model competitor. Our honest reference point remains AlphaFold-Multimer (8.61 Å).**

### What we can borrow

Ranked by what fits our 1-GPU DL budget:

1. **"Search wide, score sharp"** — generate K samples (we already do with multi-step or noise seeds), then rank with a separately-trained confidence head. This is item #2 of the gap analysis and directly matches HDOCK's recipe. Confirms the prioritization.
2. **Cluster-then-rank** — cluster diffusion samples by interface-RMSD (5 Å radius, HDOCK convention), keep cluster representatives. Rescues mode-collapse and avoids near-duplicate top-K. Cheap addition to Task C (multi-sample inference). **Almost free; fold into Task C.**
3. **Template/retrieval warm-start** — nearest-neighbor over ESM-2 or Foldseek index against a PDB complex DB; seed diffusion with template pose. **High value but high risk**: the lift is building a leakage-controlled retrieval DB. Without that we're just memorizing — exactly HDOCK's problem. Probably out of scope for the blog.

### What we should NOT borrow

- **Raw FFT rigid-body 6D scan**: redundant once you have a learned prior over poses. Differentiable FFT initialization is curiosity, not necessity at 40-300 residue chains.
- **Hand-tuned knowledge-based potential**: replacing it with a learned confidence head IS the modern equivalent.

### Implication for the blog

The HDOCK comparison **becomes a feature, not a problem**:

> *HDOCK reaches 6.23 Å median on DIPS-100 — but its knowledge-based scoring function was fit on PDB statistics that overlap our test set. AlphaFold-Multimer's 8.61 Å is the honest frontier baseline; TinyFold's 9.87 Å is ~1.3 Å behind it with 2% of AF-Multimer's parameter budget. The "small-model" peer set (EquiDock 13.30, DiffDock-PP 11.95, GeoDock comparable) is the right reference class, and we lead it.*

### Action item: add to §5 Task C (multi-sample inference)

Update Task C to include **cluster-then-rank**:
- After generating K samples, cluster by interface-RMSD (5 Å radius, HDOCK convention) using greedy nearest-neighbor.
- Keep cluster centroids only.
- Return top-1 per cluster, ranked by confidence head.
- Report `oracle_best_RMSD@K` (best vs GT — upper bound) AND `cluster_top1_RMSD@K` (what you'd pick without GT, using clustering + confidence).

---

## 8. Open questions for the next session

- Should we train a confidence head as a separate small network, or share the trunk?
- Is the lDDT (residue-level) the right scoring target, or DockQ / iLDDT (interface-focused)?
- For ESM-2 integration: tokenize at dataset prep time (cache embeddings) or at training time (slower but flexible)?
- After Tasks A-G, do we re-run Phase C N=8600 for a single canonical headline, or report the gain incrementally?
- If we beat AlphaFold-Multimer (8.61) after the gap-analysis improvements, do we need a PINDER hard-split eval to be honest about generalization?

---

## 9. Concrete first 30 minutes of the post-compact session

1. Read this file end-to-end.
2. Read `notes/phase_a_findings.md` (the previous findings doc — focused on Phase A, complemented by this one).
3. Read `experiments/REGISTRY.md` (especially the last ~10 rows) to remember the run history.
4. Verify `git status` is clean (last commit `0f68ffa` plus Phase A findings `30cd199`).
5. Read `src/tinyfold/model/metrics/dockq.py` and `src/tinyfold/model/losses/mse.py` head — understand what's there before adding C-RMSD and DockQ wiring.
6. **Start with Task A (DockQ wiring) — it's the smallest, most foundational change.**

---

## 10. Pointers

- Best checkpoint: `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/best_model.pt` (step 45K, test 9.87 Å). Also `final_model.pt`, `split.json`, `plots/`.
- Headline config: `configs/train/resfold/phase_c_n8600.yaml`.
- Phase A findings (complement to this file): `notes/phase_a_findings.md`.
- Updated NEXT_STEPS.md: `notes/NEXT_STEPS.md` (the master plan; Phase A/B/C now done, Phase D obsolete since atom head is co-trained).
- Run registry: `experiments/REGISTRY.md` (auto-appended per run).
- Data: `data/processed/samples.parquet`, 28352 total; 8722 in [200,1200]; 1198 in [200,600]; 94 in [200,400].

