# SOTA Landscape for Joint Fold + Dock PPI (2024-2026)

**Date:** 2026-05-25
**Purpose:** Position TinyFold against the current state of the field. DiffDock-PP (2023) is no longer a representative competitor; the action has moved to AF3-style co-folding.

## TL;DR

1. The "rigid docking" track (DPP, EquiDock) is essentially dormant. The honest non-AF3 docking baseline is now **DFMDock** (NeurIPS 2024).
2. The "joint fold + dock" track (AF3-class) is where everyone publishes. Six open or semi-open models in this class as of mid-2026: AF3, Boltz-1/2, Chai-1, Protenix-v1, HelixFold3, OpenFold3.
3. **Protenix-Mini** (sub-300M params, ESM substitution, two-step ODE sampler) is the smallest published AF3-class model and our most direct positioning rival.
4. **PINDER** (2.3M dimers, leakage-controlled splits) is the new expected benchmark. DIPS/DB5 are increasingly considered leaky.
5. Architectural consensus: Pairformer trunk + token-then-atom diffusion + recycling. MSAs de-emphasized; single-sequence mode supported by Boltz/Chai.

## Categories

### Joint fold+dock (AF3-class) — our category

| Model | Group | Year | Params | Open | Notes |
|---|---|---|---:|---|---|
| AlphaFold 3 | DeepMind / Isomorphic | 2024 | ~hundreds of M | Weights, non-commercial | Pairformer (48 blocks) + atomic diffusion. PoseBusters ~76%. |
| Boltz-1 | MIT (Wohlwend, Corso, Jaakkola, Barzilay) | 2024 | ~AF3 scale | MIT license, full | First fully open AF3 clone. Trunk-run confidence head. |
| Boltz-2 | MIT + Recursion | 2025 | ~AF3 scale | MIT license, full | Adds affinity module + steering. FEP-level Kd at 1000× speedup. |
| Chai-1 | Chai Discovery | 2024 | unstated | Weights, non-commercial | Single-sequence mode supported. PoseBusters 77%. |
| Chai-2 | Chai Discovery | 2025 | unstated | Closed, early-access | Ab-Ag DockQ > 0.8 on 34% (vs Chai-1 16%). |
| Protenix-v1 | ByteDance AI4Science | 2025-26 | 368M | Apache 2.0, full | Matches AF3 at matched compute. PXMeter eval suite. |
| HelixFold3 | Baidu | 2024 | unstated | Open, academic | Beats AF-Multimer on interfaces; lags AF3. |
| OpenFold3 | AQLab consortium | 2025 | unstated | Apache 2.0 | Bitwise AF3 reproduction. 10M training set released. |
| RoseTTAFold All-Atom | Baker Lab | 2024 | unstated | Open | Non-diffusion; 42% complex success. |

**Architectural consensus 2024-2026:** Pairformer trunk + token-then-atom diffusion decoder + self-conditioning/recycling. Atom-level diffusion has decisively replaced IPA/frame regression.

### Rigid docking (DPP-class) — effectively dead as a publication track

- **DFMDock** (NeurIPS 2024) — denoising force-matching + energy head. On DB5.5: **44% sampling success, 16% top-1** vs DPP **8% / 0%**. This is the honest rigid baseline now.
- **ElliDock** (Jan 2024) — fastest pure-regression rigid docker; ElliDock ≈ DPP on quality, much faster.
- No 2025-2026 pure E(3)-equivariant rigid-docking diffusion paper of note.

### Small / efficient AF3-class — TinyFold's niche

- **Protenix-Mini** (Jul 2025) — ESM in place of MSA; redundant Pairformer + diffusion blocks pruned; **two-step ODE sampler** replaces multi-step AF3 sampler. 1-5% benchmark degradation vs full Protenix.
- **Protenix-Mini+** (Oct 2025) — scalable Pairformer for lightweight inference.
- **Pallatom** (Aug 2024) — all-atom dual-track AF3-style generator; "atom14" residue representation; efficient AF3 alternative.
- **AlphaFlow / ESMFlow** (ICML 2024) — flow-matching repurposing of AF2/ESMFold; demonstrates fold backbones → generative samplers cheaply.

No published model sits at "~12M trainable + frozen pLM + AF3-style atomic diffusion for PPI." Protenix-Mini at sub-300M is the smallest. **TinyFold occupies a real gap.**

## Benchmarks

- **PINDER** (bioRxiv 2024-07; Proxima/NVIDIA/MIT) — 2.3M dimer systems, structurally-clustered splits, holo+apo+predicted structures, 180-dimer leakage-free test set. **Now the expected PPI eval for publication.** They explicitly retrained DPP on clean splits and showed the original numbers were leakage-inflated (vindicating our 94% leakage finding).
- **CASP16** (2024 → published 2025) — verdict: complex prediction remains unsolved. Top server: **MassiveFold** (mass-sampling AFM/AF3 wrapper). Top group: MULTICOM4 at TM 0.797 / DockQ 0.558 (Phase 1).
- **CAPRI Rounds 56-58** (2024-25) — heavily AF-Multimer/AF3 dominated; HADDOCK/ClusPro relevant for re-ranking.
- **DB5.5** — still used for rigid docking comparisons. DFMDock reports here.
- **Ab-Ag specialized:** AbBiBench (Jun 2025), DeepAAAssembly (Nov 2025, +12.9% DockQ over raw AF3).
- **Metric shift:** field reports DockQ + ipTM + PoseBusters; raw C-RMSD increasingly viewed as inadequate.

## Diffusion-specific advances

- **Flow matching vs diffusion:** all SOTA structure prediction (AF3, Boltz, Chai, Protenix, HelixFold3) is score-based diffusion. FM (AlphaFlow, FrameFlow, FoldFlow, P2DFlow) wins on ensembles and efficiency. Pragmatic split: FM for ensembles/speed, diffusion for single-structure SOTA.
- **Self-conditioning / recycling** is standard inside the diffusion sampler; Pallatom has "dual-track recycling" inside diffusion blocks.
- **Few-step sampling:** Protenix-Mini's two-step ODE and Boltz-2's "steering" are evidence that AF3 samplers don't need 200 steps.
- **Confidence calibration for diffusion samples is *not* solved.** Best post-AF3 work: actifpTM (Bioinformatics 2025), ipSAE (2025). Boltz-2 still lists calibration as open. DFMDock's dual force+energy head is one alternative direction. Our K=40 ranker gap is the same problem the field has.

## Positioning implications for TinyFold

**Drop "DPP successor" framing entirely.** The honest framing is:

> "Smallest open AF3-class joint fold + dock PPI model. ~12M trainable parameters with frozen ESM-2; single-GPU; no MSAs; DIPS-Plus only."

**Baselines required for publication:**
1. **Co-folding (our category):** Boltz-1 (open, MIT), Protenix-v1 (Apache, 368M), AF3-server.
2. **Rigid sanity:** DFMDock on DB5.5 (not DPP).
3. **Benchmarks:** PINDER interface-clean splits, optional CASP16 multimer subset, optional Ab-Ag DockQ subset.

**Risk:** Protenix-Mini and Boltz are moving down-market quickly. Protenix-Mini's ESM substitution + two-step ODE is uncomfortably close to our direction. **Ship before scooped.**

## SOTA-at-our-scale: where TinyFold needs to land

No published model sits in the sub-50M-trainable AF3-class joint fold+dock bucket. The smallest published is Protenix-Mini (16 Pairformer + 8 diffusion blocks, frozen ESM2-3B) and they don't even disclose trainable params. **There is no published params-vs-DockQ scaling curve** for AF3-class models — we'd be drawing the first data point in our regime.

**Concrete target to credibly claim SOTA-at-our-scale:**

| Metric | Target | Reference |
|---|---:|---|
| PINDER-AF2 DockQ success rate (DockQ > 0.23) | **≥ 0.30** | Boltz-1 0.625 / Chai-1 (no MSA) 0.698 — these set the ceiling, not our target |
| PINDER-AF2 interface lDDT | **≥ 0.45** | Protenix-Mini 0.49, Protenix-Tiny 0.43 — we'd target between them |
| Trainable params | < 50M | Protenix-Mini doesn't disclose; we'd own this column |
| Inference per complex | < 5s | Protenix-Mini 2-step ODE delivers ~50% FLOPs reduction |

**PINDER-AF2** (180 dimers, leakage-filtered against AF2-MM training) is the primary report-target. PINDER-S (250) as the fast iteration set. PINDER-XL (1955) for the final paper.

We have no clean PINDER number for AF-Multimer or Boltz-1 in the available extracts — that needs to be looked up from PINDER paper Table 2/3 before we quote a "gap closed" number.

## Architectural ideas worth borrowing (ranked by value/effort)

### 1. Protenix-Mini's 2-step ODE sampler — DO THIS FIRST

Pure **sampler change, no retrain.** Set γ₀=0, η=1.0, use ODE-form Euler with 2 steps on a model trained with the normal multi-step recipe. Performance "nearly identical to 200-step baseline across interaction types." ~30 LoC change to our VE/Karras sampler. Highest-value lowest-effort import in the entire survey.

Source: Protenix-Mini arXiv:2507.11839 §3.1.

### 2. AF3 detached mini-rollout self-conditioning

At training, run a 20-step mini-rollout from pure noise **with no gradients**, then feed the result back as input to a single trained diffusion step. Detached. Inside the sampler at inference, outside the optimizer at training. Canonical AF3 recipe, inherited by Boltz-1. We never implemented this in v1. ~40 LoC + retrain.

This is what v2 Loop 04 was meant to be but our planned version was inference-only recycling. The real recipe is at train time.

### 3. Pairmixer trunk — directly tests our Pairformer hypothesis

**Keep triangle multiplication, drop triangle attention + sequence attention.** Pairmixer 12-layer matches Pairformer 12-layer (lDDT 0.73 vs 0.74) on RCSB; 4× faster on long seqs, 34% cheaper training. ~50 LoC (delete two attention modules from each block).

Source: "Triangle Multiplication Is All You Need" arXiv:2510.18870, Algorithm 1, §4.

### 4. DFMDock energy head as ranker replacement

Train an auxiliary energy head whose gradient must match the predicted score; rank samples by energy. Gives K-sample reranking as a side effect of the loss, eliminating the need for a separate confidence head. DFMDock got 16% top-1 on DB5.5 vs DiffDock-PP's 0% using this — that's exactly the ranker-gap symptom we have. ~80 LoC + retrain.

Risk: DFMDock is rigid SE(3); adapting to flexible centroid+backbone diffusion is non-trivial.

### 5. PINDER leakage-control reproduction for our DIPS-Plus split

Foldseek + MMseqs cluster, iAlign-score the interface, filter test against any training cluster crossing the threshold. PINDER's GitHub (`pinder-org/pinder`) has reference code we can mostly adopt. ~300 LoC of scripting. Required before publication.

### Lower-value ideas (probably skip)

- **Pallatom atom14**: overkill for backbone-only (N/CA/C/O). The transferable bit is *inside-diffusion recycling* (recycle every K blocks instead of between full passes), ~100 LoC.
- **Boltz-1 full-trunk confidence head**: doubles trunk params; DFMDock-style energy head gets us 80% of the value at a fraction of the cost.
- **Pairformer-block parameter sharing (ALBERT-style)**: speculative, no published evidence in AF3-class.

## Status of the pair-representation question

**First, a correction to what TinyFold actually is:** the active `ResFoldOneStep` model in `src/tinyfold/model/resfold/` has **no pair representation**. Trunk = `nn.TransformerEncoder` (single-track), denoiser = stack of single-track AdaLN+MHA blocks. The Pairformer code at `src/tinyfold/model/pairformer/` exists but is wired into older / archived paths, not into the trained model. So TinyFold is closer to **SimpleFold** (pure-transformer monomer fold, no pair rep) than to AF3/Boltz/Protenix.

This means the question is **not** "should we strip Pairformer to make it lighter," it's **"should we add a pair representation at all."**

**What the literature says about pair representations at small scale (none of it answers our question directly):**

- **Protenix-Mini** Table 1: 48→16 Pairformer blocks costs only 1-5% interface lDDT. Says pruning depth is fine. Says nothing about going to zero.
- **Pairmixer** (arXiv:2510.18870): triangle multiplication + FFN, no triangle attention, no sequence attention. Matches Pairformer at 12-layer scale on monomer lDDT. Says triangle attention is removable; says nothing about going to zero.
- **SimpleFold** (arXiv:2509.18480): zero pair rep, 100M params, gets ~90% of best monomer fold lDDT. **Does not test multimer.**
- **Protenix-Mini+** Table 3: DockQ > 0.23 drops 31% relative when pair capacity is cut aggressively (but not to zero). Interface metrics degrade faster than monomer metrics under pair-rep starvation.

**What's unknown that we'd need to test:**

- Whether the cliff we observe is *caused* by the missing pair rep, or by other factors (data, sampler, ranker, denoiser depth, ESM size).
- Whether adding a small pair track at our scale closes the cliff, leaves it open, or hurts (by burning trainable capacity).
- Whether a Pairmixer-style minimal pair track (triangle multiplication only) is enough, or if the cliff requires the full Pairformer machinery.

Both "pair rep is harmful at our scale" and "pair rep is necessary for PPI at our scale" are hypotheses. The cleanest experiment is an A/B at matched parameter count, with the PINDER-style cliff metric as the readout, and accept whichever way it lands. The result is informative either direction:
- If pair rep closes the cliff: we know the bottleneck.
- If it doesn't: the cliff is data- or sampler-limited, and we should stop spending capacity on architecture.

## Bottom-line v2 design recommendation

Based on the deep dive, the highest-value v2 changes (in order):

1. **2-step ODE sampler** — Protenix-Mini verbatim, no retrain (~30 LoC)
2. **AF3 detached mini-rollout self-conditioning at train time** (~40 LoC + retrain)
3. **Pairmixer trunk**: keep 6 blocks but strip triangle attention (~50 LoC + retrain)
4. **DFMDock-style energy head** for K-sample reranking (~80 LoC + retrain)
5. **PINDER-AF2 reproduction** for our DIPS-Plus splits (~300 LoC scripting)

This pivots v2 away from the current plan (scale up depth + ESM + interface loss) toward **architectural minimalism + better sampler + better ranker**. The current v2 plan is "make it more like Boltz." The deep-dive suggests "make it more like Protenix-Mini" is the better positioning play.

## References

- AlphaFold 3 (Nature 2024): https://www.nature.com/articles/s41586-024-07487-w
- Boltz-1 (bioRxiv 2024): https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1
- Boltz-2 (bioRxiv 2025): https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
- Chai-1 technical report: https://chaiassets.com/chai-1/paper/technical_report_v1.pdf
- Protenix-v1 (bioRxiv 2025): https://www.biorxiv.org/content/10.1101/2025.01.08.631967v1
- Protenix-Mini (arXiv:2507.11839): https://arxiv.org/abs/2507.11839
- Protenix-Mini+ (arXiv:2510.12842): https://arxiv.org/abs/2510.12842
- HelixFold3 (arXiv:2408.16975): https://arxiv.org/abs/2408.16975
- OpenFold3: https://github.com/aqlaboratory/openfold-3
- DFMDock (bioRxiv 2024): https://www.biorxiv.org/content/10.1101/2024.09.27.615401v1
- ElliDock (arXiv:2401.08986): https://arxiv.org/abs/2401.08986
- PINDER (bioRxiv 2024): https://www.biorxiv.org/content/10.1101/2024.07.17.603980v1
- CASP16 assessment: https://www.biorxiv.org/content/10.1101/2025.05.29.656875v1
- AlphaFlow (arXiv:2402.04845): https://arxiv.org/abs/2402.04845
- Pallatom (bioRxiv 2024): https://www.biorxiv.org/content/10.1101/2024.08.16.608235v2
- actifpTM (Bioinformatics 2025): https://academic.oup.com/bioinformatics/article/41/3/btaf107/8075121
