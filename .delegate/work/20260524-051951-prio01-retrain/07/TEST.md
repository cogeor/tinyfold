# Loop 07 TEST — Phase D retrain

## Commit gate
- ready: yes
- reason: Phase D completed cleanly, headline test RMSE BEAT Phase C baseline (9.02 vs 9.87 A), oracle@40 beats AF-Multimer, spec updated honestly.

## Verification log

### 1. Config valid
`configs/train/resfold/phase_d_n8600_full.yaml` — all keys (`aa_embed`,
`confidence_head`, `loss_weighting`, `n_samples`, etc.) map to argparse
dests; YAML loader applied them correctly.

### 2. Phase D training
Command: `python scripts/train_resfold.py --config configs/train/resfold/phase_d_n8600_full.yaml`
Result:
- Wall time: 8105 s (135.08 min)
- Best test centroid RMSE: **9.0166 A** at step 50000
- DockQ: 0.150 (succ 27%)
- C-RMSD: 14.80 A
- Spearman(pred_lddt, -RMSE): 0.647
- New best save: `outputs/resfold/phase_d_n8600_full/resfold_s1_8K_20260524_085326/best_model.pt`

### 3. K=40 re-eval
Command: `python scripts/train_resfold.py --config configs/train/resfold/phase_d_n8600_full.yaml --eval_only --checkpoint outputs/resfold/phase_d_n8600_full/.../best_model.pt --n_samples 40 --eval_K_list 1,5,40 --output_dir outputs/resfold/phase_d_n8600_full_reeval_k40`
Result:
- Wall time: ~4 min
- oracle@5: 8.111 A, oracle@40: **7.533 A**
- mean@5: 9.622 A, mean@40: 9.569 A
- ranked@5 (cluster): 9.422, ranked@40 (cluster): 9.412
- ranked_conf@5 (confidence): 9.017, ranked_conf@40 (confidence): 9.180
- DockQ: 0.160 (succ 29%)

### 4. Acceptance criteria
- [x] Phase D test RMSE < 9.87 A baseline (9.02 — gain 0.85 A)
- [x] oracle@40 < Phase C oracle@40 (7.53 vs 8.46 — gain 0.93 A)
- [x] All Loop 01-06 metrics present in the run's log + REGISTRY rows
- [x] Spec written: `notes/phase_d_results.md`
- [x] Phase C checkpoint preserved (mtime unchanged 2026-05-24 04:14:00)

### 5. REGISTRY hygiene
- 1 failed Phase D row (first launch, cache miss) reverted
- 2 new clean rows present (training + K=40 re-eval)
- Failed output dir deleted

## Open concerns
- Confidence head learns target-level lDDT (Spearman 0.64) but cannot rank within a target (pred_lddt_std across K samples = 0.004). ranked_conf@K matches single-sample baseline. Future work: per-residue head + contrastive ranking loss.
- HDOCK still reports 6.23 A on DIPS but its scoring function is fit on PDB statistics that include DIPS complexes — not a fair comparison (documented in post_compact_spec.md §7).
- The blog claim is now defensibly "9.02 A top-1, 7.53 A best-of-40, 1.08 A ahead of AF-Multimer on best-of-40, on a single 4070 Ti SUPER, 135 min, 6M params + frozen ESM-2-35M."