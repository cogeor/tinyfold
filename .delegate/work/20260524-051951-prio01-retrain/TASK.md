# TASK — Priority 0 + 1 + Retrain

## Source

User invocation: `/dg:work implement priority 0 + 1, retrain, see if we gain anything.`

Definitive scope from `notes/post_compact_spec.md` §5 (Priority 0 + Priority 1):

- **A** — Wire `compute_dockq` into eval (already implemented at `tinyfold.model.metrics.dockq`, just unused). Log DockQ avg + success rate (>=0.23). Append to REGISTRY outcome.
- **B** — Add `compute_c_rmsd` in `tinyfold/model/losses/mse.py`: Kabsch-align predicted chain A to GT chain A only, apply that same transform to chain B, RMSD over **all** Calpha. Wire into eval. Re-eval Phase C N=8600 `best_model.pt` and append metrics to REGISTRY.
- **C** — Multi-sample inference: `--n_samples K` CLI flag; generate K samples per target at sigma_max from different seeds; report `oracle_best_RMSD@K`, `mean_RMSD@K` at K=1/5/40. Includes HDOCK-style **cluster-then-rank** (5 A interface-RMSD greedy cluster, return cluster representatives). Oracle ranking for now; G replaces with confidence head.
- **D** — Boltz Kabsch-interpolation sampler: in `sample_centroids_ve`, rigid-align `x_new` to `x_prev` after each Euler step (NOT x0_pred to x). New `kabsch_interp=True` arg. Success target: multi-step < 10.5 A on N=8600 test (would beat one-shot 10.59).
- **E** — EDM lambda(sigma) loss weighting. Existing `--loss_weighting` flag must use `(sigma^2 + sigma_data^2)/(sigma * sigma_data)^2`. Verify, fix if wrong.
- **F** — ESM-2 frozen embeddings + projection. `aa_embed: {learned, esm2_35M, esm2_150M}`. Tokenize at dataset prep, cache embeddings on disk. Projection to c_token.
- **G** — Confidence head (real version of C ranking). Tiny MLP on pooled denoiser tokens, regresses per-target lDDT vs GT. Used to rank multi-sample outputs without oracle.

## Final retrain + measurement

After A-G land:
- `configs/train/resfold/phase_d_n8600_full.yaml` — ESM-2-35M, EDM weighting, atom head, confidence head, K=5 + K=40 eval, Kabsch-interp sampler (if D worked) else one-shot.
- Re-run on full N=8600 (~2.5h on the 4070 Ti SUPER).
- Add REGISTRY row with: centroid RMSE, atom RMSE, C-RMSD, DockQ avg, DockQ success%, oracle@5, oracle@40, ranked@5, ranked@40.
- Update `notes/post_compact_spec.md` table with the delta vs the 9.87 A baseline.

## Out of scope

- Task H (pair representation + relpos) — Priority 2, deferred.
- Task I (self-conditioning) — conditional on D unlocking multi-step.
- PINDER hard-split eval — wait until we beat AF-Multimer.

## Constraints

- Keep `best_model.pt` from `outputs/resfold/phase_c_n8600/resfold_s1_8K_20260524_020409/` intact — needed for A/B/C/D re-eval, do NOT overwrite.
- Single-GPU (RTX 4070 Ti SUPER).
- Bare PyTorch.
- `src/` layout; tests in `tests/`.
- Each loop = one commit. Do not push.

## Acceptance

- All loops 01-07 commit cleanly with passing TEST.md commit gate.
- Phase D retrain registers a numerically lower test centroid RMSE than 9.87 A, OR the experimental delta is explained in the spec's "what didn't work" section.
- `notes/post_compact_spec.md` updated with the new headline number and the per-task delta.
