# stage1_10M

- **Model:** resfold_stage1
- **Date:** 2026-01-28 03:20:53
- **What was tried:** ResFold Stage 1 (trunk-only diffusion), 6-layer trunk + 6-block denoiser, 8,600 train / 100 test, 50,000 steps at eff_batch 256, lr=1e-4, continuous-sigma + rotation augment + align-per-step. This is the "current best" baseline -- it is **LIVE**.
- **Outcome:** converged train=6.68 A / test=11.30 A at step 28,000 (last `>>> Train Centroid RMSE` line before kill). Last metric line: `Step 29400 | loss: 0.089759 | mse: 0.0794 | dst: 0.0486 | lr: 4.27e-05 | 24680s`.
- **Why stopped:** manually killed at ~step 29,400 (still improving slowly).

**LIVE** -- the live directory at `outputs/stage1_10M/` is preserved; this archive copy exists for registry index completeness only. Best checkpoint stays at `outputs/stage1_10M/best_model.pt`. Loop 04 will relocate the binary to its long-term home and update this entry's pointer; do not delete this archive copy.
