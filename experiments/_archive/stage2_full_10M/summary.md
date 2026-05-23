# stage2_full_10M

- **Model:** resfold_stage2
- **Date:** 2026-01-27 23:49:55
- **What was tried:** Freeze a Stage 1 checkpoint (`stage1_50k_small/best_model.pt`, 9-layer trunk + 7-block denoiser) and train a 10-layer atom assembler for 50,000 steps at batch 8 x grad_accum 4 (eff_batch 32), lr=1e-4, with full geometry-loss weights (bond=1.0, angle=0.5, omega=0.5, chirality=0.5).
- **Outcome:** crashed at init: log ends at the "Training for 50000 steps..." banner with no metric lines and no `Finished:` line (~1.5 KB).
- **Why stopped:** crashed at init due to pipeline.py:49 import bug; not a Stage 2 failure signal.

The bug is fixed as of commit bb4cff6; rerun before drawing any Stage 2 conclusions. The reconstructed config in `config_reconstructed.yaml` captures the intended hyperparameters, so this archive entry is sufficient evidence to replay the experiment once the import is fixed.
