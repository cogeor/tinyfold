# stage2_full_8k

- **Model:** resfold_stage2
- **Date:** 2026-01-27 23:45:23
- **What was tried:** Freeze the same Stage 1 checkpoint (`stage1_50k_small/best_model.pt`) and train an oversized 19-layer atom assembler for 50,000 steps at batch 8 x grad_accum 4 (eff_batch 32), lr=1e-4. Largest assembler in the sweep at ~20M trainable params.
- **Outcome:** crashed at init: log ends at the "Training for 50000 steps..." banner with no metric lines and no `Finished:` line (~1.5 KB).
- **Why stopped:** crashed at init due to pipeline.py:49 import bug; not a Stage 2 failure signal.

The bug is fixed as of commit bb4cff6; rerun before drawing any Stage 2 conclusions. The "_8k" in the name refers to the `n_train` target (8,600), not the step count; the assembler is the largest of the three `stage2_full_*` variants.
