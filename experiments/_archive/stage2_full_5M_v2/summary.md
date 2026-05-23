# stage2_full_5M_v2

- **Model:** resfold_stage2
- **Date:** 2026-01-28 01:08:47
- **What was tried:** Freeze the same Stage 1 checkpoint (`stage1_50k_small/best_model.pt`) and train a smaller 5-layer atom assembler for 50,000 steps at batch 16 x grad_accum 4 (eff_batch 64), lr=1e-4. Same loss weights as `stage2_full_10M`; identical geometry loss.
- **Outcome:** crashed at init: log ends at the "Training for 50000 steps..." banner with no metric lines and no `Finished:` line (~1.5 KB).
- **Why stopped:** crashed at init due to pipeline.py:49 import bug; not a Stage 2 failure signal.

The bug is fixed as of commit bb4cff6; rerun before drawing any Stage 2 conclusions. This was the "v2" of the assembler-size sweep (5M-param variant); rerun jointly with `stage2_full_10M` and `stage2_full_8k` to recover the size-vs-quality curve.
