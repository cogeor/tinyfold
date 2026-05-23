# stage2_10M

- **Model:** resfold_stage2
- **Date:** 2026-01-28 10:43:12
- **What was tried:** Freeze the live `stage1_10M/best_model.pt` (6-layer trunk + 6-block denoiser) and train a 6-layer atom assembler for 50,000 steps at batch 8 x grad_accum 4 (eff_batch 32), lr=1e-4. Loss weights bond=1.0, angle=0.5, omega=0.5, chirality=0.5.
- **Outcome:** one metric line was emitted (`Step 100 | loss: 2.6335 | atom: 2.4439 | geom: 1.2192 | lr: 1.00e-04 | 4397s`) before the process died; the log was left NUL-padded by the abnormal shutdown.
- **Why stopped:** crashed at init due to pipeline.py:49 import bug; not a Stage 2 failure signal.

The bug is fixed as of commit bb4cff6; rerun before drawing any Stage 2 conclusions. Unlike the three `stage2_full_*` runs, this one survived to step 100 before the eval-time import hit, so the one recorded loss value (`2.6335`) is a useful sanity check on the run-config but not a quality signal.
