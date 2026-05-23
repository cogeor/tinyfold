# oneshot_plots

- **Model:** visualization only
- **Date:** 2026-01-30 00:41 (mtime of the first PNG)
- **What was tried:** No training. Rendered one-shot prediction PNGs from an existing checkpoint to compare against ground truth.
- **Outcome:** Visualization run -- no training, no checkpoint. Produced N=15 PNGs and PDBs under `train/` and `test/`. Kept here for index completeness; the PNGs themselves are the evidence.
- **Why stopped:** completed (not a training job)

This entry exists purely so the registry index is exhaustive. The source directory `outputs/oneshot_plots/` contained 5 PNGs at the top level (~1.8 MB total) plus `train/` (5 PNGs) and `test/` (5 PNGs) subdirs of rendered comparisons. The PNGs themselves are not copied into the archive because they are large (~300 KB each, exceeding the 100 KB cutoff) and the visualization can be regenerated cheaply from any Stage 1 checkpoint.
