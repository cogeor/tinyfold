# stage1_50k_small

- **Model:** resfold_stage1
- **Date:** 2026-01-27 14:20:57
- **What was tried:** --mode stage1_only --n_train 8600 --n_test 100 --n_eval_train 50 --n_steps 50000 --eval_every 2000 --batch_size 16 --grad_accum 8 --lr 1e-4 --min_lr 1e-6 --min_atoms 320 --max_atoms 1200 --align_per_step --trunk_layers 9 --denoiser_blocks 7 --continuous_sigma --augment_rotation --output_dir outputs/stage1_50k_small
- **Outcome:** >>> Train Centroid RMSE (50): 7.1768 A | Test Centroid RMSE (100): 12.5198 A
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: resfold_stage1. Trained on 8600 samples / tested on 100 for up to 50000 steps at lr=0.0001. See ``train.log`` for full configuration; key outcome metrics above.
