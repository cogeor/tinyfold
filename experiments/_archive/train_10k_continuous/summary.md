# train_10k_continuous

- **Model:** resfold_stage1
- **Date:** 2026-01-24 11:42:18
- **What was tried:** --mode stage1_only --n_train 8600 --n_test 100 --n_eval_train 50 --n_steps 50000 --eval_every 2000 --batch_size 64 --grad_accum 4 --output_dir outputs/train_10k_continuous --T 50 --augment_rotation --lr 1e-4 --min_lr 1e-6 --min_atoms 200 --max_atoms 1200 --align_per_step --trunk_layers 14 --denoiser_blocks 10 --continuous_sigma
- **Outcome:** Step 21500 | loss: 0.101040 | mse: 0.0628 | dst: 0.0420 | lr: 6.13e-05 | 32039s
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: resfold_stage1. Trained on 8600 samples / tested on 100 for up to 50000 steps at lr=0.0001. See ``train.log`` for full configuration; key outcome metrics above.
