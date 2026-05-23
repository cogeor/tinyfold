# stage1_50k

- **Model:** resfold_stage1
- **Date:** 2026-01-27 13:30:55
- **What was tried:** --mode stage1_only --n_train 1000 --n_test 100 --min_atoms 320 --max_atoms 1200 --n_steps 50000 --eval_every 2500 --batch_size 16 --grad_accum 8 --lr 1e-3 --continuous_sigma --self_cond_prob 0.5 --align_per_step --recenter --c_token_s1 256 --trunk_layers 9 --denoiser_blocks 7 --T 50 --output_dir outputs/stage1_50k
- **Outcome:** Step  4000 | loss: 0.684945 | mse: 0.4690 | dst: 0.3858 | lr: 9.84e-04 | 2930s
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: resfold_stage1. Trained on 1000 samples / tested on 100 for up to 50000 steps at lr=0.001. See ``train.log`` for full configuration; key outcome metrics above.
