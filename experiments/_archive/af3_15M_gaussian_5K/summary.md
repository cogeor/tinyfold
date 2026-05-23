# af3_15M_gaussian_5K

- **Model:** archive af3_style
- **Date:** 2026-01-20 00:34:18
- **What was tried:** --model af3_style --noise_type gaussian --n_steps 100000 --n_train 5000 --n_test 500 --eval_every 5000 --output_dir outputs/af3_15M_gaussian_5K --h_dim 128 --n_layers 6 --batch_size 32 --grad_accum 4 --lr 1e-3 --min_atoms 320 --max_atoms 1000
- **Outcome:** Step 50200 | loss: 0.006970 | lr: 5.02e-04 | 45012s
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: archive af3_style. Trained on 5000 samples / tested on 500 for up to 100000 steps at lr=0.001. See ``train.log`` for full configuration; key outcome metrics above.
