# iterfold_geom_decoder

- **Model:** iterfold
- **Date:** 2026-01-28 19:05:47
- **What was tried:** --n_train 100 --n_test 20 --n_eval_train 50 --n_steps 5000 --eval_every 500 --batch_size 32 --grad_accum 1 --lr 1e-4 --output_dir outputs/iterfold_geom_decoder --c_token 256 --trunk_layers 6 --decoder_layers 6 --c_atom 128 --min_atoms 100 --max_atoms 600 --rotation_augment --mask_ratio_min 0.5 --mask_ratio_max 1.0 --geom_weight 1.0 --dist_weight 5.0 --use_geometric_decoder
- **Outcome:** completed, best test RMSE 13.5361 A, best DockQ 0.0715
- **Why stopped:** completed (n_steps budget reached)

Model family: iterfold. Trained on ? samples / tested on ? for up to 5000 steps at lr=0.0001. See ``train.log`` for full configuration; key outcome metrics above.
