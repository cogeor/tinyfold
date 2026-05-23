# frame_decoder_v2

- **Model:** archived FrameDecoder
- **Date:** 2026-01-28 20:25:09
- **What was tried:** --n_train 100 --n_test 20 --n_eval_train 50 --n_steps 5000 --eval_every 500 --batch_size 32 --grad_accum 1 --lr 1e-4 --use_frame_decoder --c_atom 128 --decoder_layers 4 --trunk_layers 6 --centroid_weight 1.0 --dist_weight 5.0 --geom_weight 0.1 --min_atoms 100 --max_atoms 600 --rotation_augment --mask_ratio_min 0.5 --mask_ratio_max 1.0 --output_dir outputs/frame_decoder_v2
- **Outcome:** >>> Test (20):  RMSE=16.90� | DockQ=0.068
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: archived FrameDecoder. Trained on ? samples / tested on ? for up to 5000 steps at lr=0.0001. See ``train.log`` for full configuration; key outcome metrics above.
