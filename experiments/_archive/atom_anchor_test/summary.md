# atom_anchor_test

- **Model:** archive atom_anchor variant
- **Date:** 2026-01-30 00:23:01
- **What was tried:** --n_train 5 --n_test 2 --n_steps 100 --eval_every 50 --batch_size 2 --grad_accum 1 --use_atom_anchor_decoder --output_dir outputs/atom_anchor_test --c_atom 64
- **Outcome:** completed, best test RMSE 15.8733 A, best DockQ 0.3868
- **Why stopped:** completed (n_steps budget reached)

Model family: archive atom_anchor variant. Trained on ? samples / tested on ? for up to 100 steps at lr=0.0001. See ``train.log`` for full configuration; key outcome metrics above.
