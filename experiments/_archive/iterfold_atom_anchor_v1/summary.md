# iterfold_atom_anchor_v1

- **Model:** archive atom_anchor variant
- **Date:** 2026-01-30 00:26:05
- **What was tried:** --n_train 20 --n_test 5 --n_steps 2000 --eval_every 500 --batch_size 8 --lr 1e-3 --use_atom_anchor_decoder --geom_weight 0 --c_atom 64 --output_dir outputs/iterfold_atom_anchor_v1
- **Outcome:** >>> Test (5):  RMSE=6.58� | DockQ=0.141
- **Why stopped:** manually killed before completion (no Finished: line)

Model family: archive atom_anchor variant. Trained on ? samples / tested on ? for up to 2000 steps at lr=0.001. See ``train.log`` for full configuration; key outcome metrics above.
