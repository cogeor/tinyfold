# Loop 05 TEST — ESM-2 frozen embeddings

## Commit gate
- ready: yes
- reason: ESM cache prep + dataset loader + model gating all green; learned mode bit-identical to pre-Loop-05; all regression tests pass.

## Verification log

### 1. New unit tests
Command: `.venv/Scripts/python.exe -m pytest tests/test_esm2_cache.py -v`
Result: 6 passed in 2.26s.
Covers: cache shape, residue-encoder construction (learned + esm), forward runs, assertion when esm_embed missing, param-delta vs learned, no EsmModel params leaked.

### 2. Regression suite (prior loops)
Command: `.venv/Scripts/python.exe -m pytest tests/test_edm_loss_weight.py tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py tests/unit/test_kabsch_rigid.py tests/test_pose_clustering.py tests/test_multisample_eval.py tests/test_kabsch_interp_sampler.py -v`
Result: 30 passed, 1 skipped.
All Loop 01-04 tests green after the encoder/dataloader changes (esm_embed signature threaded as optional `None` for backward compat).

### 3. ESM cache prep
Command: `python scripts/prepare_esm2_embeddings.py --parquet data/processed/samples.parquet --output-dir data/processed/esm2_35M --variant 35M --device cuda --filter-residues 200-1200`
Result:
- 25050 NPZ files written, 8.19 GB on disk
- 23.94 min wall time (combined 2 sessions: 5435 + 19616, all skipped duplicates handled correctly via the idempotent skip-if-exists path)
- 0 failures

### 4. ESM-2 mode smoke
Command:
```
.venv/Scripts/python.exe scripts/train_resfold.py \
    --model_kind onestep --mode stage1_only --aa_embed esm2_35M \
    --n_train 4 --n_test 2 --n_steps 10 --eval_every 5 --batch_size 2 \
    --c_token_s1 128 --trunk_layers 2 --denoiser_blocks 2 --continuous_sigma \
    --min_atoms 800 --max_atoms 1600 \
    --output_dir outputs/_loop05_smoke_esm
```
Result: exit 0, total time 9 s. Train RMSE 24.5 -> 22.7, test 28.1 -> 25.8 across 2 eval steps. ESM cache loaded successfully via mmap, projection trains, no EsmModel forward at training time.

### 5. Learned-mode regression smoke
Command: `.venv/Scripts/python.exe scripts/train_resfold.py --config configs/train/resfold/phase_b_n4.yaml --n_steps 10 --eval_every 5 --output_dir outputs/_loop05_smoke_learned`
Result: exit 0, test RMSE 14.1151 A — bit-identical to Loop 01's smoke. Default `aa_embed=learned` path preserved.

### 6. REGISTRY hygiene
5 stray rows from earlier pytest integration tests + smoke runs reverted. Both smoke output dirs deleted. Phase C checkpoint untouched (no `--checkpoint` arg used).

## Open concerns

- The cache uses 8.19 GB locally. Acceptable on a 4070 Ti SUPER box; flag if the box hits disk-space pressure during Phase D retrain.
- ESM mode smoke RMSE (25.8 A on N=4) is worse than learned-mode (14.1 A on the same N=4 `phase_b_n4.yaml` — but different filter, different chains). Numbers aren't directly comparable; the smoke only checks that ESM path runs cleanly. The real ESM vs learned comparison lands in Loop 07 (Phase D retrain).
- ESM-2-150M variant is plumbed but not exercised here. Phase D defaults to 35M per plan.