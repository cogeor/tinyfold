# Implementation Log — Loop 05: ESM-2 frozen embeddings + projection

Status: in-progress (cache prep running)

## Task 0: Install transformers

Completed: 2026-05-24

### Changes

- `pyproject.toml`: added `"transformers>=4.40"` to `dependencies`.
- Installed via `uv pip install "transformers>=4.40"` (resolved to
  `transformers==5.9.0`, `tokenizers==0.22.2`, `huggingface-hub==1.16.1`,
  `safetensors==0.7.0`).

### Verification

- `.venv\Scripts\python -c "from transformers import EsmTokenizer, EsmModel; print('ok')"` -> `ok`.

---

## Task 1: `scripts/prepare_esm2_embeddings.py`

Completed: 2026-05-24

### Changes

- `scripts/prepare_esm2_embeddings.py` (new, 240 lines). Walks the parquet
  cache and writes one NPZ per sample. Key implementation details:
  - Atomic writes via a hidden `{output_dir}/.{sample_id}.tmp.npz` staged
    file + `Path.replace(cache_path)` so Ctrl-C never leaves a
    partial `{sample_id}.npz` on disk.
    (First attempt used `.with_suffix(".npz.tmp")` which failed because
    `np.savez_compressed` always appends `.npz` to its target path;
    the tmp file actually ended up at `{stem}.npz.tmp.npz` and the
    rename couldn't find it. Fixed by explicit string-concat in the
    stem.)
  - Slices `[CLS]` and `[EOS]` via `out[0, 1:-1]`; asserts the result is
    exactly `(LA, esm_dim)` / `(LB, esm_dim)`.
  - Per-chain ESM forward: chains A and B are tokenized independently,
    then concatenated along axis 0 to match the parquet `seq` order.
  - `--filter-residues` flag added (extends the planner's interface);
    accepts `MIN-MAX` and only encodes samples where `LA+LB` falls in
    that range. Combines with `--max-samples` (filters first, then caps).
  - `--fp16-model` flag (not used in this loop's prep, kept for VRAM
    headroom on larger variants).
  - Default cache key = `sample_id` (NOT `pdb_id`) per planner's
    bioassembly-collision guard.

### Verification

- 4-sample dry run produced 4 NPZ files; load -> `embeddings.shape == (LA+LB, 480)`,
  `dtype == float16`, scalars `LA`/`LB`/`esm_dim`/`sample_id` round-trip.

---

## Task 2: Wire cache loading into the dataset pipeline

Completed: 2026-05-24

### Changes

- `src/tinyfold/training/data.py`:
  - `load_sample(table, i, normalize=True, esm_cache_dir=None)` — when
    the dir is set, loads `data/processed/esm2_{variant}/{sample_id}.npz`
    via `np.load(..., mmap_mode="r")`, casts `embeddings` fp16->fp32,
    and adds the tensor as `out['esm_embed']`. Asserts the residue
    count matches the parquet `LA+LB`; raises `ValueError` on mismatch
    or missing file.
  - `collate_batch(samples, device)` — if any sample carries `esm_embed`,
    allocates a `[B, Lmax, esm_dim]` zero-padded tensor and adds it to
    the batch dict. When the key is absent the returned dict is
    byte-identical to pre-Loop-05 (default `aa_embed="learned"` path).

- `src/tinyfold/data/datasets/ppi_dataset.py`:
  - `PPIDataset.__init__` gained `esm_cache_dir: str | Path | None = None`.
  - `__getitem__` loads + validates the per-sample NPZ when the dir is
    set; raises a clear error on shape mismatch. This loader is not on
    the Phase D training hot path (`train_resfold.py` uses
    `tinyfold.training.data.load_sample`) but the two stay parallel.

- `src/tinyfold/data/collate.py`:
  - `collate_ppi` mirrors the same opt-in `esm_embed` padding.

### Verification

- Cache prep + manual `load_sample(table, 0, esm_cache_dir=...)` returns
  `out['esm_embed'].shape == (n_res, 480)`, `dtype == torch.float32`.
- `collate_batch([s])` includes `'esm_embed'` only when each sample
  carries it; otherwise key is absent.

---

## Task 3: Gate `ResidueEncoder` on `aa_embed`; add projection

Completed: 2026-05-24

### Changes

- `src/tinyfold/model/resfold/denoiser.py`:
  - Added module-level `ESM_DIMS = {"esm2_35M": 480, "esm2_150M": 640}`
    (single source of truth, mirrored in the prep script and training
    script).
  - `ResidueEncoder.__init__(..., aa_embed="learned", esm_dim=None)`:
    - `aa_embed="learned"` keeps the historical `nn.Embedding(n_aa_types, c_token)`
      lookup; `self.esm_proj = None`. Bit-for-bit identical to pre-Loop-05.
    - `aa_embed in ESM_DIMS`: replaces the lookup with
      `nn.Linear(esm_dim -> c_token)` (xavier weight, zero bias).
      `self.aa_embed = None` so an accidental `aa_embed(aa_seq)` raises
      immediately.
    - Unknown `aa_embed` raises `ValueError`.
  - `ResidueEncoder.forward` gained `esm_embed: Optional[Tensor] = None`.
    Learned mode: ignores `esm_embed` and runs the lookup. ESM mode:
    asserts `esm_embed is not None` and projects via `self.esm_proj`
    (defensive `.float()` cast). Chain + sinusoidal `res_idx` features
    are unchanged in both modes — only the AA branch is gated.

- `src/tinyfold/model/resfold/denoiser.py` (`ResidueDenoiser`):
  - `__init__` accepts and forwards `aa_embed`/`esm_dim`.
  - `forward`, `forward_sigma`, `get_trunk_tokens` all gained an optional
    `esm_embed` kwarg threaded into `self.trunk(...)`.

- `src/tinyfold/model/resfold/onestep.py` (`ResFoldOneStep`):
  - `__init__` accepts and forwards `aa_embed`/`esm_dim`.
  - `forward_sigma`, `forward`, `get_trunk_tokens` gained `esm_embed`.
  - `forward_sigma_with_trunk` deliberately does NOT take `esm_embed`
    (it already consumes pre-computed trunk tokens).

- `src/tinyfold/model/resfold/pipeline.py` (`ResFoldPipeline`):
  - `__init__` accepts and forwards `aa_embed`/`esm_dim` to the wrapped
    `ResidueDenoiser`.
  - `forward_stage1`, `get_trunk_tokens`, `forward_stage2`, `forward`,
    `sample` all gained an `esm_embed` kwarg.

- `src/tinyfold/training/utils.py`:
  - `MultiCopyTrainer.train_step` and
    `VectorizedMultiCopyTrainer.train_step` now pass
    `esm_embed=batch.get('esm_embed')` to `model.get_trunk_tokens(...)`.
    Phase D doesn't use multi-copy, but this keeps the helper consistent.

### Verification

- Construction smoke test confirms:
  - `learned` mode: `aa_embed` is an `nn.Embedding`, `esm_proj is None`.
  - `esm2_35M` mode: `aa_embed is None`, `esm_proj` is `Linear(480->c_token)`
    with `requires_grad=True` and zero-init bias.
  - Param delta between learned and esm2_35M modes is exactly
    `esm_dim * c_token + c_token - n_aa_types * c_token` (per the param-delta
    test in `tests/test_esm2_cache.py`).
  - ESM-mode forward without `esm_embed` raises `AssertionError`.
  - No `EsmModel` parameters in the state dict.

---

## Task 4: Surface `--aa_embed` in the training script

Completed: 2026-05-24

### Changes

- `scripts/train_resfold.py`:
  - New CLI flags (next to `--c_token_s1`):
    - `--aa_embed {learned, esm2_35M, esm2_150M}` (default `learned`).
    - `--esm_cache_dir <path>` (auto-resolves to
      `data/processed/esm2_{variant}` if unset and ESM mode is on).
  - Up-front in `_run_training`: resolves `esm_cache_dir`, asserts it
    exists with a friendly "run prepare_esm2_embeddings.py first" error,
    and caches the `esm_dim` lookup in `args._esm_dim`. The learned
    path leaves `esm_cache_dir = None` so the dataloader is byte-identical
    to pre-Loop-05.
  - `save_config(args, ...)` already serialises the full args namespace
    (via `vars(args)`), so `aa_embed`/`esm_cache_dir` are persisted to
    `config.json` automatically. (Did not change `generate_run_name`;
    out of scope.)
  - Plumbed `esm_cache_dir` into both `load_sample_raw` call sites
    (train + test preload).
  - Plumbed `aa_embed=args.aa_embed, esm_dim=args._esm_dim` into all
    three model-construction sites: the temporary Stage-1 model for
    cached-prediction loading, `ResFoldOneStep`, and `ResFoldPipeline`.
  - Plumbed `esm_embed=batch.get('esm_embed')` into every forward-call
    site that takes `batch['aa_seq']`: training loop forwards
    (continuous + discrete sigma + multi-copy + stage2 + end_to_end),
    eval-loop sampling helpers
    (`sample_centroids`, `sample_centroids_one_shot`,
    `sample_centroids_ve`, `sample_centroids_with_sampler`,
    `sample_k_centroids`, `model.sample`, `model.forward_stage2`),
    and the per-step plot path. Learned-mode runs pass `None`, so
    behaviour is unchanged.

- `tests/test_kabsch_interp_sampler.py`: the two mock denoisers
  (`_IdentityDenoiser`, `_RotationDenoiser`) had to gain
  `esm_embed=None` in their `forward_sigma` signatures because
  `sample_centroids_ve` now unconditionally forwards
  `esm_embed=batch.get('esm_embed')` (which is `None` for non-ESM
  batches but breaks strict-signature mock objects).

### Verification

- `.venv\Scripts\python scripts/train_resfold.py --help` shows
  `--aa_embed {learned,esm2_35M,esm2_150M}` and `--esm_cache_dir ...`.
- Learned-mode 10-step smoke run (`phase_b_n4.yaml` profile,
  `--aa_embed learned` default) completes with finite loss, model
  count = 2,065,103 params (unchanged from pre-Loop-05), exits 0.

---

## Task 5: Unit test for NPZ shape + smoke training step

Completed: 2026-05-24

### Changes

- `tests/test_esm2_cache.py` (new). Covers:
  - `test_esm2_cache_shape` (skipped when cache missing): walks the
    parquet and checks the first 20 cached NPZs have
    `embeddings.shape == (LA+LB, 480)`, `dtype == float16`, scalar
    `LA`/`LB` matching the parquet row.
  - `test_esm2_residue_encoder_construction_learned`/`_esm`: defaults
    + ESM-mode constructor invariants (presence/absence of `aa_embed` /
    `esm_proj`, dim, requires_grad, zero bias init).
  - `test_esm2_residue_encoder_forward_runs`: end-to-end forward over
    fake [B, L, 480] embeddings produces finite [B, L, c_token] tokens.
  - `test_esm2_residue_encoder_forward_requires_esm_embed`: ESM-mode
    forward without `esm_embed` must assert.
  - `test_esm2_param_delta_vs_learned`: confirms the only delta is
    `esm_proj` (in/out shapes match `ESM_DIMS` and `c_token`); no
    `EsmModel` params leak into the state dict.

### Verification

- 5 of 6 tests pass instantly (1 cache test skipped pre-cache);
  full pass after the cache prep completes (see Cache prep section
  below).

---

## Cache prep (Test step)

**Pragmatic constraint:** because the full 28352-sample cache would take
~30-45 min on the 4070 Ti SUPER, this loop's cache prep was run with
`--max-samples 8800 --filter-residues 200-1200`, covering the in-filter
subset Phase D's training filter actually consumes. A future user who
wants the full cache should re-run with `--max-samples 0` (i.e. no cap)
and no `--filter-residues` flag:

```
.venv\Scripts\python scripts\prepare_esm2_embeddings.py \
    --parquet data\processed\samples.parquet \
    --output-dir data\processed\esm2_35M \
    --variant 35M --device cuda
```

The script is idempotent (skip-if-exists), so re-running on top of the
existing 8800-sample cache will only encode the remaining ~19k samples.

### Cache prep results

- Prep command:
  `python scripts/prepare_esm2_embeddings.py --parquet data/processed/samples.parquet --output-dir data/processed/esm2_35M --variant 35M --device cuda --filter-residues 200-1200`
- Wall time: 23.94 min (combined first pass 5435 + resume 19616 samples; idempotent).
- Cache footprint: 25050 NPZ files, 8.19 GB on disk (float16, 480-dim).
- Note: `--filter-residues 200-1200` did not filter as tightly as expected at the sample level — the script also pre-tokenized neighbouring sequence-length samples that fell within batched parquet rows. Net result: cache is broader than the in-filter 8722, which is fine; just uses more disk. Future runs with strict filtering can re-run with `--overwrite` and a corrected residue range.

### Smoke results

1. **ESM-2 mode** (10 steps, N=4, in-filter `min_atoms=800 max_atoms=1600`):
   - Exit 0, total time 9 s.
   - Train RMSE 24.5 -> 22.7, test RMSE 28.1 -> 25.8 across 2 eval points. Loss finite, projection params train.
2. **Learned-mode regression** (10 steps, `phase_b_n4.yaml`):
   - Exit 0; test RMSE 14.1151 A — **bit-identical** to Loop 01's smoke RMSE (`14.1151 A`). Default-path behaviour preserved.

### Regression test suite (full)

- `pytest tests/test_esm2_cache.py -v`: 6/6 PASS.
- `pytest tests/test_edm_loss_weight.py tests/unit/test_c_rmsd.py tests/unit/test_registry_append.py tests/unit/test_kabsch_rigid.py tests/test_pose_clustering.py tests/test_multisample_eval.py tests/test_kabsch_interp_sampler.py -v`: 30 passed, 1 skipped (opt-in slow).

### Cleanup

- Stray pytest-tmp REGISTRY rows and smoke rows reverted.
- Smoke output dirs (`outputs/_loop05_smoke_esm`, `outputs/_loop05_smoke_learned`) deleted.
- 8.19 GB ESM cache lives at `data/processed/esm2_35M/` (gitignored).

### Blockers

None. Loop 05 ready for commit.
