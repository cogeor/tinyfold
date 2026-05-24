# Loop 05: ESM-2 frozen embeddings + projection (Task F)

## Overview

Add a frozen ESM-2 sequence-embedding pathway to the trunk. The aim is to
replace the learned `nn.Embedding(n_aa_types, c_token)` lookup inside
`ResidueEncoder` with a frozen ESM-2 forward + a small trainable projection
to `c_token`. ESM-2 itself is **never loaded at training time**: we run
ESM-2 once over the full DIPS-Plus parquet cache and write per-sample
NPZ files to disk; the dataloader mmaps the cache and the model only sees
the precomputed embeddings.

Concretely:

* `aa_embed: {learned, esm2_35M, esm2_150M}` config field (default `learned`).
  Mirrored as a `--aa_embed` CLI flag in `scripts/train_resfold.py`.
* New `scripts/prepare_esm2_embeddings.py` to populate the cache once.
* New code-path in `ResidueEncoder.forward` that consumes a cached
  `esm_embed` tensor and projects through `nn.Linear(esm_dim -> c_token)`
  instead of the AA embedding lookup. The learned path stays bit-for-bit
  identical (gated by `aa_embed=="learned"`).
* Dataset + collate plumbing to load the per-sample NPZ, concat chain A
  + chain B along the L axis (to match the existing residue ordering),
  and pad to `Lmax` in the batch.
* Smoke training run (10 steps, 4-target subset, `aa_embed=esm2_35M`)
  to confirm loss is finite and projection params show `requires_grad=True`.

### Locked design decisions

| # | Decision |
|---|----------|
| 1 | Cache file path: `data/processed/esm2_{variant}/{sample_id}.npz` (one file per sample, key=`sample_id` from parquet, e.g. `10gs.pdb1_0`). |
| 2 | NPZ contents: `embeddings` float16 array of shape `[LA+LB, esm_dim]` (already concatenated to match `seq`/`chain_id_res` ordering), plus scalar `LA`, `LB` for shape verification. |
| 3 | Per-chain ESM-2 forward: we tokenize chain A and chain B **separately** (each is an independent biological chain), strip `[CLS]`/`[EOS]`, then concat. This matches how the trunk treats them (no cross-chain attention in the AA lookup itself). |
| 4 | Default variant: `esm2_35M` (`facebook/esm2_t12_35M_UR50D`, 480-dim, 6 layers). 150M (`facebook/esm2_t30_150M_UR50D`, 640-dim) is opt-in. |
| 5 | Tokenizer library: HuggingFace `transformers` (`EsmTokenizer`, `EsmModel`). Not currently installed -> Task 0 installs it. |
| 6 | Projection: `nn.Linear(esm_dim -> c_token)` lives in `ResidueEncoder`; cast input from float16 -> float32 inside the forward. Bias init=0, weight init=xavier_uniform_. |
| 7 | The `aa_embed`, `chain_embed`, and sinusoidal `res_idx` features remain. In ESM mode we **replace only** the AA-lookup branch; chain and positional features still go through `input_proj`. |
| 8 | Cache prep is run as part of this loop's TEST step (~15 min for 35M on the 8722 in-filter set; we cache the **full** 28352 to keep splits flexible). Subsequent runs are no-ops because each sample skips if its NPZ already exists. |

### Out of scope

* Training ESM-2 (frozen forever).
* 150M variant performance check (Phase D uses 35M; 150M kept as a flag).
* Pair-representation routing (Loop H, out of scope for the entire task series).
* PINDER eval, alternate tokenizers, ProtBERT/Ankh.

## Tasks

### Task 0: Install transformers

**Goal:** Make `transformers` importable from the project venv.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `pyproject.toml` |

**Steps:**
1. Add `"transformers>=4.40"` to the `dependencies` list in `pyproject.toml`.
2. Run `.venv\Scripts\pip install transformers` (or `pip install -e .` if
   the implementer prefers a full reinstall).
3. Verify: `.venv\Scripts\python -c "from transformers import EsmTokenizer, EsmModel; print('ok')"`.

**Verify:** Import succeeds with no `ModuleNotFoundError`.

---

### Task 1: Add `scripts/prepare_esm2_embeddings.py`

**Goal:** One-shot script that walks `data/processed/samples.parquet`, runs
ESM-2 over chain A + chain B of every sample, and writes per-sample NPZ
files to `data/processed/esm2_{variant}/{sample_id}.npz`.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `scripts/prepare_esm2_embeddings.py` |

**Interface:**
```
python scripts/prepare_esm2_embeddings.py \
    --parquet data/processed/samples.parquet \
    --output-dir data/processed/esm2_35M \
    --variant 35M \
    [--device cuda] [--batch-size 1] [--max-samples N] [--overwrite]
```

**Variant mapping (hardcoded constant in the script):**
```python
ESM_VARIANTS = {
    "35M":  ("facebook/esm2_t12_35M_UR50D",  480),
    "150M": ("facebook/esm2_t30_150M_UR50D", 640),
}
```

**Steps:**
1. Parse args (parquet path, output dir, variant, device, batch_size,
   max_samples, overwrite).
2. Load `EsmTokenizer.from_pretrained(model_id)` and
   `EsmModel.from_pretrained(model_id).eval().to(device)`. Set
   `torch.set_grad_enabled(False)`. `output_dir.mkdir(parents=True, exist_ok=True)`.
3. Read parquet `samples.parquet`. For each row, reconstruct the per-chain
   AA1 string:
   - `seq_idx = row['seq']`  # length LA+LB
   - `chain  = row['chain_id_res']`  # 0 for chain A, 1 for chain B
   - Map each integer back to a 1-letter code using `tinyfold.constants.IDX_TO_AA`
     (treat index 20 / "X" as `X` -> ESM tokenizer handles it).
   - `seq_a = "".join(IDX_TO_AA[i] for i in seq_idx[:LA])`
   - `seq_b = "".join(IDX_TO_AA[i] for i in seq_idx[LA:LA+LB])`
   - Sanity: `assert len(seq_a) == LA and len(seq_b) == LB`.
4. Cache path: `output_dir / f"{sample_id}.npz"`. If it exists and
   `not args.overwrite`, skip.
5. For each chain string `s` in `(seq_a, seq_b)`:
   - `tok = tokenizer(s, return_tensors="pt", add_special_tokens=True).to(device)`
   - `out = model(**tok).last_hidden_state`  # [1, L+2, esm_dim]
   - Slice off `[CLS]` and `[EOS]`: `emb = out[0, 1:-1].cpu().numpy().astype(np.float16)`
   - Assert `emb.shape == (len(s), esm_dim)`. If not, log + skip the sample.
6. Concatenate `emb_a` and `emb_b` along axis 0 -> `emb_all` shape `[LA+LB, esm_dim]`.
7. `np.savez_compressed(cache_path, embeddings=emb_all, LA=LA, LB=LB,
   sample_id=sample_id, esm_dim=esm_dim)`.
8. tqdm progress bar; log every 500 samples (rate, ETA).
9. At the end print a summary: total processed, skipped (already cached),
   failed (with reason), disk usage of the output dir.

**Wall-time budget:** ESM-2-35M is ~50 ms per ~250-aa chain on the
4070 Ti SUPER. `28352 samples * 2 chains * 50 ms ~= 47 min` worst case;
in practice ~25-35 min because shorter chains dominate. The 8722 in-filter
subset alone (`200 <= LA+LB <= 1200`) is ~15 min.

**Disk footprint:** float16, 480-dim, mean L~=420 -> ~400 KB per sample
-> ~11 GB for the 28352-sample cache. (For 150M it doubles to ~22 GB.)

**Verify:** After a 4-sample dry run (`--max-samples 4`), open one NPZ:
```python
import numpy as np
d = np.load("data/processed/esm2_35M/10gs.pdb1_0.npz")
assert d["embeddings"].shape == (d["LA"] + d["LB"], 480)
assert d["embeddings"].dtype == np.float16
```

---

### Task 2: Wire cache loading into the dataset pipeline

**Goal:** When `aa_embed != "learned"`, the per-sample loader returns an
extra tensor `esm_embed` of shape `[L, esm_dim]` that the collate function
pads to `[B, Lmax, esm_dim]`.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/training/data.py` |
| MODIFY | `src/tinyfold/data/datasets/ppi_dataset.py` |
| MODIFY | `src/tinyfold/data/collate.py` |

**Steps:**

1. **`src/tinyfold/training/data.py`** (primary path used by `train_resfold.py`):
   - Extend `load_sample(table, i, normalize=True, esm_cache_dir: str | None = None)`.
     When `esm_cache_dir is not None`:
     - Compute `cache_path = Path(esm_cache_dir) / f"{sample_id}.npz"`.
     - Load with `np.load(cache_path, mmap_mode="r")` (mmap so we
       don't blow RAM with parallel workers).
     - Read `emb = torch.from_numpy(np.asarray(d["embeddings"])).float()`
       (cast fp16 -> fp32 here, off the hot path).
     - Assert `emb.shape[0] == n_res` (matches the parquet row's `LA+LB`).
       Raise a clear `ValueError("ESM cache missing/mismatched for {sample_id}")`
       if not.
     - Add `'esm_embed': emb` to the returned dict.
   - In `collate_batch`, when any sample has `'esm_embed'`:
     - Find `esm_dim = samples[0]['esm_embed'].shape[1]`.
     - Allocate `esm_embed_batch = torch.zeros(B, max_res, esm_dim)`.
     - Fill each sample's slice `[i, :L]`.
     - Move to `device` and add to the output dict under key `'esm_embed'`.
   - When `esm_cache_dir` is None, behavior is **identical to today** (no new
     key in the batch dict, no extra disk I/O).

2. **`src/tinyfold/data/datasets/ppi_dataset.py`** (the secondary loader
   used by tests / Phase B eval if any): same change in spirit -- add an
   optional `esm_cache_dir` constructor arg, load+return `esm_embed` in
   `__getitem__` when set. (One-line addition since we already read by `sample_id`.)
   If this dataset isn't on the Phase D training path, document that the
   change is for future use and only add the constructor arg; do not call
   sites need to change.

3. **`src/tinyfold/data/collate.py`** (`collate_ppi`): mirror the same
   `esm_embed` padding logic if any sample has the key.

**Verify:**
- Manual: instantiate `load_sample(table, 0, esm_cache_dir="data/processed/esm2_35M")`
  on a cached sample, assert `out['esm_embed'].shape == (out['n_res'], 480)`
  and `dtype == torch.float32`.
- Batch-level: `collate_batch([s1, s2])` produces `batch['esm_embed']`
  shape `[2, max(L1, L2), 480]` with proper zero-padding.

---

### Task 3: Gate `ResidueEncoder` on `aa_embed`; add projection

**Goal:** `ResidueEncoder` learns either the existing `nn.Embedding`
lookup (`aa_embed="learned"`) or projects a cached ESM tensor
(`aa_embed in {"esm2_35M", "esm2_150M"}`). The rest of the encoder
(chain embed, sinusoidal `res_idx`, transformer, output norm) is unchanged.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `src/tinyfold/model/resfold/denoiser.py` (`ResidueEncoder`) |
| MODIFY | `src/tinyfold/model/resfold/onestep.py` (`ResFoldOneStep.__init__`, `forward_sigma`, `get_trunk_tokens`, etc.) |
| MODIFY | `src/tinyfold/model/resfold/pipeline.py` (`ResFoldPipeline.__init__`) -- pass `aa_embed`/`esm_dim` through to `ResidueDenoiser`/`ResidueEncoder`. |

**Steps:**

1. **`ResidueEncoder.__init__`** — add two args with backward-compatible
   defaults:
   ```python
   aa_embed: str = "learned",   # "learned" | "esm2_35M" | "esm2_150M"
   esm_dim: int | None = None,  # required when aa_embed != "learned"
   ```
   - Resolve `esm_dim` from a small dict if not provided:
     `{"esm2_35M": 480, "esm2_150M": 640}`. Store as `self.aa_embed_mode`,
     `self.esm_dim`.
   - If `aa_embed == "learned"`: keep `self.aa_embed = nn.Embedding(n_aa_types, c_token)`
     exactly as today. Set `self.esm_proj = None`.
   - Else:
     - Do NOT create `nn.Embedding`. Either `self.aa_embed = None`, or
       create a dummy `nn.Identity()` so that loading old "learned"
       checkpoints with `strict=False` is unambiguous. (Prefer `None`.)
     - `self.esm_proj = nn.Linear(esm_dim, c_token)` with
       `nn.init.xavier_uniform_(self.esm_proj.weight); nn.init.zeros_(self.esm_proj.bias)`.

2. **`ResidueEncoder.forward`** — add optional kwarg `esm_embed: Tensor | None = None`:
   ```python
   def forward(self, aa_seq, chain_ids, res_idx, mask=None, esm_embed=None):
       if self.aa_embed_mode == "learned":
           aa_emb = self.aa_embed(aa_seq)              # [B, L, c_token]
       else:
           assert esm_embed is not None, "ESM mode requires batch['esm_embed']"
           aa_emb = self.esm_proj(esm_embed.float())   # [B, L, c_token]
       # ... rest of forward unchanged (chain_emb, res_emb, input_proj, transformer, norm)
   ```
   When ESM mode is on, `aa_seq` is still passed (used for `chain_ids` shape
   inference) but its values are ignored for the AA branch -- the caller is
   free to keep passing the integer seq tensor for backward compatibility.

3. **`ResFoldOneStep`** and **`ResidueDenoiser`**:
   - Add `aa_embed: str = "learned"` and `esm_dim: int | None = None`
     constructor args; pass them through to `ResidueEncoder`.
   - Thread `esm_embed` through:
     - `ResFoldOneStep.forward_sigma(..., esm_embed=None)` -> `self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)`
     - `get_trunk_tokens(..., esm_embed=None)` (same)
     - `ResidueDenoiser.get_trunk_tokens` / equivalent in `denoiser.py`.
     - `ResFoldPipeline.forward_stage1(..., esm_embed=None)` and
       `forward(..., esm_embed=None)`.

4. **Caller updates in `scripts/train_resfold.py`**:
   - Every `model.forward_stage1(...)` / `model.forward_sigma(...)` /
     `get_trunk_tokens(...)` call site that already takes
     `batch['aa_seq'], batch['chain_ids'], batch['res_idx']` gains an
     extra `esm_embed=batch.get('esm_embed')` kwarg. With `aa_embed=learned`
     this is `None` and the model is byte-identical to before.
   - Grep target list (from earlier scan): lines around
     117, 181, 255, 302, 369, 1336, 1400, 1408, 1420, 1425, 1525, 1534,
     1586, 1715, 1778, plus any in `tinyfold.training.utils` helpers
     (e.g. `MultiCopyTrainer.compute_loss`). Use a single helper
     `_call_forward_sigma(model, x, batch, sigma, ...)` if the diff
     becomes large, otherwise keep inline.

**Verify:**
- `aa_embed=learned` path: existing tests + a 1-step training run produce
  the same loss as before (within fp noise). No new params on the
  optimizer when the flag is left at the default.
- `aa_embed=esm2_35M` path: a forward pass on a single cached sample
  produces a finite loss; `model.trunk.esm_proj.weight.requires_grad`
  is True; `model.trunk.aa_embed is None`.

---

### Task 4: Surface `--aa_embed` in the training script

**Goal:** New CLI/config field, plumbed through model construction and
data loading.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `scripts/train_resfold.py` |

**Steps:**

1. Add the argparse flag near the existing model-shape flags
   (around line 565, next to `--c_token_s1`):
   ```python
   parser.add_argument(
       "--aa_embed", type=str, default="learned",
       choices=["learned", "esm2_35M", "esm2_150M"],
       help="AA representation: learned nn.Embedding (default) or frozen "
            "ESM-2 cached embeddings (35M=480d, 150M=640d).",
   )
   parser.add_argument(
       "--esm_cache_dir", type=str, default=None,
       help="Directory containing per-sample ESM NPZ files. Required when "
            "--aa_embed != learned. Default: data/processed/esm2_{variant}.",
   )
   ```
2. After args parsing: if `args.aa_embed != "learned"` and
   `args.esm_cache_dir is None`, derive
   `args.esm_cache_dir = f"data/processed/esm2_{args.aa_embed.split('_')[1]}"`
   (i.e. `esm2_35M` -> `data/processed/esm2_35M`). Assert the dir exists,
   otherwise print a friendly "run scripts/prepare_esm2_embeddings.py first" error.
3. Resolve `esm_dim` from a tiny dict (mirror of the one in the prep script):
   `ESM_DIMS = {"esm2_35M": 480, "esm2_150M": 640}`.
4. Pass `aa_embed=args.aa_embed, esm_dim=ESM_DIMS.get(args.aa_embed)`
   into both `ResFoldOneStep(...)` (line ~1138) and `ResFoldPipeline(...)`
   (lines ~1084 and ~1160).
5. Plumb `esm_cache_dir` into every `load_sample(table, i, ...)` /
   `load_sample_raw(...)` call site (grep the file). Default `None` -> no
   change in behavior.
6. Persist `aa_embed` in `save_config(args, ...)` so the run output dir
   captures it (it should already be included if `save_config` serializes
   the full args namespace; verify).
7. Optional: include `aa_embed` in `generate_run_name(...)` so run dirs
   are self-documenting (`..._esm2_35M_...`); fine to skip if it complicates
   the existing naming scheme.

**Verify:** `python scripts/train_resfold.py --help` shows the new flag.
Running with `--aa_embed learned` (default) is byte-identical to the
prior commit (compare training loss curve over 10 steps).

---

### Task 5: Unit test for NPZ shape + smoke training step

**Goal:** Lock in the shape contract of the cached embedding and confirm
the ESM-2 training path runs end-to-end.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `tests/test_esm2_cache.py` |

**Steps:**

1. **Shape test** (skips with `pytest.skip` if no cache dir on disk):
   ```python
   def test_esm2_cache_shape():
       cache_dir = Path("data/processed/esm2_35M")
       if not cache_dir.exists():
           pytest.skip("ESM-2 cache not built (run prepare_esm2_embeddings.py)")
       table = pq.read_table("data/processed/samples.parquet",
                             columns=["sample_id", "LA", "LB"])
       df = table.to_pandas()
       n_checked = 0
       for _, row in df.iterrows():
           p = cache_dir / f"{row['sample_id']}.npz"
           if not p.exists():
               continue
           d = np.load(p)
           assert d["embeddings"].shape == (row["LA"] + row["LB"], 480)
           assert d["embeddings"].dtype == np.float16
           assert int(d["LA"]) == row["LA"] and int(d["LB"]) == row["LB"]
           n_checked += 1
           if n_checked >= 20:
               break
       assert n_checked > 0, "no cached samples found"
   ```

2. **Smoke training test** (also `pytest.skip` if cache missing):
   ```bash
   .venv\Scripts\python scripts/train_resfold.py \
       --model_kind onestep --mode stage1_only \
       --aa_embed esm2_35M \
       --n_train 4 --n_test 2 --n_steps 10 --batch_size 2 \
       --c_token_s1 128 --trunk_layers 2 --denoiser_blocks 2 \
       --continuous_sigma \
       --output_dir outputs/resfold/_smoke_loop05
   ```
   - Tester runs this from PowerShell.
   - Assertions (manually in the run log): `loss` is finite for all 10 steps,
     no `ModuleNotFoundError`, run dir contains a `config.yaml` mentioning
     `aa_embed: esm2_35M`.

3. **Frozen-ESM assertion** (inline in the test or in the train script
   startup print): the model has NO `EsmModel` parameter and the only new
   trainable tensor relative to the `learned` baseline is `trunk.esm_proj`
   (`Linear(480, c_token)` -> `480*c_token + c_token` params).

**Verify:**
- `.venv\Scripts\pytest tests/test_esm2_cache.py -v` -> 1 pass + (1 pass after cache built).
- Smoke training command above completes 10 steps, prints finite loss,
  exits zero.

---

## Acceptance Criteria

- [ ] `transformers` importable from `.venv`.
- [ ] `scripts/prepare_esm2_embeddings.py` runs end-to-end on the full
      28352-sample parquet for the 35M variant and writes one NPZ per
      sample under `data/processed/esm2_35M/`.
- [ ] Each cached NPZ has `embeddings.shape == (LA+LB, 480)` and dtype
      `float16`; spot-check 20 samples passes (Task 5 unit test).
- [ ] `ResidueEncoder` exposes `aa_embed` + `esm_dim` constructor args
      and a new `esm_embed` forward kwarg; `aa_embed="learned"` is
      byte-identical to the prior commit (loss curve matches in a
      10-step smoke run).
- [ ] `--aa_embed esm2_35M` smoke training (Task 5) runs 10 steps with
      finite loss on the 4070 Ti SUPER, peak VRAM under the existing
      budget (the trunk only grows by `480 * c_token + c_token` params
      and the cached embedding tensor is `[B, Lmax, 480]` fp32, ~1 MB
      at Lmax=500 / B=2).
- [ ] `trunk.esm_proj.weight.requires_grad` is True; no `EsmModel`
      parameters exist in the model state dict (we only carry the
      projection -- ESM-2 itself is not in the model).
- [ ] `pyproject.toml` updated; commit gate green.

## Risks + mitigations

- **Download size**: ESM-2-35M weights are ~150 MB, fetched on first
  prep-script run to `~/.cache/huggingface`. If the network is gated,
  pre-stage with `huggingface-cli download facebook/esm2_t12_35M_UR50D`.
- **Cache prep wall-time** (~30-45 min): run with a tqdm bar; the script
  is idempotent (skip-if-exists), so a Ctrl-C resume is cheap.
- **Disk footprint** (~11 GB for 35M, ~22 GB for 150M): float16 + per-file
  NPZ already minimises this. If disk pressure shows up, switch to a
  single sharded `.npz` per split (deferred).
- **dtype drift**: ESM-2 native is fp32 in `EsmModel`; we explicitly cast
  to fp16 on save and back to fp32 in `load_sample`. The projection sees
  fp32 throughout -> no `MatMul` dtype surprises.
- **Token alignment**: ESM tokenization prepends `[CLS]` and appends
  `[EOS]`. The prep script slices `out[0, 1:-1]` and the per-chain
  length assertion catches any tokenizer quirk on rare residues (e.g.
  `B`, `Z`, `U`). If an assert trips, the sample is logged + skipped,
  and the dataset loader falls back to a clear error at training time
  (cache-miss => raise).
- **150M variant not used in Phase D**: the flag exists, but Loop 07's
  Phase D config defaults to `esm2_35M`. 150M is a single-arg flip
  if/when we want to try it.

## Loop boundary

Last action of the loop = the unit test passes + the smoke training run
prints 10 finite losses. Implementer commits with message
`feat(esm2): frozen ESM-2-35M embedding cache + projection (Task F)`.
