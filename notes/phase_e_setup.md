# Phase E setup — full-parquet DIPS + VRAM headroom

Date: 2026-05-24
Status: **Config + scripts shipped. Training NOT yet started.**

## Two questions answered

### 1. How much larger of a model fits in VRAM?

Measured via `scripts/vram_probe.py` (synthetic forward + backward at
batch=32, L=1200, with confidence head + ESM-2 projection) on the
RTX 4070 Ti SUPER (16,376 MiB total):

| Config (c_token, trunk+denoiser) | Params | Peak alloc | Reserved | % 16 GiB |
|---|---:|---:|---:|---:|
| **Phase D baseline (256, 6+6)** | 11.8M | 9.3 GiB | 9.8 GiB | 60% |
| Deeper (256, 8+8) | 15.2M | 12.0 GiB | 12.6 GiB | 77% |
| Deeper (256, 10+10) | 18.6M | 14.7 GiB | 15.3 GiB | 93% |
| Wider (384, 6+6) | 26.4M | 13.8 GiB | 14.6 GiB | 89% |
| Wider+Deeper (384, 8+8) | 34.1M | 17.8 GiB | 18.7 GiB | 114% (spill) |
| Wider (512, 6+6) | 46.8M | 18.5 GiB | 19.3 GiB | 118% (spill) |
| Wider (512, 8+8) | 60.5M | 23.8 GiB | 24.7 GiB | 151% (spill) |
| Big (768, 6+6) | 105M | 27.7 GiB | 28.9 GiB | 177% (spill) |
| Big (768, 8+8) | — | — | — | **OOM** |

"Spill" = exceeds physical 16 GiB; PyTorch falls back to shared
host memory (slow, not practical for training but doesn't OOM).

**Practical scaling envelope at batch_size=32, L=1200:**

- **3-4× params (up to ~25M trainable)** — fits with headroom.
  Recommended sweet spot for Phase F: width 384 OR depth 10+10.
- **~30-50M trainable** — feasible with `batch_size=16, grad_accum=2`
  (halves activation memory, preserves effective batch).
- **~100M trainable** — needs `batch_size=8, grad_accum=4` AND mixed
  precision (which we don't currently use). Worth checking if bf16
  is wired in `scripts/train_resfold.py`.

Caveat: VRAM probe uses synthetic data with the worst-case L=1200,
but the actual parquet caps at LA+LB ≤ 600 residues — real training
peaks lower. Phase D's actual run held under 12 GiB the whole time.
That means **width 512 (~47M params) likely fits comfortably in
production at the real data distribution.**

### 2. Apples-to-apples full-DIPS setup

`scripts/dataset_stats.py` ran on `data/processed/samples.parquet`:

```
Total samples: 28,352
LA distribution: min 40, p25 139, med 197, p75 241, p95 281, max 300
LB distribution: min 40, p25 129, med 194, p75 232, p95 280, max 300
LA+LB:           min 80, p25 275, med 384, p75 463, p95 555, max 600
```

**The big surprise:** the parquet is itself pre-filtered. Per
`CLAUDE.md`: "Chain length: 40-300 residues". The data prep script
caps both chains at 300 residues. So:

- Our previous `[200, 1200]` filter retained 25,050 / 28,352 (88.4%)
- The "1200 residue ceiling" was effectively a no-op — parquet maxes
  at LA+LB = 600
- The "we cut 69% of DIPS" framing I floated earlier was WRONG;
  actual cut was 11.6% (the smallest complexes, <200 residues total)

**Phase E config (`configs/train/resfold/phase_e_full_dips.yaml`):**

- `min_atoms: 0, max_atoms: 99999` → uses all 28,352 parquet rows
- `dynamic_batch: true, use_bucketing: true, max_tokens: 30000` →
  variable batch size by token count, safe for the mixed
  size distribution
- `n_train: 28,000, n_test: 200` (vs Phase D's 8600 / 100)
- Everything else identical to Phase D (architecture, ESM-2, EDM
  weighting, confidence head, sampler)
- Same seed=42 so deltas vs Phase D are interpretable

**What Phase E IS apples-to-apples for:**
- DiffDock-PP, EquiDock, GeoDock — these use chain ranges
  comparable to our parquet (50-600 residues per chain). Phase E
  number vs theirs is a fair comparison.

**What Phase E is NOT apples-to-apples for:**
- AlphaFold-Multimer, Boltz, AF3 — they evaluate on much larger
  complexes (often 1000+ residues per chain). Their numbers are
  averaged over a harder distribution than ours.
- True apples-to-apples with AF-M requires re-running
  `scripts/prepare_data.py` with chain cap raised to ~600+ residues.
  That's a data-pipeline job, not a config knob.

## What's already done

- `scripts/vram_probe.py` — VRAM sweep script (synthetic loads only).
- `scripts/dataset_stats.py` — parquet size-distribution reporter.
- `configs/train/resfold/phase_e_full_dips.yaml` — full-parquet config.

## What's NOT done (deferred)

- The actual Phase E training run. Expected wall time: 50K steps
  with ~3.3× the data of Phase D → roughly 6-7 hours on the
  4070 Ti SUPER (Phase D was 135 min on 8600 samples; per-epoch
  cost is similar but more steps per epoch worth seeing).
- Family-filtered splits (DiffDock-PP / EquiDock convention) — our
  test set is random-split. To match their evaluation methodology
  we'd need to cluster by sequence identity (BLAST or MMSeqs2)
  and split families. Not done.
- Re-prep parquet with larger chain cap for true AF-M comparison.
- Run AF-Multimer / ColabFold on our test set (cheap, ~6 h GPU,
  would make any "comparable to AF-M" claim defensible).

## Recommended next-step sequence

1. Run Phase E (~6-7 h) — produces an honest "no filter" baseline.
2. While Phase E trains: run AF-Multimer / ColabFold on the same
   100-target Phase D test set. Get a real on-our-split number.
3. Decide if scaling up (Phase F) is worth the compute given the
   ranker-failure finding from the previous loop.
