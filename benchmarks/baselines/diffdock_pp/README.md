# DiffDock-PP baseline

Score-based generative model for rigid PPI docking from
Ketata, Corso et al. (ICLR MLDD workshop, 2023). ~1.6M parameters
— same order as TinyFold's 6M.

## Upstream

- Repo: https://github.com/ketatam/DiffDock-PP
- Pinned commit: **set this after first clone** (record in `clone.sh`).
- Paper: arxiv.org/abs/2304.03889

## Why we care

DiffDock-PP reports top-1 median C-RMSD ~4.85 Å on DIPS. The paper does
not stratify by complex size. Our hypothesis: those numbers are heavily
weighted toward small complexes (LA+LB ≲ 300), and DiffDock-PP shows the
same cliff we do. This benchmark either confirms or refutes that.

## How to run

```bash
# 1. Clone at pinned commit (idempotent)
bash benchmarks/baselines/diffdock_pp/clone.sh

# 2. Build a separate Python venv with torch 2.4 + PyG (NOT the TinyFold venv).
#    Apply Windows portability patches.
bash benchmarks/baselines/diffdock_pp/setup_env.sh

# 3. Convert our parquet -> DPP input format (one-time per split)
python benchmarks/baselines/diffdock_pp/adapt.py \
  --split benchmarks/splits/clean_400_600.json \
  --out benchmarks/baselines/diffdock_pp/repo/datasets/clean_400_600/

# 4. Run pretrained checkpoint, writes NPZ predictions (Loop 03)
python benchmarks/baselines/diffdock_pp/run_pretrained.py \
  --split benchmarks/splits/clean_400_600.json \
  --out benchmarks/predictions/diffdock_pp/clean_400_600/

# 5. Score via shared metric pipeline
python benchmarks/scripts/compute_metrics.py --model diffdock_pp
```

## Known gotchas

- DiffDock-PP was trained on DIPS. Our test bins also come from DIPS.
  **Always run `scripts/check_leakage.py` before publishing numbers.**
- DiffDock-PP uses CA-only graphs internally and writes back full
  predictions via rigid alignment. Our metric pipeline needs N/CA/C/O,
  so the adapter must reconstruct the missing backbone atoms (or we
  score CA-only and disclose it).
