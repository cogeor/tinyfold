# Benchmarks

Apples-to-apples comparison of TinyFold against external baselines on a
size-stratified PPI test set. The goal is a **single comparison table /
cliff plot** that runs every model through one metric pipeline so the
numbers are directly comparable.

## Layout

```
benchmarks/
├── README.md                          # this file
├── baselines/
│   └── diffdock_pp/                   # one dir per external baseline
│       ├── README.md                  # pinned commit, run cmds, notes
│       ├── repo/                      # gitignored clone target
│       ├── clone.sh                   # idempotent clone at pinned commit
│       ├── adapt.py                   # parquet ↔ baseline input format
│       ├── run_pretrained.py          # eval pretrained checkpoint
│       └── train.py                   # retrain on our split (optional)
├── splits/                            # symlinks/copies of stratified bins
│   ├── le200.json                     # LA+LB ≤ 200
│   ├── 200_400.json
│   ├── 400_600.json
│   ├── 600_1000.json
│   └── ge1000.json
├── predictions/                       # gitignored — large NPZ outputs
│   └── {model}/{split}/{sample_id}.npz
├── scripts/
│   ├── compute_metrics.py             # SHARED metric path for all baselines
│   ├── check_leakage.py               # baseline train ∩ our test
│   ├── eval_tinyfold.py               # (Loop 04) run our checkpoints
│   └── make_comparison.py             # join CSVs → cliff plot
└── results/
    ├── {model}.csv                    # per-sample (sample_id, bin, c_rmsd, dockq, ...)
    └── cliff_comparison.csv           # joined master table
```

## Ground rules

1. **One metric path.** Every model's predictions (TinyFold and external)
   land as NPZ in `predictions/{model}/{split}/{sample_id}.npz` carrying
   exactly: `pred_atoms` (L, 4, 3) in Angstroms, plus `sample_id` (str).
   `scripts/compute_metrics.py` ingests those and writes `results/{model}.csv`.
   No baseline gets to compute its own DockQ.
2. **Leakage check before publish.** `scripts/check_leakage.py` dumps the
   intersection between a baseline's training set and our test bins.
   If the overlap is non-trivial, we either retrain the baseline on our
   split or disclose the overlap.
3. **Test bins are fixed.** Stratified bins (sourced from
   `data/processed/splits/phase_d_bin_*.json`) are the canonical
   evaluation set. Don't add new bins without updating this README.

## Adding a baseline

1. Create `baselines/{name}/` with `README.md` (pin the upstream commit
   and checkpoint), `clone.sh`, `adapt.py`, and one run script per
   evaluation mode (pretrained / retrained).
2. Make the run script write predictions to `predictions/{name}/{split}/`.
3. Add the leakage check to `scripts/check_leakage.py`.
4. Run `scripts/compute_metrics.py --model {name}` to populate
   `results/{name}.csv`.
5. Re-run `scripts/make_comparison.py` to refresh the cliff plot.
