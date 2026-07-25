# TinyFold Scripts

Scripts for training, evaluation, data prep, and visualization. The only model is
**ResFoldOneStep** (`--model_kind onestep`); the legacy two-stage pipeline has
been removed. Paths below are verified against the tree.

## Training & evaluation

| Script | Description |
|--------|-------------|
| `train_resfold.py` | Train ResFoldOneStep. Driven by YAML configs in `configs/train/resfold/`. Recycling / multiplicity / EMA / self-conditioning / FAPE / ODE sampling are all off by default (see `--n_recycle`, `--diffusion_multiplicity`, `--ema_decay`, `--self_cond_steps`, `--fape_weight`, `--sampler`). Cluster-clean splits are enforced by `--require_clean_split` (on). |
| `eval_leakage_split.py` | Score a checkpoint on its own `split.json` under permutation-aware DockQ, annotated with leakage strata + CAPRI bands. The honest read-out. |
| `eval_dockq_histogram.py` | Per-complex DockQ + CAPRI-band histogram on the test set and OOD size bins. |
| `eval_sampler_sweep.py` | Sweep sampler settings (K, steps) for a checkpoint. |
| `eval_sidechains.py` | All-atom sidechain-RMSD eval for a torsion-head checkpoint. |
| `predict.py` | Run a checkpoint on an input complex. |

```bash
python scripts/train_resfold.py --config configs/train/resfold/small_specialist_le200.yaml
python scripts/eval_leakage_split.py \
    --checkpoint outputs/resfold/small_specialist_le200/<run>/best_model.pt \
    --split      outputs/resfold/small_specialist_le200/<run>/split.json
```

## Data preparation (`scripts/data/`)

| Script | Description |
|--------|-------------|
| `prepare_data.py` | Download DIPS-Plus, process, cache to Parquet |
| `../prepare_esm2_embeddings.py` | Precompute & cache frozen ESM-2 embeddings (`data/processed/esm2_35M/`; 150M/650M via `tinyfold.cli.prepare_esm2`) |
| `prepare_atom14.py` | Cache atom14 sidechain coords (for the torsion sidechain head) |
| `prepare_templates.py` | Cache retrieved-template features |
| `prepare_msa_chains.py` / `prepare_msa_features.py` / `measure_msa_depth.py` / `msa_unpack_to_chain_keys.py` | Paired-MSA / coevolution pipeline. **Kept as a verified artifact but no longer wired into training** — see `src/tinyfold/msa/__init__.py` for why. |
| `cluster_interfaces.py` / `make_cluster_split.py` | Foldseek/sequence clustering and cluster-holdout split generation |
| `audit_crops.py` | Static crop auditor (no training): reports the E1.a PASS/KILL verdict for `InterfaceCrop` at a given `crop_size` |
| `../build_diffdockpp_split.py` | Build the DiffDock-PP comparison split |
| `../dataset_stats.py` | Print dataset length/size statistics |

## Standalone training / diagnostics (`scripts/train/`)

| Script | Description |
|--------|-------------|
| `train/overfit_sidechain_torsion.py` | Overfit gate for the torsion sidechain head (χ on SO(2)⁴) |
| `train/roundtrip_sidechain_torsion.py` | atom14 ↔ torsion round-trip check |

## Visualization (`scripts/visualization/`)

| Script | Description |
|--------|-------------|
| `plot_architecture.py` | Render the model architecture diagram |
| `visualize_structure.py` | Render predicted vs ground-truth structures |

## Web (`scripts/web/`)

| Script | Description |
|--------|-------------|
| `build_showcase.py` | Generate `assets/showcase_samples.json` for the `web-light/` viewer from a checkpoint |

## Benchmarking

The standardized comparison harness lives at `benchmarks/scripts/`
(`eval_tinyfold.py`, `compute_metrics.py`, `check_leakage.py`,
`make_comparison.py`), not under `scripts/`.

## Utilities

| Script | Description |
|--------|-------------|
| `script_utils.py` | Shared utilities (Logger, config loading, checkpoints) |
| `test_positional_invariance.py` | Diagnostic for positional-encoding size dependence |
| `vram_probe.py` | Probe VRAM usage at various sequence lengths |
