# TinyFold Scripts

Core scripts for training, evaluation, data prep, and visualization.

## Training & evaluation (headline path)

| Script | Description |
|--------|-------------|
| `train_resfold.py` | Train the ResFold model (`--model_kind onestep` is the supported headline model). Driven by YAML configs in `configs/train/resfold/`. |
| `eval_dockq_histogram.py` | Per-complex DockQ + CAPRI-band histogram for a trained checkpoint, on the small test set and the OOD size bins. |

Reproduce the headline experiment (see the repo README for the full sequence):

```bash
python scripts/train_resfold.py --config configs/train/resfold/small_specialist_le200.yaml
python scripts/eval_dockq_histogram.py \
    --checkpoint outputs/resfold/small_specialist_le200/<run>/best_model.pt \
    --small_split outputs/resfold/small_specialist_le200/<run>/split.json
```

## Data preparation

| Script | Description |
|--------|-------------|
| `data/prepare_data.py` | Download DIPS-Plus, process, and cache to Parquet |
| `prepare_esm2_embeddings.py` | Precompute & cache frozen ESM-2 embeddings (`data/processed/esm2_35M/`) |
| `build_diffdockpp_split.py` | Build the DiffDock-PP comparison split |
| `dataset_stats.py` | Print dataset length/size statistics |

## Web

| Script | Description |
|--------|-------------|
| `web/build_showcase.py` | Generate `assets/showcase_samples.json` (zero-friction `web-light/` viewer) from a checkpoint |
| `web/prepare_web_light_showcase.py` | Legacy showcase builder (predictions.json → showcase) |

## Visualization

`scripts/visualization/`: `visualize_preds.py`, `visualize_diffusion.py`,
`visualize_structure.py`, `plot_architecture.py`, `plot_coiled_coils.py`,
`plot_coil_predictions.py`.

## Benchmarking

`scripts/benchmark/` — standardized evaluation for the resfold model line
(`cli.py`, `runner.py`, `model_adapter.py`, `data_loader.py`, `metrics.py`).

## Utilities

| Script | Description |
|--------|-------------|
| `script_utils.py` | Shared utilities (Logger, data loading, checkpoints) |
| `test_positional_invariance.py` | Diagnostic for positional-encoding size dependence |
| `vram_probe.py` | Probe VRAM usage at various sequence lengths |
| `tools/scaffold_extension.py` | Scaffold a new model extension |
