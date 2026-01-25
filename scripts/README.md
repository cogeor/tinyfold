# TinyFold Scripts

This directory contains the core scripts for training, evaluation, and visualization.

## Data Pipeline

| Script | Description |
|--------|-------------|
| `prepare_data.py` | Download DIPS-Plus dataset, process, and cache to Parquet |
| `data_split.py` | Utility module for deterministic train/test splits |

## Training

| Script | Description |
|--------|-------------|
| `train.py` | Main atom-level diffusion training (af3_style model) |
| `train_resfold.py` | Two-stage ResFold training (centroids + atoms) |
| `train_stage2_continuous.py` | Stage 2 atom refinement training |
| `train_coil_experiment.py` | Coiled-coil experiment comparing noise types |

### Usage Examples

```bash
# Atom-level diffusion (af3_style model)
python scripts/train.py \
    --model af3_style \
    --h_dim 128 \
    --trunk_layers 5 \
    --denoiser_blocks 5 \
    --continuous_sigma \
    --augment_rotation \
    --output_dir outputs/atom_diffusion

# ResFold Stage 1 only
python scripts/train_resfold.py \
    --mode stage1_only \
    --n_train 8600 \
    --output_dir outputs/resfold_s1

# ResFold end-to-end
python scripts/train_resfold.py \
    --mode end_to_end \
    --n_train 8600 \
    --output_dir outputs/resfold_e2e
```

## Evaluation

| Script | Description |
|--------|-------------|
| `eval_stage1.py` | Evaluate Stage 1 model (centroids) |
| `eval_two_stage.py` | Evaluate full pipeline: Stage 1 -> Stage 2 -> DockQ |

### Usage Examples

```bash
# Evaluate two-stage pipeline
python scripts/eval_two_stage.py \
    --stage1_checkpoint outputs/train_10k_continuous/best_model.pt \
    --stage2_checkpoint outputs/stage2_continuous_rot/best_model.pt \
    --load_split outputs/train_10k_continuous/split.json
```

## Visualization

Scripts in `scripts/visualization/`:

| Script | Description |
|--------|-------------|
| `visualize_preds.py` | Visualize predictions vs ground truth |
| `visualize_diffusion.py` | Visualize diffusion sampling process |
| `visualize_structure.py` | Publication-ready structure visualization |
| `plot_architecture.py` | Generate architecture diagrams |
| `plot_coiled_coils.py` | Visualize coiled-coil dataset |
| `plot_coil_predictions.py` | Visualize coil prediction results |

## Utilities

| Script | Description |
|--------|-------------|
| `script_utils.py` | Shared utilities (Logger, data loading, checkpoints) |
| `profile_train.py` | Profile training bottlenecks |
| `select_coils.py` | Select coiled-coil samples for experiments |

## Archive

One-off debug/test scripts are in `scripts/archive/`. These were used during development but are not part of the core pipeline.

## Models Subpackage

The `scripts/models/` directory contains model definitions and utilities:

**Active Models:**
- `af3_style.py` - AF3-style decoder with atom attention
- `resfold.py` - ResidueDenoiser for centroid diffusion
- `resfold_pipeline.py` - Combined Stage 1 + Stage 2 pipeline
- `atomrefine_v2.py` - Stage 2 atom refiner (used by resfold_pipeline)
- `atomrefine_continuous.py` - Stage 2 continuous atom refinement

**Utilities:**
- `diffusion.py` - Diffusion schedules, noisers (VE, Karras, linear chain)
- `training_utils.py` - Augmentation and training utilities
- `dockq_utils.py` - DockQ evaluation
- `self_conditioning.py` - Self-conditioning for diffusion
- `multi_sample.py` - Multi-sample inference utilities
- `samplers.py` - DDPM, DDIM, Heun, EDM samplers
- `base.py` - Base decoder class
- `attention_v2.py` - Attention-based decoder

**Archived (in `models/archive/`):**
- `atomrefine.py` - Original Stage 2 (replaced by atomrefine_continuous)
- `hierarchical.py` - Old hierarchical decoder
- `pairformer_decoder.py` - Old Pairformer-based decoder
