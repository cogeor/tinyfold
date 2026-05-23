# TinyFold Web Frontend

## Modes

The frontend supports two modes:

| Mode | Server | Use Case |
|------|--------|----------|
| **Full** | `web/server.py` | Evaluate models: browse dataset, run/load predictions, inspect metrics |
| **Light** | `web-light/server.py` | Static showcase of a few train/test GT+prediction examples |

## Quick Start

```bash
cd web
../.venv/Scripts/python.exe server.py
# Open http://127.0.0.1:5001
```

## Generating Cached Predictions

For fast loading in the frontend, pre-compute predictions using `predict_all.py`.

### AF3-Style Model

```bash
cd web

# All samples (train + test)
python predict_all.py \
    --model af3_style \
    --checkpoint ../outputs/af3_15M_gaussian_5K/best_model.pt \
    --output predictions.json

# Test set only (faster)
python predict_all.py \
    --model af3_style \
    --checkpoint ../outputs/af3_15M_gaussian_5K/best_model.pt \
    --split test \
    --output predictions_test.json
```

### ResFold Two-Stage Model

ResFold has separate stage 1 (residue) and stage 2 (atom) models. Run them separately to manage memory:

```bash
# Stage 1: Predict residue centroids
python predict_all.py \
    --model resfold \
    --stage 1 \
    --checkpoint ../outputs/resfold_s1_5K_50K/best_model.pt \
    --output predictions_s1.json

# Stage 2: Predict atoms from cached centroids
python predict_all.py \
    --model resfold \
    --stage 2 \
    --checkpoint ../outputs/resfold_s2/best_model.pt \
    --stage1_cache predictions_s1.json \
    --output predictions.json
```

If memory allows, run both stages together:

```bash
python predict_all.py \
    --model resfold \
    --stage both \
    --checkpoint ../outputs/resfold_full/best_model.pt \
    --output predictions.json
```

### Script Options

| Option | Description |
|--------|-------------|
| `--model` | `af3_style` or `resfold` |
| `--checkpoint` | Path to model checkpoint |
| `--stage` | For resfold: `1`, `2`, or `both` |
| `--stage1_cache` | Path to stage 1 predictions (required for `--stage 2`) |
| `--noise_type` | `gaussian` or `linear_chain` (af3 only) |
| `--split` | `train`, `test`, or `both` |
| `--output` | Output JSON file path |
| `--n_train` | Number of train samples (default: 5000) |
| `--n_test` | Number of test samples (default: 1000) |

## Using Cached Predictions

1. Generate predictions (see above)

2. Update `config.yaml`:
   ```yaml
   predictions:
     cache_path: "predictions.json"
   ```

3. Restart the server:
   ```bash
   ../.venv/Scripts/python.exe server.py
   ```

The frontend will:
- Show a **P** indicator next to samples with cached predictions
- Display "Load Prediction" instead of "Run Prediction"
- Show "cached" tag in results
- Load instantly instead of running inference

## Output Format

The predictions JSON file contains:

```json
{
  "sample_id_1": {
    "coords": [[x, y, z], ...],
    "rmsd": 1.75,
    "inference_time": 1.5
  },
  ...
}
```

For ResFold stage 1:
```json
{
  "sample_id_1": {
    "centroids": [[x, y, z], ...],
    "rmsd_ca": 1.2,
    "inference_time": 0.8
  },
  ...
}
```

---

## Light Mode (Embeddable Viewer)

The light mode provides an embeddable viewer without model loading or sample browsing.

### Quick Start

```bash
cd web-light
python server.py --port 5002
# Open http://127.0.0.1:5002
```

`web-light` uses Python stdlib only (`http.server`) and is runnable right after cloning.

### Showcase Data

`web-light` reads `/assets/showcase_samples.json`.
The simplified flow uses fixed predictions in `assets/`:

- `assets/showcase_predictions.json`

```bash
python scripts/web/prepare_web_light_showcase.py
```

This script reads prediction IDs from `assets/showcase_predictions.json`,
pulls matching ground truth directly from `data/processed/samples.parquet`,
and writes `assets/showcase_samples.json`.
