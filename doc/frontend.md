# TinyFold Web Frontend

## Modes

The frontend supports two modes:

| Mode | Server | Use Case |
|------|--------|----------|
| **Full** | `web/server.py` | Browse samples, run predictions, view cached results |
| **Light** | `web-light/server.py` | Embed viewer in other apps, load coords via JS API |

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
../.venv/Scripts/python.exe server.py --port 5002
# Open http://127.0.0.1:5002
```

### URL Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `mode` | `embed`, `light` | Force embed mode (auto-detected from server) |
| `minimal` | `true` | Hide all controls, show only viewer |

Examples:
- `http://localhost:5002/` - Embed mode with controls
- `http://localhost:5002/?minimal=true` - Viewer only

### JavaScript API

For same-origin usage, the `window.tinyfold` API is available:

```javascript
// Load structures from coordinate arrays
window.tinyfold.load({
    groundTruth: [[x, y, z], ...],  // Nx3 array
    prediction: [[x, y, z], ...],   // Nx3 array (optional)
    atomTypes: ['N', 'CA', 'C', 'O'],  // Optional, defaults to backbone
    sequence: ['ALA', 'GLY', ...]      // Optional residue names
});

// Other methods
window.tinyfold.clear();              // Clear all models
window.tinyfold.setStyle('cartoon');  // 'cartoon', 'stick', 'sphere', 'line'
window.tinyfold.setColorScheme('chain');  // 'chain', 'spectrum', 'ss'
window.tinyfold.resetView();          // Reset camera
```

### postMessage API (for iframes)

For cross-origin iframe embedding:

```html
<iframe id="viewer" src="http://localhost:5002/?minimal=true"></iframe>

<script>
const viewer = document.getElementById('viewer');

// Wait for iframe to load
viewer.onload = () => {
    // Load structures
    viewer.contentWindow.postMessage({
        type: 'load',
        groundTruth: [[0, 0, 0], [1.46, 0, 0], ...],
        prediction: [[0, 0.1, 0], [1.5, 0, 0], ...],
    }, '*');

    // Other commands
    viewer.contentWindow.postMessage({ type: 'setStyle', style: 'stick' }, '*');
    viewer.contentWindow.postMessage({ type: 'clear' }, '*');
    viewer.contentWindow.postMessage({ type: 'resetView' }, '*');
};
</script>
```

### Integration Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>My App with TinyFold Viewer</title>
</head>
<body>
    <div style="display: flex; height: 100vh;">
        <!-- Your app content -->
        <div style="flex: 1; padding: 20px;">
            <h1>Protein Analysis</h1>
            <button onclick="loadExample()">Load Example</button>
        </div>

        <!-- Embedded TinyFold viewer -->
        <iframe
            id="tinyfold"
            src="http://localhost:5002/"
            style="flex: 1; border: none;"
        ></iframe>
    </div>

    <script>
    function loadExample() {
        // Example: backbone coords for 3 residues (12 atoms)
        const coords = [
            [0, 0, 0], [1.46, 0, 0], [2.5, 1.2, 0], [2.4, 2.3, 0],      // Res 1
            [3.8, 0.8, 0], [5.2, 1.1, 0], [6.3, 0.1, 0], [6.2, -1.1, 0], // Res 2
            [7.6, 0.5, 0], [9.0, 0.3, 0], [10.1, 1.4, 0], [10.0, 2.6, 0] // Res 3
        ];

        document.getElementById('tinyfold').contentWindow.postMessage({
            type: 'load',
            groundTruth: coords
        }, '*');
    }
    </script>
</body>
</html>
```
