# TinyFold Web Visualizer

Web-based protein structure visualization using [3Dmol.js](https://3dmol.csb.pitt.edu/).

## Overview

Local web application for:
- Browsing training/test dataset samples
- Visualizing ground truth structures with cartoon/stick/sphere rendering
- Running model predictions and overlaying with ground truth
- Comparing multiple structures side-by-side

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Sample     │  │  Style      │  │  3Dmol.js Viewer        │  │
│  │  Browser    │  │  Controls   │  │  - Ground truth (blue)  │  │
│  │  - Search   │  │  - Cartoon  │  │  - Prediction (orange)  │  │
│  │  - Filter   │  │  - Stick    │  │  - Overlap mode         │  │
│  │  - List     │  │  - Sphere   │  │  - Per-chain coloring   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Python Server (Backend)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Data API   │  │  Predict    │  │  PDB Generation         │  │
│  │  /samples   │  │  API        │  │  coords_to_pdb_string() │  │
│  │  /sample/id │  │  /predict   │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                         │                                        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Model Loading & Inference                                   ││
│  │  - Load checkpoint                                           ││
│  │  - Run diffusion sampling                                    ││
│  │  - Kabsch alignment                                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  data/processed/train.parquet  |  data/processed/test.parquet   │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
web/
├── server.py              # Flask/FastAPI backend
├── static/
│   ├── index.html         # Main page
│   ├── css/
│   │   └── style.css      # UI styling
│   └── js/
│       ├── app.js         # Main application logic
│       ├── viewer.js      # 3Dmol.js wrapper
│       └── api.js         # Backend API client
└── templates/             # (if using Jinja2)
```

## Backend API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/samples` | List all samples with metadata |
| GET | `/api/samples?split=train` | Filter by split |
| GET | `/api/samples?search=1a2k` | Search by sample_id |
| GET | `/api/sample/{id}` | Get sample details |
| GET | `/api/sample/{id}/pdb` | Get ground truth PDB string |
| POST | `/api/predict/{id}` | Run prediction, return PDB |
| GET | `/api/models` | List available checkpoints |
| POST | `/api/predict/{id}?model=checkpoint.pt` | Predict with specific model |

### Sample List Response

```json
{
  "samples": [
    {
      "sample_id": "1a2k_A_B",
      "split": "train",
      "n_residues": 245,
      "chain_a_len": 120,
      "chain_b_len": 125
    }
  ],
  "total": 28352,
  "page": 1,
  "per_page": 50
}
```

### Sample Detail Response

```json
{
  "sample_id": "1a2k_A_B",
  "split": "train",
  "n_residues": 245,
  "n_atoms": 980,
  "sequence_a": "MKTAYIAKQR...",
  "sequence_b": "GPLGSVTEA...",
  "pdb_string": "ATOM      1  N   MET A   1..."
}
```

### Prediction Response

```json
{
  "sample_id": "1a2k_A_B",
  "model": "outputs/best_model.pt",
  "pdb_string": "ATOM      1  N   MET A   1...",
  "rmsd": 2.34,
  "inference_time_ms": 150
}
```

## Configuration

Server configuration via `web/config.yaml`:

```yaml
# web/config.yaml
model:
  checkpoint: "../outputs/best_model.pt"
  architecture: "af3_style"  # or "attention_v2"

inference:
  noise_type: "linear_chain"  # or "gaussian"
  n_timesteps: 50
  clamp_val: 3.0

data:
  train_path: "../data/processed/train.parquet"
  test_path: "../data/processed/test.parquet"

server:
  host: "127.0.0.1"
  port: 5000
```

## Backend Implementation

```python
# web/server.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import torch
import pyarrow.parquet as pq
import yaml
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../scripts')

from tinyfold.data.cache import dict_to_sample
from tinyfold.viz.io.structure_writer import coords_to_pdb_string
from models import create_model, create_schedule, create_noiser
from train import ddpm_sample, kabsch_align

# Global state (initialized at startup)
class AppState:
    model = None
    noiser = None
    schedule = None
    config = None
    data = {"train": None, "test": None}
    device = None

state = AppState()

def load_config(path: str = "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_model_and_data():
    """Load model checkpoint and data at startup."""
    state.config = load_config()
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(
        state.config["model"]["checkpoint"],
        map_location=state.device
    )
    state.model = create_model(state.config["model"]["architecture"])
    state.model.load_state_dict(checkpoint["model_state_dict"])
    state.model.to(state.device)
    state.model.eval()

    # Create noiser
    state.schedule = create_schedule(state.config["inference"]["n_timesteps"])
    state.noiser = create_noiser(
        state.config["inference"]["noise_type"],
        state.schedule
    )

    # Load data
    state.data["train"] = pq.read_table(state.config["data"]["train_path"])
    state.data["test"] = pq.read_table(state.config["data"]["test_path"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Response Models ---

class SampleMeta(BaseModel):
    sample_id: str
    split: str
    n_residues: int
    chain_a_len: int
    chain_b_len: int

class SampleListResponse(BaseModel):
    samples: list[SampleMeta]
    total: int

class SampleDetailResponse(BaseModel):
    sample_id: str
    split: str
    n_residues: int
    n_atoms: int
    pdb_string: str

class PredictionResponse(BaseModel):
    sample_id: str
    pdb_string: str
    rmsd: float
    inference_time_ms: float

# --- Helper Functions ---

def find_sample(sample_id: str) -> tuple[dict, str] | None:
    """Find sample in train or test set."""
    for split_name in ["train", "test"]:
        table = state.data[split_name]
        df = table.to_pandas()
        mask = df["sample_id"] == sample_id
        if mask.any():
            row = df[mask].iloc[0].to_dict()
            return dict_to_sample(row), split_name
    return None

def sample_to_pdb(sample: dict) -> str:
    """Convert sample to PDB string."""
    return coords_to_pdb_string(
        xyz=sample["xyz"],
        atom_to_res=sample["atom_to_res"],
        atom_type=sample["atom_type"],
        chain_id_res=sample["chain_id_res"],
        res_idx=sample["res_idx"],
        seq=sample["seq"],
        atom_mask=sample.get("mask"),
    )

# --- Endpoints ---

@app.get("/api/samples", response_model=SampleListResponse)
def list_samples(split: str = "all", search: str = "", page: int = 1, per_page: int = 50):
    samples = []
    for split_name in ["train", "test"]:
        if split != "all" and split != split_name:
            continue
        df = state.data[split_name].to_pandas()
        for _, row in df.iterrows():
            if search and search.lower() not in row["sample_id"].lower():
                continue
            samples.append(SampleMeta(
                sample_id=row["sample_id"],
                split=split_name,
                n_residues=len(row["seq"]),
                chain_a_len=int((row["chain_id_res"] == 0).sum()),
                chain_b_len=int((row["chain_id_res"] == 1).sum()),
            ))

    total = len(samples)
    start = (page - 1) * per_page
    samples = samples[start:start + per_page]
    return SampleListResponse(samples=samples, total=total)

@app.get("/api/sample/{sample_id}", response_model=SampleDetailResponse)
def get_sample(sample_id: str):
    result = find_sample(sample_id)
    if not result:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample, split = result
    return SampleDetailResponse(
        sample_id=sample_id,
        split=split,
        n_residues=len(sample["seq"]),
        n_atoms=len(sample["xyz"]),
        pdb_string=sample_to_pdb(sample),
    )

@app.get("/api/sample/{sample_id}/pdb", response_class=PlainTextResponse)
def get_sample_pdb(sample_id: str):
    result = find_sample(sample_id)
    if not result:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample, _ = result
    return sample_to_pdb(sample)

@app.post("/api/predict/{sample_id}", response_model=PredictionResponse)
def predict(sample_id: str):
    import time
    result = find_sample(sample_id)
    if not result:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample, _ = result

    start = time.time()

    # Prepare batch (single sample)
    batch = {k: torch.tensor(v).unsqueeze(0).to(state.device)
             for k, v in sample.items() if k != "sample_id"}

    # Run inference
    with torch.no_grad():
        pred_coords = ddpm_sample(
            model=state.model,
            atom_types=batch["atom_type"],
            atom_to_res=batch["atom_to_res"],
            aa_seq=batch["seq"],
            chain_ids=batch["chain_ids"],
            noiser=state.noiser,
            mask=batch.get("mask"),
            clamp_val=state.config["inference"]["clamp_val"],
            noise_type=state.config["inference"]["noise_type"],
        )

    # Kabsch align prediction to ground truth
    pred_aligned, _ = kabsch_align(pred_coords, batch["xyz"], batch.get("mask"))
    pred_np = pred_aligned[0].cpu().numpy()
    gt_np = batch["xyz"][0].cpu().numpy()

    # Compute RMSD
    diff = pred_np - gt_np
    rmsd = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    inference_time = (time.time() - start) * 1000

    # Generate PDB with predicted coords
    pred_sample = sample.copy()
    pred_sample["xyz"] = pred_np

    return PredictionResponse(
        sample_id=sample_id,
        pdb_string=sample_to_pdb(pred_sample),
        rmsd=rmsd,
        inference_time_ms=inference_time,
    )

if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
```

## Frontend Implementation

### HTML Structure

```html
<!-- web/static/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>TinyFold Visualizer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <!-- Sidebar: Sample Browser -->
        <aside class="sidebar">
            <div class="search-box">
                <input type="text" id="search" placeholder="Search samples...">
                <select id="split-filter">
                    <option value="all">All</option>
                    <option value="train">Train</option>
                    <option value="test">Test</option>
                </select>
            </div>
            <div id="sample-list" class="sample-list">
                <!-- Populated by JS -->
            </div>
        </aside>

        <!-- Main: Viewer + Controls -->
        <main class="viewer-area">
            <!-- Style Controls -->
            <div class="controls">
                <div class="control-group">
                    <label>Style:</label>
                    <button data-style="cartoon" class="active">Cartoon</button>
                    <button data-style="stick">Stick</button>
                    <button data-style="sphere">Sphere</button>
                    <button data-style="line">Line</button>
                </div>
                <div class="control-group">
                    <label>Color:</label>
                    <select id="color-scheme">
                        <option value="chain">By Chain</option>
                        <option value="spectrum">Spectrum</option>
                        <option value="ss">Secondary Structure</option>
                        <option value="residue">By Residue Type</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Show:</label>
                    <label><input type="checkbox" id="show-gt" checked> Ground Truth</label>
                    <label><input type="checkbox" id="show-pred"> Prediction</label>
                    <label><input type="checkbox" id="show-surface"> Surface</label>
                </div>
                <div class="control-group">
                    <button id="btn-predict" class="primary">Run Prediction</button>
                    <span id="rmsd-display"></span>
                </div>
            </div>

            <!-- 3Dmol Viewer -->
            <div id="viewer" class="mol-viewer"></div>

            <!-- Info Panel -->
            <div id="info-panel" class="info-panel">
                <h3 id="sample-title">Select a sample</h3>
                <div id="sample-info"></div>
            </div>
        </main>
    </div>

    <script src="js/api.js"></script>
    <script src="js/viewer.js"></script>
    <script src="js/app.js"></script>
</body>
</html>
```

### 3Dmol.js Wrapper

```javascript
// web/static/js/viewer.js

class ProteinViewer {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.viewer = $3Dmol.createViewer(this.element, {
            backgroundColor: 'white'
        });
        this.models = {
            groundTruth: null,
            prediction: null
        };
        this.style = 'cartoon';
        this.colorScheme = 'chain';
    }

    /**
     * Load ground truth structure from PDB string
     */
    loadGroundTruth(pdbString) {
        if (this.models.groundTruth !== null) {
            this.viewer.removeModel(this.models.groundTruth);
        }
        this.models.groundTruth = this.viewer.addModel(pdbString, 'pdb');
        this.applyStyle('groundTruth');
        this.viewer.zoomTo();
        this.viewer.render();
    }

    /**
     * Load prediction structure from PDB string
     */
    loadPrediction(pdbString) {
        if (this.models.prediction !== null) {
            this.viewer.removeModel(this.models.prediction);
        }
        this.models.prediction = this.viewer.addModel(pdbString, 'pdb');
        this.applyStyle('prediction');
        this.viewer.render();
    }

    /**
     * Apply visual style to a model
     */
    applyStyle(modelKey) {
        const model = this.models[modelKey];
        if (!model) return;

        const baseColor = modelKey === 'groundTruth' ?
            {chain: {A: '#3498db', B: '#2ecc71'}} :  // Blue/Green for GT
            {chain: {A: '#e74c3c', B: '#f39c12'}};   // Red/Orange for Pred

        let styleSpec = {};

        switch (this.style) {
            case 'cartoon':
                styleSpec = {cartoon: this.getColorSpec(modelKey)};
                break;
            case 'stick':
                styleSpec = {stick: {radius: 0.2, ...this.getColorSpec(modelKey)}};
                break;
            case 'sphere':
                styleSpec = {sphere: {radius: 0.5, ...this.getColorSpec(modelKey)}};
                break;
            case 'line':
                styleSpec = {line: this.getColorSpec(modelKey)};
                break;
        }

        model.setStyle({}, styleSpec);
    }

    /**
     * Get color specification based on current scheme
     */
    getColorSpec(modelKey) {
        const isPred = modelKey === 'prediction';

        switch (this.colorScheme) {
            case 'chain':
                // Different colors for GT vs Pred
                if (isPred) {
                    return {colorfunc: (atom) =>
                        atom.chain === 'A' ? '#e74c3c' : '#f39c12'
                    };
                } else {
                    return {colorfunc: (atom) =>
                        atom.chain === 'A' ? '#3498db' : '#2ecc71'
                    };
                }
            case 'spectrum':
                return {color: 'spectrum'};
            case 'ss':
                return {color: 'ss'};
            case 'residue':
                return {colorscheme: 'amino'};
            default:
                return {};
        }
    }

    /**
     * Set rendering style for all models
     */
    setStyle(style) {
        this.style = style;
        Object.keys(this.models).forEach(key => this.applyStyle(key));
        this.viewer.render();
    }

    /**
     * Set color scheme
     */
    setColorScheme(scheme) {
        this.colorScheme = scheme;
        Object.keys(this.models).forEach(key => this.applyStyle(key));
        this.viewer.render();
    }

    /**
     * Toggle visibility of a model
     */
    toggleModel(modelKey, visible) {
        const model = this.models[modelKey];
        if (!model) return;

        if (visible) {
            this.applyStyle(modelKey);
        } else {
            model.setStyle({}, {});
        }
        this.viewer.render();
    }

    /**
     * Add molecular surface
     */
    addSurface(modelKey, opacity = 0.7) {
        const model = this.models[modelKey];
        if (!model) return;

        this.viewer.addSurface($3Dmol.SurfaceType.VDW, {
            opacity: opacity,
            color: modelKey === 'groundTruth' ? '#3498db' : '#e74c3c'
        }, {model: model});
        this.viewer.render();
    }

    /**
     * Remove all surfaces
     */
    removeSurfaces() {
        this.viewer.removeAllSurfaces();
        this.viewer.render();
    }

    /**
     * Clear all models
     */
    clear() {
        this.viewer.removeAllModels();
        this.models = {groundTruth: null, prediction: null};
        this.viewer.render();
    }

    /**
     * Reset view
     */
    resetView() {
        this.viewer.zoomTo();
        this.viewer.render();
    }
}
```

### API Client

```javascript
// web/static/js/api.js

const API = {
    baseUrl: '/api',

    async getSamples(options = {}) {
        const params = new URLSearchParams();
        if (options.split) params.append('split', options.split);
        if (options.search) params.append('search', options.search);
        if (options.page) params.append('page', options.page);

        const response = await fetch(`${this.baseUrl}/samples?${params}`);
        return response.json();
    },

    async getSample(sampleId) {
        const response = await fetch(`${this.baseUrl}/sample/${sampleId}`);
        return response.json();
    },

    async getSamplePDB(sampleId) {
        const response = await fetch(`${this.baseUrl}/sample/${sampleId}/pdb`);
        return response.text();
    },

    async predict(sampleId, modelPath = null) {
        const options = {method: 'POST'};
        let url = `${this.baseUrl}/predict/${sampleId}`;
        if (modelPath) {
            url += `?model=${encodeURIComponent(modelPath)}`;
        }
        const response = await fetch(url, options);
        return response.json();
    },

    async getModels() {
        const response = await fetch(`${this.baseUrl}/models`);
        return response.json();
    }
};
```

### Main Application

```javascript
// web/static/js/app.js

class App {
    constructor() {
        this.viewer = new ProteinViewer('viewer');
        this.currentSample = null;
        this.setupEventListeners();
        this.loadSamples();
    }

    setupEventListeners() {
        // Search
        document.getElementById('search').addEventListener('input',
            debounce(() => this.loadSamples(), 300));

        // Split filter
        document.getElementById('split-filter').addEventListener('change',
            () => this.loadSamples());

        // Style buttons
        document.querySelectorAll('[data-style]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-style]').forEach(b =>
                    b.classList.remove('active'));
                btn.classList.add('active');
                this.viewer.setStyle(btn.dataset.style);
            });
        });

        // Color scheme
        document.getElementById('color-scheme').addEventListener('change',
            (e) => this.viewer.setColorScheme(e.target.value));

        // Visibility toggles
        document.getElementById('show-gt').addEventListener('change',
            (e) => this.viewer.toggleModel('groundTruth', e.target.checked));
        document.getElementById('show-pred').addEventListener('change',
            (e) => this.viewer.toggleModel('prediction', e.target.checked));
        document.getElementById('show-surface').addEventListener('change',
            (e) => {
                if (e.target.checked) {
                    this.viewer.addSurface('groundTruth');
                } else {
                    this.viewer.removeSurfaces();
                }
            });

        // Predict button
        document.getElementById('btn-predict').addEventListener('click',
            () => this.runPrediction());
    }

    async loadSamples() {
        const search = document.getElementById('search').value;
        const split = document.getElementById('split-filter').value;

        const data = await API.getSamples({
            search: search || undefined,
            split: split !== 'all' ? split : undefined
        });

        this.renderSampleList(data.samples);
    }

    renderSampleList(samples) {
        const container = document.getElementById('sample-list');
        container.innerHTML = samples.map(s => `
            <div class="sample-item" data-id="${s.sample_id}">
                <span class="sample-id">${s.sample_id}</span>
                <span class="sample-meta">${s.n_residues} res</span>
            </div>
        `).join('');

        container.querySelectorAll('.sample-item').forEach(item => {
            item.addEventListener('click', () =>
                this.selectSample(item.dataset.id));
        });
    }

    async selectSample(sampleId) {
        // Update UI
        document.querySelectorAll('.sample-item').forEach(el =>
            el.classList.remove('selected'));
        document.querySelector(`[data-id="${sampleId}"]`)
            ?.classList.add('selected');

        // Load sample
        const sample = await API.getSample(sampleId);
        this.currentSample = sample;

        // Update info panel
        document.getElementById('sample-title').textContent = sampleId;
        document.getElementById('sample-info').innerHTML = `
            <p>Residues: ${sample.n_residues}</p>
            <p>Atoms: ${sample.n_atoms}</p>
            <p>Split: ${sample.split}</p>
        `;

        // Load structure
        const pdb = await API.getSamplePDB(sampleId);
        this.viewer.clear();
        this.viewer.loadGroundTruth(pdb);

        // Reset prediction checkbox
        document.getElementById('show-pred').checked = false;
        document.getElementById('rmsd-display').textContent = '';
    }

    async runPrediction() {
        if (!this.currentSample) {
            alert('Select a sample first');
            return;
        }

        const btn = document.getElementById('btn-predict');
        btn.disabled = true;
        btn.textContent = 'Running...';

        try {
            const result = await API.predict(this.currentSample.sample_id);

            this.viewer.loadPrediction(result.pdb_string);
            document.getElementById('show-pred').checked = true;
            document.getElementById('rmsd-display').textContent =
                `RMSD: ${result.rmsd.toFixed(2)} A`;
        } catch (err) {
            alert('Prediction failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Run Prediction';
        }
    }
}

// Utility
function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
```

## CSS Styling

```css
/* web/static/css/style.css */

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
}

.container {
    display: flex;
    height: 100vh;
}

/* Sidebar */
.sidebar {
    width: 280px;
    background: white;
    border-right: 1px solid #ddd;
    display: flex;
    flex-direction: column;
}

.search-box {
    padding: 12px;
    border-bottom: 1px solid #ddd;
}

.search-box input, .search-box select {
    width: 100%;
    padding: 8px;
    margin-bottom: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.sample-list {
    flex: 1;
    overflow-y: auto;
}

.sample-item {
    padding: 10px 12px;
    border-bottom: 1px solid #eee;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
}

.sample-item:hover {
    background: #f9f9f9;
}

.sample-item.selected {
    background: #e3f2fd;
    border-left: 3px solid #2196f3;
}

.sample-id {
    font-family: monospace;
    font-size: 13px;
}

.sample-meta {
    color: #888;
    font-size: 12px;
}

/* Main viewer area */
.viewer-area {
    flex: 1;
    display: flex;
    flex-direction: column;
}

.controls {
    background: white;
    padding: 12px;
    border-bottom: 1px solid #ddd;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
}

.control-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.control-group label {
    font-size: 13px;
    color: #666;
}

.control-group button {
    padding: 6px 12px;
    border: 1px solid #ddd;
    background: white;
    border-radius: 4px;
    cursor: pointer;
}

.control-group button.active {
    background: #2196f3;
    color: white;
    border-color: #2196f3;
}

.control-group button.primary {
    background: #4caf50;
    color: white;
    border-color: #4caf50;
}

#rmsd-display {
    font-weight: bold;
    color: #e74c3c;
}

/* 3Dmol viewer */
.mol-viewer {
    flex: 1;
    position: relative;
}

/* Info panel */
.info-panel {
    background: white;
    padding: 12px;
    border-top: 1px solid #ddd;
}

.info-panel h3 {
    margin-bottom: 8px;
    font-family: monospace;
}

.info-panel p {
    font-size: 13px;
    color: #666;
    margin: 4px 0;
}
```

## Implementation Order

1. **Phase 1: Static Frontend**
   - Set up basic HTML/CSS layout
   - Integrate 3Dmol.js viewer
   - Test with hardcoded PDB data
   - Implement style/color controls

2. **Phase 2: Backend Data API**
   - Flask server with sample listing
   - Load parquet data
   - PDB generation endpoint
   - Sample search/filter

3. **Phase 3: Prediction API**
   - Model loading from checkpoint
   - Inference endpoint
   - Kabsch alignment
   - RMSD computation

4. **Phase 4: Integration**
   - Connect frontend to backend
   - Handle loading states
   - Error handling
   - Polish UI

## Running Locally

```bash
# Install dependencies
pip install fastapi uvicorn pyyaml

# Start server
cd web
python server.py

# Open browser
# http://127.0.0.1:5000
# API docs at http://127.0.0.1:5000/docs
```

## Dependencies

Backend:
- FastAPI + uvicorn
- PyYAML (for config)
- PyArrow (already used)
- PyTorch (for inference)

Frontend:
- 3Dmol.js (CDN, no install needed)

## References

- [3Dmol.js Documentation](https://3dmol.csb.pitt.edu/)
- [3Dmol.js GitHub](https://github.com/3dmol/3Dmol.js)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
