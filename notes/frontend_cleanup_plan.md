# Frontend Cleanup Implementation Plan

## Overview

Refactor the TinyFold web frontends to clearly separate concerns between:
1. **`web`**: Full-featured frontend with lazy model loading and live predictions
2. **`web-light`**: Lightweight demo using pre-extracted static JSON data (no server-side model)

---

## Current Issues

### Try-Except Imports to Remove

The following try-except imports in `web/server.py` need to be removed/cleaned:

| File | Lines | Import | Purpose |
|------|-------|--------|---------|
| [server.py](file:///c:/Users/costa/src/tinyfold/web/server.py#L30-L39) | 30-39 | `from models import create_model, create_schedule, create_noiser` | Optional model loading |
| [server.py](file:///c:/Users/costa/src/tinyfold/web/server.py#L41-L46) | 41-46 | `from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices` | Data split utilities |

> [!IMPORTANT]
> These try-excepts exist because the web server is designed to work both **with** and **without** the model imports available. The `web` frontend should **always** require model imports (clean imports), while `web-light` should **never** need them.

---

## Proposed Changes

### Component 1: `web` Frontend - Lazy Model Loading

#### [MODIFY] [server.py](file:///c:/Users/costa/src/tinyfold/web/server.py)

1. **Remove try-except imports** - Convert to clean direct imports:
   ```python
   from models import create_model, create_schedule, create_noiser
   from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices
   ```

2. **Implement lazy model loading**:
   - Remove model loading from `init_app()` startup
   - Add `_ensure_model_loaded()` helper that loads model on first prediction request
   - Model is only loaded when user clicks "Run Prediction" button
   - Benefits: Faster server startup, lower memory usage until needed

3. **Update `predict()` endpoint** to call lazy loader before inference

---

### Component 2: `web-light` Frontend - Static Demo

#### [NEW] `web-light/data/demo_predictions.json`

Create a small JSON file containing a curated subset of predictions (~10-20 samples) with:
- Sample IDs
- Pre-computed prediction coordinates  
- Ground truth coordinates (extracted from parquet)
- RMSD values

#### [NEW] `web-light/static/` (separate static files)

Create standalone static files for `web-light`:
- `index.html` - Simplified UI without backend API calls
- `js/app.js` - Local-only data loading from static JSON
- `css/style.css` - (can symlink or copy from web)

#### [MODIFY] [web-light/server.py](file:///c:/Users/costa/src/tinyfold/web-light/server.py)

1. Serve static demo files from `web-light/static/`
2. Add endpoint to serve `demo_predictions.json`
3. No model imports required - purely static file server

#### [NEW] `scripts/extract_demo_data.py`

Script to extract ground truths and bundle with predictions:
- Read samples from parquet file
- Match with predictions from `web/predictions.json`
- Output condensed `demo_predictions.json` for web-light

---

## Verification Plan

### Manual Testing

1. **Test `web` frontend with lazy loading**:
   - Start server: `cd web && python server.py`
   - Verify server starts WITHOUT loading model (check startup logs)
   - Browse samples in UI (should work - no model needed)
   - Click "Run Prediction" on a sample
   - Verify model loads on first prediction (check logs for "Loading model...")
   - Subsequent predictions should be fast (model already loaded)

2. **Test `web-light` frontend**:
   - Start server: `cd web-light && python server.py`
   - Verify server starts quickly (no model loading)
   - Browse pre-loaded samples in UI
   - Select a sample - should show ground truth
   - View cached prediction (from static JSON)
   - Verify no network errors in browser console

### Questions for User

1. How many samples should be included in the `web-light` demo JSON? (Suggested: 10-20)
2. Should `web-light` include both train and test split samples?
3. For the lazy loading in `web`, should the model load be triggered by:
   - First prediction request (current suggestion)
   - Server startup on separate thread
   - Explicit `/api/load-model` endpoint
