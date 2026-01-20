# ResFold: Two-Stage PPI Structure Prediction

A hierarchical architecture that decouples residue-level diffusion from atomic refinement.

## Model Name: **ResFold**

*"Residue-First Folding"* - emphasizing the hierarchical approach.

---

## Overview

ResFold addresses a key insight: predicting 3D protein structure can be decomposed into:
1. **Global positioning**: Where do residues go in 3D space? (Coarse, diffusion-based)
2. **Local geometry**: Given residue positions, where do atoms go? (Fine, one-shot)

This matches biological intuition: backbone topology is the hard problem; local bond geometry is well-constrained.

### Architecture Comparison

| Aspect | AF3-Style (Current) | ResFold (Proposed) |
|--------|---------------------|-------------------|
| **Output granularity** | 4 atoms/residue | 2 stages: 1 position/residue → 4 atoms/residue |
| **Diffusion target** | All 4×L atoms | L residue centroids only |
| **Atom positioning** | Implicit in diffusion | Explicit one-shot refinement |
| **Diffusion complexity** | O(4L) tokens | O(L) tokens (4× smaller) |
| **Training modes** | End-to-end only | Stage 1 only, Stage 2 only, or end-to-end |

---

## Stage 1: Residue-Level Diffusion (ResidueDenoiser)

### Input/Output Specification

```
Input:
  - x_t_res: Noisy residue positions [B, L, 3]  (centroid of 4 atoms)
  - aa_seq: Amino acid sequence [B, L]
  - chain_ids: Chain assignments [B, L]
  - res_idx: Residue indices [B, L]
  - t: Diffusion timestep [B]
  - mask: Valid residue mask [B, L]

Output:
  - x0_res_pred: Predicted clean residue positions [B, L, 3]
```

### Architecture (Same structure as AF3StyleDecoder, simplified)

```
┌──────────────────────────────────────────────────────────────┐
│                    ResidueDenoiser                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ╔══════════════════════════════════════════════════════╗   │
│  ║ ResidueEncoder (Trunk) - runs ONCE                   ║   │
│  ╠══════════════════════════════════════════════════════╣   │
│  ║  • aa_embed(aa_seq)       [B, L, c_token]            ║   │
│  ║  • chain_embed(chain_ids) [B, L, c_token//4]         ║   │
│  ║  • res_pos_enc(res_idx)   [B, L, c_token]            ║   │
│  ║  • coord_proj(x0_res)     [B, L, c_token//2]         ║   │
│  ║                                                       ║   │
│  ║  → input_proj → Transformer(n_layers) → trunk_tokens ║   │
│  ╚══════════════════════════════════════════════════════╝   │
│                          ↓                                   │
│  ╔══════════════════════════════════════════════════════╗   │
│  ║ ResidueDenoiser Module - runs EACH diffusion step    ║   │
│  ╠══════════════════════════════════════════════════════╣   │
│  ║  Inputs: noisy x_t_res, trunk_tokens, time_embed(t)  ║   │
│  ║                                                       ║   │
│  ║  • coord_embed(x_t_res)  [B, L, c_token]             ║   │
│  ║  • add trunk_tokens conditioning                     ║   │
│  ║  • DiffusionTransformer(n_blocks) with AdaLN         ║   │
│  ║  • output_proj → coord_delta [B, L, 3]               ║   │
│  ║                                                       ║   │
│  ║  x0_pred = x_t + scale(t) * coord_delta              ║   │
│  ╚══════════════════════════════════════════════════════╝   │
│                          ↓                                   │
│  Output: x0_res_pred [B, L, 3]                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Differences from AF3-Style

1. **No AtomAttentionEncoder/Decoder** - no local atom attention, only residue-level
2. **3× fewer tokens** - L instead of 4L (4 atoms/residue)
3. **Simpler architecture** - DiffusionTransformer operates directly on residue tokens

### Residue Centroid Definition

```python
# During training data preparation:
# coords: [B, N_atoms, 3] where N_atoms = 4 * L
# Reshape and compute mean:
coords_res = coords.view(B, L, 4, 3)  # [B, L, 4, 3]
centroids = coords_res.mean(dim=2)    # [B, L, 3]  (mean of N, CA, C, O)
```

---

## Stage 2: Atom Refinement Network (AtomRefiner)

### Input/Output Specification

```
Input:
  - x_res: Residue centroid positions [B, L, 3]
  - aa_seq: Amino acid sequence [B, L]
  - chain_ids: Chain assignments [B, L]
  - res_idx: Residue indices [B, L]
  - mask_res: Valid residue mask [B, L]

Output:
  - x_atoms: 3D atom positions [B, L, 4, 3]
```

### Architecture (Attention-based, one-shot)

```
┌──────────────────────────────────────────────────────────────┐
│                    AtomRefiner                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ╔══════════════════════════════════════════════════════╗   │
│  ║ Input Embedding                                       ║   │
│  ╠══════════════════════════════════════════════════════╣   │
│  ║  • coord_proj(x_res)      [B, L, c_token]            ║   │
│  ║  • aa_embed(aa_seq)       [B, L, c_token]            ║   │
│  ║  • chain_embed(chain_ids) [B, L, c_token//4]         ║   │
│  ║  • res_pos_enc(res_idx)   [B, L, c_token]            ║   │
│  ║                                                       ║   │
│  ║  → input_proj [B, L, c_token]                        ║   │
│  ╚══════════════════════════════════════════════════════╝   │
│                          ↓                                   │
│  ╔══════════════════════════════════════════════════════╗   │
│  ║ Global Context Encoder (Transformer)                  ║   │
│  ╠══════════════════════════════════════════════════════╣   │
│  ║  TransformerEncoder(n_layers) → tokens [B, L, c]     ║   │
│  ╚══════════════════════════════════════════════════════╝   │
│                          ↓                                   │
│  ╔══════════════════════════════════════════════════════╗   │
│  ║ Atom Position Prediction                              ║   │
│  ╠══════════════════════════════════════════════════════╣   │
│  ║  Option A: Direct prediction                          ║   │
│  ║    • atom_proj: [B, L, c] → [B, L, 4, 3]             ║   │
│  ║    • atom_pos = x_res.unsqueeze(2) + atom_offsets    ║   │
│  ║                                                       ║   │
│  ║  Option B: Local atom attention (from AF3-style)     ║   │
│  ║    • Broadcast tokens → [B, L, 4, c_atom]            ║   │
│  ║    • LocalAtomTransformer blocks                     ║   │
│  ║    • output_proj → [B, L, 4, 3]                      ║   │
│  ╚══════════════════════════════════════════════════════╝   │
│                          ↓                                   │
│  Output: x_atoms [B, L, 4, 3]                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Design Choice: Direct Prediction (Option A)

For Stage 2, we use **direct prediction** because:
1. **Local geometry is constrained** - N-CA-C-O has fixed bond lengths/angles
2. **Simpler** - just predict 4×3 offsets from centroid
3. **Faster** - no per-atom attention needed

The network predicts **relative offsets** from the residue centroid to each of the 4 backbone atoms.

---

## Combined Pipeline (ResFoldPipeline)

```
┌──────────────────────────────────────────────────────────────┐
│                    ResFoldPipeline                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Residue-level diffusion (T steps)                 │
│  ─────────────────────────────────────────                  │
│    x_T = randn(B, L, 3)                                     │
│    for t in reversed(range(T)):                             │
│        x0_res_pred = residue_denoiser(x_t, t, ...)          │
│        x_{t-1} = ddpm_step(x_t, x0_res_pred, t)             │
│    x_res = x_0                                              │
│                                                              │
│  Phase 2: Atom refinement (1 step)                          │
│  ─────────────────────────────────                          │
│    x_atoms = atom_refiner(x_res, ...)                       │
│    x_atoms: [B, L, 4, 3]                                    │
│                                                              │
│  Output: x_atoms reshaped to [B, N_atoms, 3]                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Training Modes

| Mode | Stage 1 | Stage 2 | Use Case |
|------|---------|---------|----------|
| `stage1_only` | Train | Frozen/None | Pre-train residue diffusion |
| `stage2_only` | Frozen | Train | Pre-train atom refinement (use GT residue positions) |
| `end_to_end` | Train | Train | Fine-tune both stages together |

---

## Proposed Changes

### [NEW] resfold.py

Stage 1 residue-level diffusion model:
- `ResidueEncoder` - Trunk network (runs once per sample)
- `ResidueDiffusionTransformer` - Denoiser (runs each diffusion step)
- `ResidueDenoiser` - Main Stage 1 model

Reuses from `af3_style.py`:
- `AdaLN` - Adaptive layer normalization
- `SwiGLU` - Feedforward block
- `DiffusionTransformerBlock` - Global attention with AdaLN

---

### [NEW] atomrefine.py

Stage 2 atom refinement network:
- `AtomRefiner` - One-shot prediction of atom positions from residue centroids

Architecture:
- Input embedding layer
- Transformer encoder for global context
- Atom offset prediction head

---

### [NEW] resfold_pipeline.py

Combined two-stage pipeline:
- `ResFoldPipeline` - Orchestrates Stage 1 → Stage 2
- `sample()` method for inference
- Support for training modes: `stage1_only`, `stage2_only`, `end_to_end`

---

### [MODIFY] __init__.py

Add new models to factory:
```python
from .resfold import ResidueDenoiser
from .atomrefine import AtomRefiner
from .resfold_pipeline import ResFoldPipeline

# Add to _MODELS dict
_MODELS["resfold_stage1"] = ResidueDenoiser
_MODELS["resfold_stage2"] = AtomRefiner
_MODELS["resfold"] = ResFoldPipeline
```

---

### [NEW] train_resfold.py

Separate training script for ResFold:

**Key differences from `train.py`:**
1. Data preprocessing computes residue centroids
2. Training modes: `--mode stage1_only|stage2_only|end_to_end`
3. Only Gaussian noise (no linear_chain/linear_flow)
4. No curriculum
5. Separate loss components for each stage

**Command line interface:**
```bash
# Train Stage 1 only (residue diffusion)
python train_resfold.py --mode stage1_only --n_train 80 --n_steps 10000

# Train Stage 2 only (atom refinement, using GT residue positions)
python train_resfold.py --mode stage2_only --n_train 80 --n_steps 5000

# End-to-end training
python train_resfold.py --mode end_to_end --n_train 80 --n_steps 15000
```

---

## Model Configuration (Baseline from AF3-Style)

Based on successful overfitting experiments with AF3-style on 15M parameter models:

### Stage 1: ResidueDenoiser

| Parameter | Value | Notes |
|-----------|-------|-------|
| `c_token` | 256 | Hidden dimension |
| `trunk_layers` | 9 | ResidueEncoder layers |
| `denoiser_blocks` | 7 | DiffusionTransformer blocks |
| `n_heads` | 8 | Attention heads |
| `T` | 50 | Diffusion timesteps |
| `schedule` | cosine | Alpha-bar schedule |

**Estimated parameters**: ~8-10M (smaller than AF3-style due to no atom attention)

### Stage 2: AtomRefiner

| Parameter | Value | Notes |
|-----------|-------|-------|
| `c_token` | 128 | Hidden dimension |
| `n_layers` | 4 | Transformer layers |
| `n_heads` | 4 | Attention heads |

**Estimated parameters**: ~2-3M

### Combined Pipeline

**Total parameters**: ~10-13M (comparable to AF3-style 15M)

---

## Verification Plan

### Automated Tests

1. **Unit tests** (`tests/test_resfold.py`):
   - Shape tests for each module
   - Forward pass verification
   - Gradient flow checks

2. **Integration tests**:
   - Training loop with 10 steps
   - Full Stage1 → Stage2 sampling

### Overfitting Test

```bash
# Stage 1 overfit test
python train_resfold.py --mode stage1_only --n_train 5 --n_test 2 --n_steps 5000

# Stage 2 overfit test
python train_resfold.py --mode stage2_only --n_train 5 --n_test 2 --n_steps 2000

# End-to-end overfit test
python train_resfold.py --mode end_to_end --n_train 5 --n_test 2 --n_steps 5000
```

**Success criteria**:
- Stage 1: Train RMSE < 1.0 Å on residue centroids
- Stage 2: Train RMSE < 0.5 Å on atom positions (given GT residues)
- End-to-end: Train RMSE < 2.0 Å on atoms

---

## Implementation Order

1. **Phase 1**: Create model files
   - `resfold.py` (Stage 1)
   - `atomrefine.py` (Stage 2)
   - `resfold_pipeline.py` (Combined)
   - Update `__init__.py`

2. **Phase 2**: Create training script
   - `train_resfold.py`
   - Data loading with residue centroid computation
   - Training loop for all three modes

3. **Phase 3**: Tests and verification
   - Unit tests in `tests/test_resfold.py`
   - Run overfit tests
   - Compare with AF3-style baseline
