# Plan: README Polish & Architecture Visualization

## Current State

The README has rough placeholder text that needs professional formatting. Key sections to address:
- Title and introduction
- Architecture explanation with diagram
- Training details
- Dataset description
- Future directions
- License

## Inspiration: AF3 Diagram Style

Based on [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/):

**Color Conventions:**
- Blue for "single" representations (token-level)
- Orange/Gold for "pair" representations or secondary paths
- Green for outputs
- Gray for inputs

**Layout Conventions:**
- Rectangular blocks for tensors with labeled dimensions
- Arrows showing data flow
- Modular boxes for architectural components
- Size roughly proportional to tensor dimensions

## README Structure (Polished)

```markdown
# TinyFold

[One-line tagline]

[Hero image: test set prediction visualization]

## Overview
- What it does (binary PPI structure prediction)
- Key insight (residue-first, then atoms)
- Model size (<30M params, single GPU trainable)

## Architecture
[Architecture diagram - generated SVG/PNG]

### Two-Stage Design
1. **Stage 1: Residue Diffusion** - predict residue centroids
2. **Stage 2: Atom Refinement** - predict 4 backbone atoms per residue

### Why This Approach?
- Efficiency: L tokens vs 4L tokens
- Biological intuition: topology is hard, local geometry is constrained
- Practical: works when one protein structure is known

## Results
[Table or figure of metrics on test set]

## Installation & Usage
```bash
pip install -e .
python scripts/predict.py --pdb1 chain_a.pdb --pdb2 chain_b.pdb
```

## Training
- Dataset: DIPS-Plus (28K binary complexes)
- Hardware: Single RTX 4070 Ti Super
- Time: ~8 hours for Stage 1

## Future Directions
- Boltz-2 per-step alignment
- Energy-based refinement
- Small molecule extension

## License
MIT

## Citation
[If applicable]
```

## Architecture Diagram Script

### File: `scripts/plot_architecture.py`

**Approach:** Use matplotlib with custom drawing (patches, arrows, text) to create a clean architectural diagram. This is more maintainable than graphviz and produces publication-quality output.

### Diagram Layout (Top to Bottom)

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                   │
│  Sequence [L]  Chain IDs [L]  Noisy Centroids x_t [L, 3]        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Residue Diffusion                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ResidueEncoder (Trunk) - runs once                      │    │
│  │   AA Embed + Chain Embed + Position Enc + Coord Proj    │    │
│  │   → Transformer (9 layers) → trunk_tokens [L, 256]      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ DiffusionTransformer (Denoiser) - runs T times          │    │
│  │   coord_embed(x_t) + trunk_tokens + time_embed(t)       │    │
│  │   → Transformer (7 blocks, AdaLN) → x0_pred [L, 3]      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Predicted Centroids
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Atom Refinement                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ GlobalTransformer                                        │    │
│  │   centroid_embed + sequence_embed + position_enc        │    │
│  │   → Transformer (6 layers) → residue_tokens [L, 192]    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LocalAtomAttention                                       │    │
│  │   Broadcast to [L, 4, c] → Local attention within res   │    │
│  │   → atom_offsets [L, 4, 3]                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│              Backbone Atoms [L, 4, 3] (N, CA, C, O)             │
└─────────────────────────────────────────────────────────────────┘
```

### Color Scheme

```python
COLORS = {
    'input': '#FFE4B5',      # Moccasin (light orange)
    'stage1': '#ADD8E6',     # Light blue
    'stage2': '#90EE90',     # Light green
    'output': '#DDA0DD',     # Plum (light purple)
    'module': '#F5F5F5',     # White smoke (inner modules)
    'arrow': '#333333',      # Dark gray
    'text': '#1a1a1a',       # Near black
}
```

### Implementation Plan

```python
# scripts/plot_architecture.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, w, h, label, sublabel=None, color='white', ...):
    """Draw a rounded rectangle with label."""

def draw_arrow(ax, start, end, ...):
    """Draw a connecting arrow."""

def draw_stage(ax, y_start, stage_name, modules, color):
    """Draw a stage box containing multiple modules."""

def plot_resfold_architecture(save_path='docs/images/architecture.png'):
    fig, ax = plt.subplots(figsize=(12, 16))

    # Draw from top to bottom
    draw_input_box(ax, ...)
    draw_stage(ax, ..., "Stage 1: Residue Diffusion", [...], COLORS['stage1'])
    draw_stage(ax, ..., "Stage 2: Atom Refinement", [...], COLORS['stage2'])
    draw_output_box(ax, ...)

    # Connect with arrows
    ...

    plt.savefig(save_path, dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    plot_resfold_architecture()
```

### Additional: Example Prediction Plot

Also create a function to plot an example prediction showing:
- Chain A (blue) and Chain B (orange) residue centroids
- Ground truth vs predicted overlay
- Optional: backbone trace

```python
def plot_prediction_example(pred_centroids, gt_centroids, chain_ids, save_path):
    """3D scatter plot of predicted vs ground truth structure."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot chains with different colors
    # Plot GT as solid, pred as transparent/dashed
    ...
```

## Files to Create/Modify

1. **MODIFY: `README.md`**
   - Complete rewrite with professional formatting
   - Embed architecture diagram
   - Add installation, usage, results sections

2. **CREATE: `scripts/plot_architecture.py`**
   - Architecture diagram generation
   - Example prediction visualization
   - CLI interface for regenerating figures

3. **CREATE: `docs/images/` directory**
   - `architecture.png` - main architecture diagram
   - `example_prediction.png` - test set example
   - `training_curves.png` - loss/RMSE over training (optional)

## Execution Order

1. Create `scripts/plot_architecture.py` with architecture diagram function
2. Generate `docs/images/architecture.png`
3. Create example prediction plot (need to load model and run inference)
4. Polish README.md with all content and embedded images

## Questions

1. Should we include a results table with metrics (RMSE, lDDT, etc.)?
2. Do you want a web frontend section in the README?
3. Any specific test cases you want visualized?
