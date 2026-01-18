# TinyFold Architecture

## Overview

TinyFold is a protein-protein interaction (PPI) structure predictor that uses an attention-based diffusion model to predict backbone atom coordinates (N, CA, C, O) for two interacting protein chains.

```
                    AttentionDiffusionV2
┌────────────────────────────────────────────────────────┐
│                                                        │
│   Inputs:                                              │
│   ─────────────────────────────────────────────        │
│   • Noisy coords x_t    [B, N_atom, 3]                │
│   • Atom types          [B, N_atom]    (N/CA/C/O)     │
│   • Residue indices     [B, N_atom]                   │
│   • Amino acid sequence [B, N_atom]                   │
│   • Chain IDs           [B, N_atom]    (0 or 1)       │
│   • Timestep t          [B]                           │
│                                                        │
│   ┌──────────────────────────────────────────────┐    │
│   │              Token Embeddings                 │    │
│   │  atom_type + aa_seq + chain + res_pos + time │    │
│   │           + coord_proj(x_t)                  │    │
│   └──────────────────┬───────────────────────────┘    │
│                      │                                 │
│                      ▼                                 │
│   ┌──────────────────────────────────────────────┐    │
│   │         Transformer Encoder                   │    │
│   │    (n_layers x TransformerEncoderLayer)      │    │
│   │    • Self-attention across all atoms         │    │
│   │    • FFN with 4x expansion                   │    │
│   │    • Pre-norm (LayerNorm before attention)   │    │
│   └──────────────────┬───────────────────────────┘    │
│                      │                                 │
│                      ▼                                 │
│   ┌──────────────────────────────────────────────┐    │
│   │            Output Projection                  │    │
│   │         LayerNorm → Linear(h_dim, 3)         │    │
│   └──────────────────┬───────────────────────────┘    │
│                      │                                 │
│                      ▼                                 │
│   Output: x0_pred [B, N_atom, 3]                      │
│   (predicted clean coordinates)                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Design Philosophy

### Why Attention-Based?

1. **Simplicity**: A standard Transformer encoder is simpler to implement and debug than specialized equivariant GNNs.

2. **Global context**: Self-attention allows every atom to directly attend to every other atom, capturing long-range interactions without message passing iterations.

3. **Proven scaling**: Transformer architectures scale well with model size and data.

4. **x0 prediction**: We predict clean coordinates directly rather than noise, which works better for coordinate prediction.

### Trade-offs

- **No equivariance**: Unlike EGNN, this architecture is not SE(3)-equivariant. The model must learn rotation/translation invariance from data.
- **Quadratic attention**: O(N²) attention limits sequence length, but backbone-only proteins (4 atoms/residue) keep N manageable.

## Model Configuration

| Parameter | 1M Model | 5M Model | Description |
|-----------|----------|----------|-------------|
| h_dim | 128 | 256 | Hidden dimension |
| n_heads | 8 | 8 | Attention heads |
| n_layers | 6 | 6 | Transformer layers |
| n_timesteps | 50 | 50 | Diffusion steps |

## Input Embeddings

Each atom is embedded as the sum/concatenation of:

```python
h = cat([
    atom_type_embed(atom_types),     # [B, N, h_dim//4]  N/CA/C/O
    aa_embed(aa_seq),                # [B, N, h_dim]     amino acid
    chain_embed(chain_ids),          # [B, N, h_dim//4]  chain 0/1
    sinusoidal_pos(residue_idx),     # [B, N, h_dim]     position
    time_embed(t),                   # [B, N, h_dim]     timestep
    coord_proj(x_t),                 # [B, N, h_dim]     coordinates
])
h = input_proj(h)                    # [B, N, h_dim]
```

### Sinusoidal Position Encoding

Residue positions use standard sinusoidal encoding:

```python
def sinusoidal_pos_enc(positions, dim):
    half_dim = dim // 2
    emb = log(10000) / (half_dim - 1)
    emb = exp(arange(half_dim) * -emb)
    emb = positions.unsqueeze(-1) * emb
    return cat([sin(emb), cos(emb)], dim=-1)
```

## Diffusion Process

### Forward Process (Training)

1. Sample random timestep t ~ Uniform(0, T)
2. Add noise: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
3. Predict clean coordinates: x0_pred = model(x_t, features, t)
4. Loss: Kabsch-aligned MSE between x0_pred and x0

### Reverse Process (Sampling)

```python
x = randn(B, N, 3)  # Start from noise
for t in reversed(range(T)):
    x0_pred = model(x, features, t)
    x = ddpm_step(x, x0_pred, t)  # Move toward x0_pred
return x
```

### Cosine Schedule

We use a cosine noise schedule with T=50 steps:

```python
f_t = cos((t/T + s) / (1 + s) * pi/2)^2
alpha_bar = f_t / f_t[0]
```

## Training

### Loss Function

Kabsch-aligned MSE loss:
1. Optimally align predicted and target coordinates (removes rotation/translation)
2. Compute MSE on aligned coordinates

```python
pred_aligned, target_centered = kabsch_align(x0_pred, x0, mask)
loss = ((pred_aligned - target_centered)**2).sum(-1)
loss = (loss * mask).sum() / mask.sum()
```

### Data Normalization

Per-sample coordinate normalization:
1. Center coordinates (subtract centroid)
2. Divide by standard deviation
3. Undo normalization for final RMSE calculation

## Key Dimensions

| Symbol | Description |
|--------|-------------|
| B | Batch size |
| N | Total atoms (L * 4 for L residues) |
| L | Total residues (LA + LB) |
| h_dim | Hidden dimension (128 or 256) |
| T | Diffusion timesteps (50) |

## Module Organization

```
scripts/
├── prepare_data.py    # Data preprocessing
├── train.py           # Main training script
└── visualize_preds.py # Visualization

src/tinyfold/
├── constants.py       # Amino acids, atom types
└── data/              # Data loading
    ├── cache.py       # Parquet serialization
    └── ...
```

## Limitations and Future Work

### Current Limitations

1. **No sequence encoding**: The model processes atoms independently without explicit pairwise residue features. This limits sample efficiency.

2. **Fixed atom count**: 4 atoms per residue (backbone only), no sidechain support.

3. **No equivariance**: Must learn rotation invariance from data augmentation (implicit in random initialization).

### Planned Improvements

1. **Better position encoding**: The current sinusoidal encoding doesn't capture residue identity strongly enough. Consider:
   - Relative position encodings between atoms
   - Separate encodings for within-chain vs cross-chain positions
   - Learnable position embeddings

2. **Pairwise features**: Add residue-residue distance/contact predictions as auxiliary task.

3. **Sequence pre-encoding**: Process sequence with a separate encoder before conditioning the diffusion model.
