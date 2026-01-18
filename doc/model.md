# Model Architecture

## Overview

TinyFold uses `AttentionDiffusionV2`, a Transformer-based diffusion model for protein structure prediction. The model predicts clean coordinates (x0) from noisy coordinates (x_t) conditioned on sequence and structural features.

```
                         AttentionDiffusionV2
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Embeddings:                    Transformer:        Output:      │
│  ────────────                   ────────────        ───────      │
│  atom_type ──┐                                                   │
│  aa_seq ─────┤                  ┌────────────┐                   │
│  chain_id ───┼──► concat ──────►│ Encoder    │──► LayerNorm ──►  │
│  res_pos ────┤   (input_proj)   │ (6 layers) │    Linear(3)     │
│  timestep ───┤                  └────────────┘         │         │
│  coords ─────┘                                         ▼         │
│                                                    x0_pred       │
└──────────────────────────────────────────────────────────────────┘
```

## AttentionDiffusionV2

### Constructor

```python
class AttentionDiffusionV2(nn.Module):
    def __init__(
        self,
        h_dim: int = 128,      # Hidden dimension
        n_heads: int = 8,       # Attention heads
        n_layers: int = 6,      # Transformer layers
        n_timesteps: int = 50,  # Diffusion steps
        dropout: float = 0.0,
        n_aa_types: int = 21,   # 20 amino acids + unknown
        n_chains: int = 2,      # Chain A and B
    ):
```

### Embeddings

The model uses several embedding layers to encode input features:

```python
# Atom type: N, CA, C, O (4 types)
self.atom_type_embed = nn.Embedding(4, h_dim // 4)

# Amino acid sequence (21 types including unknown)
self.aa_embed = nn.Embedding(21, h_dim)

# Chain identity (0 or 1)
self.chain_embed = nn.Embedding(2, h_dim // 4)

# Timestep (learnable, not sinusoidal)
self.time_embed = nn.Embedding(n_timesteps, h_dim)

# Coordinate projection
self.coord_proj = nn.Linear(3, h_dim)
```

### Position Encoding

Residue positions use sinusoidal encoding (not learnable):

```python
def sinusoidal_pos_enc(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
    emb = positions.float().unsqueeze(-1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
```

**Current limitation**: The position encoding only captures absolute residue index (atom_to_res), not atom position within the residue or chain-relative position. This is a key area for improvement.

### Input Projection

All embeddings are concatenated and projected to h_dim:

```python
# Concatenate all embeddings
# Total input dim = h_dim//4 + h_dim + h_dim//4 + h_dim + h_dim + h_dim
#                 = 4.5 * h_dim
input_dim = (h_dim // 4) + h_dim + (h_dim // 4) + h_dim + h_dim + h_dim

self.input_proj = nn.Linear(input_dim, h_dim)

# In forward:
h = torch.cat([atom_emb, aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
h = self.input_proj(h)
```

### Transformer Encoder

Standard PyTorch TransformerEncoder with pre-norm:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=h_dim,
    nhead=n_heads,
    dim_feedforward=h_dim * 4,  # 4x expansion in FFN
    dropout=dropout,
    batch_first=True,
    norm_first=True,  # Pre-norm (more stable)
)
self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=n_layers,
    enable_nested_tensor=False  # Required for variable-length masking
)
```

### Output

```python
self.output_norm = nn.LayerNorm(h_dim)
self.output_proj = nn.Linear(h_dim, 3)

# In forward:
h = self.output_norm(h)
return self.output_proj(h)  # [B, N, 3] predicted coordinates
```

### Forward Pass

```python
def forward(self, x_t, atom_types, atom_to_res, aa_seq, chain_ids, t, mask=None):
    B, N, _ = x_t.shape

    # Compute embeddings
    atom_emb = self.atom_type_embed(atom_types)     # [B, N, h_dim//4]
    aa_emb = self.aa_embed(aa_seq)                  # [B, N, h_dim]
    chain_emb = self.chain_embed(chain_ids)         # [B, N, h_dim//4]
    res_emb = self.sinusoidal_pos_enc(atom_to_res, self.h_dim)  # [B, N, h_dim]
    time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)  # [B, N, h_dim]
    coord_emb = self.coord_proj(x_t)                # [B, N, h_dim]

    # Concatenate and project
    h = torch.cat([atom_emb, aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
    h = self.input_proj(h)  # [B, N, h_dim]

    # Apply attention mask for padding
    attn_mask = ~mask if mask is not None else None
    h = self.transformer(h, src_key_padding_mask=attn_mask)

    # Output
    h = self.output_norm(h)
    return self.output_proj(h)  # [B, N, 3]
```

## Diffusion Components

### CosineSchedule

```python
class CosineSchedule:
    def __init__(self, T: int = 50, s: float = 0.008):
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1]])
        self.betas = 1 - self.alphas
```

### Adding Noise

```python
def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
    noise = torch.randn_like(x0)
    sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
    sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
    x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
    return x_t, noise
```

### DDPM Sampling

```python
@torch.no_grad()
def ddpm_sample(model, atom_types, atom_to_res, aa_seq, chain_ids, schedule, mask=None):
    device = atom_types.device
    B, N = atom_types.shape
    x = torch.randn(B, N, 3, device=device)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # Prevent divergence

        if t > 0:
            # DDPM update step
            ab_t = schedule.alpha_bar[t]
            ab_prev = schedule.alpha_bar[t - 1]
            beta = schedule.betas[t]
            alpha = schedule.alphas[t]

            coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
            coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
            mean = coef1 * x0_pred + coef2 * x

            var = beta * (1 - ab_prev) / (1 - ab_t)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = x0_pred

    return x
```

## Loss Function

### Kabsch Alignment

Before computing loss, we optimally align predicted and target coordinates:

```python
def kabsch_align(pred, target, mask=None):
    B, N, _ = pred.shape

    # Compute masked centroids
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        pred_mean = (pred * mask_exp).sum(dim=1, keepdim=True) / n_valid
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

    # Center
    pred_c = pred - pred_mean
    target_c = target - target_mean

    if mask is not None:
        pred_c = pred_c * mask_exp
        target_c = target_c * mask_exp

    # SVD for optimal rotation
    H = torch.bmm(pred_c.transpose(1, 2), target_c)
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=pred.device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))
    pred_aligned = torch.bmm(pred_c, R.transpose(1, 2))

    return pred_aligned, target_c
```

### Loss Computation

```python
def compute_loss(pred, target, mask=None):
    pred_aligned, target_c = kabsch_align(pred, target, mask)
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)  # [B, N]

    if mask is not None:
        loss = (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        loss = sq_diff.mean()

    return loss
```

## Parameter Counts

For reference, parameter counts for different configurations:

| Configuration | h_dim | n_layers | Total Params |
|---------------|-------|----------|--------------|
| 1M Model | 128 | 6 | ~1.27M |
| 5M Model | 256 | 6 | ~6.63M |

Main parameter sources:
- TransformerEncoder: ~90% of parameters
- Embeddings: ~5%
- Input/output projections: ~5%

## Known Issues and Improvements

### Current Limitations

1. **Weak position encoding**: `atom_to_res` only captures which residue an atom belongs to, not:
   - Position within residue (N=0, CA=1, C=2, O=3)
   - Position within chain
   - Relative positions between atoms

2. **No pairwise features**: Unlike Pairformer, there's no explicit pairwise representation capturing residue-residue relationships.

3. **Atom type underutilized**: The atom type embedding is h_dim//4, but could carry more structural information.

### Proposed Improvements

1. **Enhanced position encoding**:
   ```python
   # Separate encodings for different position aspects
   res_pos_emb = self.E_res_pos(residue_idx)  # Position in sequence
   atom_pos_emb = self.E_atom_pos(atom_type)  # N/CA/C/O position
   chain_pos_emb = self.E_chain_pos(chain_residue_idx)  # Position within chain
   ```

2. **Relative position bias**: Add pairwise distance/position features to attention:
   ```python
   # In attention computation
   rel_pos = residue_idx.unsqueeze(1) - residue_idx.unsqueeze(0)
   attn_bias = self.rel_pos_embed(rel_pos)
   attn_logits = attn_logits + attn_bias
   ```

3. **Sequence pre-encoding**: Process sequence through a separate encoder before conditioning.
