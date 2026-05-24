# IterFold Architecture Plan

## Overview

Alternative to diffusion-based ResFold that directly predicts atom positions via iterative masked prediction. Instead of denoising random noise, we condition on known residue positions (anchors) and predict unknown ones.

## Architecture Summary

```
Sequence (aa, chain, res_idx)
    │
    ▼
┌─────────────────────────┐
│   Trunk Encoder         │  (same as Stage 1)
│   [B, L, c_token]       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│   Anchor-Conditioned Atom Decoder                            │
│                                                              │
│   Input: anchor_pos [B, L, 3]  (residue positions)          │
│   - Known positions: ground truth centroid values            │
│   - Unknown positions: zeros (inferred from tensor)          │
│                                                              │
│   Output: [B, L, 4, 3] atom coordinates                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Trunk Encoder (Reuse Existing)

Reuse `ResidueEncoder` from `scripts/models/resfold.py`:
- Input: `aa_seq`, `chain_ids`, `res_idx`
- Output: `trunk_tokens [B, L, c_token]`
- No changes needed

### 2. Anchor-Conditioned Atom Decoder (New)

```python
class AnchorDecoder(nn.Module):
    """
    Decodes atom positions conditioned on known anchor positions.

    Anchors are residue centroids: non-zero = known, zero = unknown.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        n_atom_layers: int = 8,
    ):
        # Position embedding: [B, L, 3] -> [B, L, c_token]
        self.pos_embed = nn.Linear(3, c_token)

        # Learnable embedding for unknown (zero) positions
        self.unknown_embed = nn.Parameter(torch.randn(1, 1, c_token))

        # Main transformer (sequence-level, processes residues)
        self.transformer = nn.TransformerEncoder(...)

        # Atom query embeddings [1, 1, 4, c_token] for N, CA, C, O
        self.atom_type_embed = nn.Parameter(torch.randn(1, 1, 4, c_token))

        # Atom refinement layers (cross-attention to residues)
        self.atom_layers = nn.ModuleList([
            AtomRefinementLayer(c_token, n_heads)
            for _ in range(n_atom_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        trunk_tokens: Tensor,  # [B, L, c_token]
        anchor_pos: Tensor,    # [B, L, 3] residue positions (0 = unknown)
        mask: Tensor,          # [B, L] bool, True = valid residue
    ) -> Tensor:
        """
        Returns atom coordinates [B, L, 4, 3]
        """
        B, L, _ = trunk_tokens.shape

        # Infer known mask from anchor positions (non-zero = known)
        is_known = (anchor_pos.abs().sum(dim=-1) > 1e-6)  # [B, L]

        # Embed anchor positions
        pos_feat = self.pos_embed(anchor_pos)  # [B, L, c_token]

        # Replace unknown positions with learnable embedding
        pos_feat = torch.where(
            is_known.unsqueeze(-1),
            pos_feat,
            self.unknown_embed.expand(B, L, -1)
        )

        # Combine with trunk tokens
        h = trunk_tokens + pos_feat  # [B, L, c_token]

        # Transformer processing
        h = self.transformer(h, src_key_padding_mask=~mask)  # [B, L, c_token]

        # Create atom queries
        atom_queries = h.unsqueeze(2) + self.atom_type_embed  # [B, L, 4, c_token]
        atom_queries = atom_queries.view(B, L*4, -1)

        # Refine with cross-attention
        for layer in self.atom_layers:
            atom_queries = layer(atom_queries, h, mask)

        # Project to coordinates
        atom_offsets = self.out_proj(atom_queries.view(B, L, 4, -1))  # [B, L, 4, 3]

        # RESIDUAL CONNECTION: Add anchor position broadcasted to 4 atoms
        atom_coords = atom_offsets + anchor_pos.unsqueeze(2)  # [B, L, 4, 3]

        return atom_coords
```

### 3. Training Strategy

#### Data Augmentation

**Rotation augmentation (DISABLED by default for IterFold):**

Unlike diffusion models, IterFold should NOT use rotation augmentation because:
- Non-anchored residues have `anchor_pos = 0`, losing all orientation info
- The target atoms are rotated: `R @ atoms`
- Model receives SAME input (zeros) but must produce DIFFERENT outputs
- This is impossible without inferring R from anchored residues

For diffusion models, rotation augmentation works because noisy `x_t` carries
orientation info for ALL residues. For IterFold, only anchored residues have
position info in `anchor_pos`.

```python
# Default: rotation_augment = False for IterFold
```

#### Masked Iterative Training

For each training batch:

```python
def train_step(batch):
    # Get ground truth
    gt_atoms = batch['atom_coords']      # [B, L, 4, 3]
    gt_centroids = gt_atoms.mean(dim=2)  # [B, L, 3] residue centroid

    # 0. Rotation augmentation (default: enabled)
    if augment_rotation:
        gt_atoms, gt_centroids = apply_rotation_augment(gt_atoms, gt_centroids)

    # 1. Encode sequence (run trunk once)
    trunk_tokens = trunk_encoder(aa_seq, chain_ids, res_idx, mask)

    # 2. Sample anchor mask (tracked internally, NOT fed to network)
    anchor_mask = sample_anchor_mask(L, ratio_range=(0.1, 0.3))  # [B, L]

    # 3. Build anchor_pos: GT for anchored, zeros for unknown
    # Network infers known/unknown from zeros - we don't pass anchor_mask
    anchor_pos = gt_centroids * anchor_mask.unsqueeze(-1).float()  # [B, L, 3]

    # 4. Predict atoms
    pred_atoms = decoder(trunk_tokens, anchor_pos, mask)

    # 5. Select K next residues via clustering (closest to anchored)
    next_mask = select_next_k_residues(pred_atoms, anchor_mask, k=8)

    # 6. Compute loss ONLY on anchored + next K residues
    loss_anchor = mse_loss(pred_atoms[anchor_mask], gt_atoms[anchor_mask])
    loss_next = mse_loss(pred_atoms[next_mask], gt_atoms[next_mask])

    # Other atoms (unknown and not in next K) don't count towards loss
    total_loss = loss_anchor + loss_next
    return total_loss
```

#### Anchor Sampling Strategies

```python
def sample_anchor_mask(L: int, strategy: str = "random") -> Tensor:
    """Sample which residues have known anchor positions."""

    if strategy == "random":
        # Random 10-30% anchored
        ratio = random.uniform(0.1, 0.3)
        return torch.rand(L) < ratio

    elif strategy == "terminal":
        # N-terminal or C-terminal anchored
        anchored = torch.zeros(L, dtype=torch.bool)
        anchored[:3] = True  # First 3 residues
        return anchored

    elif strategy == "central":
        # Central region anchored
        anchored = torch.zeros(L, dtype=torch.bool)
        center = L // 2
        anchored[center-2:center+3] = True
        return anchored

    elif strategy == "interface":
        # Interface residues anchored (from batch data)
        return batch['interface_mask']
```

### 4. Inference: Iterative Assembly

Uses clustering to select next residues to anchor (reuse logic from `scripts/models/clustering.py`):

```python
@torch.no_grad()
def inference(model, batch, n_iter: int = 10, K_per_iter: int = None):
    """
    Iteratively assemble structure starting from seed anchor.
    """
    # Encode sequence
    trunk_tokens = model.trunk(aa_seq, chain_ids, res_idx, mask)

    L = trunk_tokens.shape[1]
    K_per_iter = K_per_iter or (L // n_iter)

    # Initialize anchor_pos: all zeros (nothing known yet)
    anchor_pos = torch.zeros(1, L, 3, device=device)

    # Seed with most central residue (or terminal, or interface residue)
    # For true inference, use a predicted or heuristic initial position
    center_idx = L // 2
    anchor_pos[0, center_idx] = initial_estimate[0, center_idx]

    for i in range(n_iter):
        # Predict all atoms conditioned on current anchors
        pred_atoms = model.decoder(trunk_tokens, anchor_pos, mask)

        # Select next K residues to anchor (closest to current anchors)
        is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)  # [B, L]
        next_res = select_next_residues(pred_atoms, is_anchored, K_per_iter)

        # Update anchor_pos with predicted centroids for selected residues
        pred_centroids = pred_atoms.mean(dim=2)  # [B, L, 3]
        anchor_pos[0, next_res] = pred_centroids[0, next_res]

    # Final prediction with all positions anchored
    final_atoms = model.decoder(trunk_tokens, anchor_pos, mask)
    return final_atoms
```

## Loss Functions

### 1. Anchor Loss (MSE on anchored atoms)

```python
def anchor_loss(pred, gt, anchor_mask):
    """MSE on atoms of anchored residues."""
    anchor_atom_mask = anchor_mask.unsqueeze(-1).expand(-1, -1, 4)  # [B, L, 4]
    diff_sq = (pred - gt).pow(2).sum(dim=-1)  # [B, L, 4]
    return (diff_sq * anchor_atom_mask).sum() / anchor_atom_mask.sum()
```

### 2. Next Loss (MSE on K next atoms)

```python
def next_loss(pred, gt, next_mask):
    """MSE on atoms of K next residues (selected by clustering)."""
    next_atom_mask = next_mask.unsqueeze(-1).expand(-1, -1, 4)  # [B, L, 4]
    diff_sq = (pred - gt).pow(2).sum(dim=-1)  # [B, L, 4]
    return (diff_sq * next_atom_mask).sum() / next_atom_mask.sum()
```

### Total Loss

```python
total_loss = anchor_loss + next_loss
```

**Note**: No geometry loss for simplicity. Other atoms (unknown and not in next K) don't count towards loss.

## Model Configuration

```python
@dataclass
class IterFoldConfig:
    # Trunk (reuse)
    c_token: int = 256
    n_trunk_layers: int = 9
    n_heads: int = 8

    # Decoder
    n_decoder_layers: int = 12
    n_atom_layers: int = 8

    # Training
    anchor_ratio_range: Tuple[float, float] = (0.1, 0.3)
    k_next: int = 8  # Next residues to include in loss (clustering-based)
    rotation_augment: bool = True  # SO(3) augmentation (default: on)

    # Inference
    n_inference_iters: int = 10
```

## File Structure

```
scripts/
├── models/
│   ├── resfold.py              # Existing (trunk encoder)
│   ├── resfold_assembler.py    # Existing Stage 2
│   └── iterfold.py             # NEW: Anchor-conditioned decoder
├── train_iterfold.py           # NEW: Training script
```

## Implementation Steps

### Phase 1: Core Model (DONE)

1. **Create `iterfold.py`** - `scripts/models/iterfold.py`
   - [x] `AnchorDecoder` class
   - [x] Position embedding layer
   - [x] Unknown position learnable embedding (for zero positions)
   - [x] Atom query generation (reuse pattern from assembler)
   - [x] Cross-attention atom refinement layers
   - [x] Residual connection (anchor_pos broadcasted 4x)

2. **Wrapper model**
   - [x] `IterFold` combining trunk + decoder
   - [x] Forward pass (infer anchor mask from zeros in anchor_pos)
   - [x] Inference method with iterative assembly (`sample_iterative`)

### Phase 2: Training (DONE)

3. **Create `train_iterfold.py`** - `scripts/train_iterfold.py`
   - [x] Data loading (reuse existing)
   - [x] Rotation augmentation (default: enabled via `--rotation_augment`)
   - [x] Anchor sampling (random, terminal, central strategies)
   - [x] Simple loss: position_loss + geometry_loss on all atoms
   - [x] DockQ evaluation during validation
   - [x] Training loop with logging

### Phase 3: Evaluation (DONE - included in training script)

4. **Inference in training script**
   - [x] Iterative assembly loop (`sample_iterative`)
   - [x] Final evaluation with `evaluate_iterative()`
   - [x] Metrics: RMSD, DockQ (Fnat, iRMS, LRMS), quality distribution

## Evaluation Metrics

### DockQ (Primary Metric for PPI)

DockQ is the standard metric for protein-protein docking quality. Computed on predicted vs ground truth complex:

```python
def compute_dockq(pred_atoms, gt_atoms, chain_ids, mask):
    """
    Compute DockQ score for binary complex.

    Components:
    - Fnat: Fraction of native contacts preserved
    - LRMS: Ligand RMSD (smaller chain after receptor alignment)
    - iRMS: Interface RMSD

    Returns: DockQ in [0, 1], higher is better
    """
    # Split chains
    chain_a_mask = chain_ids == 0
    chain_b_mask = chain_ids == 1

    # Align on receptor (larger chain), compute ligand RMSD
    # Compute interface contacts and iRMS
    # Combine into DockQ score
    ...
```

**DockQ thresholds:**
- Incorrect: < 0.23
- Acceptable: 0.23 - 0.49
- Medium: 0.49 - 0.80
- High: >= 0.80

### Other Metrics

- **RMSD**: Overall backbone RMSD after alignment
- **GDT-TS**: Global Distance Test (robustness to outliers)
- **Bond geometry**: Bond lengths, angles, omega dihedrals

## Comparison with Diffusion Approach

| Aspect | Diffusion (ResFold) | IterFold (Proposed) |
|--------|---------------------|-------------------------|
| Training signal | Denoise at varying sigma | Predict from partial anchors |
| Conditioning | Noise level (sigma) | anchor_pos (0 = unknown) |
| Inference | 50 denoising steps | ~10 iterative assembly steps |
| Samples | K=5 diverse samples | Deterministic (or with dropout) |
| Residual | None (predicts absolute) | Add anchor_pos broadcasted 4x |
| Speed | Slower (many forward passes) | Faster (fewer iterations) |

## Design Rationale

### Why residual connection?

The residual connection `atom_coords = offsets + anchor_pos.unsqueeze(2)` serves two purposes:

1. **Anchor to known structure**: For anchored residues, anchor_pos provides the approximate atom location (centroid), and the model only needs to predict small offsets for each backbone atom relative to this.

2. **Gradient flow**: Direct path for gradients to flow from loss to input, similar to ResNet architecture benefits.

3. **Natural curriculum**: Anchored atoms automatically have lower loss (residual helps), so model focuses learning on the harder unknown atoms.

### Why simple all-atom loss?

Single MSE loss on all atoms with residual connection naturally creates curriculum:
- Anchored residues: low loss (offset from GT centroid is small)
- Unknown residues: higher loss (must predict from scratch)

No need for separate loss terms - the architecture handles the weighting implicitly.

### Why iterative inference?

1. **Reduces drift**: Each prediction conditions on previously-placed atoms, preventing error accumulation.

2. **Physical intuition**: Protein folding is hierarchical - global topology before local details.

3. **Matches training**: Model sees partial structures during training, same as inference.
