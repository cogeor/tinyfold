# Decoder Redesign Plan

## Overview

This document outlines two major improvements to the TinyFold decoder:
1. **Hierarchical Decoder**: Predict residue positions first, then decode to atoms
2. **Linear Chain Noise**: Replace Gaussian noise with diffusion from extended chain

## Background

### Current Architecture

The current `AttentionDiffusionV2` predicts atom coordinates directly:
- Input: noisy coords `x_t [B, N_atoms, 3]`, features
- Output: predicted clean coords `x0_pred [B, N_atoms, 3]`
- Each atom is an independent token in the transformer

### Data Properties

- **Atoms per residue**: Always exactly 4 (N, CA, C, O backbone atoms)
- **Atom ordering**: Fixed within residue (N=0, CA=1, C=2, O=3)
- **Bond lengths**: N-CA: 1.46Å, CA-C: 1.52Å, C-O: 1.23Å, peptide C-N: 1.33Å
- **CA-CA distance**: ~3.8Å in extended chain

---

## Part 1: Hierarchical Decoder

### Concept

Instead of predicting all atom coordinates independently, decompose the prediction:

```
                    Hierarchical Decoder
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Step 1: Predict residue-level representations              │
│  ─────────────────────────────────────────────              │
│  Input: sequence features, noisy CA positions               │
│  Output: residue positions (CA) + local frames              │
│                                                             │
│  Step 2: Decode atoms from residue frames                   │
│  ─────────────────────────────────────────────              │
│  Input: residue frames, atom type                           │
│  Output: atom 3D positions                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why This Helps Sample Efficiency

1. **Fewer tokens**: L residues instead of 4L atoms → 4x smaller attention matrix
2. **Structural prior**: Backbone geometry is fixed, only learn residue placement
3. **Local frames**: Atoms within residue have fixed relative positions (up to local rotation)

### Architecture Design

#### Option A: Two-Stage Sequential

```python
class HierarchicalDecoder(nn.Module):
    def __init__(self, h_dim, n_layers_res, n_layers_atom):
        # Stage 1: Residue transformer
        self.residue_encoder = TransformerEncoder(
            h_dim, n_layers=n_layers_res, n_heads=8
        )

        # Stage 2: Atom decoder (small, per-residue)
        self.atom_decoder = nn.Sequential(
            nn.Linear(h_dim + 4, h_dim),  # h_dim features + 4 atom types one-hot
            nn.GELU(),
            nn.Linear(h_dim, 3 * 4)  # 4 atoms × 3 coords
        )

    def forward(self, x_t, features, t):
        B, N_atoms, _ = x_t.shape
        N_res = N_atoms // 4

        # Extract CA positions (atom_type=1)
        x_ca = x_t[:, 1::4, :]  # [B, N_res, 3]

        # Stage 1: Process at residue level
        res_features = self.residue_encoder(x_ca, features, t)  # [B, N_res, h_dim]

        # Stage 2: Decode to atoms
        atom_offsets = self.atom_decoder(res_features)  # [B, N_res, 12]
        atom_offsets = atom_offsets.view(B, N_res, 4, 3)  # [B, N_res, 4, 3]

        # Add offsets to CA position
        x0_pred = x_ca.unsqueeze(2) + atom_offsets  # [B, N_res, 4, 3]
        x0_pred = x0_pred.view(B, N_atoms, 3)

        return x0_pred
```

#### Option B: Frame-Based (AlphaFold-style)

```python
class FrameDecoder(nn.Module):
    """Predict local coordinate frames for each residue, then place atoms."""

    def __init__(self, h_dim):
        self.residue_transformer = TransformerEncoder(h_dim, n_layers=6)

        # Predict frame: translation (3) + rotation (quaternion 4 or 6D)
        self.frame_head = nn.Linear(h_dim, 3 + 6)  # translation + 6D rotation

        # Fixed atom positions in local frame (learnable)
        self.local_atom_coords = nn.Parameter(torch.tensor([
            [-1.46, 0.0, 0.0],   # N relative to CA
            [0.0, 0.0, 0.0],     # CA (origin)
            [1.52, 0.0, 0.0],    # C relative to CA
            [2.4, 1.2, 0.0],     # O relative to CA (approximate)
        ]))

    def forward(self, x_t_ca, features, t):
        B, N_res, _ = x_t_ca.shape

        # Get residue representations
        h = self.residue_transformer(x_t_ca, features, t)  # [B, N_res, h_dim]

        # Predict frames
        frames = self.frame_head(h)  # [B, N_res, 9]
        trans = frames[:, :, :3]      # [B, N_res, 3]
        rot_6d = frames[:, :, 3:]     # [B, N_res, 6]
        rot_mat = rotation_6d_to_matrix(rot_6d)  # [B, N_res, 3, 3]

        # Apply frames to local atom coords
        local = self.local_atom_coords  # [4, 3]
        rotated = torch.einsum('bnij,aj->bnai', rot_mat, local)  # [B, N_res, 4, 3]
        atoms = rotated + trans.unsqueeze(2)  # [B, N_res, 4, 3]

        return atoms.view(B, N_res * 4, 3)
```

### Recommendation

**Start with Option A (Two-Stage Sequential)** because:
- Simpler to implement and debug
- Doesn't require rotation representations
- Can add frame-based approach later if needed

### Changes Required

1. **Data loading**: Extract CA positions separately
2. **Model**: New `HierarchicalDecoder` class
3. **Training loop**: Minimal changes (same loss function)
4. **Sampling**: Extract CA from noise, run decoder

---

## Part 2: Linear Chain Noise (Extended → Folded Diffusion)

### Concept

Instead of:
- **Standard diffusion**: Clean structure ↔ Gaussian noise

Use:
- **Linear chain diffusion**: Folded structure ↔ Extended chain

```
t=0 (clean)                    t=T (fully noised)
┌─────────────────┐            ┌─────────────────────────────────┐
│    Folded       │            │         Extended Chain          │
│   Structure     │   ──────►  │  ● ─ ● ─ ● ─ ● ─ ● ─ ● ─ ●    │
│   (compact)     │            │  (straight line, random orient) │
└─────────────────┘            └─────────────────────────────────┘
```

### Why This Helps

1. **Physical intuition**: Mimics protein folding from unstructured to folded
2. **Simpler noise structure**: Linear chain is more structured than Gaussian
3. **Better conditioning**: Model learns folding trajectories, not arbitrary noise removal
4. **Evaluation**: Start from extended chain, watch it fold

### Mathematical Formulation

#### Standard Diffusion

```
x_t = √(ᾱ_t) × x_0 + √(1-ᾱ_t) × ε,  where ε ~ N(0, I)
```

#### Linear Chain Diffusion

```
x_t = √(ᾱ_t) × x_0 + √(1-ᾱ_t) × x_linear
```

Where `x_linear` is the extended chain configuration:
- CA atoms placed at `(i × 3.8, 0, 0)` for residue i
- Backbone atoms offset from CA using ideal geometry
- Random 3D rotation applied for orientation invariance

### Extended Chain Generation

```python
def generate_extended_chain(n_residues, n_atoms, atom_to_res, atom_type, device):
    """Generate extended chain coordinates."""
    CA_CA_DIST = 3.8  # Angstroms

    # Ideal atom offsets from CA in local frame
    ATOM_OFFSETS = torch.tensor([
        [-1.46, 0.0, 0.0],   # N
        [0.0, 0.0, 0.0],     # CA
        [1.52, 0.0, 0.0],    # C
        [2.4, 1.0, 0.0],     # O
    ], device=device)

    # Place CA atoms along x-axis
    ca_positions = torch.zeros(n_residues, 3, device=device)
    ca_positions[:, 0] = torch.arange(n_residues, device=device) * CA_CA_DIST

    # Center the chain
    ca_positions = ca_positions - ca_positions.mean(dim=0, keepdim=True)

    # Build all atom positions
    x_linear = torch.zeros(n_atoms, 3, device=device)
    for i in range(n_atoms):
        res_idx = atom_to_res[i]
        atom_idx = atom_type[i]
        x_linear[i] = ca_positions[res_idx] + ATOM_OFFSETS[atom_idx]

    # Apply random rotation for orientation invariance
    R = random_rotation_matrix(device)
    x_linear = x_linear @ R.T

    return x_linear
```

### Modified Diffusion Process

```python
class LinearChainSchedule:
    """Diffusion schedule that interpolates between folded and extended."""

    def __init__(self, T=50, s=0.008):
        self.T = T
        # Same alpha_bar as cosine schedule
        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        self.alpha_bar = f_t / f_t[0]
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def add_noise(self, x0, t, x_linear):
        """Interpolate between x0 (folded) and x_linear (extended)."""
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        # Key difference: use x_linear instead of random noise
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * x_linear

        return x_t, x_linear  # Return x_linear as "noise" for loss computation
```

### Training Changes

```python
def training_step(model, batch, schedule):
    x0 = batch['coords']  # Folded structure

    # Generate extended chain for this batch
    x_linear = generate_extended_chain(
        batch['n_residues'], batch['n_atoms'],
        batch['atom_to_res'], batch['atom_type'],
        device=x0.device
    )

    # Add noise (interpolate toward extended)
    t = torch.randint(0, schedule.T, (batch_size,))
    x_t, _ = schedule.add_noise(x0, t, x_linear)

    # Predict x0
    x0_pred = model(x_t, batch['features'], t)

    # Same loss as before
    loss = compute_loss(x0_pred, x0, batch['mask'])
    return loss
```

### Sampling Changes

```python
@torch.no_grad()
def sample_from_extended(model, features, schedule):
    """Start from extended chain and fold."""
    # Generate starting extended chain
    x = generate_extended_chain(n_residues, n_atoms, atom_to_res, atom_type, device)
    x = x.unsqueeze(0)  # Add batch dim

    # Denoise (fold) from T-1 to 0
    for t in reversed(range(schedule.T)):
        t_batch = torch.full((1,), t, device=device)
        x0_pred = model(x, features, t_batch)

        if t > 0:
            # Generate fresh extended chain for interpolation
            x_linear = generate_extended_chain(...)
            x = ddpm_step_linear(x, x0_pred, x_linear, t, schedule)
        else:
            x = x0_pred

    return x
```

### Considerations

1. **Normalization**: Extended chain is much longer than folded → need to normalize differently
   - Option A: Normalize both by same std (folded protein std)
   - Option B: Normalize extended chain to unit std separately

2. **Two chains**: For PPI, generate two separate extended chains
   - Chain A: residues 0 to LA-1 along one line
   - Chain B: residues LA to L-1 along parallel line, offset in y

3. **Rotation invariance**: Apply random rotation to extended chain each time
   - Training: Different rotation each batch
   - Evaluation: Fixed rotation for reproducibility

---

## Implementation Plan

### Phase 1: Hierarchical Decoder (Work Package 1)

**Files to modify**: `scripts/train.py`

**Step 1.1**: Extract CA positions from data
```python
def extract_ca_positions(coords, atom_to_res):
    """Extract CA atom positions (atom_type=1, every 4th atom starting at 1)."""
    ca_coords = coords[:, 1::4, :]  # [B, L, 3]
    return ca_coords
```

**Step 1.2**: Add `HierarchicalDecoder` class
- Residue-level transformer (L tokens instead of 4L)
- Atom offset MLP decoder
- Input: noisy CA positions + sequence features
- Output: all 4 atom positions per residue

**Step 1.3**: Update training loop
- Extract CA from ground truth
- Run diffusion on CA positions only
- Decoder outputs full atom coords
- Loss on all atoms (or CA only for simplicity)

**Step 1.4**: Update sampling
- Start from noise at CA level
- Decode to full atoms at each step (or only at end)

**Step 1.5**: Test on 10-sample overfit

### Phase 2: Linear Chain Noise (Work Package 2)

**Files to modify**: `scripts/train.py`

**Step 2.1**: Add `generate_extended_chain_ca()` function
```python
def generate_extended_chain_ca(n_residues, chain_ids, device):
    """Generate extended chain CA positions (residue-level)."""
    CA_CA_DIST = 3.8

    # Separate chains
    chain_a_mask = chain_ids == 0
    chain_b_mask = chain_ids == 1
    LA = chain_a_mask.sum()
    LB = chain_b_mask.sum()

    ca_linear = torch.zeros(n_residues, 3, device=device)

    # Chain A along x-axis at y=0
    ca_linear[chain_a_mask, 0] = torch.arange(LA, device=device) * CA_CA_DIST

    # Chain B parallel at y=offset
    CHAIN_OFFSET = 20.0  # Angstroms
    ca_linear[chain_b_mask, 0] = torch.arange(LB, device=device) * CA_CA_DIST
    ca_linear[chain_b_mask, 1] = CHAIN_OFFSET

    # Center around origin
    ca_linear = ca_linear - ca_linear.mean(dim=0, keepdim=True)

    return ca_linear
```

**Step 2.2**: Add `LinearChainSchedule` class
- Same alpha_bar as cosine schedule
- `add_noise()` interpolates toward extended + adds small Gaussian
- Track both folded and extended normalization stats

**Step 2.3**: Modify sampling to start from extended chain

**Step 2.4**: Add folding trajectory visualization

**Step 2.5**: Test on 10-sample overfit

### Phase 3: Combine Both

**After validating each separately**:
1. Hierarchical decoder (residue-level) + linear chain noise (residue-level)
2. Both operate at CA level → natural fit
3. Evaluate synergy on sample efficiency

---

## Design Decisions

### 1. Normalization Strategy

**Decision**: Normalize both folded and extended to same scale, unnormalize at output.

```python
# During training:
x0_norm = (x0 - centroid) / std          # Normalize folded
x_linear_norm = (x_linear - centroid_linear) / std_linear  # Normalize extended

# Interpolate in normalized space
x_t = √ᾱ × x0_norm + √(1-ᾱ) × x_linear_norm

# At output, scale back to Ångströms
x0_pred_angstrom = x0_pred * std + centroid
```

**Alternatives considered**:
- B) Train in raw Ångströms (no normalization) - may have gradient issues
- C) Normalize extended to match folded std - loses 3.8Å physical meaning

### 2. Multi-chain Geometry

**Decision**: Two parallel lines centered around origin, aligned along X-axis.

```python
# Chain A: along x-axis at y=0
ca_A = [(i * 3.8, 0, 0) for i in range(LA)]

# Chain B: parallel line at y=offset
ca_B = [(i * 3.8, offset, 0) for i in range(LB)]

# Center both around origin
```

**Future**: Add augmentations (random separation, random orientation).

### 3. Decoder Architecture

**Decision**: Option A (two-stage sequential) - no rotation math needed.

### 4. Hybrid Noise

**Decision**: Interpolate toward extended chain, then add small Gaussian noise for stochasticity.

```python
# Step 1: Interpolate toward extended
x_interp = √ᾱ × x0 + √(1-ᾱ) × x_linear

# Step 2: Add small Gaussian noise
x_t = x_interp + σ_small × ε
```

This gives deterministic interpolation path with stochastic perturbation.

### 5. Hierarchical Decoder + Linear Chain Interaction

**Decision**: Define linear chain at **residue level (CA positions only)**. Atom positions are decoded from residue representations.

```python
# Linear chain defined for CA only
ca_linear = [(i * 3.8, 0, 0) for i in range(L)]  # [L, 3]

# Diffusion operates on CA positions
ca_t = √ᾱ × ca_0 + √(1-ᾱ) × ca_linear

# Hierarchical decoder:
# 1. Transformer processes ca_t → residue features
# 2. MLP decodes 4 atom offsets from each residue
# 3. Final atoms = predicted_CA + atom_offsets
```

This keeps diffusion simple (residue-level) while decoder handles atom geometry
---

## Success Metrics

1. **Sample efficiency**: Achieve same RMSE with fewer training samples
2. **Convergence speed**: Reach target RMSE in fewer steps
3. **Physical plausibility**: Bond lengths and angles within tolerance
4. **Folding visualization**: Smooth trajectory from extended to folded
