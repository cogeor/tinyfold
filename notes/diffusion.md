# Diffusion Implementation Report - TinyFold

> **Purpose**: Thorough documentation of how diffusion is implemented for each model, including training logic, inference rollout, loss weighting, and noise schedules.

---

## Table of Contents

1. [Overview](#overview)
2. [Noise Schedules](#noise-schedules)
3. [Noise Types](#noise-types)
4. [Model-Specific Implementations](#model-specific-implementations)
   - [AttentionV2](#attentionv2)
   - [AF3StyleDecoder](#af3styledecoder)
   - [ResFold (Stage 1)](#resfold-stage-1)
5. [Training Logic](#training-logic)
6. [Inference/Sampling (Rollout)](#inferencesampling-rollout)
7. [Loss Formulations](#loss-formulations)
8. [Comparison Table](#comparison-table)

---

## Overview

TinyFold implements **x0-prediction** diffusion (also called direct prediction), where the model predicts the clean coordinates `x0` from noisy input `x_t` at each timestep. This differs from epsilon-prediction which predicts the noise added.

**Key Files:**
- `scripts/models/diffusion.py` - Schedules, noisers, and utilities
- `scripts/models/af3_style.py` - AF3StyleDecoder model
- `scripts/models/attention_v2.py` - AttentionV2 model  
- `scripts/models/resfold.py` - ResidueDenoiser (Stage 1)
- `scripts/models/resfold_pipeline.py` - Full ResFold pipeline
- `scripts/train.py` - Training for AF3/AttentionV2
- `scripts/train_resfold.py` - Training for ResFold

---

## Noise Schedules

Located in `scripts/models/diffusion.py` (lines 22-66).

### CosineSchedule (Nichol & Dhariwal 2021)

```python
class CosineSchedule:
    def __init__(self, T: int = 50, s: float = 0.008):
        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
```

**Properties computed:**
- `alpha_bar[t]`: Cumulative product of alphas at timestep t
- `sqrt_alpha_bar[t]`: For forward noising
- `sqrt_one_minus_alpha_bar[t]`: For noise magnitude
- `alphas[t]`: Step-wise alpha (derived from alpha_bar)
- `betas[t] = 1 - alphas[t]`

**Behavior:** Alpha_bar decays slowly at start and end, faster in middle. Better for preserving signal early and late in diffusion.

### LinearSchedule

```python
class LinearSchedule:
    def __init__(self, T: int = 50):
        t = torch.arange(T + 1, dtype=torch.float32)
        alpha_bar = 1 - t / T  # Direct linear interpolation
        alpha_bar = alpha_bar.clamp(min=1e-6)  # Avoid exactly 0
```

**Behavior:** Simple linear decay of alpha_bar from 1 to 0. Used as default in ResFold.

---

## Noise Types

### 1. GaussianNoise (Standard DDPM)

```python
class GaussianNoise:
    def add_noise(self, x0: Tensor, t: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(x0)
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        return x_t, noise  # Returns (noisy input, added noise)
```

**Forward process:** $x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$

where $\epsilon \sim \mathcal{N}(0, I)$

**Fully noised state:** Pure Gaussian noise (isotropic, no structure)

### 2. LinearChainNoise (Structure-preserving)

```python
class LinearChainNoise:
    def add_noise(self, x0, t, atom_to_res, atom_type, chain_ids, **kwargs):
        # Generate extended chain for each sample
        x_linear = generate_extended_chain(...)
        # Normalize extended chain to match x0 scale
        x_linear = (x_linear - x_linear.mean()) / x_linear.std()
        
        # Interpolate: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * x_linear
        x_interp = sqrt_ab * x0 + sqrt_one_minus_ab * x_linear
        
        # Add small Gaussian noise for stochasticity
        x_t = x_interp + noise_scale * randn_like(x0)
        return x_t, x_linear  # Returns (noisy input, extended chain)
```

**Forward process:** $x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot x_{\text{linear}} + \sigma \cdot \epsilon$

**Fully noised state:** Extended peptide chain (all residues in straight lines)

**Extended chain generation (`generate_extended_chain`):**
- Places CA atoms at regular intervals (3.8Å apart)
- Two chains offset by 20Å in Y direction
- Atom offsets from CA: N(-0.4,0,0), CA(0,0,0), C(+0.4,0,0), O(+0.6,+0.3,0)

**Reverse step:**
```python
def reverse_step(self, x_t, x0_pred, t, x_linear):
    if t == 0:
        return x0_pred
    # Interpolate back toward x0_pred
    x_prev = sqrt_ab_{t-1} * x0_pred + sqrt_(1-ab_{t-1}) * x_linear
    return x_prev + noise_scale * randn()
```

### 3. LinearChainFlow (Iterative refinement)

```python
class LinearChainFlow:
    # Training: input is x_{t-1}, target is x_0
    # Model always predicts x0 from current state
    def add_noise(self, x0, t, ...):
        x_t = alpha * x0 + (1-alpha) * x_linear + noise
        return x_t, x0  # Target is x0, not the intermediate
```

**Forward process:** Same as LinearChainNoise but different training interpretation

**Inference:** Iteratively refine from x_linear toward x0_pred, interpolating based on timestep.

---

## Model-Specific Implementations

### AttentionV2

**File:** `scripts/models/attention_v2.py` (114 lines)

**Architecture:** Simple all-atom attention transformer
- Input: Noisy atom coordinates [B, N_atoms, 3]
- Embeddings: atom_type, aa_type, chain_id, residue_pos, timestep, coordinate
- Single transformer encoder
- Output: Predicted clean coordinates [B, N_atoms, 3]

**Forward pass:**
```python
def forward(self, x_t, atom_types, atom_to_res, aa_seq, chain_ids, t, mask=None):
    # Compute embeddings
    atom_emb = self.atom_type_embed(atom_types)  # [B, N, h_dim//4]
    aa_emb = self.aa_embed(aa_seq)               # [B, N, h_dim]
    chain_emb = self.chain_embed(chain_ids)      # [B, N, h_dim//4]
    res_emb = sinusoidal_pos_enc(atom_to_res, h_dim)  # [B, N, h_dim]
    time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)  # [B, N, h_dim]
    coord_emb = self.coord_proj(x_t)             # [B, N, h_dim]
    
    # Concatenate and project
    h = cat([atom_emb, aa_emb, chain_emb, res_emb, time_emb, coord_emb])
    h = self.input_proj(h)
    
    # Standard transformer
    h = self.transformer(h, src_key_padding_mask=~mask)
    
    # Direct x0 prediction
    return self.output_proj(h)  # [B, N, 3]
```

**Output mode:** **Direct x0 prediction** - no residual scaling

---

### AF3StyleDecoder

**File:** `scripts/models/af3_style.py` (827 lines)

**Architecture:** AlphaFold3-inspired with trunk/denoiser separation
- **Trunk (runs ONCE):** ResidueEncoder - transforms residue features
- **Denoiser (runs EACH step):** 
  - AtomAttentionEncoder: atoms → tokens with local attention
  - DiffusionTransformer: global token attention with Adaptive LayerNorm
  - AtomAttentionDecoder: tokens → atom coordinate updates

**Trunk (ResidueEncoder):**
```python
def forward(self, atom_coords, aa_seq, chain_ids, res_idx, mask=None):
    # [B, L, 4, 3] -> [B, L, c_token]
    aa_emb = self.aa_embed(aa_seq)
    chain_emb = self.chain_embed(chain_ids)
    res_emb = sinusoidal_pos_enc(res_idx, c_token)
    atom_info = self.atom_info_proj(atom_coords.flatten(-2))  # [B, L, 12] -> [B, L, c_token//2]
    
    h = cat([aa_emb, chain_emb, res_emb, atom_info])
    h = self.transformer(self.input_proj(h))
    return self.output_norm(h)  # [B, L, c_token]
```

**Denoiser forward (with Gaussian noise):**
```python
def forward(self, x_t, atom_types, atom_to_res, aa_seq, chain_ids, t, mask=None):
    # Reshape atoms to residue structure
    x_res = x_t.view(B, N_res, 4, 3)

    # TRUNK (once per sample)
    trunk_tokens = self.trunk(x_res, aa_res, chain_res, res_idx, mask)  # [B, L, c_token]
    
    # DENOISER
    time_cond = self.time_embed(t).expand(-1, N_res, -1)  # [B, L, c_token]
    tokens, skip_states = self.atom_encoder(x_res, atom_types, trunk_tokens, mask)
    tokens = self.diff_transformer(tokens, time_cond, mask)  # AdaLN conditioning
    coord_updates = self.atom_decoder(tokens, skip_states, atom_types, mask)
    
    # SCALING: residual with noise-level scaling
    noise_scale = (t.float() / n_timesteps).sqrt().view(-1, 1, 1)
    x0_pred = x_t + noise_scale * coord_updates
    return x0_pred
```

> **IMPORTANT:** The `noise_scale` multiplier causes gradient issues at low timesteps (near 0, the gradients vanish). This is why `forward_direct()` exists.

**Forward modes:**
1. **`forward()`**: x0 = x_t + sqrt(t/T) * coord_updates ← **Gaussian noise default**
2. **`forward_direct()`**: x0 = coord_updates directly ← **For LinearChain**
3. **`forward_flow()`**: x_next = coord_updates directly ← **For LinearFlow**

**Local Attention (within residue):**
- Operates on [B, L, 4, c_atom] tensors
- Attention only within each residue's 4 atoms
- Memory: O(L * 16) vs O((4L)^2) for full attention

**AdaLN (Adaptive Layer Normalization):**
```python
class AdaLN(nn.Module):
    def forward(self, x, cond):
        x = self.norm(x)  # LayerNorm without learnable params
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale) + shift  # Modulated by conditioning
```

---

### ResFold (Stage 1)

**File:** `scripts/models/resfold.py` (389 lines)

**Architecture:** Residue-level diffusion (4x fewer tokens than all-atom)
- **Input:** Residue centroids [B, L, 3] (not atoms!)
- **Trunk:** ResidueEncoder - like AF3 but with centroids instead of atoms
- **Denoiser:** DiffusionTransformer with AdaLN

**Key difference from AF3:** Operates on L residue centroids, not 4L atoms.

**Trunk (ResidueEncoder):**
```python
def forward(self, centroids, aa_seq, chain_ids, res_idx, mask=None):
    aa_emb = self.aa_embed(aa_seq)           # [B, L, c_token]
    chain_emb = self.chain_embed(chain_ids)  # [B, L, c_token//4]
    res_emb = sinusoidal_pos_enc(res_idx)    # [B, L, c_token]
    coord_emb = self.coord_proj(centroids)   # [B, L, c_token//2]
    
    h = cat([aa_emb, chain_emb, res_emb, coord_emb])
    h = self.transformer(self.input_proj(h))
    return self.output_norm(h)
```

**Denoiser forward:**
```python
def forward(self, x_t, aa_seq, chain_ids, res_idx, t, mask=None):
    # TRUNK (encodes clean structure for conditioning)
    trunk_tokens = self.trunk(x_t, aa_seq, chain_ids, res_idx, mask)  # [B, L, c_token]
    
    # DENOISER
    coord_emb = self.coord_embed(x_t)  # [B, L, c_token]
    tokens = coord_emb + trunk_tokens  # Additive conditioning
    
    time_cond = self.time_embed(t).expand(-1, L, -1)  # [B, L, c_token]
    tokens = self.diff_transformer(tokens, time_cond, mask)  # Uses AdaLN
    
    x0_pred = self.output_proj(tokens)  # [B, L, 3]
    return x0_pred  # Direct x0 prediction
```

**Output mode:** **Direct x0 prediction** - no residual or scaling

**Stage 2 (AtomRefinerV2):** One-shot prediction of 4 backbone atoms from centroids (no diffusion).

---

## Training Logic

### train.py (AF3/AttentionV2)

**Timestep sampling:**
```python
if noise_type == "linear_chain":
    t = torch.randint(0, noiser.T + 1, (batch_size,))  # 0 to T inclusive
else:
    t = torch.randint(0, noiser.T, (batch_size,))      # 0 to T-1 (standard DDPM)
```

**Forward noising:**
```python
x_input, target = noiser.add_noise(
    batch['coords'], t,
    atom_to_res=batch['atom_to_res'],
    atom_type=batch['atom_types'],
    chain_ids=batch['chain_ids'],
)
```

**Model prediction:**
```python
if noise_type == "linear_chain" and hasattr(model, 'forward_direct'):
    pred = model.forward_direct(x_input, ...)  # No scaling
else:
    pred = model(x_input, ...)  # Standard with scaling
```

**Loss:**
```python
use_kabsch = (noise_type != "linear_chain")  # Kabsch alignment disabled for linear_chain

if noise_type == "linear_flow":
    loss = compute_loss(pred, target, mask, use_kabsch)  # target = x0
else:
    loss = compute_loss(pred, batch['coords'], mask, use_kabsch)  # target = GT x0
```

### train_resfold.py (ResFold Stage 1)

**Timestep sampling:**
```python
t = torch.randint(0, noiser.T, (batch_size,), device=device)  # 0 to T-1
```

**Forward noising (manual, not using noiser.add_noise):**
```python
noise = torch.randn_like(batch['centroids'])
sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
x_t = sqrt_ab * batch['centroids'] + sqrt_one_minus_ab * noise
```

**Model forward:**
```python
centroids_pred = model.forward_stage1(
    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
    t, batch['mask_res']
)
```

**Loss (with auxiliary terms):**
```python
loss_mse = compute_mse_loss(centroids_pred, batch['centroids'], mask)
loss_dist = compute_distance_consistency_loss(centroids_pred, batch['centroids'], mask)
loss = loss_mse + dist_weight * loss_dist

# Optional contact loss
if contact_loss_fn is not None:
    loss = loss + contact_weight * contact_losses['stage1']
```

---

## Inference/Sampling (Rollout)

### Standard DDPM Reverse (Gaussian)

**From `train.py:ddpm_sample()`:**

```python
x = torch.randn(B, N, 3, device=device)  # Start from pure noise

for t in reversed(range(noiser.T)):  # T-1, T-2, ..., 0
    t_batch = torch.full((B,), t, device=device)
    
    x0_pred = model(x, atom_types, ..., t_batch, mask)
    x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
    
    if t > 0:
        # DDPM reverse step
        ab_t = noiser.alpha_bar[t]
        ab_prev = noiser.alpha_bar[t - 1]
        beta = noiser.betas[t]
        alpha = noiser.alphas[t]
        
        coef1 = sqrt(ab_prev) * beta / (1 - ab_t)
        coef2 = sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
        mean = coef1 * x0_pred + coef2 * x
        
        var = beta * (1 - ab_prev) / (1 - ab_t)
        x = mean + sqrt(var) * randn_like(x)
    else:
        x = x0_pred  # Final step: just use prediction

return x
```

**DDPM Posterior mean formula:**
$$\mu_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

**DDPM Posterior variance:**
$$\sigma_{t-1}^2 = \frac{\beta_t (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}$$

### Linear Chain Reverse

```python
x = x_linear.clone()  # Start from extended chain

for t in reversed(range(noiser.T + 1)):  # T, T-1, ..., 0
    x0_pred = model.forward_direct(x, ...)  # No scaling!
    x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
    
    x = noiser.reverse_step(x, x0_pred, t, x_linear)
    # Inside reverse_step:
    #   if t == 0: return x0_pred
    #   x_prev = sqrt_ab_{t-1} * x0_pred + sqrt_(1-ab_{t-1}) * x_linear + noise
```

### Linear Flow Reverse

```python
x = x_linear.clone()  # Start from extended chain

for t in range(noiser.T):  # 0, 1, ..., T-1 (forward in time!)
    x0_pred = model(x, ..., t, ...)
    x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
    
    if t < noiser.T - 1:
        # Interpolate toward x0_pred
        alpha_next = noiser.schedule.sqrt_alpha_bar[t + 1]
        one_minus_alpha = noiser.schedule.sqrt_one_minus_alpha_bar[t + 1]
        x = alpha_next * x0_pred + one_minus_alpha * x_linear
    else:
        x = x0_pred  # Final step
```

### ResFold Stage 1 Sampling

**From `train_resfold.py:sample_centroids()`:**

```python
x = torch.randn(B, L, 3, device=device)  # Start from noise

for t in reversed(range(noiser.T)):
    x0_pred = model.forward_stage1(x, aa_seq, chain_ids, res_idx, t, mask)
    x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
    
    if t > 0:
        # Standard DDPM reverse
        ab_t = noiser.alpha_bar[t]
        ab_prev = noiser.alpha_bar[t - 1]
        beta = noiser.betas[t]
        alpha = noiser.alphas[t]
        
        coef1 = sqrt(ab_prev) * beta / (1 - ab_t)
        coef2 = sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
        mean = coef1 * x0_pred + coef2 * x
        
        var = beta * (1 - ab_prev) / (1 - ab_t)
        x = mean + sqrt(var) * randn_like(x)
    else:
        x = x0_pred

return x  # [B, L, 3] centroids
```

---

## Loss Formulations

### Primary Loss: MSE after Kabsch Alignment

```python
def compute_mse_loss(pred, target, mask=None, use_kabsch=True):
    if use_kabsch:
        pred_aligned, target_c = kabsch_align(pred, target, mask)
    else:
        # Just center both (for linear_chain where frame matters)
        pred_aligned = pred - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
    
    sq_diff = ((pred_aligned - target_c) ** 2).sum(dim=-1)  # [B, N]
    
    if mask is not None:
        loss = (sq_diff * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = sq_diff.mean()
    
    return loss
```

**Kabsch alignment:** Finds optimal rotation to align pred to target, removing rotational degrees of freedom from the loss.

### Auxiliary Losses

#### Distance Consistency Loss (ResFold Stage 1)

```python
def compute_distance_consistency_loss(pred_centroids, target_centroids, mask=None):
    # Compute pairwise distances [B, L, L]
    pred_dist = torch.cdist(pred_centroids, pred_centroids)
    target_dist = torch.cdist(target_centroids, target_centroids)
    
    # MSE on pairwise distances
    dist_diff = (pred_dist - target_dist) ** 2
    
    if mask is not None:
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        loss = (dist_diff * pair_mask).sum() / pair_mask.sum()
    else:
        loss = dist_diff.mean()
    
    return loss
```

**Purpose:** Encourages predicted centroids to preserve inter-residue distances from ground truth.

**Default weight:** 0.1

#### Geometry Losses (ResFold Stage 2)

**File:** `scripts/models/geometry_losses.py`

| Loss | Formula | Expected Value | Weight |
|------|---------|----------------|--------|
| Bond length | $(d - d_{ideal})^2$ | N-CA: 1.458Å, CA-C: 1.525Å, C-O: 1.229Å, C-N: 1.329Å | 1.0 |
| Bond angle | $(\theta - \theta_{ideal})^2$ | N-CA-C: 111°, CA-C-O: 121°, etc. | 0.1 |
| Omega dihedral | $\min((|\omega| - \pi)^2, \omega^2)$ | ~180° (trans) or ~0° (cis) | 0.1 |
| O chirality | ReLU-based penalty | Carbonyl on correct side of peptide plane | 0.1 |
| Virtual CB | Improper dihedral | ~-34° for L-amino acids | 0.0 (disabled) |

**Overall geometry weight:** 0.1

#### Contact Loss (Optional)

```python
# Penalize if predicted contacts don't match GT contacts
# Inter-chain contacts weighted 2x higher
loss_contact = ContactLoss(threshold=1.0, inter_chain_weight=2.0)(...)
```

### Loss Weighting Summary (ResFold)

| Mode | Primary Loss | Auxiliary Losses |
|------|--------------|------------------|
| Stage 1 | MSE on centroids | +0.1 × distance_consistency, +contact_weight × contact |
| Stage 2 | MSE on atoms | +0.1 × geometry_loss (bond+angle+omega+chirality) |
| End-to-end | MSE centroid + MSE atoms | +0.1 × distance_consistency + 0.1 × geometry |

---

## Comparison Table

| Aspect | AttentionV2 | AF3StyleDecoder | ResFold Stage 1 |
|--------|-------------|-----------------|-----------------|
| **Token granularity** | Atom-level (4L tokens) | Atom-level (4L tokens) | Residue-level (L tokens) |
| **Trunk runs** | N/A (single pass) | Once per sample | Once per sample |
| **Denoiser runs** | T iterations | T iterations | T iterations |
| **Output mode** | Direct x0 | x_t + scale × delta | Direct x0 |
| **Scaling** | None | sqrt(t/T) × residual | None |
| **AdaLN** | No (time_embed concat) | Yes | Yes |
| **Skip connections** | No | Encoder→Decoder | No |
| **Default schedule** | linear/cosine | linear/cosine | linear |
| **Default noise** | gaussian | gaussian | gaussian |
| **Loss** | MSE + Kabsch | MSE + Kabsch | MSE + Kabsch + dist |
| **Parameters** | ~5M (128h, 6L) | ~14M | ~14M (Stage 1 only) |

---

## Known Issues and Quirks

### 1. AF3StyleDecoder Scaling Problem

The `forward()` method uses `noise_scale = sqrt(t/T)` which causes:
- At t=0: scale=0, so coord_updates have zero gradient
- At low t: gradients are very small

**Workaround:** Use `forward_direct()` for linear_chain which predicts x0 directly.

### 2. Linear Chain Timestep Range

For linear_chain noise, timesteps are sampled from `[0, T]` inclusive (T+1 values), not `[0, T-1]` like Gaussian. This ensures the model sees pure extended chain at t=T.

```python
if noise_type == "linear_chain":
    t = torch.randint(0, noiser.T + 1, (batch_size,))  # 0 to T
else:
    t = torch.randint(0, noiser.T, (batch_size,))      # 0 to T-1
```

### 3. Kabsch Disabled for Linear Chain

When using linear_chain noise, Kabsch alignment is disabled in the loss:
```python
use_kabsch = (noise_type != "linear_chain")
```

This preserves the coordinate frame which is important for the reverse step interpolation with x_linear.

### 4. ResFold Stage 1 Runs Trunk on Noisy Input

In `ResidueDenoiser.forward()`, the trunk is called with `x_t` (noisy centroids), not clean centroids:
```python
trunk_tokens = self.trunk(x_t, aa_seq, chain_ids, res_idx, mask)
```

This means conditioning is computed from noisy coordinates, which differs from AF3 where trunk typically sees cleaner geometry.

### 5. No Timestep Curriculum Currently Enabled

`TimestepCurriculum` is implemented but disabled by default (`--curriculum` flag). When enabled:
- Starts with low max_t (easy, small noise)
- Gradually increases to full T over `warmup_steps`
- Helps with stable early training

---

## Recommendations

1. **Unify forward modes**: Consider having all models use consistent output (either always direct x0 or always scaled residual).

2. **Extract sampling logic**: Current DDPM sampling is duplicated in `train.py`, `train_resfold.py`, and `ResFoldPipeline.sample()`.

3. **Fix trunk conditioning**: Consider whether trunk should see noisy or clean coordinates - this affects the conditioning signal quality.

4. **Document noise type semantics**: The target returned by `add_noise()` differs between noise types (noise vs x_linear vs x0).
