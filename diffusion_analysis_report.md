# ResFold Diffusion Architecture Analysis

## Executive Summary

The ResFold diffusion model fails to converge even when overfitting on a single sample. After 500 training steps:
- t=0 (clean input): 0.00Å RMSE (trivial - copies input)
- t=49 (near-pure noise): 9.17Å RMSE (should be ~0 for overfitting)
- Full sampling: 6.91Å RMSE on training sample

This report identifies **three critical bugs/design flaws** in the current implementation.

---

## 1. Critical Issues Identified

### Issue 1: Residual Connection with Zero Gradient at Low Timesteps

**Location:** `scripts/models/resfold.py:350-351`

```python
# Output coordinate delta
coord_delta = self.output_proj(tokens)  # [B, L, 3]

# Residual connection with sqrt noise-level scaling
noise_scale = (t.float() / self.n_timesteps).sqrt().view(-1, 1, 1)
x0_pred = x_t + noise_scale * coord_delta
```

**The Problem:**

| t | noise_scale | Gradient to coord_delta |
|---|-------------|-------------------------|
| 0 | 0.000 | **ZERO** - no learning! |
| 1 | 0.141 | Very small |
| 5 | 0.316 | Still suppressed |
| 25 | 0.707 | Reasonable |
| 49 | 0.990 | Full gradient |

At t=0, the model outputs `x0_pred = x_t + 0 * coord_delta = x_t`, meaning:
1. The prediction trivially copies the input
2. No gradient flows to `coord_delta`, so the network doesn't learn anything for clean inputs
3. This defeats the purpose of denoising - the model never learns what the target structure looks like

**Why This Matters:**
- During training, ~2% of samples have t=0 (random uniform sampling)
- But during DDPM sampling, **every sample must pass through t=0** at the final step
- The model has essentially no training signal for what the final output should look like

### Issue 2: Trunk Runs on Noisy Coordinates, Not Clean Target

**Location:** `scripts/models/resfold.py:329-330`

```python
# === TRUNK (once) ===
trunk_tokens = self.trunk(x_t, aa_seq, chain_ids, res_idx, mask)
```

**The Problem:**

The trunk encoder conditions on `x_t` (the noisy input), not on clean coordinates or sequence alone. This creates a **leaky conditioning** problem:

1. During training: trunk sees `x_t = sqrt(α) * x0 + sqrt(1-α) * noise`
2. At low t: `x_t ≈ x0`, so trunk learns to extract structural info from nearly-clean coords
3. At high t: `x_t ≈ noise`, trunk receives almost no useful structural information
4. During sampling: at high t, trunk cannot provide meaningful conditioning

**Comparison with AF3:**
- AF3's trunk also conditions on noisy coordinates
- BUT AF3 diffuses on **atoms** (4L points) with local atom attention that provides skip connections
- AF3's AtomAttentionEncoder creates skip connections that pass through the denoiser
- ResFold has no such mechanism - it's purely feed-forward from trunk to denoiser

### Issue 3: Timestep Conditioning is Weak

**Location:** `scripts/models/resfold.py:341-344`

```python
# Timestep conditioning
time_cond = self.time_embed(t).unsqueeze(1).expand(-1, L, -1)

# Diffusion transformer
tokens = self.diff_transformer(tokens, time_cond, mask)
```

**The Problem:**

The timestep embedding is:
1. A single learned embedding per timestep
2. Expanded uniformly across all positions
3. Used only in AdaLN layers

This provides weak signal about the noise level. The model struggles to distinguish between "slightly noisy" and "heavily noisy" inputs, making it difficult to learn appropriate denoising strategies for different noise levels.

---

## 2. Detailed Architecture Comparison

### ResFold Stage 1 (Current - Diffuses on Centroids)

```
Input: x_t [B, L, 3] (noisy centroids)
       t [B] (timestep)
       sequence features

Trunk (runs once):
  - Embeds: aa_seq, chain_id, res_idx, coord_proj(x_t)
  - Transformer layers
  - Output: trunk_tokens [B, L, c_token]

Denoiser (runs once per training step):
  - coord_embed(x_t) + trunk_tokens
  - Timestep AdaLN conditioning
  - DiffusionTransformer
  - coord_delta = output_proj(tokens)
  - x0_pred = x_t + sqrt(t/T) * coord_delta   <-- BUG

Output: x0_pred [B, L, 3]
```

**Problems:**
1. Single-shot prediction, no iterative refinement during training
2. sqrt(t/T) scaling kills gradients at low t
3. Trunk conditioning is corrupted at high t

### AF3 (Diffuses on Atoms)

```
Input: x_t [B, N, 3] (noisy atom coords)
       t [B] (timestep)
       sequence features

Trunk (runs ONCE per sample):
  - Processes atom_coords → residue tokens
  - Provides static conditioning

Denoiser (runs at EACH diffusion step):
  - AtomAttentionEncoder: local attention, atoms → tokens
    * Creates skip connections [B, L, 4, c_atom]
  - DiffusionTransformer: global token attention with AdaLN
  - AtomAttentionDecoder: tokens → atoms
    * Uses skip connections from encoder
  - coord_updates [B, N, 3]

  # Same problematic scaling in base version:
  - x0_pred = x_t + sqrt(t/T) * coord_updates

Output: x0_pred [B, N, 3]
```

**Key Differences:**
1. AF3 has local atom attention with skip connections
2. AF3 processes atoms (4x more points) - more supervision signal
3. AF3 has both `forward()` (scaled residual) and `forward_direct()` (no scaling)

---

## 3. Why AF3-Style Works Better (Even with Same Scaling Issue)

### 3.1 More Supervision Signal
- ResFold diffuses on L centroids (1 point per residue)
- AF3 diffuses on 4L atoms (4 points per residue)
- 4x more coordinates = 4x more gradient signal per sample

### 3.2 Local Structure Preservation
- AF3's local atom attention preserves bond/angle relationships
- AtomAttentionEncoder → skip connections → AtomAttentionDecoder
- Even if global structure is noisy, local geometry provides signal

### 3.3 More Inductive Bias
- 4 atoms per residue have known relative positions
- N-CA-C-O backbone geometry is highly constrained
- This regularizes the learning problem

---

## 4. Analysis of the Linear Schedule

```python
# Linear schedule: alpha_bar goes from 1 to 0 linearly
alpha_bar[t] = 1 - t/T
```

| t | alpha_bar | sqrt(alpha_bar) | sqrt(1-alpha_bar) | SNR |
|---|-----------|-----------------|-------------------|-----|
| 0 | 1.000 | 1.000 | 0.000 | ∞ |
| 10 | 0.800 | 0.894 | 0.447 | 2.0 |
| 25 | 0.500 | 0.707 | 0.707 | 1.0 |
| 40 | 0.200 | 0.447 | 0.894 | 0.5 |
| 49 | 0.020 | 0.141 | 0.990 | 0.14 |

The linear schedule is reasonable, but combined with the sqrt(t/T) output scaling, creates a mismatch:

- At t=49: input has SNR=0.14 (almost pure noise), but noise_scale=0.99 (good gradient)
- At t=10: input has SNR=2.0 (clear signal), but noise_scale=0.45 (suppressed gradient)
- At t=0: input has SNR=∞ (clean), but noise_scale=0.0 (zero gradient!)

---

## 5. Proposed Solutions

### Solution A: Remove Residual Scaling (Simplest Fix)

**Change in `resfold.py:350-352`:**
```python
# Current (broken):
noise_scale = (t.float() / self.n_timesteps).sqrt().view(-1, 1, 1)
x0_pred = x_t + noise_scale * coord_delta

# Proposed (direct prediction):
x0_pred = self.output_proj(tokens)  # Direct coordinate prediction
```

**Pros:**
- Simple fix
- Consistent gradients at all timesteps
- Model must learn to output coordinates directly

**Cons:**
- Harder optimization (no residual connection helps)
- May need to initialize output_proj carefully

### Solution B: Predict Noise Instead of x0 (DDPM Standard)

**Change approach:**
```python
# Instead of predicting x0, predict the noise ε
epsilon_pred = self.output_proj(tokens)  # [B, L, 3]

# Loss: ||epsilon_pred - epsilon||^2
# During sampling: use standard DDPM formulas

# Reconstruction:
x0_pred = (x_t - sqrt(1-alpha_bar) * epsilon_pred) / sqrt(alpha_bar)
```

**Pros:**
- Standard DDPM formulation
- Well-studied, known to work
- Consistent gradient magnitude across timesteps

**Cons:**
- Requires changes to training and sampling loops
- May need numerical stability fixes at t≈0

### Solution C: Use v-Prediction (Progressive Distillation Style)

```python
# v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x0
# Model predicts v instead of x0 or epsilon
v_pred = self.output_proj(tokens)

# Reconstruction:
x0_pred = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v_pred
```

**Pros:**
- Stable across all timesteps
- Used in modern diffusion models
- Better for few-step sampling

**Cons:**
- More complex implementation
- Need to derive loss and sampling formulas

### Solution D: Add Rollout Training (AF3-Style)

Instead of training on single (x_t, t) pairs, run partial diffusion trajectories:

```python
# During training, occasionally run K-step rollouts
K = 5  # or random K ~ [1, 10]
x = torch.randn(B, L, 3)  # Start from noise

for step in range(K):
    t = T - step - 1  # Going backwards
    x0_pred = model(x, ..., t)
    x = ddpm_reverse_step(x, x0_pred, t)

# Loss on final x0_pred vs ground truth
loss = mse(x0_pred, gt_centroids)
```

**Pros:**
- Trains the model on actual sampling trajectories
- Forces consistency across timesteps
- Errors compound, so model learns to avoid them

**Cons:**
- K times more compute per training step
- Memory intensive (need to store K activations for backward)
- Hyperparameter K to tune

### Solution E: Separate Trunk from Noisy Coords (Clean Conditioning)

**Current:**
```python
trunk_tokens = self.trunk(x_t, aa_seq, chain_ids, res_idx, mask)
```

**Proposed:**
```python
# Trunk conditions on sequence + position only (no coordinates)
trunk_tokens = self.trunk_seq(aa_seq, chain_ids, res_idx, mask)

# Denoiser gets noisy coords separately
coord_emb = self.coord_embed(x_t)
tokens = coord_emb + trunk_tokens
```

**Pros:**
- Trunk provides consistent conditioning regardless of noise level
- Denoiser can learn noise-level-specific processing
- Cleaner separation of concerns

**Cons:**
- Trunk loses ability to reason about structure
- May need more denoiser capacity

---

## 6. Recommended Implementation Path

### Phase 1: Quick Fixes (Test First)
1. **Remove sqrt scaling** - just use `x0_pred = output_proj(tokens)`
2. **Test overfitting** - should now see t=49 RMSE drop to near 0

### Phase 2: Better Architecture
3. **Switch to ε-prediction** - standard DDPM approach
4. **Separate sequence-only trunk** - cleaner conditioning

### Phase 3: Training Improvements
5. **Add rollout training** - occasional K-step trajectories
6. **Timestep curriculum** - start with low noise, increase gradually

---

## 7. Immediate Action Items

1. **Fix the sqrt scaling bug** - this is the primary issue
2. **Verify on 1 sample** - should overfit to <1Å RMSE at all t values
3. **Test on 100 samples** - verify generalization
4. **Consider switching to ε-prediction** if direct prediction is unstable

---

## Appendix: Code Snippets for Fixes

### Fix 1: Remove sqrt scaling (resfold.py)

```python
# In ResidueDenoiser.forward(), replace:
noise_scale = (t.float() / self.n_timesteps).sqrt().view(-1, 1, 1)
x0_pred = x_t + noise_scale * coord_delta

# With:
x0_pred = self.output_proj(tokens)
```

### Fix 2: ε-prediction (if direct prediction fails)

```python
# In forward():
epsilon_pred = self.output_proj(tokens)
return epsilon_pred  # Return noise prediction

# In training:
noise = torch.randn_like(gt_centroids)
x_t = sqrt_ab * gt_centroids + sqrt_one_minus_ab * noise
epsilon_pred = model(..., x_t, t)
loss = mse(epsilon_pred, noise)

# In sampling:
x0_pred = (x_t - sqrt_one_minus_ab * epsilon_pred) / sqrt_ab.clamp(min=1e-4)
```

---

## Conclusion

The ResFold diffusion model has a critical bug: the `sqrt(t/T)` output scaling eliminates gradients at low timesteps, preventing the model from learning what the target structure should look like. Combined with trunk conditioning on noisy coordinates, this creates a fundamentally broken training dynamic.

The fix is straightforward: remove the sqrt scaling and predict coordinates directly. More sophisticated approaches (ε-prediction, v-prediction, rollout training) can be explored if the simple fix proves insufficient.
