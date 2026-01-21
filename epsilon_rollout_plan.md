# Implementation Plan: Epsilon Prediction + Rollout Training

## Changes Overview

### 1. Epsilon Prediction (resfold.py)
- Model predicts noise `ε` instead of clean coordinates `x0`
- Training: `loss = ||ε_pred - ε||²`
- Inference: `x0 = (x_t - sqrt(1-αbar) * ε_pred) / sqrt(αbar)`

### 2. Sampling from t=1 (not t=0)
- Skip t=0 since it's trivial (no noise)
- Sample t from `[1, T)` instead of `[0, T)`

### 3. Rollout Training
- Occasionally run K-step reverse diffusion during training
- Train on trajectory consistency, not just single-step denoising

---

## File Changes

### A. `scripts/models/resfold.py`

```python
# In ResidueDenoiser.forward():

# OLD (x0 prediction with broken scaling):
coord_delta = self.output_proj(tokens)
noise_scale = (t.float() / self.n_timesteps).sqrt().view(-1, 1, 1)
x0_pred = x_t + noise_scale * coord_delta
return x0_pred

# NEW (epsilon prediction):
epsilon_pred = self.output_proj(tokens)
return epsilon_pred  # Return predicted noise
```

### B. `scripts/train_resfold_e2e.py`

**Training step changes:**
```python
# Sample t from [1, T) instead of [0, T)
t = torch.randint(1, noiser.T, (B,), device=device)

# Compute x_t from noise
noise = torch.randn_like(gt_centroids)
alpha_bar = noiser.alpha_bar[t].view(-1, 1, 1)
x_t = torch.sqrt(alpha_bar) * gt_centroids + torch.sqrt(1 - alpha_bar) * noise

# Model predicts epsilon
epsilon_pred = model.forward_stage1(x_t, ..., t, mask)

# Loss on epsilon (not x0)
loss = mse(epsilon_pred, noise)

# To get x0_pred for Stage 2:
sqrt_ab = torch.sqrt(alpha_bar)
sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar)
x0_pred = (x_t - sqrt_one_minus_ab * epsilon_pred) / sqrt_ab.clamp(min=1e-4)
```

**Rollout training:**
```python
def rollout_step(model, batch, noiser, device, K=5):
    """Run K-step reverse diffusion and compute loss on final prediction."""
    B, L = batch['aa_seq'].shape

    # Start from pure noise at t=T-1
    x = torch.randn(B, L, 3, device=device)

    for step in range(K):
        t_val = noiser.T - 1 - step
        if t_val < 1:
            break
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Predict epsilon
        epsilon_pred = model.forward_stage1(x, ..., t, mask)

        # Reconstruct x0
        ab = noiser.alpha_bar[t_val]
        x0_pred = (x - math.sqrt(1-ab) * epsilon_pred) / math.sqrt(ab)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        # DDPM reverse step to x_{t-1}
        if t_val > 1:
            # ... standard DDPM update
            x = ddpm_step(x, x0_pred, t_val, noiser)
        else:
            x = x0_pred

    # Loss on final x0_pred
    return mse_loss(x0_pred, gt_centroids)
```

### C. Sampling changes

```python
def sample_full(model, batch, noiser, device):
    x = torch.randn(B, L, 3, device=device)

    # Start from T-1, go down to 1 (skip 0)
    for t in reversed(range(1, noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict epsilon
        epsilon_pred = model.forward_stage1(x, ..., t_batch, mask)

        # Reconstruct x0
        ab = noiser.alpha_bar[t]
        x0_pred = (x - math.sqrt(1-ab) * epsilon_pred) / math.sqrt(ab)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        # DDPM reverse step
        if t > 1:
            # Standard DDPM update...
        else:
            x = x0_pred

    return x
```

---

## Implementation Order

1. Modify `resfold.py` - change forward to return epsilon
2. Modify `train_resfold_e2e.py`:
   - Change training loss to epsilon prediction
   - Add x0 reconstruction for Stage 2
   - Add rollout training function
   - Mix single-step and rollout training
3. Update sampling code
4. Test on 1 sample
