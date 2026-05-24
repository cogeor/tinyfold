# Line-Shaped Outputs in VE Training - Root Cause Analysis

## Problem
Model outputs are line-shaped instead of folded protein structure.

## Root Cause

**The model hasn't learned to denoise at high sigma (σ > 5)**

VE sampling starts from `x ~ N(0, 10²)` at σ=10, while data has std=1. If model can't predict x0 well at high sigma, Euler steps accumulate errors into a random walk that collapses to origin/line.

## Why Discrete Training Works

| Mode | Signal-to-noise at max noise | Conditioning |
|------|------------------------------|--------------|
| **VP (t=49)** | 14% x0 + 99% noise | Learned embedding |
| **VE (σ=10)** | 10% x0 + 1000% noise | Fixed Fourier features |

VP always has ~14% signal; VE at σ=10 has virtually none.

## Quick Fixes to Test

### Fix A: Lower sigma_max
```bash
--sigma_max 5.0  # Instead of 10.0
```

### Fix B: Disable loss_weighting
Loss weighting down-weights high-sigma samples, preventing learning there:
```bash
# Remove --loss_weighting flag
```

### Fix C: Check sigma embedding
Add debug to verify different sigmas produce different embeddings.

### Fix D: Try discrete timesteps (known working)
```bash
# Remove --continuous_sigma flag entirely
python scripts/train.py --model af3_style ... --T 50
```

## Recommended Next Step
Run with `--sigma_max 5.0` and no loss weighting:
```bash
python scripts/train.py --model af3_style \
    --h_dim 128 --trunk_layers 5 --denoiser_blocks 5 \
    --load_split outputs/train_10k_continuous/split.json \
    --batch_size 32 --grad_accum 4 --lr 5e-4 \
    --n_steps 2000 --eval_every 500 --T 50 \
    --continuous_sigma --sigma_min 0.002 --sigma_max 5.0 \
    --output_dir outputs/atom_diffusion_ve_low_sigma
```
