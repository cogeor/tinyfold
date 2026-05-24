# Diffusion Inference Problem: Distribution Mismatch

## Summary

Training achieves near-zero MSE loss, but inference gives 4-10Å RMSE instead of sub-1Å. The root cause is a **distribution mismatch** between training and inference.

## The Problem

### Training Distribution
```
x_t = ground_truth + σ * gaussian_noise
```
- Fresh Gaussian noise added to exact ground truth
- Model learns to predict x0 from this specific distribution
- At low σ, x_t is very close to ground truth

### Inference Distribution
```
x_t = previous_prediction + accumulated_error
```
- x_t is the result of iterative Euler updates
- Error accumulates through sampling steps
- At low σ, x_t has ~2-5Å structural drift from ground truth

### Why This Matters

At low σ (e.g., 0.01), the model expects:
- x_t ≈ ground_truth + tiny_noise

But during inference, x_t has:
- ~2-5Å structural error (not tiny random noise)
- Error pattern is correlated, not i.i.d. Gaussian

The model was never trained on this distribution, so it fails.

## Evidence

### Model Works Perfectly on Training Distribution
```
sigma=0.01, x = gt + 0.01*noise  →  RMSE = 0.18Å  ✓
sigma=0.10, x = gt + 0.10*noise  →  RMSE = 0.13Å  ✓
sigma=1.00, x = gt + 1.00*noise  →  RMSE = 0.17Å  ✓
```

### Model Fails on Inference Distribution
At σ=0.01 with structural error:
```
x = gt + 0.5Å error  →  RMSE = 0.26Å
x = gt + 1.0Å error  →  RMSE = 0.42Å
x = gt + 2.0Å error  →  RMSE = 0.73Å
x = gt + 5.0Å error  →  RMSE = 1.74Å
```

### Sampling Step Analysis
During inference, tracking x0_pred quality at each step:
```
Step  0 (σ=10.0): x0_pred RMSE = 0.59Å  ✓ (model ignores noisy x, uses sequence)
Step 10 (σ=3.5):  x0_pred RMSE = 0.29Å  ✓ (BEST - medium noise, model works well)
Step 25 (σ=0.5):  x0_pred RMSE = 0.75Å    (starting to degrade)
Step 40 (σ=0.03): x0_pred RMSE = 4.35Å  ✗ (x has drifted, model fails)
Step 49 (σ=0.003):x0_pred RMSE = 11.4Å  ✗ (complete failure)
```

### Key Insight
The **best x0_pred** occurs at step 10-15 (σ ≈ 2-3.5), achieving **0.27-0.29Å RMSE**.

Going to lower σ makes results progressively worse:
```
Stop at step 10 (σ=3.5): 0.29Å
Stop at step 20 (σ=1.0): 0.33Å
Stop at step 30 (σ=0.2): 0.42Å
Stop at step 40 (σ=0.03): 1.19Å
Stop at step 49 (σ=0.003): 6.84Å
```

## Why Standard Diffusion Doesn't Have This Problem

In standard image diffusion:
1. The model predicts noise ε, not x0 directly
2. Score matching objective is robust to distribution shift
3. Images have much more redundancy (local patches are similar)

In our coordinate diffusion:
1. We predict x0 directly
2. Coordinates have no local redundancy
3. Small errors in x compound through sampling

## Additional Problem: Rotation Sensitivity

The model is **not rotation-invariant**:
```
Original x_t:           RMSE = 0.26Å
Rotated x_t (trial 1):  RMSE = 0.62Å
Rotated x_t (trial 2):  RMSE = 0.33Å
Rotated x_t (trial 3):  RMSE = 0.75Å
```

During training, x_t is always in the same coordinate frame as ground truth (x_t = gt + noise). But during inference, x_t can have arbitrary rotation from Euler updates. The model hasn't seen rotated inputs.

**Fix**: Add rotation augmentation during training:
```python
if args.augment_rotation:
    R = random_rotation_matrix(batch_size, device)
    x_t = torch.bmm(x_t, R.transpose(1, 2))
```

Note: The loss uses Kabsch alignment (rotation-invariant), but the model's internal processing is not rotation-equivariant. Training with rotation augmentation should help.

## Attempted Solutions

### 1. Self-Conditioning
- Train model to see its previous prediction
- Helps slightly (4.68Å → 4.40Å) but doesn't solve the problem
- Model still expects x_t to be close to gt at low σ

### 2. Translation Augmentation
- Add random translations during training
- Marginal improvement
- Doesn't address the core distribution mismatch

### 3. Early Stopping
- Stop sampling at σ ≈ 1-3 instead of σ → 0
- Use x0_pred directly instead of final x
- Works! Gets 0.29Å instead of 6.8Å
- But feels like a hack, not a principled solution

## Potential Real Solutions

### 1. Noise Prediction Instead of x0 Prediction
Train to predict ε instead of x0:
```
x_t = x0 + σ * ε
model predicts ε_pred
x0_pred = x_t - σ * ε_pred
```
This might be more robust because the target (noise) is always i.i.d. Gaussian.

### 2. Flow Matching / Rectified Flow
Use a different formulation where the model learns a velocity field:
```
v = dx/dt = (x1 - x0)
```
This has better training-inference alignment.

### 3. Consistency Models
Train the model to map any x_t directly to x0 in one step, with consistency loss ensuring all noise levels map to the same x0.

### 4. Better Sampling (Second-Order / Predictor-Corrector)
- Heun's method or DPM-Solver
- Predictor-corrector with Langevin dynamics
- May reduce error accumulation

### 5. Train on Model's Own Distribution
During training, sometimes:
1. Run inference to get x_pred
2. Add noise to x_pred
3. Train model to denoise x_pred + noise → gt

This exposes the model to the inference distribution.

## Conclusion

The core issue is that VE diffusion with x0-prediction creates a distribution mismatch at low noise levels. The model achieves perfect training loss but fails at inference because the inference distribution (accumulated errors) differs from training distribution (fresh Gaussian noise around ground truth).

The model IS capable of 0.27Å predictions - we just can't access them reliably during standard sampling. A fundamental change to the diffusion formulation or training procedure is needed.
