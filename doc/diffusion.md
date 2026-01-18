# Diffusion for Structure Prediction

## Motivation

Why use diffusion for structure prediction?

1. **Generative modeling**: Diffusion naturally handles the multi-modal nature of protein structure—multiple valid conformations may exist.

2. **Iterative refinement**: Starting from noise and gradually refining mimics the folding process conceptually.

3. **Conditioning flexibility**: The denoiser can condition on any information (sequence, MSA, templates) without architectural changes.

4. **Training stability**: Predicting at each timestep provides stable gradients compared to single-shot coordinate regression.

## Mathematical Framework

### Forward Process (Adding Noise)

Given clean coordinates x₀, we define a noising process:

```
x_t = √(ᾱ_t) × x₀ + √(1 - ᾱ_t) × ε
```

where:
- ε ~ N(0, I) is Gaussian noise
- ᾱ_t is the cumulative noise schedule
- t ∈ {0, 1, ..., T-1} is the timestep

At t=0, x_t ≈ x₀ (mostly signal)
At t=T-1, x_t ≈ ε (mostly noise)

### Prediction Target: x₀ vs ε

There are two common prediction targets:

**Noise prediction (ε-prediction)**:
```
ε̂ = model(x_t, t, conditioning)
x̂₀ = (x_t - √(1-ᾱ_t) × ε̂) / √(ᾱ_t)
```

**Clean sample prediction (x₀-prediction)** [used in TinyFold]:
```
x̂₀ = model(x_t, t, conditioning)
```

We use **x₀-prediction** because:
- Directly predicts coordinates, easier to interpret
- Works better for coordinate regression tasks
- Loss can be computed directly on x̂₀

## Cosine Schedule

We use the cosine schedule from "Improved DDPM" (Nichol & Dhariwal, 2021):

```python
def cosine_schedule(T, s=0.008):
    steps = arange(T + 1)
    f = cos((steps/T + s) / (1 + s) * π/2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar
```

### Why Cosine?

```
Linear:  ᾱ_t drops quickly then slowly
Cosine:  ᾱ_t drops smoothly, more time at intermediate noise levels
```

The cosine schedule spends more steps at intermediate noise levels where learning is most effective.

### Schedule Implementation

```python
class CosineSchedule:
    def __init__(self, T=50, s=0.008):
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

## Training

### Forward Pass

```python
def training_step(model, x0, features, schedule):
    # Sample random timestep
    t = randint(0, schedule.T, (batch_size,))

    # Add noise
    x_t, noise = schedule.add_noise(x0, t)

    # Predict clean coordinates
    x0_pred = model(x_t, features, t)

    # Compute loss (Kabsch-aligned MSE)
    loss = compute_loss(x0_pred, x0, mask)

    return loss
```

### Loss Function

We use Kabsch-aligned MSE to make the loss invariant to rotation and translation:

```python
def compute_loss(pred, target, mask=None):
    # Optimally align pred to target
    pred_aligned, target_centered = kabsch_align(pred, target, mask)

    # MSE on aligned coordinates
    sq_diff = ((pred_aligned - target_centered) ** 2).sum(dim=-1)

    if mask is not None:
        loss = (sq_diff * mask.float()).sum() / mask.float().sum()
    else:
        loss = sq_diff.mean()

    return loss
```

## DDPM Sampling

### Basic DDPM Step

Given x₀ prediction, the DDPM reverse step is:

```python
def ddpm_step(x_t, x0_pred, t, schedule):
    if t == 0:
        return x0_pred

    ab_t = schedule.alpha_bar[t]
    ab_prev = schedule.alpha_bar[t - 1]
    beta = schedule.betas[t]
    alpha = schedule.alphas[t]

    # Compute mean of p(x_{t-1} | x_t, x0_pred)
    coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
    coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
    mean = coef1 * x0_pred + coef2 * x_t

    # Add noise (except at t=0)
    var = beta * (1 - ab_prev) / (1 - ab_t)
    noise = torch.randn_like(x_t)
    x_prev = mean + torch.sqrt(var) * noise

    return x_prev
```

### Full Sampling Loop

```python
@torch.no_grad()
def ddpm_sample(model, features, schedule, shape):
    device = features['atom_types'].device

    # Start from noise
    x = torch.randn(shape, device=device)

    # Denoise from T-1 to 0
    for t in reversed(range(schedule.T)):
        t_batch = torch.full((batch_size,), t, device=device)

        # Predict x0
        x0_pred = model(x, features, t_batch)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # Stability

        # Take DDPM step
        x = ddpm_step(x, x0_pred, t, schedule)

    return x
```

### Clamping for Stability

We clamp x0_pred to prevent divergence:
```python
x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
```

This works because coordinates are normalized (zero mean, unit variance) before diffusion.

## Coordinate Preprocessing

### Centering and Normalization

Coordinates are preprocessed before diffusion:

```python
# Center
centroid = coords.mean(dim=0, keepdim=True)
coords_centered = coords - centroid

# Normalize
std = coords_centered.std()
coords_normalized = coords_centered / std
```

This ensures:
- Zero mean (no translation to learn)
- Unit variance (noise scale matches)

### Undoing Normalization

After sampling, RMSE is computed in original scale:

```python
rmse_angstroms = rmse_normalized * std
```

## Why T=50?

| Model | T | Notes |
|-------|---|-------|
| DDPM (images) | 1000 | High detail, slow |
| Stable Diffusion | 50-100 | Reduced steps |
| AlphaFold3 | 16-200 | Structure has less fine detail |
| TinyFold | 50 | Balance of quality and speed |

Protein structure has less fine detail than images—fewer steps suffice. Each step requires a full model forward pass, so fewer steps means faster sampling.

## Comparison with DDIM

DDIM (Denoising Diffusion Implicit Models) allows deterministic sampling:

```python
# DDIM step (deterministic, eta=0)
def ddim_step(x_t, x0_pred, t, schedule):
    sqrt_ab_prev = schedule.sqrt_alpha_bar[t-1] if t > 0 else 1.0
    sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t]

    x_prev = sqrt_ab_prev * x0_pred + sqrt_1_ab * eps_implicit
    return x_prev
```

We use DDPM (stochastic) because:
- Slightly better sample quality
- Same number of steps anyway
- Simpler implementation

## Numerical Stability

### Division Guards

At t=T-1, √ᾱ_t is very small:

```python
# Avoid division by near-zero
x0_pred = (x_t - sqrt_1_ab * eps) / (sqrt_ab + 1e-8)
```

### Alpha Bar Clipping

```python
alpha_bar = clamp(alpha_bar, 1e-5, 1.0)
```

Prevents exactly 0 or 1, which cause numerical issues.

### Gradient Clipping

During training:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Prevents gradient explosion from bad samples.
