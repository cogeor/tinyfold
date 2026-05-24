# Diffusion Implementation Plan - TinyFold

## Executive Summary

This plan addresses testing AF3/Boltz-1 style diffusion improvements in TinyFold with:
- **Backward compatibility**: Current pipelines remain functional
- **Swappable diffusion logic**: Registry-based architecture for easy A/B testing
- **Multi-phase rollout**: Critical fixes first, then progressive enhancements

---

## Part 1: Answers to Clarifying Questions

### Q1: Are normalized coordinates std ≈ 1?

**Yes.** The data loading normalizes coordinates to approximately unit standard deviation:

```python
# scripts/train.py:198-201
centroid = coords.mean(dim=0, keepdim=True)
coords = coords - centroid
std = coords.std()
coords = coords / std  # Now coords has std ≈ 1
```

The original `std` is stored per sample for denormalization during evaluation. This means:
- **sigma_data = 1.0** is appropriate (not AF3's 16Å because AF3 uses raw Angstroms)
- Current VP noise (`sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise`) is well-scaled

### Q2: Focus on ResFold or AF3-style first?

**Recommendation: ResFold Stage 1 first.** Reasons:

| Factor | ResFold | AF3StyleDecoder |
|--------|---------|-----------------|
| Token count | L residues | 4L atoms |
| Iteration speed | Faster | 4x slower |
| Current status | Main pipeline | Less used |
| Simpler debug | Yes (no atom attention) | No |

After validating fixes on ResFold, port to AF3-style.

### Q3: Increase steps from 50 to 100-200?

**Yes, but make it configurable.** Current T=50 is fine for fast iteration. For final quality:

| Steps | Training | Inference | Recommendation |
|-------|----------|-----------|----------------|
| 50 | Keep for dev | Keep for dev | Default |
| 100 | Optional | Configurable | Good balance |
| 200 | Not needed | For final eval | Best quality |

Add `--inference_steps` flag separate from training `T`.

### Q4: Kabsch weighting: all atoms or backbone only?

**For ResFold Stage 1**: All centroids equally weighted (only L points).

**For AF3-style (atom-level)**:
- **Option A (simple)**: All atoms equally weighted
- **Option B (AF3-like)**: Weight by atom type (CA > backbone > sidechain)
- **Recommendation**: Start with Option A, add weighting later if needed

### Q5: Specific test proteins with visible drift?

**Suggested test set:**
1. **Small complex (fast iteration)**: Any ~50 residue complex
2. **Medium complex**: Any ~100 residue complex
3. **Large complex**: Any 200+ residue complex (drift amplifies with size)

**Diagnostic protocol:**
1. Run 50-step sampling, visualize every 10 steps
2. Check if structures gradually warp or collapse
3. Compare centroid trajectory vs GT centroid

---

## Part 2: Swappable Architecture Design

### Current Architecture

```
scripts/models/diffusion.py
├── Schedules: CosineSchedule, LinearSchedule
├── Noisers: GaussianNoise, LinearChainNoise, LinearChainFlow
└── Factory: create_schedule(), create_noiser()

scripts/train.py
└── ddpm_sample() - hardcoded DDPM reverse

scripts/train_resfold.py
└── sample_centroids() - hardcoded DDPM reverse
```

### Proposed Architecture

```
scripts/models/diffusion.py (extended)
├── Schedules (existing)
│   ├── CosineSchedule
│   ├── LinearSchedule
│   └── NEW: KarrasSchedule (EDM-style continuous σ)
│
├── Noisers (existing + new)
│   ├── GaussianNoise (VP-style: sqrt_alpha_bar blend)
│   ├── LinearChainNoise
│   ├── LinearChainFlow
│   └── NEW: VENoiser (additive: x + σ*ε)
│
├── NEW: Samplers (swappable reverse process)
│   ├── DDPMSampler (current, baseline)
│   ├── DDPMKabschSampler (DDPM + per-step alignment)
│   ├── EDMSampler (Euler with VE)
│   ├── EDMKabschSampler (EDM + alignment)
│   └── HeunSampler (2nd order)
│
└── Factories (extended)
    ├── create_schedule()
    ├── create_noiser()
    └── NEW: create_sampler()
```

### Sampler Protocol

All samplers implement the same interface:

```python
class BaseSampler(ABC):
    """Abstract base for diffusion samplers."""

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],  # [B, N, 3]
        model_inputs: dict,  # atom_types, aa_seq, chain_ids, etc.
        noiser: BaseNoiser,
        device: torch.device,
        clamp_val: float = 3.0,
        **kwargs,
    ) -> Tensor:
        """Run full reverse diffusion sampling.

        Returns:
            x0_pred: Final predicted coordinates [B, N, 3]
        """
        pass
```

### Backward Compatibility

```python
# Old code (still works)
x_pred = ddpm_sample(model, atom_types, ..., noiser, mask)

# New code (explicit sampler)
sampler = create_sampler("ddpm_kabsch", align_every_step=True)
x_pred = sampler.sample(model, (B, N, 3), model_inputs, noiser, device)
```

The legacy `ddpm_sample()` function remains as a wrapper.

---

## Part 3: Multi-Phase Implementation Plan

### Phase 1: Critical Fix - Per-Step Kabsch Alignment (2-3 hours)

**Goal**: Fix the drift problem without changing anything else.

**Changes**:

1. **Add `kabsch_align_to_target()` utility** in `diffusion.py`:

```python
def kabsch_align_to_target(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    """Kabsch-align pred INTO target's frame (returns aligned pred).

    Unlike kabsch_align() which returns both aligned, this returns ONLY
    the prediction transformed to match target's coordinate frame.
    """
    pred_aligned, _ = kabsch_align(pred, target, mask)
    # Add back target's centroid
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        target_mean = target.mean(dim=1, keepdim=True)
    return pred_aligned + target_mean
```

2. **Modify `ddpm_sample()`** in `train.py`:

```python
def ddpm_sample(..., align_per_step: bool = True):  # NEW FLAG
    ...
    for t in t_range:
        x0_pred = model(x, ...)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # NEW: Kabsch-align x0_pred to current x
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        if t > 0:
            # DDPM reverse (unchanged)
            ...

        # NEW: Re-center after update
        x = x - x.mean(dim=1, keepdim=True)
```

3. **Same modification to `sample_centroids()`** in `train_resfold.py`

**Testing**:
- Compare sampling with/without alignment on same model
- Visualize step-by-step evolution
- Should see coherent structures instead of warped blobs

**Backward compatibility**: Default `align_per_step=True` fixes the issue, but can be disabled for comparison.

---

### Phase 2: Sampler Registry (3-4 hours)

**Goal**: Make samplers swappable via registry pattern.

**New file**: `scripts/models/samplers.py`

```python
"""Diffusion samplers for TinyFold.

Samplers handle the reverse diffusion process. Each sampler can be
used with any compatible noiser.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Optional

from .diffusion import kabsch_align_to_target


class BaseSampler(ABC):
    """Abstract base for diffusion samplers."""

    def __init__(self, recenter: bool = True, clamp_val: float = 3.0):
        self.recenter = recenter
        self.clamp_val = clamp_val

    @abstractmethod
    def sample(self, model, shape, inputs, noiser, device, **kwargs) -> Tensor:
        pass


class DDPMSampler(BaseSampler):
    """Standard DDPM reverse sampler (baseline)."""

    def sample(self, model, shape, inputs, noiser, device, **kwargs) -> Tensor:
        B, N, _ = shape
        x = torch.randn(B, N, 3, device=device)

        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x0_pred = model(x, **inputs, t=t_batch)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            if t > 0:
                # Standard DDPM reverse
                ab_t, ab_prev = noiser.alpha_bar[t], noiser.alpha_bar[t-1]
                beta, alpha = noiser.betas[t], noiser.alphas[t]
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

        return x


class DDPMKabschSampler(BaseSampler):
    """DDPM with per-step Kabsch alignment (recommended)."""

    def sample(self, model, shape, inputs, noiser, device, **kwargs) -> Tensor:
        B, N, _ = shape
        mask = inputs.get('mask')
        x = torch.randn(B, N, 3, device=device)

        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x0_pred = model(x, **inputs, t=t_batch)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            # KEY FIX: Align x0_pred to current x's frame
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            if t > 0:
                ab_t, ab_prev = noiser.alpha_bar[t], noiser.alpha_bar[t-1]
                beta, alpha = noiser.betas[t], noiser.alphas[t]
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

            # Re-center to avoid translation drift
            if self.recenter:
                x = x - x.mean(dim=1, keepdim=True)

        return x


class HeunSampler(BaseSampler):
    """Heun (2nd order) sampler for EDM-style diffusion."""

    def sample(self, model, shape, inputs, noiser, device, **kwargs) -> Tensor:
        # Implementation as per diffusion_todo.md REQ-6
        ...


# Registry
_SAMPLERS = {
    "ddpm": DDPMSampler,
    "ddpm_kabsch": DDPMKabschSampler,
    "heun": HeunSampler,
}

def list_samplers() -> list[str]:
    return list(_SAMPLERS.keys())

def create_sampler(name: str, **kwargs) -> BaseSampler:
    if name not in _SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(_SAMPLERS.keys())}")
    return _SAMPLERS[name](**kwargs)
```

**Integration** in `train.py`:

```python
from models import create_sampler

# In main():
sampler = create_sampler(args.sampler)  # NEW ARG: --sampler ddpm_kabsch

# In eval:
x_pred = sampler.sample(model, (B, N, 3), model_inputs, noiser, device)
```

---

### Phase 3: VE Noise + EDM Sampler (4-5 hours)

**Goal**: Implement AF3-style variance-exploding diffusion as an alternative.

**New classes in `diffusion.py`**:

```python
class KarrasSchedule:
    """Karras/EDM-style continuous sigma schedule."""

    def __init__(
        self,
        n_steps: int = 200,
        sigma_min: float = 0.002,  # Scaled for normalized coords
        sigma_max: float = 10.0,   # Scaled for normalized coords
        rho: float = 7.0,
    ):
        self.n_steps = n_steps
        # Build schedule: sigma[i] = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        sigmas = (sigma_max ** (1/rho) + steps * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        self.sigmas = sigmas  # [n_steps+1], from high to low

    @property
    def T(self):
        return self.n_steps


class VENoiser:
    """Variance-Exploding noise (AF3-style additive)."""

    def __init__(self, schedule: KarrasSchedule, sigma_data: float = 1.0):
        self.schedule = schedule
        self.sigma_data = sigma_data

    @property
    def T(self):
        return self.schedule.T

    def add_noise(self, x0: Tensor, sigma: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """VE forward: x_noisy = x0 + sigma * noise"""
        noise = torch.randn_like(x0)
        sigma_view = sigma.view(-1, 1, 1)
        x_noisy = x0 + sigma_view * noise
        return x_noisy, noise

    def sample_sigma(self, batch_size: int, device) -> Tensor:
        """Sample sigma for training (log-normal like AF3)."""
        log_sigma = torch.randn(batch_size, device=device) * 1.2
        return self.sigma_data * torch.exp(log_sigma)

    def loss_weight(self, sigma: Tensor) -> Tensor:
        """AF3-style loss weighting."""
        return (sigma**2 + self.sigma_data**2) / (sigma + self.sigma_data)**2
```

**New sampler** `EDMKabschSampler`:

```python
class EDMKabschSampler(BaseSampler):
    """EDM-style Euler sampler with Kabsch alignment."""

    def sample(self, model, shape, inputs, noiser, device, **kwargs) -> Tensor:
        B, N, _ = shape
        mask = inputs.get('mask')
        sigmas = noiser.schedule.sigmas.to(device)

        # Initialize at high noise
        x = sigmas[0] * torch.randn(B, N, 3, device=device)

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Model predicts x0 given (x, sigma)
            x0_pred = model(x, **inputs, sigma=sigma)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            # Kabsch align to current frame
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            # EDM update: x = x + dt * (x - x0) / sigma
            dt = sigma_next - sigma  # negative
            delta = (x - x0_pred) / sigma
            x = x + dt * delta

            if self.recenter:
                x = x - x.mean(dim=1, keepdim=True)

        return x
```

---

### Phase 4: Training Improvements (3-4 hours)

**Goal**: Add training-time improvements for better generalization.

#### 4.1 Loss Weighting

```python
# In training loop:
if args.noise_type == "ve":
    sigma = noiser.sample_sigma(batch_size, device)
    weight = noiser.loss_weight(sigma)
    loss = weight * compute_mse_loss(pred, target, mask)
else:
    loss = compute_mse_loss(pred, target, mask)
```

#### 4.2 Random Rigid Augmentation

```python
def random_rigid_augment(x: Tensor, mask: Tensor = None) -> Tensor:
    """Apply random rotation and translation for training robustness."""
    B = x.shape[0]
    device = x.device

    # Random rotation per sample
    R = random_rotation_matrices(B, device)  # [B, 3, 3]

    # Random translation (small, scaled)
    T = torch.randn(B, 1, 3, device=device) * 0.5

    # Apply: x_aug = x @ R^T + T
    x_aug = torch.bmm(x, R.transpose(1, 2)) + T

    return x_aug
```

**Usage in training**:
```python
if args.augment:
    x0 = random_rigid_augment(batch['coords'])
else:
    x0 = batch['coords']
x_t, target = noiser.add_noise(x0, t, ...)
```

#### 4.3 Configurable Inference Steps

```python
parser.add_argument("--inference_steps", type=int, default=None,
                    help="Steps for sampling (default: same as T)")

# In sampling:
n_steps = args.inference_steps or args.T
sampler = create_sampler(args.sampler, n_steps=n_steps)
```

---

### Phase 5: Testing Infrastructure (2-3 hours)

**Goal**: Easy A/B testing of diffusion configurations.

**New script**: `scripts/compare_samplers.py`

```python
"""Compare different sampler configurations on the same model."""

def compare_samplers(
    model_path: str,
    samplers: list[str],
    test_indices: list[int],
    n_samples: int = 3,
) -> dict:
    """Run same model with different samplers, report RMSE."""

    results = {}
    for sampler_name in samplers:
        sampler = create_sampler(sampler_name)
        rmses = []
        for idx in test_indices:
            for _ in range(n_samples):
                pred = sampler.sample(model, ...)
                rmse = compute_rmse(pred, gt)
                rmses.append(rmse)
        results[sampler_name] = {
            'mean': np.mean(rmses),
            'std': np.std(rmses),
        }

    return results
```

**Usage**:
```bash
python compare_samplers.py \
    --model outputs/resfold/best_model.pt \
    --samplers ddpm ddpm_kabsch heun \
    --n_samples 5
```

---

## Part 4: Implementation Schedule

| Phase | Tasks | Est. Time | Dependencies |
|-------|-------|-----------|--------------|
| **1** | Kabsch alignment in sampling | 2-3h | None |
| **2** | Sampler registry | 3-4h | Phase 1 |
| **3** | VE noise + EDM sampler | 4-5h | Phase 2 |
| **4** | Training improvements | 3-4h | Phase 3 |
| **5** | Testing infrastructure | 2-3h | Phase 2 |

**Total estimated time**: 14-19 hours

**Recommended execution order**:
1. **Phase 1** → Immediate drift fix, test on existing models
2. **Phase 2** → Clean architecture for experimentation
3. **Phase 5** → Testing tools (can run parallel with Phase 3)
4. **Phase 3** → AF3-style diffusion
5. **Phase 4** → Training improvements

---

## Part 5: Backward Compatibility Checklist

| Component | Current API | After Changes | Breaking? |
|-----------|-------------|---------------|-----------|
| `ddpm_sample()` | `ddpm_sample(model, ...)` | Same + optional `align_per_step` | No |
| `sample_centroids()` | `sample_centroids(model, ...)` | Same + optional `align_per_step` | No |
| `create_noiser()` | `create_noiser("gaussian", schedule)` | Same, + "ve" option | No |
| `create_schedule()` | `create_schedule("cosine", T=50)` | Same, + "karras" option | No |
| Training args | `--noise_type gaussian` | Same | No |
| **NEW** | - | `--sampler ddpm_kabsch` | Additive |
| **NEW** | - | `--augment` | Additive |
| **NEW** | - | `--inference_steps` | Additive |

All changes are **additive** - existing scripts and models work unchanged.

---

## Part 6: Verification Checklist

After each phase, verify:

### Phase 1 Verification
- [ ] `ddpm_sample()` produces coherent structures with alignment
- [ ] Step-by-step visualization shows structure preserved
- [ ] RMSE comparable or better than before
- [ ] No regression on existing test set

### Phase 2 Verification
- [ ] `create_sampler("ddpm")` matches old `ddpm_sample()` behavior
- [ ] `create_sampler("ddpm_kabsch")` produces better results
- [ ] Legacy `ddpm_sample()` wrapper still works

### Phase 3 Verification
- [ ] VE noiser produces valid noisy coordinates
- [ ] EDM sampler produces valid structures
- [ ] Can train with VE noise process

### Phase 4 Verification
- [ ] Loss weighting doesn't break training stability
- [ ] Augmentation improves test set generalization
- [ ] Inference steps configurable works

### Phase 5 Verification
- [ ] Comparison script runs without errors
- [ ] Results reproducible across runs

---

## Summary

**Critical insight from diffusion_todo.md**: The model can overfit training but produces garbage at inference due to **frame inconsistency** - the denoiser outputs x0 in a different rigid frame than x_t, causing the interpolation step to warp structures.

**Primary fix**: Add Kabsch alignment before each interpolation step (REQ-1 from todo).

**Architecture**: Registry-based swappable samplers enable easy A/B testing while maintaining backward compatibility.

**Recommendation**: Start with Phase 1 immediately - it's a small change that should dramatically improve sampling quality with no risk to existing functionality.
