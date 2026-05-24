# TinyFold Codebase Refactoring Plan

> **Goal**: Clean up the codebase, homogenize model training, improve abstraction for extensibility, and establish production-quality foundations.

---

## Executive Summary

This plan addresses the current issues with code duplication across training scripts, inconsistent model handling, scattered utilities, and lack of unified training infrastructure. The refactoring will:

1. **Unify training infrastructure** with reusable primitives
2. **Type all models** with proper protocol/interface definitions
3. **Standardize logging and metrics** across all training scripts
4. **Add deterministic run naming** with timestamps
5. **Organize test files** into dedicated directories
6. **Treat data layer as production code** with proper abstractions
7. **Keep separate training scripts per model** (no if-statement soup)

---

## Current State Analysis

### Directory Structure Issues

```
scripts/
├── train.py               # 735 lines - AF3/attention_v2 training
├── train_resfold.py       # 1163 lines - ResFold Stage 1/2 training
├── train_resfold_e2e.py   # 800+ lines - ResFold end-to-end
├── train_stage1.py        # Older stage1 script
├── train_stage1_clean.py  # Another stage1 variant
├── models/                # 13 model files (28KB af3_style, 38KB geometry_losses)
│   ├── af3_style.py
│   ├── resfold.py
│   ├── diffusion.py
│   └── ...
├── test_*.py              # Test scripts mixed with training
src/tinyfold/
├── data/                  # Well-organized data layer
├── model/                 # Partial model layer (different from scripts/models)
└── viz/                   # Visualization
tests/                     # Some pytest tests (8 files)
```

### Key Problems Identified

| Issue | Current State | Impact |
|-------|--------------|--------|
| **Duplicated Logger class** | Copied in `train.py`, `train_resfold.py`, `train_resfold_e2e.py` | Maintenance burden |
| **Duplicated loss functions** | `kabsch_align`, `compute_mse_loss`, `compute_rmse` copied everywhere | Inconsistency risk |
| **Duplicated data loading** | `load_sample_raw`, `collate_batch` duplicated with slight variations | Bug duplication |
| **Inconsistent eval/logging** | Different log formats, metric names across scripts | Hard to compare runs |
| **No typing for models** | No Protocol/ABC defining model interface | Hard to add new models |
| **Mixed test locations** | Some in `tests/`, some in `scripts/test_*.py` | Confusing structure |
| **Static run naming** | Output dir must be specified manually | Overwrite risk |
| **Scattered diffusion logic** | Noising, scheduling, sampling spread across files | Hard to extend |

---

## Proposed Architecture

### New Directory Structure

```
src/tinyfold/
├── __init__.py
├── constants.py
├── types.py                    # [NEW] Protocol definitions for models
├── data/
│   ├── __init__.py
│   ├── cache.py               # Parquet I/O
│   ├── collate.py             # Collation (already exists)
│   ├── datasets/
│   ├── parsing/
│   ├── processing/
│   ├── sources/
│   └── split.py               # [NEW] Consolidate data_split.py
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── registry.py            # [NEW] Model factory/registry
│   ├── base/
│   │   ├── __init__.py
│   │   ├── decoder.py         # [MOVE] BaseDecoder from scripts/models/base.py
│   │   └── encoder.py         # [NEW] BaseEncoder protocol
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── noise.py           # [MOVE] Noiser classes from scripts/models/diffusion.py
│   │   ├── schedule.py        # [MOVE] Schedule classes
│   │   ├── sampler.py         # [NEW] DDPM sampling abstraction
│   │   └── curriculum.py      # [MOVE] TimestepCurriculum
│   ├── attention_v2/          # [NEW] Model-specific directory
│   │   ├── __init__.py
│   │   └── model.py           # [MOVE] from scripts/models/attention_v2.py
│   ├── af3_style/             # [NEW] Model-specific directory
│   │   ├── __init__.py
│   │   ├── trunk.py
│   │   └── denoiser.py
│   ├── resfold/               # [NEW] Model-specific directory
│   │   ├── __init__.py
│   │   ├── encoder.py         # ResidueEncoder
│   │   ├── denoiser.py        # DiffusionTransformer
│   │   ├── refiner.py         # AtomRefiner
│   │   └── pipeline.py        # ResFoldPipeline
│   └── losses/
│       ├── __init__.py
│       ├── mse.py             # [NEW] Kabsch + MSE losses
│       ├── geometry.py        # [MOVE] from scripts/models/geometry_losses.py
│       ├── contact.py         # [MOVE] ContactLoss
│       └── distance.py        # [NEW] Distance consistency loss
├── training/                  # [NEW] Training infrastructure
│   ├── __init__.py
│   ├── logger.py              # Unified Logger class
│   ├── metrics.py             # Metric tracking (train/test RMSE, losses)
│   ├── checkpointing.py       # Save/load/resume logic
│   ├── run_naming.py          # Deterministic run names with timestamps
│   ├── trainer.py             # Base Trainer class
│   └── callbacks.py           # Eval, plotting callbacks
└── viz/ (existing)

scripts/                       # Thin entry points only
├── train_resfold.py          # [REFACTOR] ResFold-specific training
├── train_af3.py              # [REFACTOR] AF3-style training
├── train_attention.py        # [REFACTOR] AttentionV2 training
├── prepare_data.py           # (keep as-is)
├── predict.py                # (keep as-is)
└── visualize/                # [MOVE] visualization scripts
    ├── visualize_diffusion.py
    ├── visualize_preds.py
    └── visualize_structure.py

tests/
├── conftest.py
├── fixtures/
├── unit/
│   ├── test_losses.py
│   ├── test_diffusion.py
│   └── test_models.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   └── test_training.py
└── test_bucketing.py          # (existing)
```

---

## Component Specifications

### 1. Model Typing System

Create `src/tinyfold/types.py` with Protocol definitions:

```python
from typing import Protocol, Optional, Dict, Any
from torch import Tensor
import torch.nn as nn

class DiffusionDecoder(Protocol):
    """All diffusion decoders must implement this interface."""
    
    def forward(
        self,
        x_t: Tensor,           # Noisy input [B, N, 3]
        t: Tensor,             # Timesteps [B]
        conditioning: Dict[str, Tensor],  # Model-specific conditioning
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict clean x0 from noisy x_t at timestep t."""
        ...

class ResidueEncoder(Protocol):
    """Encoder that produces per-residue representations."""
    
    def forward(
        self,
        aa_seq: Tensor,        # [B, L]
        chain_ids: Tensor,     # [B, L]
        positions: Tensor,     # [B, L, 3]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode residue features to token embeddings."""
        ...

class StructurePredictor(Protocol):
    """Full structure prediction model (any architecture)."""
    
    def predict(
        self,
        batch: Dict[str, Tensor],
        noiser: Any,
        n_steps: int = 50,
    ) -> Dict[str, Tensor]:
        """Run full inference to predict structure."""
        ...
```

### 2. Unified Training Infrastructure

Create `src/tinyfold/training/` module:

#### `logger.py`
```python
class TrainingLogger:
    """Unified logging for training scripts."""
    
    def __init__(self, output_dir: Path, run_name: str):
        self.log_path = output_dir / run_name / "train.log"
        self.console = True
        self._file = open(self.log_path, 'w', buffering=1)
        
    def log(self, msg: str = "", level: str = "INFO"):
        """Log to console and file with timestamp."""
        
    def log_config(self, config: dict):
        """Log configuration in consistent format."""
        
    def log_step(self, step: int, metrics: Dict[str, float], aux_losses: Dict[str, float] = None):
        """Log training step with consistent format for all models."""
        # Format: Step {step:5d} | loss: {total:.4f} | mse: {mse:.4f} | [aux losses] | lr: {lr:.2e} | {time}s
        
    def log_eval(self, step: int, train_metrics: Dict, test_metrics: Dict):
        """Log evaluation results consistently."""
```

#### `run_naming.py`
```python
from datetime import datetime

def generate_run_name(
    model_name: str,
    config: dict,
    timestamp: bool = True,
) -> str:
    """Generate deterministic, descriptive run name.
    
    Format: {model}_{key_params}_{YYYYMMDD_HHMMSS}
    
    Examples:
        - resfold_s1_20K_20260122_220043
        - af3_128h6L_gs50_20260122_220043
    """
    # Extract key params (n_train, n_layers, etc.)
    params = extract_key_params(model_name, config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
    return f"{model_name}_{params}_{ts}"
```

#### `metrics.py`
```python
@dataclass
class MetricTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.step_metrics: Dict[int, Dict[str, float]] = {}
        self.aux_losses: Dict[int, Dict[str, float]] = {}
        
    def update(self, step: int, loss: float, aux: Dict[str, float] = None):
        """Record step metrics."""
        
    def get_eval_metrics(self) -> Dict[str, float]:
        """Get metrics for logging."""
```

### 3. Diffusion Abstraction Layer

Create `src/tinyfold/model/diffusion/sampler.py`:

```python
class DDPMSampler:
    """Unified DDPM sampling for all diffusion models."""
    
    def __init__(
        self,
        noiser: BaseNoiser,
        clamp_val: float = 3.0,
    ):
        self.noiser = noiser
        self.clamp_val = clamp_val
        
    @torch.no_grad()
    def sample(
        self,
        model: DiffusionDecoder,
        conditioning: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
        progress_callback: Callable = None,
    ) -> Tensor:
        """Run full DDPM sampling loop.
        
        Works with any model implementing DiffusionDecoder protocol.
        """
        B, L = conditioning['shape']
        device = conditioning['device']
        
        x = self._initialize(B, L, device)
        
        for t in reversed(range(self.noiser.T)):
            x = self._step(model, x, t, conditioning, mask)
            if progress_callback:
                progress_callback(t, x)
                
        return x
```

### 4. Loss Module Consolidation

Move all losses to `src/tinyfold/model/losses/`:

```python
# mse.py
def kabsch_align(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
    """Kabsch alignment for rotation-invariant comparison."""

def compute_mse_loss(pred: Tensor, target: Tensor, mask: Tensor = None, use_kabsch: bool = True) -> Tensor:
    """MSE loss with optional Kabsch alignment."""

def compute_rmse(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    """RMSE after Kabsch alignment (for evaluation)."""

# distance.py
def compute_distance_consistency_loss(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    """Loss for preserving pairwise residue distances."""
```

### 5. Per-Model Training Scripts

Each model gets its own training script that:
- Uses shared training infrastructure
- Has model-specific forward pass logic
- Maintains separation of concerns

#### Example: `scripts/train_resfold.py` (Refactored)

```python
#!/usr/bin/env python
"""ResFold training script - Two-stage residue diffusion + atom refinement."""

from tinyfold.training import TrainingLogger, generate_run_name, MetricTracker
from tinyfold.training.callbacks import EvalCallback, PlotCallback
from tinyfold.model.resfold import ResFoldPipeline
from tinyfold.model.diffusion import DDPMSampler, create_noiser
from tinyfold.model.losses import compute_mse_loss, compute_distance_consistency_loss
from tinyfold.data import load_dataset, create_sampler

def train_stage1(config: ResFoldConfig) -> None:
    """Train Stage 1 (residue diffusion) only."""
    # Setup (10-20 lines)
    run_name = generate_run_name("resfold_s1", vars(config))
    logger = TrainingLogger(config.output_dir, run_name)
    
    # Model creation (5 lines)
    model = ResFoldPipeline.from_config(config).to(device)
    model.set_training_mode("stage1_only")
    
    # Training loop (50 lines vs 300+ today)
    for step in range(config.n_steps):
        batch = next(train_loader)
        loss, aux = train_step_stage1(model, batch, noiser)
        optimizer_step(optimizer, loss)
        logger.log_step(step, loss, aux)
```

---

## Implementation Plan

### Phase 1: Foundation (No Breaking Changes) ✅ Safe to Merge

| Task | Files | Effort |
|------|-------|--------|
| 1.1 Create `src/tinyfold/types.py` | New file | 1h |
| 1.2 Create `src/tinyfold/training/` module | New directory | 2h |
| 1.3 Implement `TrainingLogger` with consistent formatting | `training/logger.py` | 1h |
| 1.4 Implement `generate_run_name()` | `training/run_naming.py` | 0.5h |
| 1.5 Implement `MetricTracker` | `training/metrics.py` | 1h |

### Phase 2: Loss Consolidation ✅ Safe to Merge

| Task | Files | Effort |
|------|-------|--------|
| 2.1 Move/consolidate losses to `src/tinyfold/model/losses/` | Multiple | 2h |
| 2.2 Create `mse.py` with Kabsch + MSE | New file | 1h |
| 2.3 Move `geometry_losses.py` | Move from scripts/models | 0.5h |
| 2.4 Update imports in training scripts (no behavior change) | train*.py | 1h |

### Phase 3: Diffusion Abstraction ✅ Safe to Merge

| Task | Files | Effort |
|------|-------|--------|
| 3.1 Create `DDPMSampler` class | `model/diffusion/sampler.py` | 2h |
| 3.2 Consolidate noise types | `model/diffusion/noise.py` | 1h |
| 3.3 Consolidate schedules | `model/diffusion/schedule.py` | 0.5h |
| 3.4 Add curriculum as optional wrapper | `model/diffusion/curriculum.py` | 0.5h |

### Phase 4: Model Organization ✅ Safe to Merge

| Task | Files | Effort |
|------|-------|--------|
| 4.1 Move `scripts/models/` → `src/tinyfold/model/` | Multiple | 2h |
| 4.2 Create model-specific directories (resfold/, af3_style/, etc.) | New dirs | 1h |
| 4.3 Implement model registry/factory | `model/registry.py` | 1h |
| 4.4 Type all models with Protocol | All model files | 2h |

### Phase 5: Training Script Refactoring ⚠️ Behavior Changes

| Task | Files | Effort |
|------|-------|--------|
| 5.1 Refactor `train_resfold.py` to use new infrastructure | `scripts/train_resfold.py` | 3h |
| 5.2 Create `train_af3.py` from current `train.py` variations | New file | 2h |
| 5.3 Create `train_attention.py` for AttentionV2 | New file | 1h |
| 5.4 Add deterministic run naming to all scripts | All train*.py | 1h |

### Phase 6: Test Organization

| Task | Files | Effort |
|------|-------|--------|
| 6.1 Move `scripts/test_*.py` → `tests/unit/` | Multiple | 1h |
| 6.2 Organize into unit/integration structure | `tests/` | 1h |
| 6.3 Add tests for new training utilities | `tests/unit/test_training.py` | 2h |

### Phase 7: Data Layer Enhancement

| Task | Files | Effort |
|------|-------|--------|
| 7.1 Move `data_split.py` → `src/tinyfold/data/split.py` | Move file | 0.5h |
| 7.2 Add proper typing to data functions | `data/*.py` | 1h |
| 7.3 Create DataLoader wrappers for training | `data/loaders.py` | 2h |

---

## Eval/Logging Standardization

### Current Inconsistent Format

```
# train.py (AF3)
Step {step:5d} | loss: {loss:.6f} | lr: {lr:.2e} | {elapsed:.0f}s

# train_resfold.py (Stage 1)
Step {step:5d} | loss: {loss:.6f} | mse: {mse:.4f} | dst: {dist:.4f} | lr: {lr:.2e} | {elapsed:.0f}s

# train_resfold.py (Stage 2)
Step {step:5d} | loss: {loss:.4f} | mse: {mse:.4f} | geom: {geom:.4f} (bnd:{bond:.3f} ang:{angle:.3f} omg:{omega:.3f}) | lr: {lr:.2e} | {elapsed:.0f}s
```

### Proposed Unified Format

```
# All models use same format with optional aux_losses dict
Step {step:6d} | loss: {total:.4f} | main: {main:.4f} | aux: [{key}:{val:.3f}, ...] | lr: {lr:.2e} | {elapsed}s

# Eval format (also unified)
>>> Eval @ {step} | Train RMSE: {train:.4f}Å ({n_train}) | Test RMSE: {test:.4f}Å ({n_test})
```

### Auxiliary Loss Handling

```python
@dataclass
class LossComponents:
    """Structured loss output for any model."""
    total: float                    # Used for backward()
    main: float                     # Primary MSE loss
    auxiliary: Dict[str, float]     # Model-specific aux losses
    
    def to_dict(self) -> Dict[str, float]:
        return {"total": self.total, "main": self.main, **self.auxiliary}
```

---

## Run Naming Convention

### Format
```
{model}_{mode}_{n_train}_{timestamp}
```

### Examples
```
resfold_s1_20K_20260122_220043/
    train.log
    best_model.pt
    plots/
    config.json

af3_std_5K_20260123_140512/
    train.log
    best_model.pt
    plots/
    config.json
```

### Implementation

```python
def generate_run_name(model: str, config: dict) -> str:
    mode = config.get('mode', 'std')  # s1, s2, e2e, std
    n_train = config.get('n_train', 0)
    n_train_str = f"{n_train // 1000}K" if n_train >= 1000 else str(n_train)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{mode}_{n_train_str}_{ts}"
```

---

## Migration Notes

### Backward Compatibility

- All existing training scripts continue to work during migration
- New infrastructure is additive; existing code uses it via imports
- Models maintain same forward() signatures
- Checkpoints remain compatible (state_dict format unchanged)

### Testing Strategy

1. **Before refactoring**: Run existing training for 1000 steps, save metrics
2. **After refactoring**: Run same config, verify metrics match within 1%
3. **Add regression tests**: Automated comparison on small dataset

### Files to Delete After Migration

```
scripts/train_stage1.py       # Superseded by train_resfold.py --mode stage1_only
scripts/train_stage1_clean.py # Superseded
scripts/test_dihedral*.py     # Move to tests/
scripts/test_omega*.py        # Move to tests/
```

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Training script LOC (per model) | 700-1200 | 200-300 |
| Duplicated utility code | ~500 lines | 0 lines |
| Model definition locations | 2 (scripts/models, src/tinyfold/model) | 1 |
| Test file locations | 2 (scripts/, tests/) | 1 |
| Log format variations | 3+ | 1 |
| Type annotations for models | Minimal | Full Protocol coverage |
| Run name determinism | Manual | Automatic with timestamp |

---

## Next Steps

1. **Review this plan** - Check if proposed structure meets your needs
2. **Prioritize phases** - What should we tackle first?
3. **Implementation** - Start with Phase 1 (foundation) as it's risk-free
