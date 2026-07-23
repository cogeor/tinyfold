"""Checkpoint management for TinyFold training.

Provides utilities for saving, loading, and managing model checkpoints
with consistent format across all training scripts.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMA:
    """Exponential moving average of model parameters (C3).

    Keeps a shadow copy of the model's floating-point parameters, updated after
    each optimizer step as ``shadow = decay*shadow + (1-decay)*param``. Eval and
    checkpointing can then read the smoothed weights -- universal in AF2/AF3/Boltz
    (Boltz-1 inits its confidence model from EMA trunk weights). Off when
    ``decay <= 0``; the training loop simply never constructs one.

    Only float params are averaged; integer buffers/params (if any) are tracked
    by copy so a produced state dict still round-trips through
    ``load_state_dict``. Buffers are not averaged by default -- this model uses
    LayerNorm (no running stats), so there are none that matter.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            s = self.shadow[n]
            if p.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                s.copy_(p.detach())

    def state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Full model state dict with EMA params overlaid (buffers kept as-is),
        so it loads cleanly via ``model.load_state_dict``."""
        sd = model.state_dict()
        for n, v in self.shadow.items():
            sd[n] = v.detach().clone()
        return sd

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Snapshot the live params so they can be restored after eval."""
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the live params with the EMA shadow (for eval/checkpoint)."""
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Undo :meth:`copy_to`, restoring the params snapshotted by :meth:`store`."""
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            p.data.copy_(self._backup[n])
        self._backup = None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
):
    """Save training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        step: Current training step
        metrics: Optional metrics dict (e.g., {"train_rmse": 0.5, "test_rmse": 0.6})
        config: Optional training configuration
        extra: Optional extra data to save
        ema_state_dict: Optional EMA weights, saved under ``ema_state_dict``.
            ``model_state_dict`` always stays the raw weights so existing loaders
            keep working.
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    if extra is not None:
        checkpoint.update(extra)

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to (default: model's current device)
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint dict with metadata (step, metrics, config, etc.)
    """
    if device is None:
        device = next(model.parameters()).device

    # weights_only=True: checkpoints hold only tensors + plain dicts/numbers
    # (state dicts, metrics, config), so this is safe and blocks code execution
    # from an untrusted .pt.
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Load model weights
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if missing:
        logger.warning("Missing keys when loading checkpoint: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))

    # Restore optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return metadata
    return {
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


class CheckpointManager:
    """Manage multiple checkpoints with automatic cleanup.

    Keeps track of best checkpoints and recent checkpoints,
    automatically removing old ones to save disk space.

    Example:
        >>> manager = CheckpointManager(output_dir, keep_best=3, keep_recent=2)
        >>> manager.save(model, step=1000, metrics={"test_rmse": 0.5})
        >>> manager.save(model, step=2000, metrics={"test_rmse": 0.4})  # New best
    """

    def __init__(
        self,
        output_dir: str | Path,
        keep_best: int = 1,
        keep_recent: int = 2,
        metric_name: str = "test_rmse",
        lower_is_better: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            keep_best: Number of best checkpoints to keep
            keep_recent: Number of recent checkpoints to keep
            metric_name: Metric to use for determining "best"
            lower_is_better: Whether lower metric values are better
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_recent = keep_recent
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better

        # Track checkpoints
        self.best_checkpoints: list[tuple[float, int, Path]] = []  # (metric, step, path)
        self.recent_checkpoints: list[tuple[int, Path]] = []  # (step, path)

    def save(
        self,
        model: nn.Module,
        step: int,
        metrics: dict[str, float],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        config: dict[str, Any] | None = None,
        ema_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[bool, Path | None]:
        """Save checkpoint if it's a new best or meets recent criteria.

        Args:
            model: Model to save
            step: Current training step
            metrics: Current metrics dict
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            config: Optional config
            ema_state_dict: Optional EMA weights forwarded to save_checkpoint.

        Returns:
            (is_new_best, path) - whether this is a new best, and the saved path
        """
        metric_value = metrics.get(self.metric_name, float("inf"))

        # Check if this is a new best
        is_new_best = False
        if self.lower_is_better:
            is_best = all(metric_value < m for m, _, _ in self.best_checkpoints) or len(self.best_checkpoints) < self.keep_best
        else:
            is_best = all(metric_value > m for m, _, _ in self.best_checkpoints) or len(self.best_checkpoints) < self.keep_best

        # Save best checkpoint
        if is_best:
            path = self.output_dir / "best_model.pt"
            save_checkpoint(path, model, optimizer, scheduler, step, metrics, config,
                            ema_state_dict=ema_state_dict)
            is_new_best = True

            # Update best list
            self.best_checkpoints.append((metric_value, step, path))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=not self.lower_is_better)
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]

        # Save recent checkpoint
        recent_path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        save_checkpoint(recent_path, model, optimizer, scheduler, step, metrics, config,
                        ema_state_dict=ema_state_dict)

        # Update recent list and cleanup
        self.recent_checkpoints.append((step, recent_path))
        self._cleanup_recent()

        return is_new_best, recent_path

    def _cleanup_recent(self):
        """Remove old recent checkpoints."""
        while len(self.recent_checkpoints) > self.keep_recent:
            _, old_path = self.recent_checkpoints.pop(0)
            if old_path.exists() and old_path.name != "best_model.pt":
                old_path.unlink()

    def get_best_path(self) -> Path | None:
        """Get path to best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][2]
        return None

    def get_latest_path(self) -> Path | None:
        """Get path to most recent checkpoint."""
        if self.recent_checkpoints:
            return self.recent_checkpoints[-1][1]
        return None
