"""Checkpoint management for TinyFold training.

Provides utilities for saving, loading, and managing model checkpoints
with consistent format across all training scripts.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
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
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }

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
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
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

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

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
        output_dir: Union[str, Path],
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
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[Path]]:
        """Save checkpoint if it's a new best or meets recent criteria.

        Args:
            model: Model to save
            step: Current training step
            metrics: Current metrics dict
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            config: Optional config

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
            path = self.output_dir / f"best_model.pt"
            save_checkpoint(path, model, optimizer, scheduler, step, metrics, config)
            is_new_best = True

            # Update best list
            self.best_checkpoints.append((metric_value, step, path))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=not self.lower_is_better)
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]

        # Save recent checkpoint
        recent_path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        save_checkpoint(recent_path, model, optimizer, scheduler, step, metrics, config)

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

    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][2]
        return None

    def get_latest_path(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if self.recent_checkpoints:
            return self.recent_checkpoints[-1][1]
        return None
