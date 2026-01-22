"""Metric tracking for TinyFold training.

Provides structured tracking of loss components and evaluation metrics
across training runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class LossComponents:
    """Structured loss output for any model.

    Provides a consistent way to return loss components from training steps.

    Example:
        >>> loss = LossComponents(
        ...     total=0.5,
        ...     main=0.4,
        ...     auxiliary={"dist": 0.05, "geom": 0.05}
        ... )
        >>> loss.to_dict()
        {"total": 0.5, "main": 0.4, "dist": 0.05, "geom": 0.05}
    """

    total: float  # Used for backward()
    main: float   # Primary loss (e.g., MSE)
    auxiliary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        return {"total": self.total, "main": self.main, **self.auxiliary}

    def aux_string(self, sep: str = " | ") -> str:
        """Format auxiliary losses as string."""
        if not self.auxiliary:
            return ""
        return sep.join(f"{k}: {v:.4f}" for k, v in self.auxiliary.items())


class MetricTracker:
    """Track and aggregate metrics during training.

    Stores step-wise metrics and provides aggregation utilities
    for logging and analysis.

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update(step=100, loss=0.5, aux={"mse": 0.4, "dist": 0.1})
        >>> tracker.update(step=200, loss=0.4, aux={"mse": 0.3, "dist": 0.1})
        >>> tracker.get_recent_avg("loss", last_n=2)
        0.45
    """

    def __init__(self):
        self.step_metrics: Dict[int, Dict[str, float]] = {}
        self.eval_metrics: Dict[int, Dict[str, Any]] = {}
        self._rolling: Dict[str, List[float]] = defaultdict(list)

    def update(
        self,
        step: int,
        loss: float,
        aux: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """Record step metrics.

        Args:
            step: Training step number
            loss: Total loss value
            aux: Optional auxiliary loss components
            lr: Optional learning rate
        """
        metrics = {"loss": loss}
        if aux:
            metrics.update(aux)
        if lr is not None:
            metrics["lr"] = lr

        self.step_metrics[step] = metrics

        # Update rolling buffers
        self._rolling["loss"].append(loss)
        if aux:
            for k, v in aux.items():
                self._rolling[k].append(v)

    def update_eval(
        self,
        step: int,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ):
        """Record evaluation metrics.

        Args:
            step: Training step number
            train_metrics: Training set evaluation results
            test_metrics: Test set evaluation results
        """
        self.eval_metrics[step] = {
            "train": train_metrics,
            "test": test_metrics,
        }

    def get_recent_avg(self, metric: str, last_n: int = 100) -> float:
        """Get average of recent values for a metric.

        Args:
            metric: Metric name
            last_n: Number of recent values to average

        Returns:
            Average value
        """
        values = self._rolling.get(metric, [])
        if not values:
            return 0.0
        recent = values[-last_n:]
        return sum(recent) / len(recent)

    def get_best_eval(self, metric: str = "rmse", mode: str = "test") -> tuple[int, float]:
        """Get step and value of best evaluation metric.

        Args:
            metric: Metric name to check
            mode: "train" or "test"

        Returns:
            (step, value) tuple
        """
        best_step = -1
        best_value = float("inf")

        for step, data in self.eval_metrics.items():
            metrics = data.get(mode, {})
            if metric in metrics:
                value = metrics[metric]
                if value < best_value:
                    best_value = value
                    best_step = step

        return best_step, best_value

    def get_last_step(self) -> int:
        """Get the most recent step number."""
        if not self.step_metrics:
            return 0
        return max(self.step_metrics.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary."""
        return {
            "step_metrics": self.step_metrics,
            "eval_metrics": self.eval_metrics,
        }

    def clear_rolling(self):
        """Clear rolling buffers to save memory."""
        self._rolling.clear()
