"""Unified logging for TinyFold training scripts.

Provides consistent logging format across all training scripts with
dual output to console and file.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Unified logging for training scripts.

    Provides:
    - Dual output to console and file
    - Consistent formatting for training steps, evaluation, and configs
    - Line buffering for real-time log updates
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str | None = None,
        log_filename: str = "train.log",
    ):
        """Initialize logger.

        Args:
            output_dir: Base output directory
            run_name: Optional run name subdirectory
            log_filename: Name of the log file
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        if run_name:
            self.log_dir = self.output_dir / run_name
        else:
            self.log_dir = self.output_dir

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_filename
        self._file = open(self.log_path, "w", buffering=1)  # Line buffered
        self._console = True

    def log(self, msg: str = "", level: str = "INFO"):
        """Log message to console and file.

        Args:
            msg: Message to log
            level: Log level (currently unused, for future extension)
        """
        # flush=True so stdout is line-flushed even when redirected to a pipe
        # or file (non-TTY stdout is block-buffered by default, which makes
        # live monitoring of a backgrounded run show nothing until exit).
        print(msg, flush=True)
        self._file.write(msg + "\n")
        self._file.flush()

    def log_header(self, title: str, script_path: str | None = None):
        """Log standard header for training run.

        Args:
            title: Title for the training run
            script_path: Optional path to the script being run
        """
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if script_path:
            self.log(f"Script: {os.path.abspath(script_path)}")
        self.log(f"Command: python {' '.join(sys.argv)}")
        self.log("")

    def log_config(self, config: dict[str, Any], title: str = "Configuration"):
        """Log configuration in consistent format.

        Args:
            config: Configuration dictionary
            title: Section title
        """
        self.log(f"{title}:")
        max_key_len = max(len(str(k)) for k in config)
        for key, value in config.items():
            self.log(f"  {key:<{max_key_len}}: {value}")
        self.log("")

    def log_step(
        self,
        step: int,
        loss: float,
        aux_losses: dict[str, float] | None = None,
        lr: float | None = None,
        elapsed: float | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """Log training step with consistent format.

        Format: Step {step:6d} | loss: {loss:.4f} | [aux losses] | lr: {lr:.2e} | {elapsed}s

        Args:
            step: Training step number
            loss: Total loss value
            aux_losses: Optional dict of auxiliary loss components
            lr: Current learning rate
            elapsed: Time elapsed since start
            extra: Additional values to log
        """
        parts = [f"Step {step:6d}", f"loss: {loss:.4f}"]

        if aux_losses:
            aux_str = " | ".join(f"{k}: {v:.4f}" for k, v in aux_losses.items())
            parts.append(aux_str)

        if extra:
            extra_str = " | ".join(f"{k}: {v}" for k, v in extra.items())
            parts.append(extra_str)

        if lr is not None:
            parts.append(f"lr: {lr:.2e}")

        if elapsed is not None:
            parts.append(f"{elapsed:.0f}s")

        self.log(" | ".join(parts))

    def log_eval(
        self,
        step: int,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
        metric_name: str = "RMSE",
    ):
        """Log evaluation results consistently.

        Format: >>> Eval @ {step} | Train {metric}: {val} | Test {metric}: {val}

        Args:
            step: Training step number
            train_metrics: Dict with 'value' and optionally 'n_samples'
            test_metrics: Dict with 'value' and optionally 'n_samples'
            metric_name: Name of the primary metric
        """
        train_val = train_metrics.get("value", train_metrics.get(metric_name.lower(), 0))
        test_val = test_metrics.get("value", test_metrics.get(metric_name.lower(), 0))
        train_n = train_metrics.get("n_samples", "")
        test_n = test_metrics.get("n_samples", "")

        train_str = f"Train {metric_name}: {train_val:.4f}"
        if train_n:
            train_str += f" ({train_n})"

        test_str = f"Test {metric_name}: {test_val:.4f}"
        if test_n:
            test_str += f" ({test_n})"

        parts = [f">>> Eval @ {step}", train_str, test_str]

        # Add any additional metrics
        for key, val in test_metrics.items():
            if key not in ("value", "n_samples", metric_name.lower()):
                if isinstance(val, float):
                    parts.append(f"{key}: {val:.4f}")
                else:
                    parts.append(f"{key}: {val}")

        self.log("         " + " | ".join(parts))

    def log_footer(self, total_time: float, best_metric: float, metric_name: str = "RMSE"):
        """Log standard footer for training run.

        Args:
            total_time: Total training time in seconds
            best_metric: Best metric value achieved
            metric_name: Name of the metric
        """
        self.log("=" * 70)
        self.log("Training complete")
        self.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
        self.log(f"  Best test {metric_name}: {best_metric:.4f}")
        self.log("")
        self.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def close(self):
        """Close the log file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
