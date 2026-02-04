"""Unified benchmarking system for TinyFold models.

Provides standardized evaluation across all model architectures:
- af3_style: Atom-level diffusion
- resfold: Two-stage (residue -> atom)
- resfold_e2e: End-to-end with multi-sampling
- iterfold: Anchor-conditioned prediction

Usage:
    python -m scripts.benchmark.cli evaluate --model af3_style --checkpoint path/to/model.pt
    python -m scripts.benchmark.cli compare results/a.json results/b.json
"""

from .data_loader import BenchmarkSample, load_benchmark_sample
from .metrics import BenchmarkMetrics, compute_all_metrics
from .model_adapter import ModelAdapter, create_adapter
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkSample",
    "load_benchmark_sample",
    "BenchmarkMetrics",
    "compute_all_metrics",
    "ModelAdapter",
    "create_adapter",
    "BenchmarkRunner",
]
