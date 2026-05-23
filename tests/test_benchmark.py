#!/usr/bin/env python
"""Quick test of benchmark module imports."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

try:
    from benchmark import (
        BenchmarkSample,
        load_benchmark_sample,
        BenchmarkMetrics,
        compute_all_metrics,
        ModelAdapter,
        create_adapter,
        BenchmarkRunner,
    )
    print("All benchmark imports OK")

    from benchmark.model_adapter import list_adapters
    print(f"Available adapters: {list_adapters()}")

except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
