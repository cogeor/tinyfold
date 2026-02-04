#!/usr/bin/env python
"""CLI for TinyFold benchmark system.

Usage:
    # Evaluate a model
    python -m scripts.benchmark.cli evaluate \
        --model af3_style \
        --checkpoint outputs/af3_15M/best_model.pt \
        --output results/af3_benchmark.json

    # Compare multiple benchmark results
    python -m scripts.benchmark.cli compare \
        results/af3_benchmark.json \
        results/resfold_benchmark.json \
        --output results/comparison.csv

    # Quick evaluation (50 test samples)
    python -m scripts.benchmark.cli evaluate \
        --model af3_style \
        --checkpoint outputs/af3_15M/best_model.pt \
        --quick
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .model_adapter import create_adapter, list_adapters
from .runner import BenchmarkRunner, print_summary


def cmd_evaluate(args):
    """Run benchmark evaluation on a model."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create adapter
    adapter_kwargs = {}
    if args.noise_type:
        adapter_kwargs["noise_type"] = args.noise_type
    if args.n_timesteps:
        adapter_kwargs["n_timesteps"] = args.n_timesteps
    if args.clamp_val:
        adapter_kwargs["clamp_val"] = args.clamp_val
    if args.n_iter:
        adapter_kwargs["n_iter"] = args.n_iter

    adapter = create_adapter(args.model, args.checkpoint, **adapter_kwargs)

    # Create runner
    runner = BenchmarkRunner(
        adapter=adapter,
        data_path=args.data_path,
        split_path=args.split_path,
        n_train=args.n_train,
        n_test=args.n_test,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        compute_dockq=not args.no_dockq,
    )

    # Setup
    runner.setup(device)

    # Determine evaluation size
    n_test_eval = args.n_test_eval
    n_train_eval = args.n_train_eval

    if args.quick:
        n_test_eval = 50
        n_train_eval = 0

    # Run
    results = runner.run(
        n_test=n_test_eval,
        n_train_eval=n_train_eval,
        show_progress=True,
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        runner.save_results(results, args.output)


def cmd_compare(args):
    """Compare multiple benchmark results."""
    results_list = []

    for path in args.results:
        with open(path) as f:
            results_list.append(json.load(f))

    # Build comparison table
    rows = []
    for results in results_list:
        row = {
            "benchmark_id": results["benchmark_id"],
            "model_type": results["model"]["model_type"],
            "checkpoint": results["model"]["checkpoint_path"],
        }

        # Add test metrics
        test = results["summary"].get("test", {})
        if test:
            row["test_rmsd_ca"] = test.get("rmsd_ca", {}).get("mean")
            row["test_rmsd_all"] = test.get("rmsd_all_atoms", {}).get("mean")
            row["test_lddt"] = test.get("lddt", {}).get("mean")
            row["test_ilddt"] = test.get("ilddt", {}).get("mean")
            row["test_contact_f1"] = test.get("contact_f1", {}).get("mean")
            row["test_dockq"] = test.get("dockq", {}).get("mean")
            row["test_dockq_medium_pct"] = test.get("dockq_medium_pct")
            row["test_inference_ms"] = test.get("inference_time_ms", {}).get("mean")
            row["test_n_samples"] = test.get("n_samples")

        rows.append(row)

    # Print table
    print("\nComparison Summary:")
    print("-" * 120)
    header = ["model_type", "test_rmsd_ca", "test_lddt", "test_ilddt", "test_dockq", "test_dockq_medium_pct", "test_inference_ms"]
    print(f"{'model_type':<15} {'rmsd_ca':>10} {'lddt':>8} {'ilddt':>8} {'dockq':>8} {'medium%':>10} {'time_ms':>10}")
    print("-" * 120)
    for row in rows:
        print(
            f"{row.get('model_type', 'N/A'):<15} "
            f"{row.get('test_rmsd_ca', 'N/A'):>10.2f} " if row.get('test_rmsd_ca') else f"{'N/A':>10} "
            f"{row.get('test_lddt', 'N/A'):>8.3f} " if row.get('test_lddt') else f"{'N/A':>8} "
            f"{row.get('test_ilddt', 'N/A'):>8.3f} " if row.get('test_ilddt') else f"{'N/A':>8} "
            f"{row.get('test_dockq', 'N/A'):>8.3f} " if row.get('test_dockq') else f"{'N/A':>8} "
            f"{row.get('test_dockq_medium_pct', 'N/A'):>10.1f} " if row.get('test_dockq_medium_pct') else f"{'N/A':>10} "
            f"{row.get('test_inference_ms', 'N/A'):>10.1f}" if row.get('test_inference_ms') else f"{'N/A':>10}"
        )
    print("-" * 120)

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TinyFold Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_adapters(),
        help="Model type",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/<model>_benchmark.json)",
    )
    eval_parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/samples.parquet",
        help="Path to samples parquet",
    )
    eval_parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to split.json (optional)",
    )
    eval_parser.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="Number of training samples in split",
    )
    eval_parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="Number of test samples in split",
    )
    eval_parser.add_argument(
        "--n_test_eval",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)",
    )
    eval_parser.add_argument(
        "--n_train_eval",
        type=int,
        default=0,
        help="Number of train samples to evaluate (default: 0)",
    )
    eval_parser.add_argument(
        "--min_atoms",
        type=int,
        default=100,
        help="Min atoms filter",
    )
    eval_parser.add_argument(
        "--max_atoms",
        type=int,
        default=1600,
        help="Max atoms filter",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    eval_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation (50 test samples)",
    )
    eval_parser.add_argument(
        "--no_dockq",
        action="store_true",
        help="Skip DockQ computation (faster)",
    )

    # Model-specific arguments
    eval_parser.add_argument(
        "--noise_type",
        type=str,
        default=None,
        help="Noise type for af3_style (gaussian, linear_chain)",
    )
    eval_parser.add_argument(
        "--n_timesteps",
        type=int,
        default=None,
        help="Number of timesteps (auto-detected from checkpoint)",
    )
    eval_parser.add_argument(
        "--clamp_val",
        type=float,
        default=None,
        help="Clamp value for predictions",
    )
    eval_parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="Number of iterations for iterfold",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "results",
        nargs="+",
        help="Paths to benchmark result JSON files",
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        # Set default output path
        if args.output is None:
            args.output = f"results/{args.model}_benchmark.json"
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
