"""Benchmark runner for evaluating models.

Orchestrates data loading, model inference, metrics computation, and results storage.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .data_loader import (
    BenchmarkSample,
    load_benchmark_sample,
    load_parquet_table,
    get_benchmark_indices,
)
from .metrics import BenchmarkMetrics, compute_all_metrics, aggregate_metrics
from .model_adapter import BaseAdapter, create_adapter


class BenchmarkRunner:
    """Runs benchmark evaluation for a model."""

    def __init__(
        self,
        adapter: BaseAdapter,
        data_path: str,
        split_path: Optional[str] = None,
        n_train: int = 5000,
        n_test: int = 1000,
        min_atoms: int = 100,
        max_atoms: int = 1600,
        compute_dockq: bool = True,
    ):
        """Initialize benchmark runner.

        Args:
            adapter: Model adapter for inference
            data_path: Path to samples.parquet
            split_path: Path to split.json (optional)
            n_train: Number of training samples
            n_test: Number of test samples
            min_atoms: Minimum atoms filter
            max_atoms: Maximum atoms filter
            compute_dockq: Whether to compute DockQ metrics
        """
        self.adapter = adapter
        self.data_path = data_path
        self.split_path = split_path
        self.n_train = n_train
        self.n_test = n_test
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.compute_dockq = compute_dockq

        self.table = None
        self.train_indices = []
        self.test_indices = []
        self.device = None

    def setup(self, device: torch.device) -> dict:
        """Load data and model.

        Args:
            device: Torch device for inference

        Returns:
            dict with setup metadata
        """
        self.device = device

        # Load data
        print(f"Loading data from {self.data_path}")
        self.table = load_parquet_table(self.data_path)
        print(f"Loaded {len(self.table)} samples")

        # Get indices
        self.train_indices, self.test_indices = get_benchmark_indices(
            self.table,
            split_path=self.split_path,
            n_train=self.n_train,
            n_test=self.n_test,
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
        )
        print(f"Train: {len(self.train_indices)}, Test: {len(self.test_indices)}")

        # Load model
        print(f"Loading model: {self.adapter.model_type}")
        model_meta = self.adapter.load(device)
        print(f"Model loaded: {model_meta}")

        return {
            "n_total_samples": len(self.table),
            "n_train": len(self.train_indices),
            "n_test": len(self.test_indices),
            **model_meta,
        }

    def evaluate_sample(self, idx: int) -> BenchmarkMetrics:
        """Evaluate a single sample.

        Args:
            idx: Sample index in table

        Returns:
            BenchmarkMetrics for the sample
        """
        # Load sample
        sample = load_benchmark_sample(self.table, idx)
        sample_tensors = sample.to_device(self.device)

        # Run inference
        result = self.adapter.predict_atoms(sample_tensors)

        # Compute metrics
        metrics = compute_all_metrics(
            pred_atoms=result.atoms_pred,
            gt_atoms=sample.gt_atoms,
            aa_seq=sample.aa_seq,
            chain_ids=sample.chain_ids,
            inference_time_ms=result.inference_time_ms,
            sample_id=sample.sample_id,
            compute_dockq_score=self.compute_dockq,
        )

        return metrics

    def evaluate_split(
        self,
        split: str,
        n_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> list[BenchmarkMetrics]:
        """Evaluate all samples in a split.

        Args:
            split: "train" or "test"
            n_samples: Limit number of samples (optional)
            show_progress: Show progress bar

        Returns:
            List of BenchmarkMetrics
        """
        if split == "train":
            indices = self.train_indices
        elif split == "test":
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        if n_samples is not None:
            indices = indices[:n_samples]

        metrics_list = []
        iterator = tqdm(indices, desc=f"Evaluating {split}") if show_progress else indices

        for idx in iterator:
            try:
                metrics = self.evaluate_sample(idx)
                metrics_list.append(metrics)
            except Exception as e:
                sample_id = self.table["sample_id"][idx].as_py()
                print(f"Error evaluating {sample_id}: {e}")
                continue

        return metrics_list

    def run(
        self,
        n_test: Optional[int] = None,
        n_train_eval: Optional[int] = None,
        show_progress: bool = True,
    ) -> dict:
        """Run full benchmark evaluation.

        Args:
            n_test: Number of test samples to evaluate (None = all)
            n_train_eval: Number of train samples to evaluate (None = skip)
            show_progress: Show progress bar

        Returns:
            dict with full benchmark results
        """
        results = {
            "benchmark_id": f"{self.adapter.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model": self.adapter.get_config(),
            "dataset": {
                "data_path": self.data_path,
                "split_path": self.split_path,
                "n_train": len(self.train_indices),
                "n_test": len(self.test_indices),
                "n_train_eval": n_train_eval,
                "n_test_eval": n_test,
            },
            "summary": {},
            "per_sample": [],
        }

        # Evaluate test set
        print("\n=== Test Set Evaluation ===")
        test_metrics = self.evaluate_split("test", n_samples=n_test, show_progress=show_progress)
        results["summary"]["test"] = aggregate_metrics(test_metrics)
        results["per_sample"].extend([{**m.to_dict(), "split": "test"} for m in test_metrics])

        # Evaluate train set (optional)
        if n_train_eval is not None and n_train_eval > 0:
            print("\n=== Train Set Evaluation ===")
            train_metrics = self.evaluate_split("train", n_samples=n_train_eval, show_progress=show_progress)
            results["summary"]["train"] = aggregate_metrics(train_metrics)
            results["per_sample"].extend([{**m.to_dict(), "split": "train"} for m in train_metrics])

        return results

    def save_results(self, results: dict, output_path: str) -> None:
        """Save results to JSON file.

        Args:
            results: Benchmark results dict
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {output_path}")


def print_summary(results: dict) -> None:
    """Print benchmark summary to console."""
    print("\n" + "=" * 60)
    print(f"Benchmark: {results['benchmark_id']}")
    print(f"Model: {results['model']['model_type']}")
    print("=" * 60)

    for split in ["test", "train"]:
        if split not in results["summary"]:
            continue

        summary = results["summary"][split]
        print(f"\n{split.upper()} ({summary['n_samples']} samples):")
        print(f"  RMSD (CA):      {summary['rmsd_ca']['mean']:.2f} +/- {summary['rmsd_ca']['std']:.2f}")
        print(f"  RMSD (all):     {summary['rmsd_all_atoms']['mean']:.2f} +/- {summary['rmsd_all_atoms']['std']:.2f}")
        print(f"  lDDT:           {summary['lddt']['mean']:.3f} +/- {summary['lddt']['std']:.3f}")
        print(f"  ilDDT:          {summary['ilddt']['mean']:.3f} +/- {summary['ilddt']['std']:.3f}")
        print(f"  Contact F1:     {summary['contact_f1']['mean']:.3f} +/- {summary['contact_f1']['std']:.3f}")

        if summary.get("dockq", {}).get("mean") is not None:
            print(f"  DockQ:          {summary['dockq']['mean']:.3f} +/- {summary['dockq']['std']:.3f}")
            print(f"    Acceptable:   {summary['dockq_acceptable_pct']:.1f}%")
            print(f"    Medium:       {summary['dockq_medium_pct']:.1f}%")
            print(f"    High:         {summary['dockq_high_pct']:.1f}%")

        print(f"  Inference time: {summary['inference_time_ms']['mean']:.1f} ms")

    print("=" * 60)
