"""Distribution and histogram plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_distribution(
    values: list[float],
    metric_name: str = "RMSD",
    unit: str = "Å",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (6, 4),
    highlight_idx: int | None = None,
) -> plt.Figure:
    """Plot distribution of a metric across multiple samples.

    Args:
        values: List of metric values (one per sample)
        metric_name: Name of the metric
        unit: Unit string
        out_path: Optional save path
        figsize: Figure size
        highlight_idx: Index of sample to highlight (e.g., best)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_samples = len(values)
    x = np.arange(n_samples)

    bars = ax.bar(x, values, color="#3B82F6", alpha=0.7)

    if highlight_idx is not None and 0 <= highlight_idx < n_samples:
        bars[highlight_idx].set_color("#22C55E")
        bars[highlight_idx].set_alpha(1.0)

    ax.set_xlabel("Sample")
    ax.set_ylabel(f"{metric_name} ({unit})")
    ax.set_title(f"{metric_name} Distribution (n={n_samples})")

    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(mean_val, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.legend()

    # Add value labels on bars if not too many
    if n_samples <= 20:
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rmsd_comparison(
    rmsd_dict: dict[str, float],
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """Plot bar chart comparing different RMSD metrics.

    Args:
        rmsd_dict: Dict with keys like 'rmsd_complex', 'lrmsd', 'irmsd'
        out_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out NaN values
    filtered = {k: v for k, v in rmsd_dict.items() if not np.isnan(v)}

    names = list(filtered.keys())
    values = list(filtered.values())

    # Pretty names
    pretty_names = {
        "rmsd_complex": "Complex",
        "rmsd_chain_a": "Chain A",
        "rmsd_chain_b": "Chain B",
        "lrmsd": "LRMSD",
        "irmsd": "iRMSD",
    }
    labels = [pretty_names.get(n, n) for n in names]

    colors = ["#3B82F6", "#60A5FA", "#93C5FD", "#F97316", "#22C55E"]
    bars = ax.bar(labels, values, color=colors[: len(labels)])

    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD Metrics")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_contact_precision_recall(
    metrics: dict[str, float],
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (5, 4),
) -> plt.Figure:
    """Plot precision/recall/F1 as bar chart.

    Args:
        metrics: Dict with 'precision', 'recall', 'f1'
        out_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = ["Precision", "Recall", "F1"]
    values = [metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("f1", 0)]
    colors = ["#3B82F6", "#F97316", "#22C55E"]

    bars = ax.bar(names, values, color=colors)

    ax.set_ylabel("Score")
    ax.set_title("Contact Prediction Metrics")
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
