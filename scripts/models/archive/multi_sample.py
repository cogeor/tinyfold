"""Multi-sampling for improved inference.

Instead of sampling once from noise, sample K times and aggregate.
This reduces variance and can improve prediction quality.

Aggregation strategies:
- mean: Simple average of all samples
- median: Per-coordinate median (robust to outliers)
- trimmed_mean: Discard outliers, average the rest
- consensus: Only average predictions that are close to majority

Usage:
    from models.multi_sample import MultiSampler, sample_with_consensus

    # Wrap any existing sampler
    multi_sampler = MultiSampler(base_sampler, n_samples=5, aggregation="consensus")
    result = multi_sampler.sample(model, batch, noiser, device)
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Literal

from .diffusion import kabsch_align_to_target


AggregationMethod = Literal["mean", "median", "trimmed_mean", "consensus"]


def aggregate_samples(
    samples: Tensor,
    method: AggregationMethod = "mean",
    mask: Optional[Tensor] = None,
    trim_fraction: float = 0.2,
    consensus_threshold: float = 1.0,
) -> Tensor:
    """Aggregate multiple samples into a single prediction.

    Args:
        samples: [K, B, L, 3] tensor of K samples
        method: Aggregation method
        mask: Optional [B, L] boolean mask
        trim_fraction: Fraction to trim from each end for trimmed_mean
        consensus_threshold: Distance threshold for consensus (in normalized units)

    Returns:
        aggregated: [B, L, 3] aggregated prediction
    """
    K, B, L, _ = samples.shape

    if method == "mean":
        return samples.mean(dim=0)

    elif method == "median":
        return samples.median(dim=0).values

    elif method == "trimmed_mean":
        # Compute distance from mean for each sample
        mean = samples.mean(dim=0, keepdim=True)  # [1, B, L, 3]
        distances = ((samples - mean) ** 2).sum(dim=-1).sqrt()  # [K, B, L]

        # Average distance per sample
        if mask is not None:
            mask_exp = mask.unsqueeze(0).float()  # [1, B, L]
            avg_dist = (distances * mask_exp).sum(dim=-1) / mask_exp.sum(dim=-1).clamp(min=1)  # [K, B]
        else:
            avg_dist = distances.mean(dim=-1)  # [K, B]

        # For each batch element, sort samples by distance and trim
        n_trim = max(1, int(K * trim_fraction))
        n_keep = K - 2 * n_trim
        if n_keep < 1:
            n_keep = 1
            n_trim = (K - 1) // 2

        # Sort and select middle samples
        sorted_indices = avg_dist.argsort(dim=0)  # [K, B]

        result = torch.zeros(B, L, 3, device=samples.device)
        for b in range(B):
            keep_indices = sorted_indices[n_trim:n_trim + n_keep, b]  # [n_keep]
            result[b] = samples[keep_indices, b].mean(dim=0)

        return result

    elif method == "consensus":
        return _consensus_aggregation(samples, mask, consensus_threshold)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _consensus_aggregation(
    samples: Tensor,
    mask: Optional[Tensor],
    threshold: float,
) -> Tensor:
    """Consensus-based aggregation: only average predictions close to majority.

    For each residue:
    1. Compute pairwise distances between all K predictions
    2. Find the prediction closest to most others (reference)
    3. Only average predictions within threshold of reference

    Args:
        samples: [K, B, L, 3] tensor of K samples
        mask: Optional [B, L] boolean mask
        threshold: Distance threshold for including in consensus

    Returns:
        aggregated: [B, L, 3] consensus prediction
    """
    K, B, L, _ = samples.shape
    device = samples.device

    # Compute per-residue centroid predictions
    # samples: [K, B, L, 3]

    result = torch.zeros(B, L, 3, device=device)

    for b in range(B):
        for l in range(L):
            if mask is not None and not mask[b, l]:
                continue

            # Get all K predictions for this residue
            preds = samples[:, b, l, :]  # [K, 3]

            # Compute pairwise distances
            # dist[i, j] = ||pred_i - pred_j||
            diff = preds.unsqueeze(0) - preds.unsqueeze(1)  # [K, K, 3]
            dist = (diff ** 2).sum(dim=-1).sqrt()  # [K, K]

            # Count how many predictions are within threshold for each
            within_threshold = (dist < threshold).sum(dim=1)  # [K]

            # Find reference (most connected prediction)
            ref_idx = within_threshold.argmax()

            # Get predictions within threshold of reference
            close_mask = dist[ref_idx] < threshold  # [K]
            close_preds = preds[close_mask]  # [n_close, 3]

            # Average close predictions
            result[b, l] = close_preds.mean(dim=0)

    return result


def _consensus_aggregation_fast(
    samples: Tensor,
    mask: Optional[Tensor],
    threshold: float,
) -> Tensor:
    """Faster consensus aggregation using vectorized operations.

    Instead of per-residue loop, uses batch operations.
    Falls back to simple mean if consensus doesn't converge.
    """
    K, B, L, _ = samples.shape
    device = samples.device

    # Compute pairwise distances between samples (averaged over residues)
    # [K, B, L, 3] -> [K, K, B]
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)  # [K, K, B, L, 3]
    dist_per_res = (diff ** 2).sum(dim=-1).sqrt()  # [K, K, B, L]

    if mask is not None:
        mask_exp = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, B, L]
        avg_dist = (dist_per_res * mask_exp).sum(dim=-1) / mask_exp.sum(dim=-1).clamp(min=1)  # [K, K, B]
    else:
        avg_dist = dist_per_res.mean(dim=-1)  # [K, K, B]

    # For each batch, find which samples form consensus
    result = torch.zeros(B, L, 3, device=device)

    for b in range(B):
        # Count connections (samples within threshold)
        connections = (avg_dist[:, :, b] < threshold).sum(dim=1)  # [K]

        # Use samples with above-median connections
        median_conn = connections.median()
        consensus_mask = connections >= median_conn  # [K]

        if consensus_mask.sum() == 0:
            consensus_mask = torch.ones(K, dtype=torch.bool, device=device)

        # Average consensus samples
        consensus_samples = samples[consensus_mask, b]  # [n_consensus, L, 3]
        result[b] = consensus_samples.mean(dim=0)

    return result


class MultiSampler:
    """Wrapper that runs multiple samples and aggregates results.

    Can wrap any sampling function or sampler object.
    """

    def __init__(
        self,
        n_samples: int = 5,
        aggregation: AggregationMethod = "mean",
        trim_fraction: float = 0.2,
        consensus_threshold: float = 1.0,
        seeds: Optional[list] = None,
        align_before_aggregate: bool = True,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            aggregation: Aggregation method
            trim_fraction: Fraction to trim for trimmed_mean
            consensus_threshold: Threshold for consensus method
            seeds: Optional list of seeds (length n_samples). If None, uses random seeds.
            align_before_aggregate: If True, Kabsch-align all samples to first before aggregating
        """
        self.n_samples = n_samples
        self.aggregation = aggregation
        self.trim_fraction = trim_fraction
        self.consensus_threshold = consensus_threshold
        self.seeds = seeds or list(range(n_samples))
        self.align_before_aggregate = align_before_aggregate

    @torch.no_grad()
    def sample(
        self,
        sample_fn: Callable[..., Tensor],
        mask: Optional[Tensor] = None,
        **sample_kwargs,
    ) -> Tensor:
        """Run multiple samples and aggregate.

        Args:
            sample_fn: Function that returns [B, L, 3] tensor
            mask: Optional [B, L] mask for alignment
            **sample_kwargs: Arguments passed to sample_fn

        Returns:
            aggregated: [B, L, 3] aggregated prediction
        """
        samples = []

        for i in range(self.n_samples):
            # Set seed for reproducibility
            torch.manual_seed(self.seeds[i])

            # Run sampling
            pred = sample_fn(**sample_kwargs)
            samples.append(pred)

        # Stack samples: [K, B, L, 3]
        samples = torch.stack(samples, dim=0)

        # Optionally align all samples to first
        if self.align_before_aggregate and self.n_samples > 1:
            ref = samples[0]  # [B, L, 3]
            for i in range(1, self.n_samples):
                samples[i] = kabsch_align_to_target(samples[i], ref, mask)

        # Aggregate
        return aggregate_samples(
            samples,
            method=self.aggregation,
            mask=mask,
            trim_fraction=self.trim_fraction,
            consensus_threshold=self.consensus_threshold,
        )

    def sample_with_all_predictions(
        self,
        sample_fn: Callable[..., Tensor],
        mask: Optional[Tensor] = None,
        **sample_kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Run multiple samples and return both aggregated and individual predictions.

        Returns:
            aggregated: [B, L, 3] aggregated prediction
            all_samples: [K, B, L, 3] all individual predictions
        """
        samples = []

        for i in range(self.n_samples):
            torch.manual_seed(self.seeds[i])
            pred = sample_fn(**sample_kwargs)
            samples.append(pred)

        samples = torch.stack(samples, dim=0)

        if self.align_before_aggregate and self.n_samples > 1:
            ref = samples[0]
            for i in range(1, self.n_samples):
                samples[i] = kabsch_align_to_target(samples[i], ref, mask)

        aggregated = aggregate_samples(
            samples,
            method=self.aggregation,
            mask=mask,
            trim_fraction=self.trim_fraction,
            consensus_threshold=self.consensus_threshold,
        )

        return aggregated, samples


def sample_centroids_multi(
    model: nn.Module,
    batch: Dict[str, Tensor],
    noiser,
    device: torch.device,
    sample_fn: Callable,
    n_samples: int = 5,
    aggregation: AggregationMethod = "mean",
    consensus_threshold: float = 1.0,
    align_before_aggregate: bool = True,
    return_all: bool = False,
    **sample_kwargs,
) -> Tensor:
    """Convenience function for multi-sampling centroids.

    Args:
        model: ResFold model
        batch: Batch dict with aa_seq, chain_ids, etc.
        noiser: Diffusion noiser
        device: torch device
        sample_fn: Single-sample function (e.g., sample_centroids_ve)
        n_samples: Number of samples
        aggregation: Aggregation method
        consensus_threshold: Threshold for consensus
        align_before_aggregate: Align samples before aggregating
        return_all: If True, also return all individual samples
        **sample_kwargs: Additional args for sample_fn

    Returns:
        If return_all=False: [B, L, 3] aggregated centroids
        If return_all=True: ([B, L, 3] aggregated, [K, B, L, 3] all samples)
    """
    mask = batch.get('mask_res')

    multi_sampler = MultiSampler(
        n_samples=n_samples,
        aggregation=aggregation,
        consensus_threshold=consensus_threshold,
        align_before_aggregate=align_before_aggregate,
    )

    def _sample():
        return sample_fn(model, batch, noiser, device, **sample_kwargs)

    if return_all:
        return multi_sampler.sample_with_all_predictions(_sample, mask=mask)
    else:
        return multi_sampler.sample(_sample, mask=mask)


# Expose in __init__
__all__ = [
    "MultiSampler",
    "aggregate_samples",
    "sample_centroids_multi",
    "AggregationMethod",
]
