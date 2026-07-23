"""Training setup utilities for TinyFold.

Encapsulates common setup patterns from training scripts:
- get_or_create_split: Load existing split or create new one
- create_diffusion_components: Create schedule and noiser based on args
- load_model_checkpoint: Load checkpoint with architecture matching
- create_train_sampler: Create batching sampler based on args
"""

import os
from typing import Any

import torch

from .data_split import (
    DataSplitConfig,
    DynamicBatchSampler,
    LengthBucketSampler,
    get_split_info,
    get_train_test_indices,
    load_split,
    save_split,
)


class SplitLeakageError(RuntimeError):
    """Raised when a run's test set shares sequence clusters with training."""


def audit_and_gate_split(
    args,
    table,
    train_indices,
    test_indices,
    logger,
) -> dict:
    """Measure train/test cluster leakage and, by default, refuse to train on it.

    Every headline number this project produced before 2026-07-23 came from a
    ``test_strategy="random"`` split. On the le200 split that meant 183 of 200
    test complexes shared a sequence cluster with training, and re-scoring the
    same checkpoint by stratum gave DockQ 0.251 (leaked, n=183) versus 0.044
    (clean, n=17) with 0% medium-quality. The cluster-aware machinery existed
    the whole time and simply was not wired to the configs that produced the
    results, and nothing in the run output recorded which kind of split was
    used. This makes that impossible to repeat silently.

    Returns the audit dict; it is merged into the run's split_info so every run
    is self-describing.
    """
    from .cluster_split import audit_split_leakage, load_clusters

    require_clean = getattr(args, "require_clean_split", True)
    clusters_path = getattr(args, "clusters", None) or "data/processed/clusters.json"

    if not os.path.exists(clusters_path):
        msg = (
            f"Cannot audit split leakage: no clusters file at {clusters_path}. "
            "Generate one with scripts/data/cluster_interfaces.py, point "
            "--clusters at it, or pass --no-require-clean-split to train "
            "without the check (the run will be labelled UNAUDITED)."
        )
        if require_clean:
            raise SplitLeakageError(msg)
        logger.log(f"  WARNING: {msg}")
        return {"split_audited": False, "reason": "clusters file missing"}

    sample_ids = table["sample_id"]
    train_ids = [sample_ids[i].as_py() for i in train_indices]
    test_ids = [sample_ids[i].as_py() for i in test_indices]
    audit = audit_split_leakage(train_ids, test_ids, load_clusters(clusters_path))
    audit["split_audited"] = True

    logger.log("  Split leakage audit:")
    logger.log(f"    Train clusters: {audit['n_train_clusters']}")
    logger.log(f"    Test clusters:  {audit['n_test_clusters']}"
               f" (max share {100 * audit['max_test_cluster_share']:.1f}%)")
    logger.log(f"    Test samples whose cluster is in train: "
               f"{audit['n_test_leaked']}/{len(test_ids)} "
               f"({100 * audit['frac_test_leaked']:.1f}%)")
    if audit["n_test_unclustered"]:
        logger.log(f"    WARNING: {audit['n_test_unclustered']} test samples are "
                   "absent from clusters.json and could not be checked")

    if audit["n_test_leaked"]:
        msg = (
            f"LEAKY SPLIT: {audit['n_test_leaked']}/{len(test_ids)} test samples "
            f"({100 * audit['frac_test_leaked']:.1f}%) share a sequence cluster "
            "with training. Any DockQ from this split measures memorisation, not "
            "generalization. Build a clean split with "
            "scripts/data/make_cluster_split.py --per-cluster-cap 1 and pass it "
            "via --load_split, or set test_strategy=cluster."
        )
        if require_clean:
            raise SplitLeakageError(msg)
        logger.log("  " + "=" * 68)
        logger.log(f"  {msg}")
        logger.log("  Proceeding anyway because --no-require-clean-split was set.")
        logger.log("  " + "=" * 68)

    # A clean split whose test samples come from very few clusters is disjoint
    # but not informative -- effective n is the cluster count, not the sample
    # count (clean_le600 shipped 200 samples carrying 21 clusters).
    if audit["n_test_clusters"] and audit["max_test_cluster_share"] > 0.1:
        logger.log(
            f"  WARNING: one cluster is {100 * audit['max_test_cluster_share']:.0f}% "
            f"of the test set ({audit['n_test_clusters']} clusters for "
            f"{len(test_ids)} samples). Effective n is the cluster count. "
            "Rebuild with --per-cluster-cap 1."
        )
    return audit


def get_or_create_split(
    args,
    table,
    logger,
    output_dir: str | None = None,
) -> tuple[list, list, dict]:
    """Load existing split or create new one.

    Args:
        args: Namespace with load_split, n_train, n_test, min_atoms, max_atoms, select_smallest
        table: PyArrow table
        logger: Logger instance
        output_dir: Where to save new split (defaults to args.output_dir)

    Returns:
        (train_indices, test_indices, split_info)
    """
    if args.load_split:
        logger.log(f"Loading split from: {args.load_split}")
        train_indices, test_indices, loaded_info = load_split(args.load_split)
        logger.log("Data split (loaded from file):")
        logger.log(f"  Training: {len(train_indices)} samples")
        logger.log(f"  Test: {len(test_indices)} samples")
        # A loaded split is audited exactly like a generated one -- the file
        # may predate the cluster machinery or have been built with a random
        # strategy, and its provenance is not otherwise recorded.
        loaded_info["leakage_audit"] = audit_and_gate_split(
            args, table, train_indices, test_indices, logger
        )
        loaded_info["split_source"] = args.load_split
        return train_indices, test_indices, loaded_info

    # Parse optional test_size_bins (CSV string from CLI, list from YAML).
    bins_arg = getattr(args, 'test_size_bins', None)
    if isinstance(bins_arg, str) and bins_arg.strip():
        bins_arg = [int(x) for x in bins_arg.split(",")]
    elif not bins_arg:
        bins_arg = None  # let DataSplitConfig pick its default

    # Create new split
    split_config = DataSplitConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        min_atoms=getattr(args, 'min_atoms', 0),
        max_atoms=getattr(args, 'max_atoms', 1000),
        select_smallest=getattr(args, 'select_smallest', False),
        seed=getattr(args, 'seed', 42),
        test_strategy=getattr(args, 'test_strategy', 'random'),
        test_size_bins=bins_arg,
        clusters_path=getattr(args, 'clusters', None),
        per_cluster_cap=getattr(args, 'per_cluster_cap', 1),
        n_test_clusters=getattr(args, 'n_test_clusters', None),
    )
    train_indices, test_indices = get_train_test_indices(table, split_config)
    split_info = get_split_info(table, split_config)

    logger.log(f"Data split (seed={split_config.seed}, "
               f"strategy={split_config.test_strategy}):")
    if getattr(args, 'select_smallest', False):
        logger.log(f"  Selected {split_info['eligible_samples']} smallest proteins")
    else:
        logger.log(f"  Eligible samples: {split_info['eligible_samples']}")
    logger.log(f"  Training: {len(train_indices)} samples")
    logger.log(f"  Test: {len(test_indices)} samples")

    split_info["leakage_audit"] = audit_and_gate_split(
        args, table, train_indices, test_indices, logger
    )
    split_info["split_source"] = "generated"

    # Save split for reuse
    save_dir = output_dir or getattr(args, 'output_dir', '.')
    split_path = os.path.join(save_dir, "split.json")
    save_split(split_info, split_path)

    return train_indices, test_indices, split_info


def create_diffusion_components(args, device, logger) -> tuple[Any, Any]:
    """Create schedule and noiser based on args.

    Args:
        args: Namespace with continuous_sigma, T, sigma_min, sigma_max, sigma_data, schedule
        device: torch device
        logger: Logger instance

    Returns:
        (schedule, noiser)
    """
    from tinyfold.model.diffusion import create_noiser, create_schedule

    if getattr(args, 'continuous_sigma', False):
        # AF3-style: VE noise with Karras schedule
        from tinyfold.model.diffusion import KarrasSchedule, VENoiser

        schedule = KarrasSchedule(
            n_steps=args.T,
            sigma_min=getattr(args, 'sigma_min', 0.002),
            sigma_max=getattr(args, 'sigma_max', 80.0),
            rho=7.0,
        )
        noiser = VENoiser(schedule, sigma_data=getattr(args, 'sigma_data', 1.0))
        noiser = noiser.to(device)

        logger.log("Diffusion:")
        logger.log("  Mode: AF3-style continuous sigma (VE)")
        logger.log(f"  sigma_range: [{args.sigma_min}, {args.sigma_max}]")
        logger.log(f"  T: {args.T} (inference steps)")
    else:
        # Standard: VP noise with discrete timesteps
        schedule_type = getattr(args, 'schedule', 'linear')
        schedule = create_schedule(schedule_type, T=args.T)
        noiser = create_noiser("gaussian", schedule)
        noiser = noiser.to(device)

        logger.log("Diffusion:")
        logger.log(f"  Schedule: {schedule_type}")
        logger.log("  Noise type: gaussian (VP, discrete t)")

    logger.log("")
    return schedule, noiser


def load_model_checkpoint(
    model,
    checkpoint_path: str,
    mode: str,
    device,
    logger,
) -> dict:
    """Load checkpoint with automatic architecture matching.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        mode: Training mode (stage1_only, stage2_only, end_to_end)
        device: torch device
        logger: Logger instance

    Returns:
        Checkpoint dict
    """
    logger.log(f"Loading checkpoint: {checkpoint_path}")
    # weights_only=True: checkpoints hold only tensors + plain dicts/numbers,
    # so this is safe and blocks code execution from an untrusted .pt.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt['model_state_dict']

    # Filter Stage 2 keys if architecture changed
    if mode == "stage2_only":
        stage2_keys = [k for k in state_dict.keys() if k.startswith('stage2.')]
        model_stage2_keys = [k for k in model.state_dict().keys() if k.startswith('stage2.')]

        if stage2_keys and model_stage2_keys:
            sample_ckpt = state_dict[stage2_keys[0]].shape
            sample_model = model.state_dict()[model_stage2_keys[0]].shape
            if sample_ckpt != sample_model:
                logger.log("  Stage 2 architecture changed, loading only Stage 1 weights")
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('stage2.')}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.log(f"  Missing keys: {len(missing)}")
    if unexpected:
        logger.log(f"  Unexpected keys: {len(unexpected)}")
    logger.log(f"  Loaded from step {ckpt.get('step', 'unknown')}")
    logger.log("")

    return ckpt


def create_train_sampler(args, samples: dict, logger) -> Any | None:
    """Create batching sampler based on args.

    Args:
        args: Namespace with dynamic_batch, use_bucketing, batch_size, max_tokens, n_buckets
        samples: Dict of sample_idx -> sample dict
        logger: Logger instance

    Returns:
        Sampler or None
    """
    if getattr(args, 'dynamic_batch', False):
        seed = getattr(args, 'seed', 42)
        sampler = DynamicBatchSampler(
            samples,
            base_batch_size=args.batch_size,
            max_tokens=getattr(args, 'max_tokens', 4096),
            n_buckets=getattr(args, 'n_buckets', 4),
            seed=seed,
        )
        logger.log(f"  Using dynamic batch sampler (max_tokens={args.max_tokens})")
        for info in sampler.get_batch_sizes():
            logger.log(f"    Bucket {info['bucket']}: max_res={info['max_res']}, batch_size={info['batch_size']}")
        return sampler

    if getattr(args, 'use_bucketing', False):
        seed = getattr(args, 'seed', 42)
        sampler = LengthBucketSampler(
            samples,
            n_buckets=getattr(args, 'n_buckets', 4),
            seed=seed,
        )
        logger.log(f"  Using length bucketing ({args.n_buckets} buckets)")
        return sampler

    return None
