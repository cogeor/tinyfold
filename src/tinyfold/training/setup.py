"""Training setup utilities for TinyFold.

Encapsulates common setup patterns from training scripts:
- get_or_create_split: Load existing split or create new one
- create_diffusion_components: Create schedule and noiser based on args
- load_model_checkpoint: Load checkpoint with architecture matching
- create_train_sampler: Create batching sampler based on args
"""

from typing import Any, Optional, Tuple
import os

import torch
import pyarrow.parquet as pq

from .data_split import (
    DataSplitConfig, 
    get_train_test_indices, 
    get_split_info, 
    save_split, 
    load_split,
    LengthBucketSampler, 
    DynamicBatchSampler,
)


def get_or_create_split(
    args, 
    table, 
    logger,
    output_dir: Optional[str] = None,
) -> Tuple[list, list, dict]:
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
        logger.log(f"Data split (loaded from file):")
        logger.log(f"  Training: {len(train_indices)} samples")
        logger.log(f"  Test: {len(test_indices)} samples")
        return train_indices, test_indices, loaded_info
    
    # Create new split
    split_config = DataSplitConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        min_atoms=getattr(args, 'min_atoms', 0),
        max_atoms=getattr(args, 'max_atoms', 1000),
        select_smallest=getattr(args, 'select_smallest', False),
        seed=getattr(args, 'seed', 42),
    )
    train_indices, test_indices = get_train_test_indices(table, split_config)
    split_info = get_split_info(table, split_config)
    
    logger.log(f"Data split (seed={split_config.seed}):")
    if getattr(args, 'select_smallest', False):
        logger.log(f"  Selected {split_info['eligible_samples']} smallest proteins")
    else:
        logger.log(f"  Eligible samples: {split_info['eligible_samples']}")
    logger.log(f"  Training: {len(train_indices)} samples")
    logger.log(f"  Test: {len(test_indices)} samples")
    
    # Save split for reuse
    save_dir = output_dir or getattr(args, 'output_dir', '.')
    split_path = os.path.join(save_dir, "split.json")
    save_split(split_info, split_path)
    
    return train_indices, test_indices, split_info


def create_diffusion_components(args, device, logger) -> Tuple[Any, Any]:
    """Create schedule and noiser based on args.
    
    Args:
        args: Namespace with continuous_sigma, T, sigma_min, sigma_max, sigma_data, schedule
        device: torch device
        logger: Logger instance
        
    Returns:
        (schedule, noiser)
    """
    from tinyfold.model.diffusion import create_schedule, create_noiser
    
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
        
        logger.log(f"Diffusion:")
        logger.log(f"  Mode: AF3-style continuous sigma (VE)")
        logger.log(f"  sigma_range: [{args.sigma_min}, {args.sigma_max}]")
        logger.log(f"  T: {args.T} (inference steps)")
    else:
        # Standard: VP noise with discrete timesteps
        schedule_type = getattr(args, 'schedule', 'linear')
        schedule = create_schedule(schedule_type, T=args.T)
        noiser = create_noiser("gaussian", schedule)
        noiser = noiser.to(device)
        
        logger.log(f"Diffusion:")
        logger.log(f"  Schedule: {schedule_type}")
        logger.log(f"  Noise type: gaussian (VP, discrete t)")
    
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
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict']
    
    # Filter Stage 2 keys if architecture changed
    if mode == "stage2_only":
        stage2_keys = [k for k in state_dict.keys() if k.startswith('stage2.')]
        model_stage2_keys = [k for k in model.state_dict().keys() if k.startswith('stage2.')]
        
        if stage2_keys and model_stage2_keys:
            sample_ckpt = state_dict[stage2_keys[0]].shape
            sample_model = model.state_dict()[model_stage2_keys[0]].shape
            if sample_ckpt != sample_model:
                logger.log(f"  Stage 2 architecture changed, loading only Stage 1 weights")
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('stage2.')}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.log(f"  Missing keys: {len(missing)}")
    if unexpected:
        logger.log(f"  Unexpected keys: {len(unexpected)}")
    logger.log(f"  Loaded from step {ckpt.get('step', 'unknown')}")
    logger.log("")
    
    return ckpt


def create_train_sampler(args, samples: dict, logger) -> Optional[Any]:
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
