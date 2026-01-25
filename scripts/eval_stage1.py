#!/usr/bin/env python
"""
Evaluation script for Stage 1 (centroid prediction) of ResFold.

Supports:
- Single-sample and multi-sample inference
- Both discrete VP and continuous VE (sigma) models
- Multiple aggregation strategies for multi-sampling

Usage:
    # Single sample evaluation
    python eval_stage1.py --checkpoint outputs/train_10k/best_model.pt

    # Multi-sample with consensus aggregation
    python eval_stage1.py --checkpoint outputs/train_10k/best_model.pt --n_samples 5 --aggregation consensus

    # Evaluate continuous sigma model
    python eval_stage1.py --checkpoint outputs/train_10k_continuous/best_model.pt --continuous_sigma
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_resfold import (
    load_sample_raw, collate_batch, sample_centroids, sample_centroids_ve
)
from models.resfold_pipeline import ResFoldPipeline
from models.diffusion import KarrasSchedule, VENoiser
from models import create_schedule, create_noiser
from models.multi_sample import MultiSampler, sample_centroids_multi
from tinyfold.model.losses import kabsch_align, compute_rmse


def compute_contact_accuracy(pred: torch.Tensor, gt: torch.Tensor, chain_ids: torch.Tensor,
                              threshold: float = 8.0, std: float = 1.0) -> dict:
    """Compute contact prediction accuracy.

    Args:
        pred: [L, 3] predicted centroids (normalized)
        gt: [L, 3] ground truth centroids (normalized)
        chain_ids: [L] chain assignments (0 or 1)
        threshold: Contact threshold in Angstroms
        std: Coordinate scaling factor

    Returns:
        dict with precision, recall, f1 for inter-chain contacts
    """
    # Scale to Angstroms
    pred_A = pred * std
    gt_A = gt * std

    L = len(pred)

    # Compute pairwise distances
    pred_dist = torch.cdist(pred_A.unsqueeze(0), pred_A.unsqueeze(0)).squeeze(0)
    gt_dist = torch.cdist(gt_A.unsqueeze(0), gt_A.unsqueeze(0)).squeeze(0)

    # Inter-chain mask
    chain_a = chain_ids == 0
    chain_b = chain_ids == 1
    inter_mask = chain_a.unsqueeze(1) & chain_b.unsqueeze(0)
    inter_mask = inter_mask | inter_mask.T  # Symmetric

    # Contact predictions
    pred_contacts = (pred_dist < threshold) & inter_mask
    gt_contacts = (gt_dist < threshold) & inter_mask

    # Metrics
    tp = (pred_contacts & gt_contacts).sum().float()
    fp = (pred_contacts & ~gt_contacts).sum().float()
    fn = (~pred_contacts & gt_contacts).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'n_gt_contacts': gt_contacts.sum().item() // 2,  # Symmetric, so divide by 2
        'n_pred_contacts': pred_contacts.sum().item() // 2,
    }


def compute_chain_rmse(pred: torch.Tensor, gt: torch.Tensor, chain_ids: torch.Tensor,
                        std: float = 1.0) -> dict:
    """Compute per-chain RMSE after alignment.

    Args:
        pred: [L, 3] predicted centroids
        gt: [L, 3] ground truth centroids
        chain_ids: [L] chain assignments
        std: Coordinate scaling factor

    Returns:
        dict with chain_a_rmse, chain_b_rmse, interface_rmse
    """
    chain_a_mask = chain_ids == 0
    chain_b_mask = chain_ids == 1

    # Overall alignment
    pred_aligned, gt_aligned = kabsch_align(pred.unsqueeze(0), gt.unsqueeze(0))
    pred_aligned = pred_aligned.squeeze(0)
    gt_aligned = gt_aligned.squeeze(0)

    # Per-chain RMSE
    def rmse(p, g):
        return ((p - g) ** 2).mean().sqrt().item() * std

    chain_a_rmse = rmse(pred_aligned[chain_a_mask], gt_aligned[chain_a_mask])
    chain_b_rmse = rmse(pred_aligned[chain_b_mask], gt_aligned[chain_b_mask])

    # Interface residues (within 10A of other chain in GT)
    gt_A = gt * std
    dist_to_other = torch.cdist(gt_A[chain_a_mask].unsqueeze(0), gt_A[chain_b_mask].unsqueeze(0)).squeeze(0)
    interface_a = dist_to_other.min(dim=1).values < 10.0
    interface_b = dist_to_other.min(dim=0).values < 10.0

    interface_pred = torch.cat([pred_aligned[chain_a_mask][interface_a], pred_aligned[chain_b_mask][interface_b]])
    interface_gt = torch.cat([gt_aligned[chain_a_mask][interface_a], gt_aligned[chain_b_mask][interface_b]])

    if len(interface_pred) > 0:
        interface_rmse = rmse(interface_pred, interface_gt)
    else:
        interface_rmse = float('nan')

    return {
        'chain_a_rmse': chain_a_rmse,
        'chain_b_rmse': chain_b_rmse,
        'interface_rmse': interface_rmse,
        'n_interface': len(interface_pred),
    }


def evaluate_sample(
    model, sample, noiser, device, args, sample_fn
) -> dict:
    """Evaluate a single sample.

    Returns dict with all metrics.
    """
    batch = collate_batch([sample], device)
    mask = batch['mask_res']
    gt = batch['centroids']

    # Compute metrics helper
    def compute_metrics(pred, suffix=""):
        n_res = sample['n_res']
        pred_trimmed = pred[0, :n_res]
        gt_trimmed = gt[0, :n_res]
        chain_ids = batch['chain_ids'][0, :n_res]
        std = sample['std']

        # Overall RMSE (after Kabsch alignment)
        pred_aligned, gt_aligned = kabsch_align(pred_trimmed.unsqueeze(0), gt_trimmed.unsqueeze(0))
        rmse = compute_rmse(pred_aligned, gt_aligned).item() * std

        # Per-chain RMSE
        chain_metrics = compute_chain_rmse(pred_trimmed, gt_trimmed, chain_ids, std)

        # Contact accuracy
        contact_metrics = compute_contact_accuracy(pred_trimmed, gt_trimmed, chain_ids,
                                                    threshold=args.contact_threshold, std=std)

        result = {
            f'rmse{suffix}': rmse,
            f'chain_a_rmse{suffix}': chain_metrics['chain_a_rmse'],
            f'chain_b_rmse{suffix}': chain_metrics['chain_b_rmse'],
            f'interface_rmse{suffix}': chain_metrics['interface_rmse'],
            f'n_interface{suffix}': chain_metrics['n_interface'],
            f'f1{suffix}': contact_metrics['f1'],
        }
        return result

    # Run inference
    if args.n_samples > 1:
        # Get all samples
        from models.multi_sample import MultiSampler, aggregate_samples

        samples_list = []
        for i in range(args.n_samples):
            torch.manual_seed(i)
            pred = sample_fn(model, batch, noiser, device)
            samples_list.append(pred)

        all_preds = torch.stack(samples_list, dim=0)  # [K, B, L, 3]

        # Align all to first before aggregating
        from models.diffusion import kabsch_align_to_target
        ref = all_preds[0]
        for i in range(1, args.n_samples):
            all_preds[i] = kabsch_align_to_target(all_preds[i], ref, mask)

        # Compute both mean and consensus
        pred_mean = aggregate_samples(all_preds, method="mean", mask=mask)
        pred_consensus = aggregate_samples(all_preds, method="consensus", mask=mask,
                                           consensus_threshold=args.consensus_threshold)

        # Compute variance
        variance = all_preds.var(dim=0).mean().item()

        # Get metrics for both
        metrics_mean = compute_metrics(pred_mean, suffix="_mean")
        metrics_consensus = compute_metrics(pred_consensus, suffix="_cons")

        # Also compute single sample (first one) for reference
        metrics_single = compute_metrics(all_preds[0], suffix="_single")

        result = {
            'sample_id': sample['sample_id'],
            'n_res': sample['n_res'],
            'variance': variance,
            **metrics_single,
            **metrics_mean,
            **metrics_consensus,
        }
        # Add primary rmse for sorting
        result['rmse'] = metrics_mean['rmse_mean']
        return result
    else:
        torch.manual_seed(args.seed)
        pred = sample_fn(model, batch, noiser, device)

        metrics = compute_metrics(pred)
        return {
            'sample_id': sample['sample_id'],
            'n_res': sample['n_res'],
            'variance': 0.0,
            'rmse': metrics['rmse'],
            **metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 (centroid prediction)")

    # Model/checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default=None,
                        help="Path to split.json (default: same dir as checkpoint)")

    # Model architecture (will try to load from checkpoint args)
    parser.add_argument("--c_token_s1", type=int, default=None)
    parser.add_argument("--trunk_layers", type=int, default=None)
    parser.add_argument("--denoiser_blocks", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)

    # Diffusion mode
    parser.add_argument("--continuous_sigma", action="store_true",
                        help="Use continuous sigma (VE) mode")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--sigma_data", type=float, default=1.0)
    parser.add_argument("--inference_steps", type=int, default=None,
                        help="Number of inference steps (default: same as T)")

    # Sampling
    parser.add_argument("--align_per_step", action="store_true", default=True)
    parser.add_argument("--recenter", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Multi-sampling
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples for multi-sampling (1 = single sample)")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "median", "trimmed_mean", "consensus"],
                        help="Aggregation method for multi-sampling")
    parser.add_argument("--consensus_threshold", type=float, default=1.0,
                        help="Distance threshold for consensus aggregation")

    # Evaluation
    parser.add_argument("--n_eval_train", type=int, default=100,
                        help="Number of training samples to evaluate")
    parser.add_argument("--contact_threshold", type=float, default=8.0,
                        help="Contact threshold in Angstroms")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: checkpoint_dir/eval_results.json)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint).parent

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model config from checkpoint or args
    ckpt_args = ckpt.get('args', {})
    c_token_s1 = args.c_token_s1 or ckpt_args.get('c_token_s1', 256)
    trunk_layers = args.trunk_layers or ckpt_args.get('trunk_layers', 9)
    denoiser_blocks = args.denoiser_blocks or ckpt_args.get('denoiser_blocks', 7)
    T = args.T or ckpt_args.get('T', 50)
    continuous_sigma = args.continuous_sigma or ckpt_args.get('continuous_sigma', False)

    print(f"Model config: c_token_s1={c_token_s1}, trunk_layers={trunk_layers}, "
          f"denoiser_blocks={denoiser_blocks}, T={T}, continuous_sigma={continuous_sigma}")

    # Create model
    model = ResFoldPipeline(
        c_token_s1=c_token_s1,
        trunk_layers=trunk_layers,
        denoiser_blocks=denoiser_blocks,
        n_timesteps=T,
        stage1_only=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Create noiser (use inference_steps if specified, else T)
    inference_steps = args.inference_steps or T
    print(f"Inference steps: {inference_steps}")

    if continuous_sigma:
        schedule = KarrasSchedule(
            n_steps=inference_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=7.0,
        )
        noiser = VENoiser(schedule, sigma_data=args.sigma_data)
        sample_fn = lambda m, b, n, d: sample_centroids_ve(
            m, b, n, d, align_per_step=args.align_per_step, recenter=args.recenter
        )
    else:
        schedule = create_schedule("linear", T=inference_steps)
        noiser = create_noiser("gaussian", schedule)
        sample_fn = lambda m, b, n, d: sample_centroids(
            m, b, n, d, align_per_step=args.align_per_step, recenter=args.recenter
        )

    # Load data split
    split_path = args.split or (checkpoint_dir / "split.json")
    print(f"Loading split: {split_path}")
    with open(split_path) as f:
        split = json.load(f)
    train_indices = split['train_indices']
    test_indices = split['test_indices']

    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data/processed/samples.parquet"
    table = pq.read_table(data_path)

    print(f"Evaluating: {min(args.n_eval_train, len(train_indices))} train, {len(test_indices)} test")
    print(f"Multi-sampling: n_samples={args.n_samples}, aggregation={args.aggregation}")
    print("=" * 70)

    # Evaluate
    results = {'train': [], 'test': []}

    # Train set (subset)
    np.random.seed(42)
    eval_train_indices = np.random.choice(train_indices,
                                           size=min(args.n_eval_train, len(train_indices)),
                                           replace=False).tolist()

    for split_name, indices in [('train', eval_train_indices), ('test', test_indices)]:
        print(f"\n{split_name.upper()} ({len(indices)} samples)")
        print("-" * 70)

        for i, idx in enumerate(indices):
            sample = load_sample_raw(table, idx)
            metrics = evaluate_sample(model, sample, noiser, device, args, sample_fn)
            results[split_name].append(metrics)

            if args.verbose or (i + 1) % 20 == 0:
                if args.n_samples > 1:
                    print(f"  [{i+1:3d}/{len(indices)}] {metrics['sample_id']}: "
                          f"single={metrics['rmse_single']:.1f}A, "
                          f"mean={metrics['rmse_mean']:.1f}A, "
                          f"cons={metrics['rmse_cons']:.1f}A, "
                          f"F1={metrics['f1_mean']:.2f}")
                else:
                    print(f"  [{i+1:3d}/{len(indices)}] {metrics['sample_id']}: "
                          f"RMSE={metrics['rmse']:.2f}A, "
                          f"ChainA={metrics.get('chain_a_rmse', 0):.2f}A, "
                          f"ChainB={metrics.get('chain_b_rmse', 0):.2f}A, "
                          f"Contact F1={metrics.get('f1', 0):.3f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for split_name in ['train', 'test']:
        r = results[split_name]
        if not r:
            continue

        print(f"\n{split_name.upper()} ({len(r)} samples)")

        if args.n_samples > 1:
            # Multi-sample: show all three methods
            single_rmses = [x['rmse_single'] for x in r]
            mean_rmses = [x['rmse_mean'] for x in r]
            cons_rmses = [x['rmse_cons'] for x in r]
            variances = [x['variance'] for x in r]

            print(f"  {'Method':<12} {'RMSE':>10} {'Std':>8}")
            print(f"  {'-'*32}")
            print(f"  {'Single':<12} {np.mean(single_rmses):>10.2f} {np.std(single_rmses):>8.2f}")
            print(f"  {'Mean (5)':<12} {np.mean(mean_rmses):>10.2f} {np.std(mean_rmses):>8.2f}")
            print(f"  {'Consensus':<12} {np.mean(cons_rmses):>10.2f} {np.std(cons_rmses):>8.2f}")
            print(f"  Sample variance: {np.mean(variances):.4f}")

            # Interface RMSE
            interface = [x['interface_rmse_mean'] for x in r if not np.isnan(x['interface_rmse_mean'])]
            if interface:
                print(f"  Interface RMSE (mean): {np.mean(interface):.2f} +/- {np.std(interface):.2f} A")

            # Contact F1
            f1s = [x['f1_mean'] for x in r]
            print(f"  Contact F1 (mean): {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        else:
            rmses = [x['rmse'] for x in r]
            chain_a = [x.get('chain_a_rmse', 0) for x in r]
            chain_b = [x.get('chain_b_rmse', 0) for x in r]
            interface = [x.get('interface_rmse', float('nan')) for x in r]
            interface = [x for x in interface if not np.isnan(x)]
            f1s = [x.get('f1', 0) for x in r]

            print(f"  Overall RMSE:    {np.mean(rmses):6.2f} +/- {np.std(rmses):.2f} A")
            print(f"  Chain A RMSE:    {np.mean(chain_a):6.2f} +/- {np.std(chain_a):.2f} A")
            print(f"  Chain B RMSE:    {np.mean(chain_b):6.2f} +/- {np.std(chain_b):.2f} A")
            if interface:
                print(f"  Interface RMSE:  {np.mean(interface):6.2f} +/- {np.std(interface):.2f} A")
            print(f"  Contact F1:      {np.mean(f1s):6.3f} +/- {np.std(f1s):.3f}")

    # Analyze best/worst examples
    if args.n_samples > 1:
        print("\n" + "=" * 70)
        print("BEST/WORST EXAMPLES (by mean RMSE)")
        print("=" * 70)

        test_r = results['test']
        sorted_by_rmse = sorted(test_r, key=lambda x: x['rmse_mean'])

        print("\nTOP 3 (lowest RMSE):")
        for i, r in enumerate(sorted_by_rmse[:3]):
            print(f"  {i+1}. {r['sample_id']}: RMSE={r['rmse_mean']:.1f}A, "
                  f"n_res={r['n_res']}, var={r['variance']:.3f}, F1={r['f1_mean']:.2f}")

        print("\nBOTTOM 3 (highest RMSE):")
        for i, r in enumerate(sorted_by_rmse[-3:][::-1]):
            print(f"  {i+1}. {r['sample_id']}: RMSE={r['rmse_mean']:.1f}A, "
                  f"n_res={r['n_res']}, var={r['variance']:.3f}, F1={r['f1_mean']:.2f}")

        # Analyze patterns
        good = sorted_by_rmse[:10]
        bad = sorted_by_rmse[-10:]

        good_nres = np.mean([x['n_res'] for x in good])
        bad_nres = np.mean([x['n_res'] for x in bad])
        good_var = np.mean([x['variance'] for x in good])
        bad_var = np.mean([x['variance'] for x in bad])

        print(f"\nPattern analysis (top 10 vs bottom 10):")
        print(f"  Avg n_res:    good={good_nres:.0f}, bad={bad_nres:.0f}")
        print(f"  Avg variance: good={good_var:.4f}, bad={bad_var:.4f}")

    # Save results
    output_path = args.output or (checkpoint_dir / "eval_results.json")
    # Build summary based on mode
    summary = {}
    for split_name, r in results.items():
        if not r:
            continue
        if args.n_samples > 1:
            summary[split_name] = {
                'rmse_mean': float(np.mean([x['rmse_mean'] for x in r])),
                'rmse_std': float(np.std([x['rmse_mean'] for x in r])),
                'contact_f1_mean': float(np.mean([x['f1_mean'] for x in r])),
                'n_samples': len(r),
            }
        else:
            summary[split_name] = {
                'rmse_mean': float(np.mean([x['rmse'] for x in r])),
                'rmse_std': float(np.std([x['rmse'] for x in r])),
                'contact_f1_mean': float(np.mean([x.get('f1', 0) for x in r])),
                'n_samples': len(r),
            }

    output_data = {
        'checkpoint': str(args.checkpoint),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_samples': args.n_samples,
            'aggregation': args.aggregation,
            'consensus_threshold': args.consensus_threshold,
            'continuous_sigma': continuous_sigma,
            'T': T,
            'inference_steps': inference_steps,
        },
        'summary': summary,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
