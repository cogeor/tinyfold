#!/usr/bin/env python
"""
Two-Stage Evaluation: Stage 1 (centroids) → Stage 2 (atoms) → DockQ

Supports two modes:
1. Separate models: --stage1_checkpoint + --stage2_checkpoint
2. E2E model: --e2e_checkpoint (ResFoldE2E with multi-sample diffusion)

Usage:
    # Separate models
    python scripts/eval_two_stage.py \
        --stage1_checkpoint outputs/train_10k_continuous/best_model.pt \
        --stage2_checkpoint outputs/stage2_continuous_rot/best_model.pt \
        --load_split outputs/train_10k_continuous/split.json \
        --n_train_eval 100

    # E2E model
    python scripts/eval_two_stage.py \
        --e2e_checkpoint outputs/resfold_e2e/best_model.pt \
        --load_split outputs/resfold_e2e/split.json \
        --n_samples 5
"""

import sys
import random
import numpy as np
import torch
import pyarrow.parquet as pq
import argparse
import os
import json
from datetime import datetime

# Model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resfold import ResidueDenoiser
from models.atomrefine_continuous import AtomRefinerContinuous
from models.resfold_e2e import ResFoldE2E, sample_e2e
from models.dockq_utils import compute_dockq
from models import create_schedule, create_noiser, KarrasSchedule, VENoiser

# Loss imports
from tinyfold.model.losses import kabsch_align, compute_rmse
from data_split import load_split


def load_sample_raw(table, i, normalize=True):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center coordinates
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    # Compute std
    original_std = coords.std()

    # Normalize to unit variance
    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


@torch.no_grad()
def sample_centroids_ve(model, aa_seq, chain_ids, res_idx, mask, sigmas, device,
                        clamp_val=3.0, self_cond=True):
    """VE sampling for Stage 1 centroids."""
    B, L = aa_seq.shape

    # Initialize at highest noise level
    x = sigmas[0] * torch.randn(B, L, 3, device=device)
    x0_prev = None

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = sigma.expand(B)

        # Predict x0
        x0_pred = model.forward_sigma(
            x, aa_seq, chain_ids, res_idx, sigma_batch, mask,
            x0_prev=x0_prev if self_cond else None
        )
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # Kabsch align to current x
        x0_pred = kabsch_align_to_x(x0_pred, x, mask)

        # Euler step
        if sigma_next > 0:
            d = (x - x0_pred) / sigma
            x = x + (sigma_next - sigma) * d
        else:
            x = x0_pred

        x0_prev = x0_pred

    return x


def kabsch_align_to_x(x0_pred, x, mask):
    """Kabsch align x0_pred to x's frame."""
    # Simple implementation - align target to pred
    aligned, _ = kabsch_align(x0_pred, x, mask)
    return aligned


@torch.no_grad()
def sample_centroids_ddpm(model, aa_seq, chain_ids, res_idx, mask, noiser, device,
                          clamp_val=3.0):
    """DDPM sampling for Stage 1 centroids (discrete timesteps)."""
    B, L = aa_seq.shape

    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        x0_pred = model(x, aa_seq, chain_ids, res_idx, t_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if t > 0:
            ab_t = noiser.alpha_bar[t]
            ab_prev = noiser.alpha_bar[t - 1]
            beta = noiser.betas[t]
            alpha = noiser.alphas[t]

            coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
            coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
            mean = coef1 * x0_pred + coef2 * x

            var = beta * (1 - ab_prev) / (1 - ab_t)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = x0_pred

    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage Evaluation")

    # Checkpoints - either separate or E2E
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 model checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 model checkpoint")
    parser.add_argument("--e2e_checkpoint", type=str, default=None,
                        help="Path to E2E model checkpoint (ResFoldE2E)")

    # Data
    parser.add_argument("--load_split", type=str, required=True,
                        help="Path to split.json")
    parser.add_argument("--n_train_eval", type=int, default=100,
                        help="Number of train samples to evaluate")

    # Stage 1 sampling
    parser.add_argument("--T", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--continuous_sigma", action="store_true",
                        help="Use continuous sigma (VE) sampling")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=10.0)

    # E2E options
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of diffusion samples for E2E model")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint dir)")

    args = parser.parse_args()

    # Validate: either e2e or both stage1+stage2
    if args.e2e_checkpoint is None and (args.stage1_checkpoint is None or args.stage2_checkpoint is None):
        parser.error("Either --e2e_checkpoint or both --stage1_checkpoint and --stage2_checkpoint required")

    return args


def main():
    args = parse_args()

    # Determine mode
    e2e_mode = args.e2e_checkpoint is not None

    # Setup output
    if args.output_dir is None:
        if e2e_mode:
            args.output_dir = os.path.dirname(args.e2e_checkpoint)
        else:
            args.output_dir = os.path.dirname(args.stage2_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    if e2e_mode:
        print("E2E Evaluation: ResFoldE2E -> DockQ")
    else:
        print("Two-Stage Evaluation: Stage 1 -> Stage 2 -> DockQ")
    print("=" * 70)
    if e2e_mode:
        print(f"E2E checkpoint: {args.e2e_checkpoint}")
        print(f"N samples (K): {args.n_samples}")
    else:
        print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
        print(f"Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"Split: {args.load_split}")
    print(f"N train eval: {args.n_train_eval}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data/processed/samples.parquet")
    table = pq.read_table(data_path)

    # Load split
    train_indices, test_indices, split_info = load_split(args.load_split)
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print()

    if e2e_mode:
        # Load E2E model
        print("Loading E2E model...")
        e2e_ckpt = torch.load(args.e2e_checkpoint, map_location=device, weights_only=False)
        e2e_args = e2e_ckpt.get('args', {})

        e2e_model = ResFoldE2E(
            c_token=e2e_args.get('c_token', 256),
            trunk_layers=e2e_args.get('trunk_layers', 9),
            denoiser_blocks=e2e_args.get('denoiser_blocks', 7),
            n_timesteps=e2e_args.get('T', 50),
            s2_layers=e2e_args.get('s2_layers', 6),
            s2_heads=e2e_args.get('s2_heads', 8),
            n_samples=e2e_args.get('n_samples', args.n_samples),
            s2_aggregation=e2e_args.get('s2_aggregation', 'learned'),
        ).to(device)
        e2e_model.load_state_dict(e2e_ckpt['model_state_dict'])
        e2e_model.eval()

        counts = e2e_model.count_parameters()
        print(f"  Loaded E2E: {counts['total']:,} params")
        print(f"    Stage 1: {counts['stage1']:,}")
        print(f"    Stage 2: {counts['stage2']:,}")
        print()

        # Setup diffusion
        schedule = KarrasSchedule(
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            n_steps=args.T,
        )
        noiser = VENoiser(schedule, sigma_data=1.0).to(device)

        stage1_model = None
        stage2_model = None
        continuous_sigma = True

    else:
        # Load Stage 1 model
        print("Loading Stage 1 model...")
        s1_ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        s1_args = s1_ckpt.get('args', {})

        # Detect model config from checkpoint
        c_token_s1 = s1_args.get('c_token_s1', 256)
        trunk_layers = s1_args.get('trunk_layers', 14)
        denoiser_blocks = s1_args.get('denoiser_blocks', 10)
        continuous_sigma = s1_args.get('continuous_sigma', True)

        stage1_model = ResidueDenoiser(
            c_token=c_token_s1,
            trunk_layers=trunk_layers,
            denoiser_blocks=denoiser_blocks,
            n_timesteps=args.T,
        ).to(device)

        # Load weights (handle both pipeline and standalone checkpoints)
        state_dict = s1_ckpt['model_state_dict']
        if any(k.startswith('stage1.') for k in state_dict.keys()):
            # Pipeline checkpoint - extract stage1 weights
            state_dict = {k.replace('stage1.', ''): v for k, v in state_dict.items() if k.startswith('stage1.')}
        stage1_model.load_state_dict(state_dict)
        stage1_model.eval()
        print(f"  Loaded Stage 1: {sum(p.numel() for p in stage1_model.parameters()):,} params")
        print(f"  continuous_sigma: {continuous_sigma}")
        print()

        # Load Stage 2 model
        print("Loading Stage 2 model...")
        s2_ckpt = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
        s2_args = s2_ckpt.get('args', {})

        stage2_model = AtomRefinerContinuous(
            c_token=s2_args.get('c_token', 256),
            c_atom=s2_args.get('c_atom', 128),
            trunk_layers=s2_args.get('trunk_layers', 3),
            refine_layers=s2_args.get('refine_layers', 3),
            local_atom_blocks=s2_args.get('local_atom_blocks', 2),
        ).to(device)
        stage2_model.load_state_dict(s2_ckpt['model_state_dict'])
        stage2_model.eval()
        print(f"  Loaded Stage 2: {sum(p.numel() for p in stage2_model.parameters()):,} params")
        print()

        e2e_model = None

        # Setup diffusion for Stage 1
        if continuous_sigma:
            schedule = KarrasSchedule(
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                n_steps=args.T,
            )
            sigmas = schedule.sigmas.to(device)
            noiser = None
        else:
            schedule = create_schedule("cosine", T=args.T)
            noiser = create_noiser("gaussian", schedule)
            sigmas = None

    # Evaluate
    results = {'train': [], 'test': []}

    # Select train samples
    eval_train_indices = random.sample(train_indices, min(args.n_train_eval, len(train_indices)))

    for split_name, indices in [('train', eval_train_indices), ('test', test_indices)]:
        print(f"\nEvaluating {split_name} set ({len(indices)} samples)...")

        for i, idx in enumerate(indices):
            sample = load_sample_raw(table, idx, normalize=True)

            # Prepare inputs
            aa_seq = sample['aa_seq'].unsqueeze(0).to(device)
            chain_ids = sample['chain_ids'].unsqueeze(0).to(device)
            res_idx = sample['res_idx'].unsqueeze(0).to(device)
            mask = torch.ones(1, sample['n_res'], dtype=torch.bool, device=device)
            gt_centroids = sample['centroids'].unsqueeze(0).to(device)
            gt_atoms = sample['coords_res'].unsqueeze(0).to(device)

            if e2e_mode:
                # E2E model: sample_e2e returns both centroids and atoms
                result = sample_e2e(
                    e2e_model, aa_seq, chain_ids, res_idx, noiser, mask,
                    n_samples=args.n_samples,
                    self_cond=True,
                    align_per_step=True,
                    recenter=True,
                )
                pred_centroids = result['mean_centroids']
                pred_atoms = result['atoms_pred']
            else:
                # Separate models
                # Stage 1: Predict centroids
                if continuous_sigma:
                    pred_centroids = sample_centroids_ve(
                        stage1_model, aa_seq, chain_ids, res_idx, mask, sigmas, device
                    )
                else:
                    pred_centroids = sample_centroids_ddpm(
                        stage1_model, aa_seq, chain_ids, res_idx, mask, noiser, device
                    )

                # Stage 2: Predict atoms from predicted centroids
                pred_atoms = stage2_model(aa_seq, chain_ids, res_idx, pred_centroids, mask)

            # Compute centroid RMSE
            centroid_rmse = compute_rmse(pred_centroids, gt_centroids, mask).item() * sample['std']

            # Compute atom RMSE
            L = sample['n_res']
            pred_flat = pred_atoms.view(1, L * 4, 3)
            gt_flat = gt_atoms.view(1, L * 4, 3)
            mask_flat = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(1, L * 4)
            gt_aligned, pred_c = kabsch_align(gt_flat, pred_flat, mask_flat)
            atom_rmse = torch.sqrt(((pred_c - gt_aligned) ** 2).sum(-1).mean()).item() * sample['std']

            # Compute DockQ (detach tensors)
            dockq_result = compute_dockq(
                pred_atoms[0].detach(),
                gt_atoms[0].detach(),
                aa_seq[0].detach(),
                chain_ids[0].detach(),
                std=sample['std'],
            )

            results[split_name].append({
                'sample_id': sample['sample_id'],
                'n_res': sample['n_res'],
                'centroid_rmse': centroid_rmse,
                'atom_rmse': atom_rmse,
                'dockq': dockq_result['dockq'],
                'fnat': dockq_result['fnat'],
                'irms': dockq_result['irms'],
                'lrms': dockq_result['lrms'],
            })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(indices)}: {sample['sample_id']} | "
                      f"centroid RMSE: {centroid_rmse:.2f}A | "
                      f"atom RMSE: {atom_rmse:.2f}A | "
                      f"DockQ: {dockq_result['dockq'] or 0:.3f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for split_name in ['train', 'test']:
        r = results[split_name]
        if not r:
            continue

        centroid_rmses = [x['centroid_rmse'] for x in r]
        atom_rmses = [x['atom_rmse'] for x in r]
        dockqs = [x['dockq'] for x in r if x['dockq'] is not None]

        print(f"\n{split_name.upper()} ({len(r)} samples):")
        print(f"  Centroid RMSE: {np.mean(centroid_rmses):.2f} ± {np.std(centroid_rmses):.2f} A")
        print(f"  Atom RMSE:     {np.mean(atom_rmses):.2f} ± {np.std(atom_rmses):.2f} A")
        if dockqs:
            print(f"  DockQ:         {np.mean(dockqs):.3f} ± {np.std(dockqs):.3f}")
            print(f"  DockQ >= 0.23: {sum(d >= 0.23 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.23 for d in dockqs)/len(dockqs):.1f}%)")
            print(f"  DockQ >= 0.49: {sum(d >= 0.49 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.49 for d in dockqs)/len(dockqs):.1f}%)")
            print(f"  DockQ >= 0.80: {sum(d >= 0.80 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.80 for d in dockqs)/len(dockqs):.1f}%)")

    # Save results
    output_path = os.path.join(args.output_dir, 'two_stage_eval.json')
    with open(output_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'summary': {
                split_name: {
                    'centroid_rmse_mean': np.mean([x['centroid_rmse'] for x in r]),
                    'centroid_rmse_std': np.std([x['centroid_rmse'] for x in r]),
                    'atom_rmse_mean': np.mean([x['atom_rmse'] for x in r]),
                    'atom_rmse_std': np.std([x['atom_rmse'] for x in r]),
                    'dockq_mean': np.mean([x['dockq'] for x in r if x['dockq'] is not None]),
                    'dockq_std': np.std([x['dockq'] for x in r if x['dockq'] is not None]),
                    'n_samples': len(r),
                }
                for split_name, r in results.items() if r
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
