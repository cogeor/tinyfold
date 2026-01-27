#!/usr/bin/env python
"""Evaluate atom-level diffusion model with DockQ scores.

Usage:
    python scripts/eval_atom_dockq.py \
        --checkpoint outputs/atom_diffusion_50K_ve/best_model.pt \
        --load_split outputs/train_10k_continuous/split.json \
        --n_train 100 --n_test 100
"""

import os
import sys
import argparse
import random
import torch
import numpy as np
import pyarrow.parquet as pq

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.af3_style import AF3StyleDecoder
from models.diffusion import KarrasSchedule, VENoiser
from models.dockq_utils import compute_dockq
from models import kabsch_align_to_target
from tinyfold.model.losses import kabsch_align, compute_rmse
from data_split import load_split


def load_sample_raw(table, i, normalize=True):
    """Load sample without batching."""
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
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

    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Expand residue-level to atom-level
    aa_seq = seq_res.repeat_interleave(4)
    chain_ids = chain_res.repeat_interleave(4)

    return {
        'coords': coords,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'chain_ids_res': chain_res,
        'aa_seq_res': seq_res,
        'n_atoms': n_atoms,
        'n_res': n_res,
        'std': std.item(),
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples, device):
    """Collate samples into a padded batch."""
    B = len(samples)
    max_atoms = max(s['n_atoms'] for s in samples)

    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    aa_seq = torch.zeros(B, max_atoms, dtype=torch.long)
    chain_ids = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s['n_atoms']
        coords[i, :n] = s['coords']
        atom_types[i, :n] = s['atom_types']
        atom_to_res[i, :n] = s['atom_to_res']
        aa_seq[i, :n] = s['aa_seq']
        chain_ids[i, :n] = s['chain_ids']
        mask[i, :n] = True

    return {
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'mask': mask.to(device),
    }


@torch.no_grad()
def sample_ve(model, atom_types, atom_to_res, aa_seq, chain_ids, noiser, mask=None,
              clamp_val=3.0, align_per_step=True, recenter=True):
    """VE (Euler) sampling for af3_style with continuous sigma."""
    device = atom_types.device
    B, N = atom_types.shape

    sigmas = noiser.sigmas.to(device)
    x = sigmas[0] * torch.randn(B, N, 3, device=device)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = sigma.expand(B)

        x0_pred = model.forward_sigma(x, atom_types, atom_to_res, aa_seq, chain_ids,
                                       sigma_batch, mask)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        d = (x - x0_pred) / sigma
        dt = sigma_next - sigma
        x = x + d * dt

        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    return x


def main():
    parser = argparse.ArgumentParser(description="Evaluate atom-level model with DockQ")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--load_split", type=str, required=True, help="Path to split.json")
    parser.add_argument("--data_path", type=str, default="data/processed/dips_train.parquet")
    parser.add_argument("--n_train", type=int, default=100, help="Number of train samples to evaluate")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--T", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load split
    train_indices, test_indices, _ = load_split(args.load_split)
    print(f"Split: {len(train_indices)} train, {len(test_indices)} test")

    # Load data
    print(f"Loading data from {args.data_path}...")
    table = pq.read_table(args.data_path)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})

    # Create model (c_token = h_dim * 2 per train.py)
    h_dim = ckpt_args.get('h_dim', 128)
    trunk_layers = ckpt_args.get('trunk_layers', 5)
    denoiser_blocks = ckpt_args.get('denoiser_blocks', 5)

    model = AF3StyleDecoder(
        c_token=h_dim * 2,  # 256 for h_dim=128
        c_atom=h_dim,
        trunk_layers=trunk_layers,
        denoiser_blocks=denoiser_blocks,
        n_timesteps=args.T,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded (step {ckpt.get('step', 'unknown')})")

    # Create noiser
    karras_schedule = KarrasSchedule(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        n_steps=args.T,
    )
    noiser = VENoiser(karras_schedule, sigma_data=1.0).to(device)

    # Evaluate
    results = {'train': [], 'test': []}

    for split_name, indices, n_samples in [
        ('train', train_indices, args.n_train),
        ('test', test_indices, args.n_test),
    ]:
        eval_indices = random.sample(indices, min(n_samples, len(indices)))
        print(f"\nEvaluating {split_name} ({len(eval_indices)} samples)...")

        rmses = []
        dockqs = []

        for i, idx in enumerate(eval_indices):
            sample = load_sample_raw(table, idx)
            batch = collate_batch([sample], device)

            # Sample
            x_pred = sample_ve(model, batch['atom_types'], batch['atom_to_res'],
                               batch['aa_seq'], batch['chain_ids'], noiser, batch['mask'])

            # RMSE
            n = sample['n_atoms']
            pred = x_pred[0, :n]
            gt = batch['coords'][0, :n]
            pred_aligned, gt_aligned = kabsch_align(pred.unsqueeze(0), gt.unsqueeze(0))
            rmse = compute_rmse(pred_aligned, gt_aligned).item() * sample['std']
            rmses.append(rmse)

            # DockQ - reshape to [L, 4, 3]
            n_res = sample['n_res']
            pred_res = pred_aligned[0].view(n_res, 4, 3)
            gt_res = gt_aligned[0].view(n_res, 4, 3)

            dockq_result = compute_dockq(
                pred_res, gt_res,
                sample['aa_seq_res'], sample['chain_ids_res'],
                std=sample['std']
            )

            if dockq_result['dockq'] is not None:
                dockqs.append(dockq_result['dockq'])
                results[split_name].append({
                    'sample_id': sample['sample_id'],
                    'rmse': rmse,
                    'dockq': dockq_result['dockq'],
                    'fnat': dockq_result['fnat'],
                    'irms': dockq_result['irms'],
                    'lrms': dockq_result['lrms'],
                })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(eval_indices)} - RMSE: {rmse:.2f}A, DockQ: {dockq_result['dockq'] or 0:.3f}")

        print(f"\n{split_name.upper()} Results:")
        print(f"  RMSE:  {np.mean(rmses):.2f} ± {np.std(rmses):.2f} A")
        if dockqs:
            print(f"  DockQ: {np.mean(dockqs):.3f} ± {np.std(dockqs):.3f}")
            print(f"  DockQ >= 0.23: {sum(d >= 0.23 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.23 for d in dockqs)/len(dockqs):.1f}%)")
            print(f"  DockQ >= 0.49: {sum(d >= 0.49 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.49 for d in dockqs)/len(dockqs):.1f}%)")
            print(f"  DockQ >= 0.80: {sum(d >= 0.80 for d in dockqs)}/{len(dockqs)} ({100*sum(d >= 0.80 for d in dockqs)/len(dockqs):.1f}%)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name in ['train', 'test']:
        r = results[split_name]
        if r:
            rmses = [x['rmse'] for x in r]
            dockqs = [x['dockq'] for x in r]
            print(f"\n{split_name.upper()} ({len(r)} samples):")
            print(f"  RMSE:  {np.mean(rmses):.2f} A")
            print(f"  DockQ: {np.mean(dockqs):.3f}")


if __name__ == "__main__":
    main()
