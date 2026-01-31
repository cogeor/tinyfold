#!/usr/bin/env python
"""Profile individual components of train_step."""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.iterfold import IterFold
from models.clustering import select_next_residues_to_place
from tinyfold.model.losses import kabsch_align

def profile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    B, L = 8, 150
    n_iters = 20

    # Model
    model = IterFold(c_token=128, trunk_layers=4, decoder_layers=4, n_atom_layers=2).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Inputs
    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    anchor_pos = torch.randn(B, L, 3, device=device) * 0.3
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    gt_atoms = torch.randn(B, L, 4, 3, device=device)

    # Warmup
    model.train()
    pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
    loss = pred.sum()
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 1. Model forward
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fwd_time = (time.time() - start) / n_iters * 1000
    print(f'1. Model forward: {fwd_time:.1f}ms')

    # 2. Kabsch alignment
    pred_flat = pred.view(B, -1, 3)
    gt_flat = gt_atoms.view(B, -1, 3)
    atom_mask = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, -1)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        gt_aligned, pred_c = kabsch_align(gt_flat, pred_flat.detach(), atom_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    kabsch_time = (time.time() - start) / n_iters * 1000
    print(f'2. Kabsch align: {kabsch_time:.1f}ms')

    # 3. Clustering selection (the loop)
    anchor_mask = torch.rand(B, L, device=device) < 0.2
    pred_centroids = pred.mean(dim=2).detach()

    start = time.time()
    for _ in range(n_iters):
        next_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            next_res = select_next_residues_to_place(pred_centroids[b], anchor_mask[b], 8)
            next_mask[b, next_res] = True
    cluster_time = (time.time() - start) / n_iters * 1000
    print(f'3. Clustering selection (B={B}): {cluster_time:.1f}ms')

    # 4. Full forward + backward (no kabsch, no cluster)
    model.zero_grad()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        model.zero_grad()
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        loss = ((pred - gt_atoms) ** 2).mean()
        loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fwd_bwd_time = (time.time() - start) / n_iters * 1000
    print(f'4. Forward+backward (simple loss): {fwd_bwd_time:.1f}ms')

    # 5. Full forward + backward WITH kabsch
    model.zero_grad()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        model.zero_grad()
        pred = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        pred_flat = pred.view(B, -1, 3)
        gt_flat = gt_atoms.view(B, -1, 3)
        gt_aligned, pred_c = kabsch_align(gt_flat, pred_flat, atom_mask)
        gt_aligned = gt_aligned.detach()
        loss = ((pred_c - gt_aligned) ** 2).mean()
        loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    full_time = (time.time() - start) / n_iters * 1000
    print(f'5. Forward+backward WITH Kabsch: {full_time:.1f}ms')

    print(f'\nKabsch overhead: {full_time - fwd_bwd_time:.1f}ms')
    print(f'Expected step time (with cluster): {full_time + cluster_time:.1f}ms')


if __name__ == '__main__':
    profile()
