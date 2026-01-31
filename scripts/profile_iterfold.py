#!/usr/bin/env python
"""Profile IterFold to find bottlenecks."""

import torch
import time
from models.iterfold import IterFold

def profile_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Test different configs
    configs = [
        {'name': 'default', 'trunk_layers': 9, 'decoder_layers': 12, 'n_atom_layers': 8},
        {'name': 'small', 'trunk_layers': 4, 'decoder_layers': 4, 'n_atom_layers': 4},
        {'name': 'tiny', 'trunk_layers': 2, 'decoder_layers': 2, 'n_atom_layers': 2},
    ]

    B, L = 1, 100  # single sample, 100 residues

    for cfg in configs:
        model = IterFold(
            c_token=256,
            trunk_layers=cfg['trunk_layers'],
            decoder_layers=cfg['decoder_layers'],
            n_atom_layers=cfg['n_atom_layers'],
        ).to(device)

        params = model.count_parameters()

        aa_seq = torch.randint(0, 20, (B, L), device=device)
        chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        res_idx = torch.arange(L, device=device).unsqueeze(0)
        anchor_pos = torch.randn(B, L, 3, device=device) * 0.3
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Warmup
        model.eval()
        with torch.no_grad():
            _ = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Time forward pass
        n_iters = 10
        start = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / n_iters * 1000

        name = cfg['name']
        total_params = params['total'] / 1e6
        print(f'{name}: {total_params:.1f}M params, {elapsed:.1f}ms/forward')

        # Time backward
        model.train()
        model.zero_grad()
        start = time.time()
        for _ in range(n_iters):
            out = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
            loss = out.sum()
            loss.backward()
            model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / n_iters * 1000
        print(f'  backward: {elapsed:.1f}ms/step')


def profile_components():
    """Profile individual components to find bottleneck."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nComponent profiling (device: {device})')
    print('=' * 50)

    B, L = 1, 100
    c_token = 256

    model = IterFold(
        c_token=c_token,
        trunk_layers=4,
        decoder_layers=4,
        n_atom_layers=4,
    ).to(device)

    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0)
    anchor_pos = torch.randn(B, L, 3, device=device) * 0.3
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    n_iters = 20

    # Profile trunk
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            trunk_tokens = model.trunk(aa_seq, chain_ids, res_idx, mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    trunk_time = (time.time() - start) / n_iters * 1000
    print(f'Trunk encoder: {trunk_time:.1f}ms')

    # Profile decoder
    trunk_tokens = model.trunk(aa_seq, chain_ids, res_idx, mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model.decoder(trunk_tokens, anchor_pos, mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    decoder_time = (time.time() - start) / n_iters * 1000
    print(f'Anchor decoder: {decoder_time:.1f}ms')

    # Profile decoder internals
    decoder = model.decoder

    # Position embedding
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)
            pos_feat = decoder.pos_embed(anchor_pos)
            pos_feat = torch.where(
                is_anchored.unsqueeze(-1),
                pos_feat,
                decoder.unknown_embed.expand(B, L, -1)
            )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pos_time = (time.time() - start) / n_iters * 1000
    print(f'  - pos_embed: {pos_time:.1f}ms')

    # Main transformer
    h = trunk_tokens + pos_feat
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            h_out = decoder.transformer(h, src_key_padding_mask=~mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    transformer_time = (time.time() - start) / n_iters * 1000
    print(f'  - main transformer: {transformer_time:.1f}ms')

    # Atom queries + refinement (cross-attention only now)
    atom_queries = h_out.unsqueeze(2) + decoder.atom_type_embed
    atom_queries = atom_queries.view(B, L * 4, c_token)
    res_key_mask = ~mask

    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            aq = atom_queries.clone()
            for i in range(decoder.n_atom_layers):
                q = decoder.cross_norms[i](aq)
                attn_out, _ = decoder.cross_attn_layers[i](q, h_out, h_out, key_padding_mask=res_key_mask)
                aq = aq + attn_out
                aq = aq + decoder.ffn_layers[i](aq)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    atom_time = (time.time() - start) / n_iters * 1000
    print(f'  - atom refinement ({decoder.n_atom_layers} layers, cross-attn only): {atom_time:.1f}ms')

    print(f'\nNote: Atom refinement now O(4L * L) = O({4*L*L}) per layer (was O({(L*4)**2}))')


if __name__ == '__main__':
    profile_model()
    profile_components()
