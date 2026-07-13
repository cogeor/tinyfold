#!/usr/bin/env python
"""Measure peak VRAM usage at a range of model sizes.

Runs one forward + backward pass with synthetic ESM-shaped inputs at
batch_size 32, L=1200 (Phase D's worst-case shape) and reports peak
allocated + peak reserved CUDA memory. Sweeps c_token / trunk_layers /
denoiser_blocks to chart where the 4070 Ti SUPER's 16 GB budget runs
out. Stops on OOM, prints the surviving configs.

Usage:
    python scripts/vram_probe.py [--batch 32] [--len 1200] [--confidence-head]

The probe constructs ResFoldOneStep directly (no real data, no parquet,
no ESM forward) and exercises the trunk + denoiser + atom head paths
with fake `esm_embed` input.
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep


def measure_peak_vram(
    c_token: int,
    trunk_layers: int,
    denoiser_blocks: int,
    batch_size: int,
    seq_len: int,
    use_confidence_head: bool,
    use_esm: bool,
    esm_dim: int = 480,
    device: str = "cuda",
) -> tuple[float, float, int]:
    """Build model, run fwd+bwd at the requested shape, return peak VRAM.

    Returns:
        (peak_alloc_mib, peak_reserved_mib, n_trainable_params)
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    kwargs = dict(
        c_token=c_token,
        trunk_layers=trunk_layers,
        denoiser_blocks=denoiser_blocks,
        atom_head_layers=2,
        atom_head_heads=4,
        n_aa_types=23,
        n_chains=2,
        confidence_head=use_confidence_head,
    )
    if use_esm:
        kwargs["aa_embed"] = "esm2_35M"
        kwargs["esm_dim"] = esm_dim
    model = ResFoldOneStep(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    aa_seq = torch.randint(0, 20, (batch_size, seq_len), device=device)
    chain_ids = torch.cat([
        torch.zeros(batch_size, seq_len // 2, dtype=torch.long, device=device),
        torch.ones(batch_size, seq_len - seq_len // 2, dtype=torch.long, device=device),
    ], dim=1)
    res_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask_res = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    x_noisy = torch.randn(batch_size, seq_len, 3, device=device)
    sigma = torch.full((batch_size,), 1.0, device=device)
    esm_embed = (
        torch.randn(batch_size, seq_len, esm_dim, device=device) if use_esm else None
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    optim.zero_grad(set_to_none=True)

    out = model.forward_sigma(
        x_t=x_noisy,
        sigma=sigma,
        aa_seq=aa_seq,
        chain_ids=chain_ids,
        res_idx=res_idx,
        mask=mask_res,
        esm_embed=esm_embed,
    )
    if isinstance(out, tuple):
        centroid_pred, atoms_pred = out[0], out[1]
    else:
        centroid_pred = out
        atoms_pred = None

    loss = (centroid_pred ** 2).mean()
    if atoms_pred is not None:
        loss = loss + (atoms_pred ** 2).mean()
    loss.backward()
    optim.step()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    del model, optim, aa_seq, chain_ids, res_idx, mask_res, x_noisy, sigma, esm_embed
    del out, centroid_pred, atoms_pred, loss
    torch.cuda.empty_cache()
    gc.collect()
    return peak_alloc, peak_reserved, n_params


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--len", type=int, default=1200, dest="seq_len")
    p.add_argument("--confidence-head", action="store_true", default=True)
    p.add_argument("--no-esm", action="store_true", help="Skip ESM-2 (learned mode)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        sys.exit(1)

    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  ({gpu_total:.0f} MiB total)")
    print(f"Probe shape: batch={args.batch}, L={args.seq_len}, "
          f"conf_head={args.confidence_head}, esm={not args.no_esm}\n")

    # Configs to sweep, ordered by increasing memory pressure.
    # Each: (label, c_token, trunk, denoiser)
    configs: list[tuple[str, int, int, int]] = [
        ("Phase D baseline",      256, 6,  6),
        ("Deeper 8+8",            256, 8,  8),
        ("Deeper 10+10",          256, 10, 10),
        ("Wider 384",             384, 6,  6),
        ("Wider+Deeper 384, 8+8", 384, 8,  8),
        ("Wider 512",             512, 6,  6),
        ("Wider 512, 8+8",        512, 8,  8),
        ("Big 768",               768, 6,  6),
        ("Big 768, 8+8",          768, 8,  8),
    ]

    rows = []
    print(f"{'Config':<30} {'Params':>10}   {'Peak alloc':>11}   {'Reserved':>11}   {'% of 16GB':>9}")
    print("-" * 80)
    for label, c, tl, dl in configs:
        try:
            alloc, reserved, n_params = measure_peak_vram(
                c_token=c,
                trunk_layers=tl,
                denoiser_blocks=dl,
                batch_size=args.batch,
                seq_len=args.seq_len,
                use_confidence_head=args.confidence_head,
                use_esm=not args.no_esm,
            )
            pct = 100.0 * reserved / gpu_total
            print(f"{label:<30} {n_params/1e6:>8.2f}M  {alloc:>9.0f} MiB  {reserved:>9.0f} MiB  {pct:>7.1f}%")
            rows.append((label, c, tl, dl, n_params, alloc, reserved, pct))
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<30} OOM (param count too high or activations too large)")
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as e:
            print(f"{label:<30} ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            break

    print()
    print("Notes:")
    print("- Peak reserved is the headroom you actually need; alloc is the working set.")
    print("- ESM-2 frozen forward is NOT in this budget — embeddings are loaded from")
    print("  cache at training time (mmap).")
    print("- Real training adds ~5-10% from optimizer state warmup + DataLoader buffers.")


if __name__ == "__main__":
    main()
