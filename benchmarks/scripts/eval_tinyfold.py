"""Run a TinyFold checkpoint on stratified bins and write NPZ predictions.

Auto-detects model config (aa_embed, per_chain_res_idx, confidence_head,
sigma_data, T, sigma_min, sigma_max) from the config.json next to the
checkpoint, so a single CLI invocation works for Phase D, F-medium,
F-medium-ESM, or any future ResFoldOneStep checkpoint with a sibling
config.json.

Outputs land in benchmarks/predictions/{model_tag}/{split}/{sample_id}.npz
with `pred_atoms: [L, 4, 3]` in real Angstroms (un-normalized, restored to
the parquet's original frame), `sample_id`, and `model_tag`. Compatible with
benchmarks/scripts/compute_metrics.py.

For the VE/Karras schedule: a minimal Euler sampler is built inline rather
than imported from scripts/train_resfold.py — that file has 2000+ lines of
train-time setup and pulling it in just for sampling would be heavy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.training.data import load_sample

SPLITS = [
    "le200", "200_400", "400_600", "600_1000", "ge1000",
    "clean_le200", "clean_200_400", "clean_400_600", "clean_600_1000", "clean_ge1000",
]


def _load(checkpoint_path: Path, device: str) -> tuple[ResFoldOneStep, dict]:
    cfg_path = checkpoint_path.parent / "config.json"
    if not cfg_path.exists():
        sys.exit(f"ERROR: no config.json next to {checkpoint_path}")
    cfg = json.load(open(cfg_path))
    model = ResFoldOneStep(
        c_token=cfg.get("c_token_s1", 256),
        trunk_layers=cfg.get("trunk_layers", 6),
        denoiser_blocks=cfg.get("denoiser_blocks", 6),
        atom_head_layers=cfg.get("atom_head_layers", 2),
        atom_head_heads=cfg.get("atom_head_heads", 4),
        n_timesteps=cfg.get("T", 50),
        aa_embed=cfg.get("aa_embed", "learned"),
        confidence_head=cfg.get("confidence_head", False),
        sigma_data=cfg.get("sigma_data", 1.0),
        relpos_bias=cfg.get("relpos_bias", False),
        relpos_clip=cfg.get("relpos_clip", 32),
    ).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing keys: {len(missing)} (first: {missing[:2]})")
    if unexpected:
        print(f"  WARN unexpected keys: {len(unexpected)} (first: {unexpected[:2]})")
    model.eval()
    return model, cfg


def _build_noiser(cfg: dict, device: str) -> VENoiser:
    schedule = KarrasSchedule(
        n_steps=cfg.get("T", 50),
        sigma_min=cfg.get("sigma_min", 0.002),
        sigma_max=cfg.get("sigma_max", 10.0),
        rho=7.0,
    )
    return VENoiser(schedule, sigma_data=cfg.get("sigma_data", 1.0)).to(device)


@torch.no_grad()
def _sample_ve(
    model: ResFoldOneStep,
    noiser: VENoiser,
    aa_seq: torch.Tensor,        # [1, L]
    chain_ids: torch.Tensor,     # [1, L]
    res_idx: torch.Tensor,       # [1, L]
    mask: torch.Tensor,          # [1, L] bool
    esm_embed: torch.Tensor | None,
    device: str,
    clamp_val: float = 3.0,
    generator: torch.Generator | None = None,
    n_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal VE Euler sampler, returns (centroids, atoms) both [1, L, *, 3].

    ``n_steps``: if given and smaller than the full schedule, the Karras
    sigma grid is subsampled to ``n_steps + 1`` evenly-spaced indices
    (sigma_max and sigma_min always preserved). ``None`` uses the full
    schedule from the noiser. Tests Protenix-Mini's claim (arXiv:2507.11839)
    that a 2-step ODE sampler is nearly identical to a 200-step run.
    """
    B, L = aa_seq.shape
    sigmas_full = noiser.sigmas.to(device)
    if n_steps is not None and n_steps + 1 < len(sigmas_full):
        idx = torch.linspace(0, len(sigmas_full) - 1, n_steps + 1).long().to(device)
        sigmas = sigmas_full[idx]
    else:
        sigmas = sigmas_full
    x = sigmas[0] * torch.randn(B, L, 3, device=device, generator=generator)
    x0_prev = None
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i].expand(B)
        out = model.forward_sigma(
            x, aa_seq, chain_ids, res_idx, sigma, mask,
            x0_prev=x0_prev, esm_embed=esm_embed,
        )
        x0_pred = out[0]
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
        x0_prev = x0_pred.detach()
        d = (x - x0_pred) / sigmas[i]
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
    # Final forward at sigma_min to read the atom head.
    sigma_min_b = sigmas[-1].expand(B)
    out_final = model.forward_sigma(
        x, aa_seq, chain_ids, res_idx, sigma_min_b, mask,
        x0_prev=x0_prev, esm_embed=esm_embed,
    )
    centroids = out_final[0]
    atoms = out_final[1]
    return centroids, atoms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model_tag", required=True,
                   help="Sub-directory under benchmarks/predictions/.")
    p.add_argument("--parquet", default="data/processed/samples.parquet")
    p.add_argument("--splits", nargs="*", default=["clean_200_400", "clean_400_600", "clean_600_1000", "clean_ge1000"],
                   help=f"Subset of bins. Defaults to all clean_* (skip le200 — only 1 sample). "
                        f"Allowed: {SPLITS}.")
    p.add_argument("--splits_root", default="benchmarks/splits")
    p.add_argument("--predictions_root", default="benchmarks/predictions")
    p.add_argument("--esm_cache_dir", default=None,
                   help="Override the config's esm_cache_dir (auto-detected by default).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=None,
                   help="Subsample the Karras schedule to this many denoising "
                        "steps (default: full schedule from the noiser).")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Draws per target. K=1 keeps the prior single-generator "
                        "behavior so old baseline CSVs stay reproducible. K>1 "
                        "writes {sample_id}_k{i}.npz with a per-(sample, k) "
                        "seed = args.seed + sample_idx * K + i, giving K "
                        "distinct deterministic draws.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip samples whose NPZ already exists.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, cfg = _load(args.checkpoint, device=device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  {n_params:,} params")
    print(f"  aa_embed={cfg.get('aa_embed')}, confidence_head={cfg.get('confidence_head')}, "
          f"per_chain_res_idx={cfg.get('per_chain_res_idx', False)}")

    noiser = _build_noiser(cfg, device=device)
    print(f"Noiser: VE Karras, T={cfg.get('T')}, sigma range [{cfg.get('sigma_min')}, {cfg.get('sigma_max')}]")
    if args.n_steps is not None:
        print(f"  Sampler: {args.n_steps}-step subsampled Karras (Protenix-Mini-style)")
    if args.n_samples > 1:
        print(f"  K={args.n_samples} draws/target, seed = {args.seed} + sample_idx * K + k")

    esm_cache = args.esm_cache_dir or cfg.get("esm_cache_dir")
    per_chain = bool(cfg.get("per_chain_res_idx", False))
    if esm_cache is not None:
        esm_cache = str(esm_cache).replace("\\", "/")
        print(f"  ESM cache: {esm_cache}")

    table = pq.read_table(args.parquet)
    print(f"Parquet: {len(table)} samples")

    pred_root = REPO_ROOT / args.predictions_root / args.model_tag
    pred_root.mkdir(parents=True, exist_ok=True)

    # K=1 reuses one global generator (preserves old baseline reproducibility).
    # K>1 reseeds per (sample, k) so a single sample's draws don't depend on
    # iteration order across bins.
    K = args.n_samples
    shared_gen = torch.Generator(device=device).manual_seed(args.seed) if K == 1 else None

    def out_name(sid: str, k_idx: int) -> str:
        return f"{sid}.npz" if K == 1 else f"{sid}_k{k_idx}.npz"

    for split in args.splits:
        split_file = REPO_ROOT / args.splits_root / f"{split}.json"
        if not split_file.exists():
            print(f"  [skip] {split}: {split_file} not found")
            continue
        d = json.load(open(split_file))
        out_dir = pred_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        test_indices = d["test_indices"]
        test_ids = d["test_ids"]

        print(f"  {split}: {len(test_ids)} samples x K={K} -> {out_dir}")
        n_written = 0
        for sample_idx, (idx, sid) in enumerate(zip(test_indices, test_ids)):
            # Skip parquet load + ESM load if every K output for this sample exists.
            out_paths = [out_dir / out_name(sid, k) for k in range(K)]
            if args.skip_existing and all(p.exists() for p in out_paths):
                continue

            # Raw centroid for un-normalization (must come from parquet before
            # load_sample re-centers).
            coords_raw = np.asarray(table["atom_coords"][idx].as_py(), dtype=np.float32)
            n_atoms = coords_raw.shape[0] // 3
            raw_centroid = coords_raw.reshape(n_atoms, 3).mean(axis=0)  # [3]

            sample = load_sample(
                table, idx,
                normalize=True,
                esm_cache_dir=esm_cache,
                per_chain_res_idx=per_chain,
            )
            L = sample["n_res"]
            aa = sample["aa_seq"].unsqueeze(0).to(device)
            chains = sample["chain_ids"].unsqueeze(0).to(device)
            res_idx = sample["res_idx"].unsqueeze(0).to(device)
            mask = torch.ones(1, L, dtype=torch.bool, device=device)
            esm = (
                sample["esm_embed"].unsqueeze(0).to(device)
                if "esm_embed" in sample else None
            )
            std = float(sample["std"])

            for k_idx, out_path in enumerate(out_paths):
                if args.skip_existing and out_path.exists():
                    continue
                if K == 1:
                    gen = shared_gen
                else:
                    gen = torch.Generator(device=device).manual_seed(
                        args.seed + sample_idx * K + k_idx
                    )
                _, atoms = _sample_ve(
                    model, noiser, aa, chains, res_idx, mask, esm,
                    device=device, generator=gen, n_steps=args.n_steps,
                )
                atoms_np = atoms.squeeze(0).cpu().numpy().reshape(L, 4, 3)
                atoms_real = atoms_np * std + raw_centroid  # back to parquet frame
                np.savez(
                    out_path,
                    pred_atoms=atoms_real.astype(np.float32),
                    sample_id=sid,
                    model_tag=args.model_tag,
                    k_idx=k_idx,
                )
                n_written += 1
        print(f"    wrote {n_written} NPZs (skipped {len(test_ids) * K - n_written})")

    print(f"\nScore with: python benchmarks/scripts/compute_metrics.py --model {args.model_tag}")


if __name__ == "__main__":
    main()
