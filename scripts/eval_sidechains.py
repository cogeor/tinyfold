"""All-atom sidechain evaluation for a trained third-stage model (C4 readout).

Standalone on purpose: it does NOT touch the training/eval hot path (so a running
job stays byte-identical). Loads a sidechain checkpoint, samples backbone (atom
diffusion) + chi (torsion diffusion), places all-atom coords, aligns to GT, and
reports symmetry-corrected sidechain RMSD -- overall and BINNED by backbone quality
(a poor number under a poor backbone is a backbone problem, not a packer problem).

    uv run python scripts/eval_sidechains.py \
        --run-dir outputs/resfold/sc_le200_SC1_sidechain/<run> \
        --atom14-dir data/processed/atom14 --n 200

Reports nothing for SC0 (no sidechain head) -- run it on the SC1 checkpoint; SC0 is
the backbone-quality reference the bins are read against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyfold.inference.samplers import sample_atoms_diffusion, sample_chi_diffusion
from tinyfold.model.geometry import kabsch_rigid
from tinyfold.model.resfold.config import ResFoldConfig
from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.model.resfold.sidechain_eval import build_atom14_geometry, sidechain_rmsd_binned
from tinyfold.sidechain_frames import place_chi


def load_model(run_dir: Path, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = ResFoldOneStep(**ResFoldConfig.from_config(cfg).to_kwargs()).to(device)
    ckpt = run_dir / "best_model.pt"
    if not ckpt.exists():
        ckpt = run_dir / "final_model.pt"
    state = torch.load(ckpt, map_location=device)
    # train_resfold saves under "model_state_dict"; fall back for other layouts.
    sd = state.get("model_state_dict") or state.get("model") or state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    sc_loaded = [k for k in sd if k.startswith("sc_head")]
    if not sc_loaded:
        raise ValueError(f"checkpoint has no sc_head weights: {ckpt}")
    model.eval()
    model._atom_eval_steps = cfg.get("atom_steps", 8)
    return model, cfg


@torch.no_grad()
def eval_one(model, cfg, npz_path: Path, geom, device, n_chi_steps: int):
    """One complex -> (per_res_se, per_res_n_atoms, per_res_bb_ca_err, res_mask)."""
    with np.load(npz_path) as z:
        gt14 = torch.from_numpy(np.asarray(z["coords_atom14"])).float().to(device)  # [L,14,3] A
        gt_mask = torch.from_numpy(np.asarray(z["mask_atom14"])).bool().to(device)  # [L,14]
        aatype = torch.from_numpy(np.asarray(z["seq_indices"])).long().to(device)   # [L]
    L = gt14.shape[0]
    # esm cache is required by the esm2 trunk; skip complexes without it.
    esm_dir = Path(cfg.get("esm_cache_dir") or "data/processed/esm2_35M")
    epath = esm_dir / f"{npz_path.stem}.npz"
    if not epath.exists():
        return None
    with np.load(epath, mmap_mode="r") as ez:
        esm = torch.from_numpy(np.asarray(ez["embeddings"])).float().to(device)
    if esm.shape[0] != L:
        return None

    scale = float(cfg.get("global_scale") or 11.0)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)
    batch = {
        "aa_seq": aatype.unsqueeze(0),
        "chain_ids": torch.zeros(1, L, dtype=torch.long, device=device),  # single-chain proxy for trunk pos
        "res_idx": torch.arange(L, device=device).unsqueeze(0),
        "mask_res": mask,
        "esm_embed": esm.unsqueeze(0),
    }
    # One-shot init draws at the largest sigma (== Karras sigmas[0] == sigma_max).
    sig0 = torch.tensor([float(cfg.get("sigma_max", 10.0))], device=device)
    x = sig0.view(1, 1, 1) * torch.randn(1, L, 3, device=device)
    centroid_pred, tokens, _ = model.centroid_tokens(
        x, batch["aa_seq"], batch["chain_ids"], batch["res_idx"], sig0, mask,
        x0_prev=None, esm_embed=batch["esm_embed"],
    )
    atoms_pred = sample_atoms_diffusion(
        model, tokens, centroid_pred, mask, n_steps=model._atom_eval_steps,
        sigma_min=model.atom_sigma_min, sigma_max=model.atom_sigma_max)   # [1,L,4,3] normalized
    chi = sample_chi_diffusion(model, tokens, atoms_pred, batch["aa_seq"], mask,
                               n_steps=n_chi_steps)
    placed = place_chi(chi[0], atoms_pred[0], batch["aa_seq"][0], geom)   # [L,14,3] normalized
    placed = placed * scale                                              # -> Angstrom

    # Kabsch-align predicted backbone (slots 0-3, resolved) to GT, apply to all atoms.
    bb_m = gt_mask[:, :4].reshape(-1)
    pred_bb = placed[:, :4, :].reshape(-1, 3)[bb_m]
    gt_bb = gt14[:, :4, :].reshape(-1, 3)[bb_m]
    R, t, _ = kabsch_rigid(pred_bb.unsqueeze(0), gt_bb.unsqueeze(0))
    placed_al = torch.einsum("ij,laj->lai", R[0], placed) + t[0]

    # per-residue backbone CA error (slot 1) after alignment
    ca_err = (placed_al[:, 1, :] - gt14[:, 1, :]).norm(dim=-1)            # [L]
    res_mask = gt_mask[:, 1]                                              # CA resolved
    return placed_al.unsqueeze(0), gt14.unsqueeze(0), gt_mask.unsqueeze(0), \
        aatype.unsqueeze(0), res_mask.unsqueeze(0), ca_err.unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--atom14-dir", type=Path, default=Path("data/processed/atom14"))
    ap.add_argument("--split", type=Path, default=None,
                    help="split json to pick test ids; default = first --n atom14 files")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chi-steps", type=int, default=32)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.run_dir, device)
    if not getattr(model, "sidechain_diffusion", False):
        print("this checkpoint has no sidechain head (SC0). Run on the SC1 checkpoint.")
        return 1
    geom = build_atom14_geometry(args.atom14_dir, n_complexes=200, device=device)

    if args.split is not None:
        ids = json.loads(args.split.read_text())["test_ids"][: args.n]
        paths = [args.atom14_dir / f"{s}.npz" for s in ids]
    else:
        paths = sorted(args.atom14_dir.glob("*.npz"))[: args.n]

    agg_se = agg_n = 0.0
    rows = []
    done = 0
    for p in paths:
        if not p.exists():
            continue
        out = eval_one(model, cfg, p, geom, device, args.chi_steps)
        if out is None:
            continue
        placed, gt14, gt_mask, aatype, res_mask, ca_err = out
        # quality in (0,1]: 1/(1+CA_err_A) -> higher is better, lDDT-like bins.
        bb_q = 1.0 / (1.0 + ca_err)
        r = sidechain_rmsd_binned(placed, gt14, gt_mask, aatype, res_mask, bb_quality=bb_q)
        rows.append((p.stem, r["overall"], r["n"]))
        done += 1

    if not rows:
        print("no complexes evaluated (missing ESM cache?).")
        return 1
    overall = float(np.average([r[1] for r in rows], weights=[r[2] for r in rows]))
    print(f"\nevaluated {done} complexes")
    print(f"OVERALL symmetry-corrected sidechain RMSD: {overall:.3f} A")
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "sidechain_rmsd_A", "n_res"])
            w.writerows(rows)
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
