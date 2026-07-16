"""S5 -- the stage-3 sidechain OVERFIT GATE (notes/2026-07-14-...-SPEC.md §11).

The go/no-go for Phase 3. Same discipline that de-risked the backbone atom
diffusion (Atom-RMSE 1.37 -> 0.32 A before any real run): if a small head cannot
overfit ONE complex's sidechains given the GT backbone, no data or scaling will
help, and the torsion/torus variant is not worth building.

    uv run python scripts/train/overfit_sidechain.py            # first cache complex
    uv run python scripts/train/overfit_sidechain.py --sample 3ozf.pdb1_0

GATE: symmetry-corrected sidechain RMSD < 0.5 A on the training complex.

Setup (spec §4, §7 phase-1):
* Backbone (atom14 slots 0-3) is FROZEN; we only diffuse the sidechains.
* Everything is in each residue's LOCAL backbone frame, so the problem is
  rotation/translation invariant and there is no aug_R bookkeeping.
* EDM/Karras denoising, exactly as AtomDiffusionHead. sigma_data is set to the
  MEASURED std of local-frame sidechain coordinates (~1.3 A), not a normalized
  guess -- this gate runs in raw Angstroms so the < 0.5 A threshold is literal.
* Loss is a plain EDM MSE on local sidechain coords: overfitting ONE structure
  with fixed GT labels needs no symmetry correction (the head just matches the
  labelling it sees). The symmetry-corrected RMSD is still the reported metric.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.atom14 import restype_atom14_mask
from tinyfold.model.losses.sidechain import sidechain_rmsd
from tinyfold.model.resfold.sidechain_diffusion import (
    SidechainDiffusionHead,
    assemble_atom14,
    sidechain_to_global,
    sidechain_to_local,
)


def load_complex(path: Path, device):
    z = np.load(path)
    coords = torch.from_numpy(z["coords_atom14"]).float().unsqueeze(0).to(device)  # [1,L,14,3]
    present = torch.from_numpy(z["mask_atom14"]).bool().unsqueeze(0).to(device)    # [1,L,14]
    aatype = torch.from_numpy(z["seq_indices"]).long().unsqueeze(0).to(device)     # [1,L]

    # An atom14 slot counts only if the residue TYPE has it AND it is resolved in
    # this structure. restype mask guards against stray atoms in the wrong slot.
    type_mask = torch.from_numpy(restype_atom14_mask()).bool().to(device)[aatype]  # [1,L,14]
    mask = present & type_mask
    return coords, mask, aatype


def karras_sigmas(sigma_max, sigma_min, steps, rho, device):
    """EDM (Karras) sigma schedule: dense at LOW sigma, where refinement lives."""
    i = torch.arange(steps, device=device) / max(steps - 1, 1)
    inv = sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    sig = inv ** rho
    return torch.cat([sig, torch.zeros(1, device=device)])   # append sigma=0


@torch.no_grad()
def sample(head, backbone, bb_feats, scale, aatype, res_mask, sigma_max, sigma_min,
           steps, device, rho=7.0, warm_start=None):
    """Deterministic Euler over a Karras schedule -- the PROVEN recipe.

    Mirrors sample_atoms_diffusion (which reached 0.32 A on backbone atoms):
    plain Euler, no churn, no Heun, and -- critically -- the sampler's
    [sigma_min, sigma_max] MATCHES the log-uniform training range, so every
    sigma the trajectory visits was trained equally. The earlier failures came
    from a log-normal training schedule that under-covered the extremes the
    sampler actually traverses. Works in NORMALIZED units; decode by *scale.
    """
    B, L = aatype.shape
    sig = karras_sigmas(sigma_max, sigma_min, steps, rho, device)
    if warm_start is None:
        x = torch.randn(B, L, 10, 3, device=device) * sig[0]
    else:
        # Diagnostic: start from gt + sigma_max*noise (target PRESENT) instead of
        # pure noise (target ABSENT). If cold fails but warm passes, the head
        # learned to EXTRACT the signal, not GENERATE it from conditioning.
        x = warm_start + torch.randn_like(warm_start) * sig[0]
    for i in range(steps):
        x0 = head(x, aatype, sig[i].expand(B), mask=res_mask, backbone_feats=bb_feats)
        # Last step: sig[i+1]=0 -> x becomes exactly x0 (the clean estimate).
        x = x + (x - x0) / sig[i] * (sig[i + 1] - sig[i])
    return sidechain_to_global(x * scale, backbone)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14-dir", type=Path, default=Path("data/processed/atom14"))
    ap.add_argument("--sample", default=None, help="sample_id (default: first in the cache)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample-steps", type=int, default=32, help="reverse-diffusion steps at eval")
    ap.add_argument("--gate", type=float, default=0.5, help="pass threshold, sidechain RMSD (A)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = (args.atom14_dir / f"{args.sample}.npz" if args.sample
            else sorted(args.atom14_dir.glob("*.npz"))[0])
    if not path.exists():
        print(f"not found: {path}")
        return 1
    coords, mask, aatype = load_complex(path, device)
    L = coords.shape[1]
    backbone = coords[:, :, :4, :]                       # frozen
    res_mask = mask[:, :, 1]                              # residue valid iff CA present
    sc_mask = mask                                       # [1,L,14]

    # NORMALIZED units (÷ global_scale), matching the PROVEN atom-diffusion regime
    # (sample_atoms_diffusion: sigma_data~0.15, sigma in [0.002, 1.0]). Working in
    # the same numeric range lets us reuse its exact recipe -- log-uniform training
    # over the sampler's range + deterministic Euler -- which is what fixed the
    # train/sample mismatch. The gate metric decodes back to raw Angstroms.
    GLOBAL_SCALE = 11.0
    gt_local = sidechain_to_local(coords) / GLOBAL_SCALE  # [1,L,10,3] normalized
    sc_local_mask = sc_mask[:, :, 4:]                     # [1,L,10]
    sd = float(gt_local[sc_local_mask.bool()].std())
    # Match the working atom head's absolute sampling range exactly.
    sigma_min, sigma_max = 0.002, 1.0

    # Backbone conditioning: 4 atoms/residue, per-complex-centered on the mean CA
    # and normalized -- a strong per-residue geometric identity (spec §5).
    ca_mean = backbone[:, :, 1, :].mean(dim=1, keepdim=True).unsqueeze(2)  # [1,1,1,3]
    bb_feats = (backbone - ca_mean) / GLOBAL_SCALE       # [1,L,4,3] normalized

    head = SidechainDiffusionHead(
        c_token=128, n_layers=3, n_heads=4, sigma_data=sd, use_tokens=False,
        backbone_cond=True,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_res = int(res_mask.sum())
    n_sc = int(sc_local_mask.sum())
    print(f"complex {path.stem}: L={L} residues={n_res} sidechain-atoms={n_sc}")
    print(f"sigma_data (normalized local std) = {sd:.4f} ; sigma in "
          f"[{sigma_min}, {sigma_max}] ; params = "
          f"{sum(p.numel() for p in head.parameters()):,}")

    m10 = sc_local_mask.unsqueeze(-1).float()            # [1,L,10,1]
    # LOG-UNIFORM training sigma over EXACTLY the sampler's [sigma_min, sigma_max]
    # -- every sigma the trajectory visits is trained equally (the proven recipe;
    # log-normal under-covered the extremes and the sampler drifted there). Batch
    # nsig draws per step for a low-variance gradient.
    lo, hi, nsig = math.log(sigma_min), math.log(sigma_max), 16
    gt = gt_local.expand(nsig, -1, -1, -1)               # [nsig,L,10,3]
    aab = aatype.expand(nsig, -1)
    mb = res_mask.expand(nsig, -1)
    mb10 = m10.expand(nsig, -1, -1, -1)
    bbb = bb_feats.expand(nsig, -1, -1, -1)
    for step in range(1, args.steps + 1):
        head.train()
        sig = torch.exp(torch.rand(nsig, device=device) * (hi - lo) + lo)
        x_t = gt + sig.view(nsig, 1, 1, 1) * torch.randn_like(gt)
        x0 = head(x_t, aab, sig, mask=mb, backbone_feats=bbb)
        # EDM loss weight lambda = (sig^2 + sd^2) / (sig*sd)^2, per-sigma.
        lam = ((sig ** 2 + sd ** 2) / (sig * sd) ** 2).view(nsig, 1, 1, 1)
        se = ((x0 - gt) ** 2 * lam * mb10).sum() / (mb10.sum() / nsig).clamp(min=1) / nsig
        loss = se
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0 or step == 1:
            head.eval()
            with torch.no_grad():
                pred_global = sample(head, backbone, bb_feats, GLOBAL_SCALE, aatype,
                                     res_mask, sigma_max, sigma_min, args.sample_steps, device)
                rmsd = sidechain_rmsd(assemble_atom14(backbone, pred_global),
                                      coords, sc_mask, aatype).item()
                # Teacher-forced recon (normalized) at a low sigma -- the
                # representation floor, separate from the sampler.
                tf_x0 = head(gt_local + sigma_min * torch.randn_like(gt_local),
                             aatype, torch.full((1,), sigma_min, device=device),
                             mask=res_mask, backbone_feats=bb_feats)
                tf = sidechain_rmsd(
                    assemble_atom14(backbone, sidechain_to_global(tf_x0 * GLOBAL_SCALE, backbone)),
                    coords, sc_mask, aatype).item()
            print(f"  step {step:5d} | loss {loss.item():6.3f} | sampled {rmsd:.3f} A "
                  f"| teacher-forced {tf:.3f} A")

    head.eval()
    with torch.no_grad():
        pred_global = sample(head, backbone, bb_feats, GLOBAL_SCALE, aatype, res_mask,
                             sigma_max, sigma_min, args.sample_steps, device)
        warm = sample(head, backbone, bb_feats, GLOBAL_SCALE, aatype, res_mask,
                      sigma_max, sigma_min, args.sample_steps, device, warm_start=gt_local)
    rmsd = sidechain_rmsd(assemble_atom14(backbone, pred_global),
                          coords, sc_mask, aatype).item()
    warm_rmsd = sidechain_rmsd(assemble_atom14(backbone, warm),
                               coords, sc_mask, aatype).item()
    print(f"\nDIAGNOSTIC warm-start (target present) sampled RMSD: {warm_rmsd:.3f} A")
    print(f"FINAL cold sampled RMSD: {rmsd:.3f} A  (gate < {args.gate} A)")
    passed = rmsd < args.gate
    print("GATE:", "PASS -- a small head overfits sidechains; Phase 3 is worth "
          "building out (monomer packer -> torsion variant)." if passed else
          "FAIL -- if a head cannot overfit ONE structure, packing is mis-specified; "
          "debug before any data work.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
