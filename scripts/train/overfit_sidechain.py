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


@torch.no_grad()
def sample(head, backbone, aatype, res_mask, sigma_max, sigma_min, steps, device):
    """Karras/EDM deterministic sampler in local frame -> global sidechains."""
    B, L = aatype.shape
    sigmas = torch.exp(torch.linspace(
        math.log(sigma_max), math.log(sigma_min), steps + 1, device=device))
    x = torch.randn(B, L, 10, 3, device=device) * sigma_max
    for i in range(steps):
        s = sigmas[i].expand(B)
        x0 = head(x, aatype, s, mask=res_mask)
        d = (x - x0) / sigmas[i]
        x = x + (sigmas[i + 1] - sigmas[i]) * d            # Euler step
    x0 = head(x, aatype, sigmas[-1].expand(B), mask=res_mask)
    return sidechain_to_global(x0, backbone)


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

    # Target in local frame; measured sigma_data (raw Angstroms).
    gt_local = sidechain_to_local(coords)                # [1,L,10,3]
    sc_local_mask = sc_mask[:, :, 4:]                    # [1,L,10]
    sd = float(gt_local[sc_local_mask.bool()].std())
    sigma_min, sigma_max = 0.05 * sd, 10.0 * sd

    head = SidechainDiffusionHead(
        c_token=128, n_layers=3, n_heads=4, sigma_data=sd, use_tokens=False,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_res = int(res_mask.sum())
    n_sc = int(sc_local_mask.sum())
    print(f"complex {path.stem}: L={L} residues={n_res} sidechain-atoms={n_sc}")
    print(f"sigma_data (measured local std) = {sd:.3f} A ; params = "
          f"{sum(p.numel() for p in head.parameters()):,}")

    m10 = sc_local_mask.unsqueeze(-1).float()            # [1,L,10,1]
    lo, hi = math.log(sigma_min), math.log(sigma_max)
    for step in range(1, args.steps + 1):
        head.train()
        sig = torch.exp(torch.rand(1, device=device) * (hi - lo) + lo)
        noise = torch.randn_like(gt_local)
        x_t = gt_local + sig.view(1, 1, 1, 1) * noise
        x0 = head(x_t, aatype, sig, mask=res_mask)
        # EDM loss weight lambda = (sig^2 + sd^2) / (sig*sd)^2.
        lam = (sig ** 2 + sd ** 2) / (sig * sd) ** 2
        se = ((x0 - gt_local) ** 2 * m10).sum() / m10.sum().clamp(min=1)
        loss = lam * se
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0 or step == 1:
            head.eval()
            pred_global = sample(head, backbone, aatype, res_mask,
                                 sigma_max, sigma_min, args.sample_steps, device)
            pred14 = assemble_atom14(backbone, pred_global)
            rmsd = sidechain_rmsd(pred14, coords, sc_mask, aatype).item()
            print(f"  step {step:5d} | loss {loss.item():7.3f} | "
                  f"sidechain RMSD {rmsd:.3f} A")

    head.eval()
    pred_global = sample(head, backbone, aatype, res_mask,
                         sigma_max, sigma_min, args.sample_steps, device)
    rmsd = sidechain_rmsd(assemble_atom14(backbone, pred_global),
                          coords, sc_mask, aatype).item()
    print(f"\nFINAL sidechain RMSD: {rmsd:.3f} A  (gate < {args.gate} A)")
    passed = rmsd < args.gate
    print("GATE:", "PASS -- a small head overfits sidechains; Phase 3 is worth "
          "building out (monomer packer -> torsion variant)." if passed else
          "FAIL -- if a head cannot overfit ONE structure, packing is mis-specified; "
          "debug before any data work.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
