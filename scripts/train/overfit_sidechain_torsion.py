"""B5 -- the torsion-variant sidechain OVERFIT GATE.

The go/no-go for the torsion sidechain path (spec §11), the counterpart of
scripts/train/overfit_sidechain.py for the free-offset prototype (which floored at
~2.0 A sampled). If a small wrapped-diffusion head cannot overfit ONE complex's
chi torsions given the GT backbone, the torsion formulation is mis-specified.

    uv run python scripts/train/overfit_sidechain_torsion.py
    uv run python scripts/train/overfit_sidechain_torsion.py --sample 3ozf.pdb1_0

GATE: symmetry-corrected sidechain RMSD < 0.5 A on the training complex.

Diagnostics printed alongside (mirrors the free-offset gate's localisation):
  * representation floor  -- extract->place round-trip (chi rep + idealized geom);
                             the head cannot beat this.
  * teacher-forced chi-MAE-- denoise gt + small noise; is the REPRESENTATION+HEAD
                             sound, separate from the sampler?
  * oracle sampler        -- run the reverse process with chi0 := gt; is the
                             SAMPLER correct?
  * cold sampled          -- the real number the gate reads.
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
from tinyfold.model.losses.torsion import chi_mae_deg, torsion_symmetry_loss
from tinyfold.model.resfold.sidechain_torsion_head import SidechainTorsionHead, wrap_angle
from tinyfold.sidechain_frames import derive_ideal_geometry, place_chi
from tinyfold.sidechain_geometry import extract_chi


def load_complex(path: Path, device):
    z = np.load(path)
    coords = torch.from_numpy(z["coords_atom14"]).float().unsqueeze(0).to(device)
    present = torch.from_numpy(z["mask_atom14"]).bool().unsqueeze(0).to(device)
    aatype = torch.from_numpy(z["seq_indices"]).long().unsqueeze(0).to(device)
    type_mask = torch.from_numpy(restype_atom14_mask()).bool().to(device)[aatype]
    return coords, present & type_mask, aatype


def karras_sigmas(sigma_max, sigma_min, steps, rho, device):
    i = torch.arange(steps, device=device) / max(steps - 1, 1)
    inv = sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    sig = inv ** rho
    return torch.cat([sig, torch.zeros(1, device=device)])


@torch.no_grad()
def sample(head, aatype, res_mask, bb_feats, sigma_max, sigma_min, steps, device,
           rho=7.0, oracle_chi=None):
    """Deterministic wrapped-Euler reverse process on the torus (x0-prediction).

    Mirrors the proven Cartesian recipe but every difference is WRAPPED to
    (-pi, pi], so the trajectory lives on SO(2)^4. ``oracle_chi`` replaces the
    head's chi0 with the GT (the sampler-correctness probe).
    """
    B, L = aatype.shape
    sig = karras_sigmas(sigma_max, sigma_min, steps, rho, device)
    chi = wrap_angle(torch.rand(B, L, 4, device=device) * 2 * math.pi - math.pi) * sig[0]
    chi = wrap_angle(chi)
    for i in range(steps):
        chi0 = oracle_chi if oracle_chi is not None else head(
            chi, aatype, sig[i].expand(B), mask=res_mask, backbone_feats=bb_feats)
        delta = wrap_angle(chi - chi0)
        chi = wrap_angle(chi0 + delta * (sig[i + 1] / sig[i]))
    return chi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14-dir", type=Path, default=Path("data/processed/atom14"))
    ap.add_argument("--sample", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample-steps", type=int, default=32)
    ap.add_argument("--gate", type=float, default=0.5)
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
    backbone = coords[:, :, :4, :]
    res_mask = mask[:, :, 1]

    # Idealized geometry + GT chi from THIS complex (isolates the head from the
    # dataset-average representation error).
    geom = derive_ideal_geometry(coords, aatype, mask)
    gt_chi, chi_mask = extract_chi(coords, aatype, mask)

    # Backbone conditioning: per-complex-centered, normalized (spec §5).
    GLOBAL_SCALE = 11.0
    ca_mean = backbone[:, :, 1, :].mean(dim=1, keepdim=True).unsqueeze(2)
    bb_feats = (backbone - ca_mean) / GLOBAL_SCALE

    # Representation floor: extract -> place with this complex's geometry.
    placed_rep = place_chi(gt_chi, backbone, aatype, geom)
    rep_floor = sidechain_rmsd(placed_rep, coords, mask, aatype).item()

    sigma_min, sigma_max = 0.02, 3.0
    head = SidechainTorsionHead(
        c_token=128, n_layers=3, n_heads=4, use_tokens=False, backbone_cond=True,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_chi = int(chi_mask.sum())
    print(f"complex {path.stem}: L={L} valid-chis={n_chi} "
          f"params={sum(p.numel() for p in head.parameters()):,}")
    print(f"representation floor (extract->place): {rep_floor:.3f} A")
    print(f"sigma in [{sigma_min}, {sigma_max}] (radians)")

    lo, hi, nsig = math.log(sigma_min), math.log(sigma_max), 16
    gtb = gt_chi.expand(nsig, -1, -1)
    aab = aatype.expand(nsig, -1)
    mb = res_mask.expand(nsig, -1)
    cmb = chi_mask.expand(nsig, -1, -1)
    bbb = bb_feats.expand(nsig, -1, -1, -1)
    for step in range(1, args.steps + 1):
        head.train()
        sig = torch.exp(torch.rand(nsig, device=device) * (hi - lo) + lo)
        # Wrapped-normal forward noise.
        chi_t = wrap_angle(gtb + sig.view(nsig, 1, 1) * torch.randn_like(gtb))
        _, vec = head(chi_t, aab, sig, mask=mb, backbone_feats=bbb, return_vec=True)
        loss = torsion_symmetry_loss(vec, gtb, cmb, aab)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0 or step == 1:
            head.eval()
            with torch.no_grad():
                chi_s = sample(head, aatype, res_mask, bb_feats, sigma_max, sigma_min,
                               args.sample_steps, device)
                rmsd = sidechain_rmsd(place_chi(chi_s, backbone, aatype, geom),
                                      coords, mask, aatype).item()
                # Teacher-forced: denoise gt + small noise -> chi-MAE.
                tf_chi = head(wrap_angle(gt_chi + sigma_min * torch.randn_like(gt_chi)),
                              aatype, torch.full((1,), sigma_min, device=device),
                              mask=res_mask, backbone_feats=bb_feats)
                tf_mae = chi_mae_deg(tf_chi, gt_chi, chi_mask, aatype).item()
            print(f"  step {step:5d} | loss {loss.item():.4f} | sampled {rmsd:.3f} A "
                  f"| teacher-forced chi-MAE {tf_mae:5.1f} deg")

    head.eval()
    with torch.no_grad():
        chi_s = sample(head, aatype, res_mask, bb_feats, sigma_max, sigma_min,
                       args.sample_steps, device)
        rmsd = sidechain_rmsd(place_chi(chi_s, backbone, aatype, geom),
                              coords, mask, aatype).item()
        chi_o = sample(head, aatype, res_mask, bb_feats, sigma_max, sigma_min,
                       args.sample_steps, device, oracle_chi=gt_chi)
        oracle_rmsd = sidechain_rmsd(place_chi(chi_o, backbone, aatype, geom),
                                     coords, mask, aatype).item()
    print(f"\nDIAGNOSTIC oracle-sampler (chi0=gt) RMSD: {oracle_rmsd:.3f} A "
          f"(== representation floor {rep_floor:.3f} if the sampler is correct)")
    print(f"FINAL cold sampled RMSD: {rmsd:.3f} A  (gate < {args.gate} A)")
    passed = rmsd < args.gate
    print("GATE:", "PASS -- a small torsion head overfits sidechains; the torsion "
          "variant is worth building out." if passed else
          "FAIL -- localise via the diagnostics above (representation vs head vs sampler).")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
