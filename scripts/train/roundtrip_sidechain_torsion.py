"""B2 representation gate: extract chi -> place chi round-trip on real atom14 data.

The torsion-variant analogue of the free-offset "teacher-forced 0.03 A" check: it
isolates the REPRESENTATION (can chi + idealized geometry reconstruct a real
sidechain?) from the DENOISER (can a head predict chi?). If this round-trip does
not clear the packing gate, no head will -- the chi representation is lossy and
the variant is mis-specified.

    uv run python scripts/train/roundtrip_sidechain_torsion.py --n 200

Reports symmetry-corrected sidechain RMSD. Geometry is derived from the sampled
complexes (dataset-averaged idealized bond lengths/angles), NOT per-complex, so
this is the honest generalization number a packer would inherit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tinyfold.atom14 import restype_atom14_mask
from tinyfold.model.losses.sidechain import sidechain_rmsd
from tinyfold.sidechain_frames import derive_ideal_geometry, place_chi
from tinyfold.sidechain_geometry import extract_chi


def _load(path: Path):
    z = np.load(path)
    coords = torch.from_numpy(z["coords_atom14"]).float()
    aatype = torch.from_numpy(z["seq_indices"]).long()
    tmask = torch.from_numpy(restype_atom14_mask()).bool()[aatype]
    mask = torch.from_numpy(z["mask_atom14"]).bool() & tmask
    return coords, aatype, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14-dir", type=Path, default=Path("data/processed/atom14"))
    ap.add_argument("--n", type=int, default=200, help="complexes to sample")
    ap.add_argument("--gate", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(args.atom14_dir.glob("*.npz"))[: args.n]
    if not files:
        print(f"no atom14 cache in {args.atom14_dir}")
        return 1

    # Derive idealized geometry from ALL sampled complexes (concatenated residues).
    cs, as_, ms = [], [], []
    for p in files:
        c, a, m = _load(p)
        cs.append(c); as_.append(a); ms.append(m)
    C, A, Mk = torch.cat(cs), torch.cat(as_), torch.cat(ms)
    geom = derive_ideal_geometry(C.unsqueeze(0), A.unsqueeze(0), Mk.unsqueeze(0))

    rmsds = []
    for c, a, m in zip(cs, as_, ms):
        coords, aatype, mask = c.unsqueeze(0), a.unsqueeze(0), m.unsqueeze(0)
        chi, _ = extract_chi(coords, aatype, mask)
        placed = place_chi(chi, coords[:, :, :4], aatype, geom)
        r = sidechain_rmsd(placed, coords, mask, aatype).item()
        rmsds.append(r)

    rmsds = np.array(rmsds)
    print(f"complexes         : {len(rmsds)}")
    print(f"sidechain RMSD    : median {np.median(rmsds):.3f} A | mean {rmsds.mean():.3f} "
          f"| p90 {np.percentile(rmsds, 90):.3f} | max {rmsds.max():.3f}")
    passed = np.median(rmsds) < args.gate
    print("GATE:", f"PASS (median < {args.gate} A) -- the chi representation is faithful; "
          "a torsion head is worth building." if passed else
          f"FAIL (median >= {args.gate} A) -- idealized-geometry placement is too lossy; "
          "needs per-atom geometry or a table fix before any head.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
