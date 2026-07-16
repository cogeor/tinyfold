"""Rigid placement of sidechain atoms from chi torsions (variant A, B2).

Design authority: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §4A ("place
heavy atoms analytically via rigid-group transforms... valid bond lengths/angles
BY CONSTRUCTION"). Companion of ``sidechain_geometry.extract_chi``: this is the
inverse map, chi + backbone -> atom14 coords.

Approach: NeRF (Natural Extension Reference Frame) placement atom-by-atom from a
per-residue-type PLACEMENT table. Each sidechain atom X is placed from a triple
of already-placed reference atoms ``(a, b, c)`` using an idealized bond length
|c-X|, bond angle (b, c, X), and a dihedral (a, b, c, X). The dihedral is a chi
angle for the atoms that DEFINE a chi (so they move with it) and a FIXED offset
for branch/ring atoms rigidly attached within a chi group.

Idealized geometry (bond length / angle / fixed dihedral) is NOT lifted from AF2
-- it is DERIVED from the atom14 data by :func:`derive_ideal_geometry`, so it is
grounded in the same DIPS structures the model trains on. Correctness is checked
by the extract->place round-trip (see the B2 gate), which must reconstruct real
sidechains to well under the 0.5 A packing gate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from tinyfold.atom14 import ATOM14_NAMES, NUM_ATOM14, aa1_to_aa3
from tinyfold.constants import AA_CODES

# Per-residue sidechain placement schedule. Each entry:
#   (atom, (ref_a, ref_b, ref_c), chi_index or -1 for a fixed dihedral)
# Every ref MUST be an earlier-placed atom (backbone N/CA/C/O or an earlier
# sidechain entry). Fixed atoms use refs that share the atom's rigid group so the
# stored dihedral is chi-independent (verified by the round-trip test).
_BB = ("N", "CA", "C")
PLACEMENT: dict[str, list[tuple[str, tuple[str, str, str], int]]] = {
    "ALA": [("CB", _BB, -1)],
    "ARG": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD", ("CA", "CB", "CG"), 1),
        ("NE", ("CB", "CG", "CD"), 2),
        ("CZ", ("CG", "CD", "NE"), 3),
        ("NH1", ("CD", "NE", "CZ"), -1),
        ("NH2", ("CD", "NE", "CZ"), -1),
    ],
    "ASN": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("OD1", ("CA", "CB", "CG"), 1),
        ("ND2", ("CB", "CG", "OD1"), -1),
    ],
    "ASP": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("OD1", ("CA", "CB", "CG"), 1),
        ("OD2", ("CB", "CG", "OD1"), -1),
    ],
    "CYS": [("CB", _BB, -1), ("SG", ("N", "CA", "CB"), 0)],
    "GLN": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD", ("CA", "CB", "CG"), 1),
        ("OE1", ("CB", "CG", "CD"), 2),
        ("NE2", ("CG", "CD", "OE1"), -1),
    ],
    "GLU": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD", ("CA", "CB", "CG"), 1),
        ("OE1", ("CB", "CG", "CD"), 2),
        ("OE2", ("CG", "CD", "OE1"), -1),
    ],
    "GLY": [],
    "HIS": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("ND1", ("CA", "CB", "CG"), 1),
        ("CD2", ("CB", "CG", "ND1"), -1),
        ("CE1", ("CG", "ND1", "CD2"), -1),
        ("NE2", ("ND1", "CD2", "CE1"), -1),
    ],
    "ILE": [
        ("CB", _BB, -1),
        ("CG1", ("N", "CA", "CB"), 0),
        ("CG2", ("CA", "CB", "CG1"), -1),
        ("CD1", ("CA", "CB", "CG1"), 1),
    ],
    "LEU": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD1", ("CA", "CB", "CG"), 1),
        ("CD2", ("CB", "CG", "CD1"), -1),
    ],
    "LYS": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD", ("CA", "CB", "CG"), 1),
        ("CE", ("CB", "CG", "CD"), 2),
        ("NZ", ("CG", "CD", "CE"), 3),
    ],
    "MET": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("SD", ("CA", "CB", "CG"), 1),
        ("CE", ("CB", "CG", "SD"), 2),
    ],
    "PHE": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD1", ("CA", "CB", "CG"), 1),
        ("CD2", ("CB", "CG", "CD1"), -1),
        ("CE1", ("CG", "CD1", "CD2"), -1),
        ("CE2", ("CG", "CD2", "CD1"), -1),
        ("CZ", ("CD1", "CE1", "CE2"), -1),
    ],
    "PRO": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD", ("CA", "CB", "CG"), 1),
    ],
    "SER": [("CB", _BB, -1), ("OG", ("N", "CA", "CB"), 0)],
    "THR": [
        ("CB", _BB, -1),
        ("OG1", ("N", "CA", "CB"), 0),
        ("CG2", ("CA", "CB", "OG1"), -1),
    ],
    "TRP": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD1", ("CA", "CB", "CG"), 1),
        ("CD2", ("CB", "CG", "CD1"), -1),
        ("NE1", ("CG", "CD1", "CD2"), -1),
        ("CE2", ("CD1", "CD2", "NE1"), -1),
        ("CE3", ("CG", "CD2", "CE2"), -1),
        ("CZ2", ("CD2", "CE2", "NE1"), -1),
        ("CZ3", ("CD2", "CE3", "CE2"), -1),
        ("CH2", ("CE2", "CZ2", "CE3"), -1),
    ],
    "TYR": [
        ("CB", _BB, -1),
        ("CG", ("N", "CA", "CB"), 0),
        ("CD1", ("CA", "CB", "CG"), 1),
        ("CD2", ("CB", "CG", "CD1"), -1),
        ("CE1", ("CG", "CD1", "CD2"), -1),
        ("CE2", ("CG", "CD2", "CD1"), -1),
        ("CZ", ("CD1", "CE1", "CE2"), -1),
        ("OH", ("CE1", "CE2", "CZ"), -1),
    ],
    "VAL": [
        ("CB", _BB, -1),
        ("CG1", ("N", "CA", "CB"), 0),
        ("CG2", ("CA", "CB", "CG1"), -1),
    ],
}

N_AA = len(AA_CODES) + 1  # +1 for X/unknown -> GLY


def _slot(aa3: str, name: str) -> int:
    return ATOM14_NAMES[aa3].index(name)


def build_placement_tables() -> dict[str, np.ndarray]:
    """Compile PLACEMENT into dense per-restype tables indexed by atom14 slot.

    Returns arrays over the project AA order ([N_AA, 14, ...]):
      * ``order``     [N_AA, 10] int  : sidechain slots in placement order (-1 pad)
      * ``ref``       [N_AA, 14, 3] int: ref slots per placed slot (0 where unused)
      * ``chi_src``   [N_AA, 14]   int: chi index driving the slot, or -1 (fixed)
      * ``place_mask``[N_AA, 14]  bool: slots that get placed (sidechain, present)
    """
    order = np.full((N_AA, NUM_ATOM14 - 4), -1, dtype=np.int64)
    ref = np.zeros((N_AA, NUM_ATOM14, 3), dtype=np.int64)
    chi_src = np.full((N_AA, NUM_ATOM14), -1, dtype=np.int64)
    place_mask = np.zeros((N_AA, NUM_ATOM14), dtype=bool)
    for i, aa1 in enumerate(list(AA_CODES) + ["X"]):
        aa3 = aa1_to_aa3(aa1) if aa1 != "X" else "GLY"
        for j, (atom, refs, chi) in enumerate(PLACEMENT[aa3]):
            s = _slot(aa3, atom)
            order[i, j] = s
            ref[i, s] = [_slot(aa3, r) for r in refs]
            chi_src[i, s] = chi
            place_mask[i, s] = True
    return {"order": order, "ref": ref, "chi_src": chi_src, "place_mask": place_mask}


def _place_atom(a: Tensor, b: Tensor, c: Tensor, length: Tensor, angle: Tensor,
                dihedral: Tensor, eps: float = 1e-8) -> Tensor:
    """NeRF: place X from refs a-b-c with bond |c-X|=length, angle (b,c,X), and
    dihedral (a,b,c,X). All inputs broadcast on a leading batch; returns ``[...,3]``.
    """
    bc = c - b
    bc = bc / (bc.norm(dim=-1, keepdim=True) + eps)          # c->? unit along b->c
    ab = b - a
    n = torch.cross(ab, bc, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)
    m = torch.cross(n, bc, dim=-1)                            # bc, m, n orthonormal
    # Local displacement of X from c in the (bc, m, n) frame.
    d = torch.stack([
        -length * torch.cos(angle),
        length * torch.sin(angle) * torch.cos(dihedral),
        length * torch.sin(angle) * torch.sin(dihedral),
    ], dim=-1)                                                # [...,3]
    basis = torch.stack([bc, m, n], dim=-1)                   # [...,3,3] columns
    return c + torch.einsum("...ij,...j->...i", basis, d)


def _angle(b: Tensor, c: Tensor, x: Tensor, eps: float = 1e-8) -> Tensor:
    """Angle at vertex c of b-c-x (radians)."""
    u = b - c
    v = x - c
    cos = (u * v).sum(-1) / ((u.norm(dim=-1) + eps) * (v.norm(dim=-1) + eps))
    return torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))


_TABLES = None


def placement_tables(device: torch.device):
    """Cached dense placement tables as torch tensors on ``device``."""
    global _TABLES
    if _TABLES is None or _TABLES["order"].device != device:
        t = build_placement_tables()
        _TABLES = {k: torch.from_numpy(v).to(device) for k, v in t.items()}
    return _TABLES


def derive_ideal_geometry(
    coords_atom14: Tensor,   # [..., L, 14, 3]
    aatype: Tensor,          # [..., L]
    atom_mask: Tensor,       # [..., L, 14] bool
) -> dict[str, Tensor]:
    """Measure idealized (bond length, bond angle, fixed dihedral) per restype/slot.

    Averages over every residue instance in the input (a batch of structures).
    Fixed dihedrals use a circular mean. Returns tensors ``[N_AA, 14]`` on the
    input device; slots that are never observed keep zeros.
    """
    from tinyfold.sidechain_geometry import dihedral as _dih

    device = coords_atom14.device
    tb = placement_tables(device)
    ref, chi_src, place_mask = tb["ref"], tb["chi_src"], tb["place_mask"]

    coords = coords_atom14.reshape(-1, NUM_ATOM14, 3)         # [M,14,3]
    aa = aatype.reshape(-1)                                   # [M]
    amask = atom_mask.reshape(-1, NUM_ATOM14)                 # [M,14]
    M = coords.shape[0]
    ar = torch.arange(M, device=device)

    length = torch.zeros(N_AA, NUM_ATOM14, device=device)
    angle = torch.zeros(N_AA, NUM_ATOM14, device=device)
    dih_sin = torch.zeros(N_AA, NUM_ATOM14, device=device)
    dih_cos = torch.zeros(N_AA, NUM_ATOM14, device=device)
    count = torch.zeros(N_AA, NUM_ATOM14, device=device)

    for slot in range(4, NUM_ATOM14):
        placed_type = place_mask[:, slot][aa]                 # [M] restype has this slot
        a_i, b_i, c_i = ref[aa, slot, 0], ref[aa, slot, 1], ref[aa, slot, 2]
        valid = placed_type & amask[:, slot] & amask[ar, a_i] & amask[ar, b_i] & amask[ar, c_i]
        if not valid.any():
            continue
        a, b, c = coords[ar, a_i], coords[ar, b_i], coords[ar, c_i]
        x = coords[:, slot]
        ln = (x - c).norm(dim=-1)
        an = _angle(b, c, x)
        dh = _dih(a, b, c, x)
        aav = aa[valid]
        length.index_put_((aav, torch.full_like(aav, slot)), ln[valid], accumulate=True)
        angle.index_put_((aav, torch.full_like(aav, slot)), an[valid], accumulate=True)
        dih_sin.index_put_((aav, torch.full_like(aav, slot)), torch.sin(dh[valid]), accumulate=True)
        dih_cos.index_put_((aav, torch.full_like(aav, slot)), torch.cos(dh[valid]), accumulate=True)
        count.index_put_((aav, torch.full_like(aav, slot)), torch.ones_like(ln[valid]), accumulate=True)

    denom = count.clamp(min=1)
    return {
        "bond_len": length / denom,
        "bond_ang": angle / denom,
        "fixed_dih": torch.atan2(dih_sin, dih_cos),   # circular mean
        "count": count,
    }


def place_chi(
    chi: Tensor,             # [..., L, 4] radians
    backbone: Tensor,        # [..., L, 4, 3] atom14 slots 0-3 (N, CA, C, O)
    aatype: Tensor,          # [..., L]
    geom: dict[str, Tensor],
) -> Tensor:
    """chi + backbone -> full atom14 coords ``[..., L, 14, 3]`` (NeRF placement).

    Backbone slots (0-3) are copied through; sidechain slots are placed in
    dependency order using the derived idealized geometry. Chi-driven slots use
    the supplied ``chi``; fixed slots use ``geom['fixed_dih']``.
    """
    device = chi.device
    tb = placement_tables(device)
    order, ref, chi_src = tb["order"], tb["ref"], tb["chi_src"]
    bond_len, bond_ang, fixed_dih = geom["bond_len"], geom["bond_ang"], geom["fixed_dih"]

    lead = aatype.shape
    aa = aatype.reshape(-1)                                   # [M]
    M = aa.shape[0]
    ar = torch.arange(M, device=device)
    coords = torch.zeros(M, NUM_ATOM14, 3, device=device, dtype=backbone.dtype)
    coords[:, :4] = backbone.reshape(M, 4, 3)
    chi_flat = chi.reshape(M, 4)

    n_steps = order.shape[1]
    for j in range(n_steps):
        slot = order[aa, j]                                  # [M], -1 = nothing at this step
        active = slot >= 0
        if not active.any():
            continue
        s = slot.clamp(min=0)
        a_i, b_i, c_i = ref[aa, s, 0], ref[aa, s, 1], ref[aa, s, 2]
        a, b, c = coords[ar, a_i], coords[ar, b_i], coords[ar, c_i]
        ln = bond_len[aa, s]
        an = bond_ang[aa, s]
        cs = chi_src[aa, s]                                   # [M] chi idx or -1
        chi_val = chi_flat[ar, cs.clamp(min=0)]
        dih = torch.where(cs >= 0, chi_val, fixed_dih[aa, s])
        placed = _place_atom(a, b, c, ln, an, dih)           # [M,3]
        write = active
        coords[ar[write], s[write]] = placed[write].to(coords.dtype)

    return coords.reshape(*lead, NUM_ATOM14, 3)
