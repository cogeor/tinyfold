"""B2: rigid NeRF placement of sidechains from chi (variant A).

Two levels:
  * structural — the PLACEMENT table is self-consistent (every ref is placed
    before the atom that uses it; chi-driven slots reference their chi's atoms);
  * geometric  — the NeRF primitive reproduces a planted atom, and a self-derived
    round-trip is near-exact (the representation is faithful).

The real-data round-trip (dataset-averaged geometry -> <0.3 A) is the B2 gate and
lives behind the ``integration`` marker (needs the atom14 cache).
"""

import math

import numpy as np
import torch

from tinyfold.atom14 import ATOM14_NAMES, aa1_to_aa3
from tinyfold.constants import AA_CODES
from tinyfold.sidechain_frames import (
    PLACEMENT,
    _place_atom,
    build_placement_tables,
    derive_ideal_geometry,
    place_chi,
)
from tinyfold.sidechain_geometry import dihedral, extract_chi


def test_placement_refs_are_placed_before_use():
    # Every ref of a sidechain atom must be a backbone atom or an already-listed
    # sidechain atom -- otherwise the NeRF driver reads an unfilled slot.
    for aa3, entries in PLACEMENT.items():
        placed = {"N", "CA", "C", "O"}
        for atom, refs, chi in entries:
            for r in refs:
                assert r in placed, f"{aa3}: {atom} refs {r} before it is placed"
            assert atom in ATOM14_NAMES[aa3], f"{aa3}: {atom} not in atom14 layout"
            placed.add(atom)


def test_placement_covers_every_sidechain_slot():
    # Each residue type must place ALL of its non-backbone atom14 slots.
    for aa3, names in ATOM14_NAMES.items():
        sidechain = {n for n in names[4:] if n}
        placed = {atom for atom, _, _ in PLACEMENT[aa3]}
        assert placed == sidechain, f"{aa3}: placed {placed} != sidechain {sidechain}"


def test_chi_driven_slots_match_chi_definition():
    # A chi-driven atom's dihedral is chi_k, so its refs must be the first three
    # atoms of that chi's 4-atom definition.
    from tinyfold.atom14 import CHI_ANGLES_ATOMS

    for aa3, entries in PLACEMENT.items():
        for atom, refs, chi in entries:
            if chi >= 0:
                chi_atoms = CHI_ANGLES_ATOMS[aa3][chi]
                assert list(refs) == chi_atoms[:3], f"{aa3} {atom}"
                assert atom == chi_atoms[3], f"{aa3} {atom} != chi{chi+1} 4th atom"


def test_place_atom_reproduces_known_geometry():
    # Place an atom, then re-measure its bond length / dihedral: must match.
    a = torch.tensor([[0.0, 1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.0]])
    c = torch.tensor([[1.5, 0.0, 0.0]])
    length = torch.tensor([1.52])
    angle = torch.tensor([1.911])   # ~109.5 deg
    dih = torch.tensor([1.0])
    x = _place_atom(a, b, c, length, angle, dih)
    assert math.isclose((x - c).norm().item(), 1.52, abs_tol=1e-4)
    assert math.isclose(dihedral(a, b, c, x).item(), 1.0, abs_tol=1e-4)


def test_tables_shapes():
    t = build_placement_tables()
    assert t["order"].shape == (21, 10)
    assert t["ref"].shape == (21, 14, 3)
    assert t["chi_src"].shape == (21, 14)
    assert t["place_mask"].shape == (21, 14)


def _synthetic_single_residue(aa1: str, seed: int):
    """One residue of type aa1 with random but non-degenerate atom positions."""
    aa3 = aa1_to_aa3(aa1)
    names = ATOM14_NAMES[aa3]
    rng = np.random.default_rng(seed)
    coords = torch.zeros(1, 1, 14, 3)
    # Spread atoms so refs are non-collinear.
    for s, nm in enumerate(names):
        if nm:
            coords[0, 0, s] = torch.from_numpy(rng.standard_normal(3) * 2.0).float()
    aatype = torch.tensor([[AA_CODES.index(aa1)]])
    mask = torch.zeros(1, 1, 14, dtype=torch.bool)
    mask[0, 0, [s for s, nm in enumerate(names) if nm]] = True
    return coords, aatype, mask


def test_self_geometry_roundtrip_is_near_exact():
    # With geometry derived from a SINGLE residue (so the per-type means equal that
    # residue's exact values), extract->place must reproduce it to machine noise.
    # Uses ARG: 4 chis + fixed terminal atoms -> exercises the whole chain.
    coords, aatype, mask = _synthetic_single_residue("R", seed=3)
    geom = derive_ideal_geometry(coords, aatype, mask)
    chi, _ = extract_chi(coords, aatype, mask)
    placed = place_chi(chi, coords[:, :, :4], aatype, geom)
    err = (placed[..., 4:, :] - coords[..., 4:, :]).norm(dim=-1)[mask[..., 4:]]
    assert err.max() < 1e-3, f"self-geometry round-trip not exact: max {err.max():.2e}"
