"""atom14 layout, chi definitions, and symmetric-atom swaps.

Design: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §4, §9, §10.
Tables lifted from AF2 ``residue_constants``.

WHY atom14
----------
TinyFold is backbone-only today: exactly 4 atoms/residue, hard-coded at
``constants.py:47``. So amino-acid identity carries no geometry (TRP == GLY as
4 points). atom14 is the compact per-restype heavy-atom layout: a dense
``[L, 14, 3]`` tensor plus a presence mask, keeping the residue-major shape the
whole pipeline already uses. A flat ``[N_atoms, 3]`` list (AF3-true, needed only
for ligands) would force an atom-major refactor of collate/cropping/losses.

Slots 0-3 are ALWAYS N, CA, C, O -- so ``atom14[..., :4, :]`` is exactly the
current ``coords_res`` and every existing backbone path keeps working.
TRP is the widest at 14 heavy atoms, which is where the "14" comes from.

SYMMETRY (§9, MANDATORY)
------------------------
Several sidechains are symmetric under a 180-degree flip of a terminal group:
ASP's OD1/OD2, GLU's OE1/OE2, PHE and TYR's ring CD1/CD2 + CE1/CE2, ARG's
NH1/NH2. The two labellings are PHYSICALLY IDENTICAL but numerically different,
so a naive per-atom loss double-penalises a rotamer that is actually correct.
:func:`alt_atom14_permutation` gives the renamed ("alt") ground truth; the loss
takes the min over both. Note this is the ATOM-RENAMING symmetry (what a
free-offset/variant-B loss needs), which is distinct from -- and broader than --
AF2's ``chi_pi_periodic`` (what a torsion/variant-A loss would need): ARG is
renaming-ambiguous but not chi-pi-periodic.
"""

from __future__ import annotations

import numpy as np

from tinyfold.constants import AA3_TO_AA1, AA_CODES

NUM_ATOM14 = 14

# Per-residue heavy-atom slots. Slots 0-3 are always the backbone N, CA, C, O.
# '' marks an unused slot (absent for this residue type).
ATOM14_NAMES: dict[str, list[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
}

# chi1..chi4 defined by the 4 atoms whose dihedral they are. Used by the torsion
# (variant A) path and by chi-MAE metrics; the offset (variant B) prototype does
# not need them to place atoms.
CHI_ANGLES_ATOMS: dict[str, list[list[str]]] = {
    "ALA": [],
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# Physically indistinguishable atom pairs (AF2 residue_atom_renaming_swaps).
# ARG appears here but NOT in chi_pi_periodic -- renaming ambiguity is the
# broader notion, and it is the one a per-atom (offset) loss must respect.
AMBIGUOUS_ATOM_SWAPS: dict[str, dict[str, str]] = {
    "ARG": {"NH1": "NH2"},
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# chi angles that are periodic by 180 degrees (AF2 chi_pi_periodic). Needed by a
# TORSION-space loss (variant A); kept here so the two symmetry notions live
# side by side and are not confused.
CHI_PI_PERIODIC: dict[str, list[bool]] = {
    aa3: [False, False, False, False] for aa3 in ATOM14_NAMES
}
CHI_PI_PERIODIC["ASP"] = [False, True, False, False]
CHI_PI_PERIODIC["GLU"] = [False, False, True, False]
CHI_PI_PERIODIC["PHE"] = [False, True, False, False]
CHI_PI_PERIODIC["TYR"] = [False, True, False, False]

_AA1_TO_AA3 = {v: k for k, v in AA3_TO_AA1.items()}


def aa1_to_aa3(aa1: str) -> str:
    """One-letter -> three-letter code. Unknown ('X') falls back to GLY."""
    return _AA1_TO_AA3.get(aa1, "GLY")


def atom14_names(aa: str) -> list[str]:
    """atom14 slot names for a residue given as a 1- or 3-letter code."""
    aa3 = aa if len(aa) == 3 else aa1_to_aa3(aa)
    return ATOM14_NAMES.get(aa3, ATOM14_NAMES["GLY"])


def atom14_mask(aa: str) -> np.ndarray:
    """``[14]`` bool: which slots this residue type actually has."""
    return np.array([bool(n) for n in atom14_names(aa)], dtype=bool)


def n_chi(aa: str) -> int:
    """Number of chi torsions for this residue type (0 for GLY/ALA)."""
    aa3 = aa if len(aa) == 3 else aa1_to_aa3(aa)
    return len(CHI_ANGLES_ATOMS.get(aa3, []))


def alt_atom14_permutation(aa: str) -> np.ndarray:
    """``[14]`` index permutation giving the RENAMED ("alt") ground truth.

    ``gt_alt[i] = gt[perm[i]]``. Identity for residues with no ambiguity, so it
    is always safe to apply. The permutation is an involution (applying it twice
    is the identity) because every swap is a transposition.

    This is what makes the symmetry-corrected loss possible: score the
    prediction against both ``gt`` and ``gt[perm]`` and keep the better one, so a
    correct rotamer that happens to be labelled the other way round is not
    punished.
    """
    aa3 = aa if len(aa) == 3 else aa1_to_aa3(aa)
    names = atom14_names(aa3)
    perm = np.arange(NUM_ATOM14)
    swaps = AMBIGUOUS_ATOM_SWAPS.get(aa3, {})
    for a, b in swaps.items():
        ia, ib = names.index(a), names.index(b)
        perm[ia], perm[ib] = ib, ia
    return perm


def restype_atom14_mask() -> np.ndarray:
    """``[21, 14]`` bool over the project's AA index order (X -> GLY)."""
    out = np.zeros((len(AA_CODES) + 1, NUM_ATOM14), dtype=bool)
    for i, aa1 in enumerate(AA_CODES):
        out[i] = atom14_mask(aa1)
    out[len(AA_CODES)] = atom14_mask("GLY")  # X / unknown
    return out


def restype_alt_permutation() -> np.ndarray:
    """``[21, 14]`` int64 alt permutations over the project's AA index order."""
    out = np.zeros((len(AA_CODES) + 1, NUM_ATOM14), dtype=np.int64)
    for i, aa1 in enumerate(AA_CODES):
        out[i] = alt_atom14_permutation(aa1)
    out[len(AA_CODES)] = np.arange(NUM_ATOM14)
    return out


NUM_CHI = 4


def chi_atom14_indices(aa: str) -> np.ndarray:
    """``[4, 4]`` int64 atom14 slot indices for this residue's chi1..chi4.

    Row k holds the four atom14 slots whose dihedral IS chi_(k+1) (order matters:
    the dihedral is measured a-b-c-d). Rows for chis this residue does not have
    are zero-filled and flagged absent by :func:`chi_mask`.
    """
    aa3 = aa if len(aa) == 3 else aa1_to_aa3(aa)
    names = atom14_names(aa3)
    out = np.zeros((NUM_CHI, 4), dtype=np.int64)
    for k, atoms in enumerate(CHI_ANGLES_ATOMS.get(aa3, [])):
        out[k] = [names.index(a) for a in atoms]
    return out


def chi_mask(aa: str) -> np.ndarray:
    """``[4]`` bool: which of chi1..chi4 this residue type defines."""
    aa3 = aa if len(aa) == 3 else aa1_to_aa3(aa)
    m = np.zeros(NUM_CHI, dtype=bool)
    m[: len(CHI_ANGLES_ATOMS.get(aa3, []))] = True
    return m


def restype_chi_atom14_indices() -> np.ndarray:
    """``[21, 4, 4]`` int64 chi atom14 slots over the project's AA index order."""
    out = np.zeros((len(AA_CODES) + 1, NUM_CHI, 4), dtype=np.int64)
    for i, aa1 in enumerate(AA_CODES):
        out[i] = chi_atom14_indices(aa1)
    out[len(AA_CODES)] = chi_atom14_indices("GLY")
    return out


def restype_chi_mask() -> np.ndarray:
    """``[21, 4]`` bool chi presence over the project's AA index order (X -> GLY)."""
    out = np.zeros((len(AA_CODES) + 1, NUM_CHI), dtype=bool)
    for i, aa1 in enumerate(AA_CODES):
        out[i] = chi_mask(aa1)
    out[len(AA_CODES)] = chi_mask("GLY")
    return out


def restype_chi_pi_periodic() -> np.ndarray:
    """``[21, 4]`` bool: chi angles that are symmetric under a 180-degree flip.

    Variant-A (torsion) symmetry: a torsion-space loss must treat these chis
    mod pi (ASP chi2, GLU chi3, PHE/TYR chi2). Distinct from the atom-renaming
    swaps used by the offset loss -- see the module docstring.
    """
    out = np.zeros((len(AA_CODES) + 1, NUM_CHI), dtype=bool)
    for i, aa1 in enumerate(AA_CODES):
        out[i] = CHI_PI_PERIODIC[aa1_to_aa3(aa1)]
    out[len(AA_CODES)] = CHI_PI_PERIODIC["GLY"]
    return out
