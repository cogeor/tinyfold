"""Chain cleaning utilities.

Handles alternate locations, modified residues, and other structure quirks.
"""

import numpy as np

from tinyfold.constants import AA3_TO_AA1, AA_TO_IDX, MODIFIED_AA_MAP


def resolve_altlocs(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve alternate location issues in coordinates.

    This is a placeholder - actual altloc resolution happens in structure_io.py
    during parsing. This function can be used for post-processing if needed.

    Args:
        coords: [L, 4, 3] backbone coordinates
        mask: [L, 4] atom mask

    Returns:
        Cleaned coords and mask
    """
    # Currently a no-op since altlocs are resolved during parsing
    return coords, mask


def map_modified_residue(residue_name: str) -> str:
    """
    Map modified residue to standard amino acid.

    Args:
        residue_name: 3-letter residue code

    Returns:
        1-letter standard AA code or 'X' for unknown
    """
    residue_name = residue_name.upper()

    # Standard AA
    if residue_name in AA3_TO_AA1:
        return AA3_TO_AA1[residue_name]

    # Known modified residue
    if residue_name in MODIFIED_AA_MAP:
        return MODIFIED_AA_MAP[residue_name]

    # Unknown
    return "X"


def clean_chain(
    sequence: list[str],
    seq_indices: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    residue_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean chain data by applying all cleaning rules.

    Cleaning steps:
    1. Re-map any remaining non-standard residues
    2. Ensure consistent indexing

    Args:
        sequence: 1-letter AA codes
        seq_indices: [L] AA indices
        coords: [L, 4, 3] backbone coords
        mask: [L, 4] atom mask
        residue_names: 3-letter residue names

    Returns:
        Cleaned (sequence, seq_indices, coords, mask)
    """
    # Re-verify sequence indices match sequence
    cleaned_seq = []
    cleaned_indices = []

    for i, (aa, res_name) in enumerate(zip(sequence, residue_names)):
        # Double-check mapping
        expected_aa = map_modified_residue(res_name)
        if aa != expected_aa:
            aa = expected_aa

        cleaned_seq.append(aa)
        cleaned_indices.append(AA_TO_IDX.get(aa, AA_TO_IDX["X"]))

    return (
        cleaned_seq,
        np.array(cleaned_indices, dtype=np.int64),
        coords,
        mask,
    )


def remove_terminal_missing(
    sequence: list[str],
    seq_indices: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove terminal residues with missing backbone atoms.

    Useful for cleaning up chains with disordered termini.

    Args:
        sequence: 1-letter AA codes
        seq_indices: [L] AA indices
        coords: [L, 4, 3] backbone coords
        mask: [L, 4] atom mask

    Returns:
        Trimmed (sequence, seq_indices, coords, mask)
    """
    L = len(sequence)
    if L == 0:
        return sequence, seq_indices, coords, mask

    # Find first residue with complete backbone (at least N, CA, C)
    start = 0
    for i in range(L):
        if mask[i, :3].all():  # N, CA, C present
            start = i
            break
    else:
        # No complete residue found
        return [], np.array([], dtype=np.int64), np.zeros((0, 4, 3)), np.zeros((0, 4), dtype=bool)

    # Find last residue with complete backbone
    end = L
    for i in range(L - 1, -1, -1):
        if mask[i, :3].all():
            end = i + 1
            break

    if start >= end:
        return [], np.array([], dtype=np.int64), np.zeros((0, 4, 3)), np.zeros((0, 4), dtype=bool)

    return (
        sequence[start:end],
        seq_indices[start:end],
        coords[start:end],
        mask[start:end],
    )
