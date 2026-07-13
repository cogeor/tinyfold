"""Chain cleaning utilities.

Handles alternate locations, modified residues, and other structure quirks.
"""

import numpy as np

from tinyfold.constants import AA3_TO_AA1, AA_TO_IDX, MODIFIED_AA_MAP


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
