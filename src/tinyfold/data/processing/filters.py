"""Validation filters for protein complexes."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from tinyfold.constants import (
    BOND_LENGTH_TOLERANCE,
    BOND_LENGTHS,
    BOND_TYPE_C_O,
    BOND_TYPE_CA_C,
    BOND_TYPE_N_CA,
    BOND_TYPE_PEPTIDE,
    MAX_CHAIN_LENGTH,
    MAX_INTER_CHAIN_DISTANCE,
    MIN_BACKBONE_COMPLETENESS,
    MIN_CHAIN_LENGTH,
    NUM_ATOM_TYPES,
)
from tinyfold.data.processing.atomization import compute_bond_lengths
from tinyfold.data.processing.interface import compute_min_interface_distance

# Expected length (Angstroms) for each bond type. Keyed by the 4-type encoding
# so validation checks every bond against its own reference, not a shared one.
EXPECTED_BOND_LENGTH = {
    BOND_TYPE_N_CA: BOND_LENGTHS["N-CA"],
    BOND_TYPE_CA_C: BOND_LENGTHS["CA-C"],
    BOND_TYPE_C_O: BOND_LENGTHS["C-O"],
    BOND_TYPE_PEPTIDE: BOND_LENGTHS["C-N"],
}
# Extra slack on top of the tolerance: validation is lenient and only catches
# gross errors (badly-placed atoms), not fine geometry.
BOND_LENGTH_SLACK = 0.3


class FilterReason(Enum):
    """Reasons for filtering out a sample."""

    PASSED = "passed"
    CHAIN_A_TOO_SHORT = "chain_a_too_short"
    CHAIN_B_TOO_SHORT = "chain_b_too_short"
    CHAIN_A_TOO_LONG = "chain_a_too_long"
    CHAIN_B_TOO_LONG = "chain_b_too_long"
    CHAIN_A_EMPTY = "chain_a_empty"
    CHAIN_B_EMPTY = "chain_b_empty"
    LOW_BACKBONE_COMPLETENESS = "low_backbone_completeness"
    NO_INTERACTION = "no_interaction"
    INVALID_BOND_LENGTHS = "invalid_bond_lengths"
    INVALID_COORDINATES = "invalid_coordinates"


@dataclass
class FilterResult:
    """Result of sample validation."""

    passed: bool
    reason: FilterReason
    details: str = ""

    @staticmethod
    def ok() -> "FilterResult":
        return FilterResult(passed=True, reason=FilterReason.PASSED)

    @staticmethod
    def fail(reason: FilterReason, details: str = "") -> "FilterResult":
        return FilterResult(passed=False, reason=reason, details=details)


def validate_chain_length(
    LA: int,
    LB: int,
    min_len: int = MIN_CHAIN_LENGTH,
    max_len: Optional[int] = MAX_CHAIN_LENGTH,
) -> FilterResult:
    """
    Check chain lengths are within bounds.

    Args:
        LA: Length of chain A
        LB: Length of chain B
        min_len: Minimum allowed length. Defaults to ``MIN_CHAIN_LENGTH``.
        max_len: Maximum allowed length. ``None`` disables the upper bound
            (faithful pass-through of whatever the source dataset emits).
            Defaults to ``MAX_CHAIN_LENGTH``.

    Returns:
        FilterResult
    """
    if LA == 0:
        return FilterResult.fail(FilterReason.CHAIN_A_EMPTY)
    if LB == 0:
        return FilterResult.fail(FilterReason.CHAIN_B_EMPTY)
    if LA < min_len:
        return FilterResult.fail(FilterReason.CHAIN_A_TOO_SHORT, f"LA={LA} < {min_len}")
    if LB < min_len:
        return FilterResult.fail(FilterReason.CHAIN_B_TOO_SHORT, f"LB={LB} < {min_len}")
    if max_len is not None:
        if LA > max_len:
            return FilterResult.fail(FilterReason.CHAIN_A_TOO_LONG, f"LA={LA} > {max_len}")
        if LB > max_len:
            return FilterResult.fail(FilterReason.CHAIN_B_TOO_LONG, f"LB={LB} > {max_len}")

    return FilterResult.ok()


def validate_backbone_completeness(
    atom_mask: np.ndarray,
    min_completeness: float = MIN_BACKBONE_COMPLETENESS,
) -> FilterResult:
    """
    Check backbone atom completeness.

    Args:
        atom_mask: [Natom] boolean mask
        min_completeness: Minimum fraction of atoms that must be present

    Returns:
        FilterResult
    """
    if len(atom_mask) == 0:
        return FilterResult.fail(FilterReason.LOW_BACKBONE_COMPLETENESS, "empty mask")

    completeness = atom_mask.mean()
    if completeness < min_completeness:
        return FilterResult.fail(
            FilterReason.LOW_BACKBONE_COMPLETENESS,
            f"completeness={completeness:.3f} < {min_completeness}",
        )

    return FilterResult.ok()


def validate_interaction(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    max_distance: float = MAX_INTER_CHAIN_DISTANCE,
) -> FilterResult:
    """
    Check that chains are actually interacting.

    Args:
        coords_a: [LA, 4, 3] chain A coordinates
        mask_a: [LA, 4] chain A mask
        coords_b: [LB, 4, 3] chain B coordinates
        mask_b: [LB, 4] chain B mask
        max_distance: Maximum min CA-CA distance to be considered interacting

    Returns:
        FilterResult
    """
    min_dist = compute_min_interface_distance(coords_a, mask_a, coords_b, mask_b)

    if min_dist > max_distance:
        return FilterResult.fail(
            FilterReason.NO_INTERACTION,
            f"min_distance={min_dist:.2f} > {max_distance}",
        )

    return FilterResult.ok()


def validate_coordinates(
    atom_coords: np.ndarray,
    atom_mask: np.ndarray,
) -> FilterResult:
    """
    Check coordinates are valid (no NaN/Inf).

    Args:
        atom_coords: [Natom, 3] coordinates
        atom_mask: [Natom] valid atom mask

    Returns:
        FilterResult
    """
    valid_coords = atom_coords[atom_mask]

    if len(valid_coords) == 0:
        return FilterResult.fail(FilterReason.INVALID_COORDINATES, "no valid atoms")

    if np.any(np.isnan(valid_coords)):
        return FilterResult.fail(FilterReason.INVALID_COORDINATES, "NaN in coordinates")

    if np.any(np.isinf(valid_coords)):
        return FilterResult.fail(FilterReason.INVALID_COORDINATES, "Inf in coordinates")

    return FilterResult.ok()


def validate_bond_lengths(
    atom_coords: np.ndarray,
    bonds_src: np.ndarray,
    bonds_dst: np.ndarray,
    bond_type: np.ndarray,
    atom_mask: np.ndarray,
    tolerance: float = BOND_LENGTH_TOLERANCE,
) -> FilterResult:
    """
    Check bond lengths are chemically reasonable.

    Args:
        atom_coords: [Natom, 3] coordinates
        bonds_src: [E] source atom indices
        bonds_dst: [E] destination atom indices
        bond_type: [E] bond types
        atom_mask: [Natom] valid atom mask
        tolerance: Allowed deviation from expected lengths

    Returns:
        FilterResult
    """
    if len(bonds_src) == 0:
        return FilterResult.ok()

    lengths = compute_bond_lengths(atom_coords, bonds_src, bonds_dst, atom_mask)

    # Validate every bond type against its own expected length. The 4-type
    # encoding (N-CA, CA-C, C-O, peptide C-N) means a single shared reference
    # would mis-check most bonds; a per-type band keeps each honest while
    # staying lenient (tolerance + slack) so only gross errors are rejected.
    band = tolerance + BOND_LENGTH_SLACK
    for btype, expected in EXPECTED_BOND_LENGTH.items():
        type_lengths = lengths[bond_type == btype]
        if len(type_lengths) == 0:
            continue
        if np.any(type_lengths < expected - band):
            bad = type_lengths[type_lengths < expected - band][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"bond type {btype} too short: {bad:.3f} (expected ~{expected:.3f})",
            )
        if np.any(type_lengths > expected + band):
            bad = type_lengths[type_lengths > expected + band][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"bond type {btype} too long: {bad:.3f} (expected ~{expected:.3f})",
            )

    return FilterResult.ok()


def validate_sample(
    LA: int,
    LB: int,
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    atom_coords: np.ndarray,
    atom_mask: np.ndarray,
    bonds_src: np.ndarray,
    bonds_dst: np.ndarray,
    bond_type: np.ndarray,
    min_chain_length: int = MIN_CHAIN_LENGTH,
    max_chain_length: Optional[int] = MAX_CHAIN_LENGTH,
) -> FilterResult:
    """
    Run all validation checks on a sample.

    Args:
        LA: Length of chain A
        LB: Length of chain B
        coords_a: [LA, 4, 3] chain A coordinates
        mask_a: [LA, 4] chain A mask
        coords_b: [LB, 4, 3] chain B coordinates
        mask_b: [LB, 4] chain B mask
        atom_coords: [Natom, 3] flattened coordinates
        atom_mask: [Natom] flattened mask
        bonds_src: [E] bond source indices
        bonds_dst: [E] bond destination indices
        bond_type: [E] bond types
        min_chain_length: Lower bound on per-chain length. Default from constants.
        max_chain_length: Upper bound on per-chain length, or ``None`` for no cap.
            Default from constants (None = no upper cap).

    Returns:
        FilterResult - first failing check or PASSED
    """
    # Chain length
    result = validate_chain_length(LA, LB, min_chain_length, max_chain_length)
    if not result.passed:
        return result

    # Backbone completeness
    result = validate_backbone_completeness(atom_mask)
    if not result.passed:
        return result

    # Chain interaction
    result = validate_interaction(coords_a, mask_a, coords_b, mask_b)
    if not result.passed:
        return result

    # Coordinate validity
    result = validate_coordinates(atom_coords, atom_mask)
    if not result.passed:
        return result

    # Bond lengths (lenient - just catches major issues)
    result = validate_bond_lengths(atom_coords, bonds_src, bonds_dst, bond_type, atom_mask)
    if not result.passed:
        return result

    return FilterResult.ok()
