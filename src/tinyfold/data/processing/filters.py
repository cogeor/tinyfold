"""Validation filters for protein complexes."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tinyfold.constants import (
    BOND_LENGTH_TOLERANCE,
    BOND_LENGTHS,
    MAX_CHAIN_LENGTH,
    MAX_INTER_CHAIN_DISTANCE,
    MIN_BACKBONE_COMPLETENESS,
    MIN_CHAIN_LENGTH,
    NUM_ATOM_TYPES,
)
from tinyfold.data.processing.atomization import compute_bond_lengths
from tinyfold.data.processing.interface import compute_min_interface_distance


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
    max_len: int = MAX_CHAIN_LENGTH,
) -> FilterResult:
    """
    Check chain lengths are within bounds.

    Args:
        LA: Length of chain A
        LB: Length of chain B
        min_len: Minimum allowed length
        max_len: Maximum allowed length

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

    # Check backbone bonds (type 0)
    backbone_mask = bond_type == 0
    backbone_lengths = lengths[backbone_mask]

    # Expected backbone bond lengths: ~1.2-1.6 Angstroms
    # We check they're in a reasonable range
    min_expected = min(BOND_LENGTHS.values()) - tolerance
    max_expected = max(BOND_LENGTHS.values()) + tolerance

    if len(backbone_lengths) > 0:
        if np.any(backbone_lengths < min_expected - 0.5):
            bad_idx = np.where(backbone_lengths < min_expected - 0.5)[0][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"backbone bond too short: {backbone_lengths[bad_idx]:.3f}",
            )
        if np.any(backbone_lengths > max_expected + 0.5):
            bad_idx = np.where(backbone_lengths > max_expected + 0.5)[0][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"backbone bond too long: {backbone_lengths[bad_idx]:.3f}",
            )

    # Check peptide bonds (type 1)
    peptide_mask = bond_type == 1
    peptide_lengths = lengths[peptide_mask]

    # Peptide C-N bond: ~1.33 Angstroms
    expected_peptide = BOND_LENGTHS["C-N"]

    if len(peptide_lengths) > 0:
        if np.any(peptide_lengths < expected_peptide - tolerance - 0.3):
            bad_idx = np.where(peptide_lengths < expected_peptide - tolerance - 0.3)[0][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"peptide bond too short: {peptide_lengths[bad_idx]:.3f}",
            )
        if np.any(peptide_lengths > expected_peptide + tolerance + 0.3):
            bad_idx = np.where(peptide_lengths > expected_peptide + tolerance + 0.3)[0][0]
            return FilterResult.fail(
                FilterReason.INVALID_BOND_LENGTHS,
                f"peptide bond too long: {peptide_lengths[bad_idx]:.3f}",
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

    Returns:
        FilterResult - first failing check or PASSED
    """
    # Chain length
    result = validate_chain_length(LA, LB)
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
