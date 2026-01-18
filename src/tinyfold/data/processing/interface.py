"""Interface annotation for protein-protein complexes."""

import numpy as np

from tinyfold.constants import INTERFACE_DISTANCE_THRESHOLD


def compute_interface_mask(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    threshold: float = INTERFACE_DISTANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interface residue masks for both chains.

    A residue is at the interface if its CA atom is within threshold
    distance of any CA atom from the other chain.

    Args:
        coords_a: [LA, 4, 3] chain A backbone coordinates
        mask_a: [LA, 4] chain A atom mask
        coords_b: [LB, 4, 3] chain B backbone coordinates
        mask_b: [LB, 4] chain B atom mask
        threshold: Distance threshold in Angstroms

    Returns:
        iface_a: [LA] boolean mask for interface residues in chain A
        iface_b: [LB] boolean mask for interface residues in chain B
    """
    LA = coords_a.shape[0]
    LB = coords_b.shape[0]

    # Get CA coordinates (atom index 1)
    ca_a = coords_a[:, 1, :]  # [LA, 3]
    ca_b = coords_b[:, 1, :]  # [LB, 3]

    # Get CA masks
    ca_mask_a = mask_a[:, 1]  # [LA]
    ca_mask_b = mask_b[:, 1]  # [LB]

    # Handle empty chains
    if LA == 0 or LB == 0:
        return np.zeros(LA, dtype=bool), np.zeros(LB, dtype=bool)

    # Compute pairwise distances between CA atoms
    # [LA, 1, 3] - [1, LB, 3] -> [LA, LB, 3] -> [LA, LB]
    diff = ca_a[:, None, :] - ca_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # [LA, LB]

    # Mask out invalid distances (where either CA is missing)
    valid_mask = ca_mask_a[:, None] & ca_mask_b[None, :]  # [LA, LB]
    dist = np.where(valid_mask, dist, np.inf)

    # Interface residues are those with min distance < threshold
    min_dist_a = dist.min(axis=1)  # [LA] - min distance to any chain B residue
    min_dist_b = dist.min(axis=0)  # [LB] - min distance to any chain A residue

    iface_a = (min_dist_a < threshold) & ca_mask_a
    iface_b = (min_dist_b < threshold) & ca_mask_b

    return iface_a, iface_b


def compute_interface_contacts(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    threshold: float = INTERFACE_DISTANCE_THRESHOLD,
) -> list[tuple[int, int]]:
    """
    Compute list of interface contact pairs.

    Args:
        coords_a: [LA, 4, 3] chain A backbone coordinates
        mask_a: [LA, 4] chain A atom mask
        coords_b: [LB, 4, 3] chain B backbone coordinates
        mask_b: [LB, 4] chain B atom mask
        threshold: Distance threshold in Angstroms

    Returns:
        List of (res_idx_a, res_idx_b) contact pairs
    """
    LA = coords_a.shape[0]
    LB = coords_b.shape[0]

    if LA == 0 or LB == 0:
        return []

    ca_a = coords_a[:, 1, :]
    ca_b = coords_b[:, 1, :]
    ca_mask_a = mask_a[:, 1]
    ca_mask_b = mask_b[:, 1]

    diff = ca_a[:, None, :] - ca_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    valid_mask = ca_mask_a[:, None] & ca_mask_b[None, :]
    contacts = (dist < threshold) & valid_mask

    # Get indices of contacts
    contact_pairs = list(zip(*np.where(contacts)))
    return contact_pairs


def compute_min_interface_distance(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Compute minimum CA-CA distance between chains.

    Used to check if chains are actually interacting.

    Args:
        coords_a: [LA, 4, 3] chain A backbone coordinates
        mask_a: [LA, 4] chain A atom mask
        coords_b: [LB, 4, 3] chain B backbone coordinates
        mask_b: [LB, 4] chain B atom mask

    Returns:
        Minimum CA-CA distance in Angstroms, or inf if no valid pairs
    """
    LA = coords_a.shape[0]
    LB = coords_b.shape[0]

    if LA == 0 or LB == 0:
        return float("inf")

    ca_a = coords_a[:, 1, :]
    ca_b = coords_b[:, 1, :]
    ca_mask_a = mask_a[:, 1]
    ca_mask_b = mask_b[:, 1]

    diff = ca_a[:, None, :] - ca_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    valid_mask = ca_mask_a[:, None] & ca_mask_b[None, :]
    dist = np.where(valid_mask, dist, np.inf)

    return float(dist.min())
