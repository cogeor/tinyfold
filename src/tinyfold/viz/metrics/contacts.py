"""Contact map and interface metrics."""

import numpy as np


def contact_map_CA(
    xyz: np.ndarray,
    atom_type: np.ndarray,
    atom_to_res: np.ndarray,
    chain_id_res: np.ndarray,
    cutoff: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CA-based inter-chain contact map.

    Args:
        xyz: [N_atom, 3] atom coordinates
        atom_type: [N_atom] atom type (1 = CA)
        atom_to_res: [N_atom] residue index per atom
        chain_id_res: [L] chain ID per residue
        cutoff: distance cutoff for contacts (Angstroms)

    Returns:
        A_idx: residue indices in chain A with contacts
        B_idx: residue indices in chain B with contacts
        contact_matrix: [LA, LB] boolean matrix
    """
    # Find CA atoms
    ca_mask = atom_type == 1  # CA is type 1
    ca_xyz = xyz[ca_mask]
    ca_res = atom_to_res[ca_mask]

    # Separate by chain
    L = len(chain_id_res)
    LA = (chain_id_res == 0).sum()
    LB = (chain_id_res == 1).sum()

    # Get CA coordinates per residue
    ca_coords = np.zeros((L, 3))
    ca_valid = np.zeros(L, dtype=bool)
    for i, res_idx in enumerate(ca_res):
        ca_coords[res_idx] = ca_xyz[i]
        ca_valid[res_idx] = True

    # Split by chain
    ca_A = ca_coords[:LA]
    ca_B = ca_coords[LA:]
    valid_A = ca_valid[:LA]
    valid_B = ca_valid[LA:]

    # Compute distance matrix
    # dist[i, j] = distance between CA of residue i (chain A) and residue j (chain B)
    diff = ca_A[:, np.newaxis, :] - ca_B[np.newaxis, :, :]  # [LA, LB, 3]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [LA, LB]

    # Apply validity mask
    valid_mask = valid_A[:, np.newaxis] & valid_B[np.newaxis, :]
    dist = np.where(valid_mask, dist, np.inf)

    # Contact matrix
    contact_matrix = dist < cutoff

    # Get indices of contacting residues
    A_idx, B_idx = np.where(contact_matrix)

    return A_idx, B_idx, contact_matrix


def contact_metrics(pred_contacts: np.ndarray, ref_contacts: np.ndarray) -> dict[str, float]:
    """Compute precision, recall, F1 for contact prediction.

    Args:
        pred_contacts: [LA, LB] predicted contact matrix (boolean)
        ref_contacts: [LA, LB] reference contact matrix (boolean)

    Returns:
        dict with precision, recall, f1, n_pred, n_ref
    """
    pred_flat = pred_contacts.flatten()
    ref_flat = ref_contacts.flatten()

    tp = (pred_flat & ref_flat).sum()
    fp = (pred_flat & ~ref_flat).sum()
    fn = (~pred_flat & ref_flat).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": int(pred_flat.sum()),
        "n_ref": int(ref_flat.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def interface_residues(
    contact_matrix: np.ndarray,
    chain_id_res: np.ndarray,
) -> np.ndarray:
    """Get mask of interface residues from contact matrix.

    Args:
        contact_matrix: [LA, LB] boolean contact matrix
        chain_id_res: [L] chain ID per residue

    Returns:
        interface_mask: [L] boolean mask for interface residues
    """
    LA = (chain_id_res == 0).sum()
    L = len(chain_id_res)

    interface_mask = np.zeros(L, dtype=bool)

    # Chain A residues with contacts
    has_contact_A = contact_matrix.any(axis=1)
    interface_mask[:LA] = has_contact_A

    # Chain B residues with contacts
    has_contact_B = contact_matrix.any(axis=0)
    interface_mask[LA:] = has_contact_B

    return interface_mask
