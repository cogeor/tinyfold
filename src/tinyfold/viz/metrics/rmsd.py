"""RMSD computation utilities."""

import numpy as np

from tinyfold.viz.metrics.align import kabsch_align


def compute_rmsd(pred_xyz: np.ndarray, ref_xyz: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute RMSD between two coordinate sets.

    Args:
        pred_xyz: [N, 3] predicted coordinates
        ref_xyz: [N, 3] reference coordinates
        mask: [N] optional boolean mask

    Returns:
        RMSD value in same units as input coordinates
    """
    if mask is not None:
        pred_xyz = pred_xyz[mask]
        ref_xyz = ref_xyz[mask]

    diff = pred_xyz - ref_xyz
    return np.sqrt((diff ** 2).sum() / len(pred_xyz))


def backbone_rmsd(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    atom_to_res: np.ndarray,
    chain_id_res: np.ndarray,
    atom_mask: np.ndarray | None = None,
    interface_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute various RMSD metrics for backbone atoms.

    Args:
        pred_xyz: [N_atom, 3] predicted coordinates
        ref_xyz: [N_atom, 3] reference coordinates
        atom_to_res: [N_atom] residue index per atom
        chain_id_res: [L] chain ID per residue (0 or 1)
        atom_mask: [N_atom] optional mask for valid atoms
        interface_mask: [L] optional mask for interface residues

    Returns:
        dict with:
            - rmsd_complex: RMSD after complex alignment
            - rmsd_chain_a: RMSD of chain A atoms
            - rmsd_chain_b: RMSD of chain B atoms
            - lrmsd: Ligand RMSD (align chain A, measure chain B)
            - irmsd: Interface RMSD (if interface_mask provided)
    """
    if atom_mask is None:
        atom_mask = np.ones(len(pred_xyz), dtype=bool)

    # Get chain masks at atom level
    chain_id_atom = chain_id_res[atom_to_res]
    chain_a_mask = (chain_id_atom == 0) & atom_mask
    chain_b_mask = (chain_id_atom == 1) & atom_mask

    results = {}

    # Complex-aligned RMSD
    aligned_pred, _, _ = kabsch_align(pred_xyz, ref_xyz, mask=atom_mask)
    results["rmsd_complex"] = compute_rmsd(aligned_pred, ref_xyz, atom_mask)
    results["rmsd_chain_a"] = compute_rmsd(aligned_pred, ref_xyz, chain_a_mask)
    results["rmsd_chain_b"] = compute_rmsd(aligned_pred, ref_xyz, chain_b_mask)

    # LRMSD: align on chain A, measure chain B
    aligned_on_a, _, _ = kabsch_align(pred_xyz, ref_xyz, mask=chain_a_mask)
    results["lrmsd"] = compute_rmsd(aligned_on_a, ref_xyz, chain_b_mask)

    # Interface RMSD (if mask provided)
    if interface_mask is not None:
        interface_atom_mask = interface_mask[atom_to_res] & atom_mask
        if interface_atom_mask.any():
            results["irmsd"] = compute_rmsd(aligned_pred, ref_xyz, interface_atom_mask)
        else:
            results["irmsd"] = float("nan")
    else:
        results["irmsd"] = float("nan")

    return results
