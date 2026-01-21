"""DockQ computation utilities for protein-protein docking evaluation.

Provides functions to compute DockQ scores between predicted and ground truth
protein complex structures.
"""

import os
import tempfile
import torch
from torch import Tensor

# Amino acid 3-letter codes
AA_CODES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Backbone atom names in order: N, CA, C, O
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']


def write_backbone_pdb(
    coords: Tensor,
    aa_seq: Tensor,
    chain_ids: Tensor,
    path: str,
) -> None:
    """Write backbone atoms to a PDB file.

    Args:
        coords: [L, 4, 3] backbone atom coordinates in Angstroms
        aa_seq: [L] amino acid indices (0-19)
        chain_ids: [L] chain IDs (0 or 1)
        path: Output PDB file path
    """
    coords = coords.cpu().numpy()
    aa_seq = aa_seq.cpu().numpy()
    chain_ids = chain_ids.cpu().numpy()

    L = coords.shape[0]
    chain_letters = ['A', 'B']

    lines = []
    atom_idx = 1

    for res_idx in range(L):
        aa_code = AA_CODES[aa_seq[res_idx]]
        chain = chain_letters[chain_ids[res_idx]]
        res_num = res_idx + 1

        for atom_idx_in_res, atom_name in enumerate(BACKBONE_ATOMS):
            x, y, z = coords[res_idx, atom_idx_in_res]
            # PDB format: ATOM serial name resName chain resSeq x y z occupancy tempFactor
            line = (
                f"ATOM  {atom_idx:5d}  {atom_name:<3s} {aa_code:3s} {chain}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            lines.append(line)
            atom_idx += 1

    # Add TER records between chains
    lines.append("END")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def compute_dockq(
    pred_coords: Tensor,
    gt_coords: Tensor,
    aa_seq: Tensor,
    chain_ids: Tensor,
    std: float = 1.0,
) -> dict:
    """Compute DockQ score between predicted and ground truth structures.

    Args:
        pred_coords: [L, 4, 3] predicted backbone coordinates (normalized)
        gt_coords: [L, 4, 3] ground truth backbone coordinates (normalized)
        aa_seq: [L] amino acid indices
        chain_ids: [L] chain IDs (0 or 1)
        std: Coordinate standard deviation for denormalization

    Returns:
        dict with 'dockq', 'fnat', 'irms', 'lrms' scores (or None if computation fails)
    """
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError:
        return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None}

    # Denormalize coordinates
    pred_coords_real = pred_coords * std
    gt_coords_real = gt_coords * std

    # Write temporary PDB files
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = os.path.join(tmpdir, 'pred.pdb')
        gt_path = os.path.join(tmpdir, 'native.pdb')

        write_backbone_pdb(pred_coords_real, aa_seq, chain_ids, pred_path)
        write_backbone_pdb(gt_coords_real, aa_seq, chain_ids, gt_path)

        try:
            # Load structures
            model_struct = load_PDB(pred_path)
            native_struct = load_PDB(gt_path)

            # Run DockQ
            chain_map = {'A': 'A', 'B': 'B'}
            results = run_on_all_native_interfaces(
                model_struct, native_struct, chain_map=chain_map
            )

            # Result is (dict_of_interfaces, best_dockq_score)
            if results and results[0]:
                interface_dict = results[0]
                # Get first interface (usually 'AB')
                for interface_key, scores in interface_dict.items():
                    return {
                        'dockq': scores['DockQ'],
                        'fnat': scores['fnat'],
                        'irms': scores['iRMSD'],
                        'lrms': scores['LRMSD'],
                    }

            # No interface found
            return {'dockq': 0.0, 'fnat': 0.0, 'irms': float('inf'), 'lrms': float('inf')}

        except Exception as e:
            # DockQ computation failed (e.g., no interface contacts)
            return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None}


def compute_dockq_batch(
    pred_coords_list: list,
    gt_coords_list: list,
    aa_seq_list: list,
    chain_ids_list: list,
    std_list: list,
) -> dict:
    """Compute average DockQ scores over a batch of structures.

    Args:
        pred_coords_list: List of [L, 4, 3] predicted coordinates
        gt_coords_list: List of [L, 4, 3] ground truth coordinates
        aa_seq_list: List of [L] amino acid sequences
        chain_ids_list: List of [L] chain IDs
        std_list: List of coordinate stds for denormalization

    Returns:
        dict with average scores
    """
    dockq_scores = []
    fnat_scores = []
    irms_scores = []
    lrms_scores = []

    for pred, gt, aa, chains, std in zip(
        pred_coords_list, gt_coords_list, aa_seq_list, chain_ids_list, std_list
    ):
        result = compute_dockq(pred, gt, aa, chains, std)
        if result['dockq'] is not None:
            dockq_scores.append(result['dockq'])
            fnat_scores.append(result['fnat'])
            irms_scores.append(result['irms'])
            lrms_scores.append(result['lrms'])

    if not dockq_scores:
        return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None, 'n_valid': 0}

    return {
        'dockq': sum(dockq_scores) / len(dockq_scores),
        'fnat': sum(fnat_scores) / len(fnat_scores),
        'irms': sum(irms_scores) / len(irms_scores),
        'lrms': sum(lrms_scores) / len(lrms_scores),
        'n_valid': len(dockq_scores),
    }
