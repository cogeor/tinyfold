"""DockQ computation utilities for protein-protein docking evaluation.

Provides functions to compute DockQ scores between predicted and ground truth
protein complex structures.
"""

import os
import tempfile

from torch import Tensor

# Amino acid 3-letter codes
AA_CODES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Backbone atom names in order: N, CA, C, O
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']

# CAPRI quality bands, keyed on DockQ. Read these, never C-RMSD: backbone
# C-RMSD can sit at 6-7 A while the docking is structurally wrong.
CAPRI_BANDS = ("incorrect", "acceptable", "medium", "high")


def capri_band(dockq: float | None) -> str | None:
    """CAPRI quality band for a DockQ score.

    incorrect  < 0.23 <= acceptable < 0.49 <= medium < 0.80 <= high
    """
    if dockq is None:
        return None
    if dockq < 0.23:
        return "incorrect"
    if dockq < 0.49:
        return "acceptable"
    if dockq < 0.80:
        return "medium"
    return "high"


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


def chains_are_interchangeable(aa_seq: Tensor, chain_ids: Tensor) -> bool:
    """True when chain A and chain B carry an identical residue sequence.

    For such a complex (a homodimer) the two chains are indistinguishable on
    the input side, so which one the ground truth calls "A" is an arbitrary
    crystallographic labelling and BOTH assignments are equally valid answers.
    **34.5% of DIPS-Plus complexes are exact homodimers** (14,456 / 41,883), so
    this is not an edge case.
    """
    a = aa_seq[chain_ids == 0]
    b = aa_seq[chain_ids == 1]
    if a.numel() == 0 or b.numel() == 0:
        return False
    if a.shape != b.shape:
        return False
    return bool((a == b).all())


def _run_dockq(model_struct, native_struct, chain_map: dict) -> dict | None:
    """Score one chain assignment. Returns None if DockQ could not score it."""
    from DockQ.DockQ import run_on_all_native_interfaces

    try:
        results = run_on_all_native_interfaces(
            model_struct, native_struct, chain_map=chain_map
        )
    except Exception:
        # DockQ computation failed (e.g. no interface contacts).
        return None

    # Result is (dict_of_interfaces, best_dockq_score).
    if results and results[0]:
        for _interface_key, scores in results[0].items():
            return {
                'dockq': scores['DockQ'],
                'fnat': scores['fnat'],
                'irms': scores['iRMSD'],
                'lrms': scores['LRMSD'],
            }
    # Structures loaded but no interface was found.
    return {'dockq': 0.0, 'fnat': 0.0, 'irms': float('inf'), 'lrms': float('inf')}


def compute_dockq(
    pred_coords: Tensor,
    gt_coords: Tensor,
    aa_seq: Tensor,
    chain_ids: Tensor,
    std: float = 1.0,
    allow_chain_permutation: bool = True,
) -> dict:
    """Compute DockQ score between predicted and ground truth structures.

    When the two chains are sequence-identical, both chain assignments are
    scored and the better one is returned. Standard DockQ and the CAPRI criteria
    both allow optimal chain mapping, so without this a structurally correct
    homodimer prediction with A/B exchanged is scored as a failure and our
    numbers are not comparable to anyone else's.

    Args:
        pred_coords: [L, 4, 3] predicted backbone coordinates (normalized)
        gt_coords: [L, 4, 3] ground truth backbone coordinates (normalized)
        aa_seq: [L] amino acid indices
        chain_ids: [L] chain IDs (0 or 1)
        std: Coordinate standard deviation for denormalization
        allow_chain_permutation: Try the swapped assignment for homodimers.
            Set False to recover the historical single-assignment behaviour.

    Returns:
        dict with 'dockq', 'fnat', 'irms', 'lrms' scores (or None if computation
        fails), plus 'chain_perm_used' indicating whether the swapped assignment
        won.
    """
    try:
        from DockQ.DockQ import load_PDB
    except ImportError:
        return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None,
                'chain_perm_used': False}

    # Denormalize coordinates
    pred_coords_real = pred_coords * std
    gt_coords_real = gt_coords * std

    # Only a homodimer admits a second valid assignment; trying the swap on a
    # heterodimer would compare mismatched chains.
    chain_maps = [{'A': 'A', 'B': 'B'}]
    if allow_chain_permutation and chains_are_interchangeable(aa_seq, chain_ids):
        chain_maps.append({'A': 'B', 'B': 'A'})

    # Write temporary PDB files
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = os.path.join(tmpdir, 'pred.pdb')
        gt_path = os.path.join(tmpdir, 'native.pdb')

        write_backbone_pdb(pred_coords_real, aa_seq, chain_ids, pred_path)
        write_backbone_pdb(gt_coords_real, aa_seq, chain_ids, gt_path)

        try:
            model_struct = load_PDB(pred_path)
            native_struct = load_PDB(gt_path)
        except Exception:
            return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None,
                    'chain_perm_used': False}

        # Structures are parsed once; only the assignment is re-scored.
        best, best_i = None, 0
        for i, chain_map in enumerate(chain_maps):
            scored = _run_dockq(model_struct, native_struct, chain_map)
            if scored is None:
                continue
            if best is None or scored['dockq'] > best['dockq']:
                best, best_i = scored, i

        if best is None:
            return {'dockq': None, 'fnat': None, 'irms': None, 'lrms': None,
                    'chain_perm_used': False}
        best['chain_perm_used'] = best_i == 1
        return best


def compute_dockq_batch(
    pred_coords_list: list,
    gt_coords_list: list,
    aa_seq_list: list,
    chain_ids_list: list,
    std_list: list,
    allow_chain_permutation: bool = True,
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
        result = compute_dockq(
            pred, gt, aa, chains, std,
            allow_chain_permutation=allow_chain_permutation,
        )
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
