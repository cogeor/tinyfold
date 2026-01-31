"""Data loading and batching utilities for TinyFold training.

Provides:
- load_sample: Load a single sample from parquet table
- collate_batch: Collate samples into padded batches
"""

from typing import Dict, List, Any
import torch
from torch import Tensor


def load_sample(table, i: int, normalize: bool = True) -> Dict[str, Any]:
    """Load sample at residue level (4 atoms per residue).

    Args:
        table: PyArrow table from samples.parquet
        i: Sample index
        normalize: If True, normalize coords to unit variance

    Returns:
        Dict with:
            - coords: [N_atoms, 3] all atom coordinates
            - coords_res: [L, 4, 3] atoms per residue (N, CA, C, O)
            - centroids: [L, 3] residue centroids
            - aa_seq: [L] amino acid sequence
            - chain_ids: [L] chain IDs
            - res_idx: [L] residue indices
            - std: normalization factor
            - n_atoms, n_res: counts
            - sample_id: identifier
    """
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center coordinates
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    # Compute std
    original_std = coords.std()

    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    return {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': torch.arange(n_res),
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch(samples: List[Dict], device: torch.device) -> Dict[str, Any]:
    """Collate residue-level samples into a padded batch.

    Args:
        samples: List of sample dicts from load_sample
        device: Target device

    Returns:
        Batched dict with padded tensors
    """
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    # Residue-level tensors
    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors
    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)

    stds = []

    for i, s in enumerate(samples):
        L = s['n_res']
        N = s['n_atoms']

        centroids[i, :L] = s['centroids']
        coords_res[i, :L] = s['coords_res']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True

        coords[i, :N] = s['coords']
        atom_types[i, :N] = s['atom_types']
        atom_to_res[i, :N] = s['atom_to_res']
        mask_atom[i, :N] = True

        stds.append(s['std'])

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }


# Aliases for backward compatibility
load_sample_raw = load_sample
collate_batch_residue = collate_batch
