"""Collate function for batching variable-length PPI samples."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_ppi(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-length PPI samples into a batch.

    Handles:
    - Padding residue-level tensors to max length
    - Padding atom-level tensors to max atoms
    - Merging bond graphs with offset indices

    Args:
        batch: List of sample dicts from PPIDataset

    Returns:
        Batched dict with:
        - seq: [B, Lmax] padded sequence indices
        - chain_id_res: [B, Lmax] padded chain IDs
        - res_idx: [B, Lmax] padded residue indices
        - res_mask: [B, Lmax] True for real residues
        - atom_coords: [B, Natom_max, 3] padded coordinates
        - atom_mask: [B, Natom_max] True for real/valid atoms
        - atom_to_res: [B, Natom_max] padded residue indices
        - atom_type: [B, Natom_max] padded atom types
        - edge_index: [2, E_total] merged bond edges with offsets
        - edge_type: [E_total] bond types
        - atom_batch: [Natom_total] sample index per atom (for graph batching)
        - sample_ids: list of sample IDs
        - LA: [B] chain A lengths
        - LB: [B] chain B lengths
    """
    B = len(batch)

    # Collect sample metadata
    sample_ids = [b["sample_id"] for b in batch]
    LA = torch.tensor([b["LA"] for b in batch], dtype=torch.long)
    LB = torch.tensor([b["LB"] for b in batch], dtype=torch.long)

    # Pad residue-level tensors
    seqs = [b["seq"] for b in batch]
    seq_padded = pad_sequence(seqs, batch_first=True, padding_value=0)

    chain_ids = [b["chain_id_res"] for b in batch]
    chain_id_padded = pad_sequence(chain_ids, batch_first=True, padding_value=0)

    res_idxs = [b["res_idx"] for b in batch]
    res_idx_padded = pad_sequence(res_idxs, batch_first=True, padding_value=0)

    # Residue mask
    res_mask = pad_sequence(
        [torch.ones(len(s), dtype=torch.bool) for s in seqs],
        batch_first=True,
        padding_value=False,
    )

    # Interface mask
    iface_masks = [b["iface_mask"] for b in batch]
    iface_mask_padded = pad_sequence(iface_masks, batch_first=True, padding_value=False)

    # Pad atom-level tensors
    coords = [b["atom_coords"] for b in batch]
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0.0)

    atom_masks = [b["atom_mask"] for b in batch]
    atom_mask_padded = pad_sequence(atom_masks, batch_first=True, padding_value=False)

    atom_to_res_list = [b["atom_to_res"] for b in batch]
    atom_to_res_padded = pad_sequence(atom_to_res_list, batch_first=True, padding_value=0)

    atom_types = [b["atom_type"] for b in batch]
    atom_type_padded = pad_sequence(atom_types, batch_first=True, padding_value=0)

    # Build merged edge index with offsets
    edge_indices = []
    edge_types = []
    atom_batch = []
    atom_offset = 0

    for i, b in enumerate(batch):
        n_atoms = b["atom_coords"].shape[0]

        # Offset bond indices
        src = b["bonds_src"] + atom_offset
        dst = b["bonds_dst"] + atom_offset
        edge_indices.append(torch.stack([src, dst], dim=0))
        edge_types.append(b["bond_type"])

        # Track which sample each atom belongs to
        atom_batch.append(torch.full((n_atoms,), i, dtype=torch.long))

        atom_offset += n_atoms

    # Concatenate edges
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types)
        atom_batch = torch.cat(atom_batch)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        atom_batch = torch.zeros(0, dtype=torch.long)

    return {
        "seq": seq_padded,
        "chain_id_res": chain_id_padded,
        "res_idx": res_idx_padded,
        "res_mask": res_mask,
        "iface_mask": iface_mask_padded,
        "atom_coords": coords_padded,
        "atom_mask": atom_mask_padded,
        "atom_to_res": atom_to_res_padded,
        "atom_type": atom_type_padded,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "atom_batch": atom_batch,
        "sample_ids": sample_ids,
        "LA": LA,
        "LB": LB,
    }
