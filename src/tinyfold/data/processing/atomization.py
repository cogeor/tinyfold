"""Atomization: convert residue-level data to atom-level.

Handles flattening residue coordinates to atoms and building bond graphs.
"""

import numpy as np

from tinyfold.constants import (
    BACKBONE_BONDS,
    BOND_TYPE_PEPTIDE,
    NUM_ATOM_TYPES,
)


def atomize_chains(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert residue-level coordinates to atom-level for two chains.

    Concatenates chain A and chain B, flattening the [L, 4, 3] structure
    to [Natom, 3] where Natom = 4 * (LA + LB).

    Args:
        coords_a: [LA, 4, 3] chain A backbone coordinates
        mask_a: [LA, 4] chain A atom mask
        coords_b: [LB, 4, 3] chain B backbone coordinates
        mask_b: [LB, 4] chain B atom mask

    Returns:
        atom_coords: [Natom, 3] flattened coordinates
        atom_mask: [Natom] flattened mask
        atom_to_res: [Natom] residue index for each atom
        atom_type: [Natom] atom type (0=N, 1=CA, 2=C, 3=O)
        chain_id_atom: [Natom] chain ID (0=A, 1=B) for each atom
    """
    LA = coords_a.shape[0]
    LB = coords_b.shape[0]
    L = LA + LB
    Natom = L * NUM_ATOM_TYPES

    # Concatenate chains
    coords_concat = np.concatenate([coords_a, coords_b], axis=0)  # [L, 4, 3]
    mask_concat = np.concatenate([mask_a, mask_b], axis=0)  # [L, 4]

    # Flatten to atom level
    atom_coords = coords_concat.reshape(Natom, 3)
    atom_mask = mask_concat.reshape(Natom)

    # Build atom_to_res mapping
    # Each residue has 4 atoms, so atom i belongs to residue i // 4
    atom_to_res = np.repeat(np.arange(L), NUM_ATOM_TYPES).astype(np.int64)

    # Build atom_type
    # Pattern repeats: [0, 1, 2, 3, 0, 1, 2, 3, ...]
    atom_type = np.tile(np.arange(NUM_ATOM_TYPES), L).astype(np.int64)

    # Build chain_id per atom
    chain_id_res = np.concatenate([
        np.zeros(LA, dtype=np.int64),
        np.ones(LB, dtype=np.int64),
    ])
    chain_id_atom = np.repeat(chain_id_res, NUM_ATOM_TYPES)

    return atom_coords, atom_mask, atom_to_res, atom_type, chain_id_atom


def build_bonds(
    LA: int,
    LB: int,
    atom_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build covalent bond graph for backbone atoms.

    Bonds:
    - Within-residue: N-CA, CA-C, C-O (type 0)
    - Peptide: C(i)-N(i+1) within each chain (type 1)

    Args:
        LA: Number of residues in chain A
        LB: Number of residues in chain B
        atom_mask: [Natom] boolean mask for valid atoms

    Returns:
        bonds_src: [E] source atom indices
        bonds_dst: [E] destination atom indices
        bond_type: [E] bond types (0=backbone, 1=peptide)
    """
    L = LA + LB
    bonds_src = []
    bonds_dst = []
    bond_types = []

    def atom_idx(res_idx: int, atom_type: int) -> int:
        """Get global atom index from residue and atom type."""
        return res_idx * NUM_ATOM_TYPES + atom_type

    # Within-residue backbone bonds for all residues
    # Each bond type is now specific (N-CA=0, CA-C=1, C-O=2)
    for res_idx in range(L):
        for src_type, dst_type, btype in BACKBONE_BONDS:
            src = atom_idx(res_idx, src_type)
            dst = atom_idx(res_idx, dst_type)

            # Only add if both atoms are present
            if atom_mask[src] and atom_mask[dst]:
                # Add both directions for undirected graph
                bonds_src.extend([src, dst])
                bonds_dst.extend([dst, src])
                bond_types.extend([btype, btype])

    # Peptide bonds within chain A
    for res_idx in range(LA - 1):
        # C of residue i to N of residue i+1
        src = atom_idx(res_idx, 2)  # C
        dst = atom_idx(res_idx + 1, 0)  # N

        if atom_mask[src] and atom_mask[dst]:
            bonds_src.extend([src, dst])
            bonds_dst.extend([dst, src])
            bond_types.extend([BOND_TYPE_PEPTIDE, BOND_TYPE_PEPTIDE])

    # Peptide bonds within chain B
    for res_idx in range(LA, L - 1):
        src = atom_idx(res_idx, 2)  # C
        dst = atom_idx(res_idx + 1, 0)  # N

        if atom_mask[src] and atom_mask[dst]:
            bonds_src.extend([src, dst])
            bonds_dst.extend([dst, src])
            bond_types.extend([BOND_TYPE_PEPTIDE, BOND_TYPE_PEPTIDE])

    return (
        np.array(bonds_src, dtype=np.int64),
        np.array(bonds_dst, dtype=np.int64),
        np.array(bond_types, dtype=np.int64),
    )


def compute_bond_lengths(
    atom_coords: np.ndarray,
    bonds_src: np.ndarray,
    bonds_dst: np.ndarray,
    atom_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute bond lengths for validation.

    Args:
        atom_coords: [Natom, 3] coordinates
        bonds_src: [E] source indices
        bonds_dst: [E] destination indices
        atom_mask: [Natom] valid atom mask

    Returns:
        bond_lengths: [E] distances in Angstroms
    """
    src_coords = atom_coords[bonds_src]
    dst_coords = atom_coords[bonds_dst]
    diff = src_coords - dst_coords
    lengths = np.linalg.norm(diff, axis=1)
    return lengths
