"""DIPS-Plus dill file loader.

DIPS-Plus stores pre-processed structures as pickle files (.dill) containing
atom3.pair.Pair objects with two DataFrames (df0, df1) for each chain.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tinyfold.atom14 import NUM_ATOM14, atom14_names
from tinyfold.constants import (
    AA3_TO_AA1,
    AA_TO_IDX,
    BACKBONE_ATOMS,
    MODIFIED_AA_MAP,
    map_residue_to_aa,
)
from tinyfold.data.parsing.structure_io import ChainData

if TYPE_CHECKING:
    import pandas as pd


def load_dips_pair(path: str | Path) -> Any:
    """
    Load a DIPS-Plus pair from dill file.

    Args:
        path: Path to .dill file

    Returns:
        Pair object with df0, df1 DataFrames
    """
    import dill

    with open(path, "rb") as f:
        return dill.load(f)


def _is_skippable_residue(resname: str, aa1: str) -> bool:
    """True for rows that are not amino acids (waters, ligands).

    Shared by the backbone and atom14 extractors so they agree on exactly which
    residues exist -- a divergence would silently misalign the atom14 cache
    against the backbone parquet.
    """
    if aa1 == "X" and resname not in MODIFIED_AA_MAP and resname not in AA3_TO_AA1:
        return len(resname) != 3 or resname in ["HOH", "WAT", "SOL"]
    return False


def extract_atom14_from_dataframe(df: "pd.DataFrame") -> ChainData:
    """Extract ALL heavy atoms in atom14 layout from a DIPS DataFrame.

    The sibling of :func:`extract_backbone_from_dataframe`, which keeps only
    N/CA/C/O. The DIPS DataFrame is atom-level and ALREADY CARRIES the sidechain
    atoms (``atom_name``: "N, CA, C, O, etc.") -- they are discarded at parse
    time, which is why ``samples.parquet`` has no sidechains and why stage 3
    needs the raw source re-downloaded (notes/2026-07-16-three-phase-plan.md
    §3.0).

    Returns:
        ChainData whose ``coords`` is ``[L, 14, 3]`` and ``mask`` is ``[L, 14]``.
        Slots 0-3 are N, CA, C, O, so ``coords[:, :4]`` is byte-identical to what
        the backbone extractor returns.
    """
    residue_groups = df.groupby("residue", sort=True)

    sequence = []
    seq_indices = []
    coords_list = []
    mask_list = []
    residue_names = []

    for _res_num, group in residue_groups:
        resname = group["resname"].iloc[0]
        aa1 = map_residue_to_aa(resname)
        if _is_skippable_residue(resname, aa1):
            continue

        sequence.append(aa1)
        seq_indices.append(AA_TO_IDX.get(aa1, AA_TO_IDX["X"]))
        residue_names.append(resname)

        res_coords = np.zeros((NUM_ATOM14, 3), dtype=np.float32)
        res_mask = np.zeros(NUM_ATOM14, dtype=bool)

        atom_rows = {row["atom_name"]: row for _, row in group.iterrows()}
        # Slot order is per residue TYPE, so the same slot means the same atom
        # for every residue of that type -- what makes the dense tensor legible.
        for i, atom_name in enumerate(atom14_names(aa1)):
            if atom_name and atom_name in atom_rows:
                row = atom_rows[atom_name]
                res_coords[i] = [row["x"], row["y"], row["z"]]
                res_mask[i] = True

        coords_list.append(res_coords)
        mask_list.append(res_mask)

    if len(sequence) == 0:
        return ChainData(
            sequence=[],
            seq_indices=np.array([], dtype=np.int64),
            coords=np.zeros((0, NUM_ATOM14, 3), dtype=np.float32),
            mask=np.zeros((0, NUM_ATOM14), dtype=bool),
            residue_names=[],
        )

    return ChainData(
        sequence=sequence,
        seq_indices=np.array(seq_indices, dtype=np.int64),
        coords=np.stack(coords_list),
        mask=np.stack(mask_list),
        residue_names=residue_names,
    )


def extract_backbone_from_dataframe(df: "pd.DataFrame") -> ChainData:
    """
    Extract backbone atoms from DIPS DataFrame.

    The DataFrame has atom-level rows with columns:
    - residue: residue sequence number
    - resname: 3-letter residue name
    - atom_name: atom name (N, CA, C, O, etc.)
    - x, y, z: coordinates

    Args:
        df: DataFrame with atom-level data

    Returns:
        ChainData with backbone atoms extracted
    """

    # Get unique residues in order
    residue_groups = df.groupby("residue", sort=True)

    sequence = []
    seq_indices = []
    coords_list = []
    mask_list = []
    residue_names = []

    for res_num, group in residue_groups:
        # Get residue name from first atom
        resname = group["resname"].iloc[0]

        # Skip non-amino acids (waters, ligands)
        aa1 = map_residue_to_aa(resname)
        if _is_skippable_residue(resname, aa1):
            continue

        sequence.append(aa1)
        seq_indices.append(AA_TO_IDX.get(aa1, AA_TO_IDX["X"]))
        residue_names.append(resname)

        # Extract backbone atom coordinates
        res_coords = np.zeros((4, 3), dtype=np.float32)
        res_mask = np.zeros(4, dtype=bool)

        # Create atom name to row mapping
        atom_rows = {row["atom_name"]: row for _, row in group.iterrows()}

        for i, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in atom_rows:
                row = atom_rows[atom_name]
                res_coords[i] = [row["x"], row["y"], row["z"]]
                res_mask[i] = True

        coords_list.append(res_coords)
        mask_list.append(res_mask)

    if len(sequence) == 0:
        return ChainData(
            sequence=[],
            seq_indices=np.array([], dtype=np.int64),
            coords=np.zeros((0, 4, 3), dtype=np.float32),
            mask=np.zeros((0, 4), dtype=bool),
            residue_names=[],
        )

    return ChainData(
        sequence=sequence,
        seq_indices=np.array(seq_indices, dtype=np.int64),
        coords=np.stack(coords_list),
        mask=np.stack(mask_list),
        residue_names=residue_names,
    )


def get_chains_from_dips_pair(pair: Any) -> tuple[ChainData, ChainData]:
    """
    Extract both chains from a DIPS Pair object.

    Args:
        pair: DIPS Pair object with df0, df1 attributes

    Returns:
        Tuple of (chain_a_data, chain_b_data)
    """
    chain_a = extract_backbone_from_dataframe(pair.df0)
    chain_b = extract_backbone_from_dataframe(pair.df1)
    return chain_a, chain_b


