"""DIPS-Plus dill file loader.

DIPS-Plus stores pre-processed structures as pickle files (.dill) containing
atom3.pair.Pair objects with two DataFrames (df0, df1) for each chain.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
        if aa1 == "X" and resname not in MODIFIED_AA_MAP and resname not in AA3_TO_AA1:
            # Check if it's a known amino acid or modified residue
            # Skip if it's clearly not an amino acid
            if len(resname) != 3 or resname in ["HOH", "WAT", "SOL"]:
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


