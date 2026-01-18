"""Structure I/O using gemmi for mmCIF/PDB parsing."""

import gzip
from pathlib import Path
from typing import NamedTuple

import gemmi
import numpy as np

from tinyfold.constants import (
    AA3_TO_AA1,
    AA_TO_IDX,
    BACKBONE_ATOMS,
    MODIFIED_AA_MAP,
)


class ChainData(NamedTuple):
    """Parsed chain data."""

    sequence: list[str]  # 1-letter AA codes
    seq_indices: np.ndarray  # [L] int, AA indices (0-20)
    coords: np.ndarray  # [L, 4, 3] backbone atom coords
    mask: np.ndarray  # [L, 4] bool, True if atom present
    residue_names: list[str]  # 3-letter residue names


def load_structure(path: str | Path) -> gemmi.Structure:
    """
    Load structure from PDB or mmCIF file.

    Handles gzipped files automatically.
    """
    path = Path(path)

    # Handle gzipped files
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            content = f.read()
        # Determine format from inner extension
        inner_name = path.stem
        if inner_name.endswith(".cif"):
            doc = gemmi.cif.read_string(content)
            return gemmi.make_structure_from_block(doc.sole_block())
        else:
            return gemmi.read_pdb_string(content)

    # Uncompressed files
    if path.suffix == ".cif":
        doc = gemmi.cif.read(str(path))
        return gemmi.make_structure_from_block(doc.sole_block())
    else:
        return gemmi.read_pdb(str(path))


def extract_chain(structure: gemmi.Structure, chain_id: str, model_idx: int = 0) -> gemmi.Chain | None:
    """
    Extract a single chain from structure.

    Args:
        structure: Loaded gemmi Structure
        chain_id: Chain identifier (e.g., "A", "B")
        model_idx: Model index to use (default 0, i.e., first model)

    Returns:
        Chain object or None if not found
    """
    if model_idx >= len(structure):
        return None

    model = structure[model_idx]

    for chain in model:
        if chain.name == chain_id:
            return chain

    return None


def resolve_altloc(residue: gemmi.Residue, atom_name: str) -> gemmi.Atom | None:
    """
    Get atom from residue, resolving alternate locations.

    Takes highest occupancy, tiebreaks by altloc ID order.
    """
    candidates = []
    for atom in residue:
        if atom.name == atom_name:
            candidates.append(atom)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Sort by occupancy (descending), then altloc (ascending)
    candidates.sort(key=lambda a: (-a.occ, a.altloc))
    return candidates[0]


def map_residue_to_aa(residue_name: str) -> str:
    """
    Map residue name to 1-letter AA code.

    Handles standard residues, modified residues, and unknowns.
    """
    residue_name = residue_name.upper()

    # Standard amino acid
    if residue_name in AA3_TO_AA1:
        return AA3_TO_AA1[residue_name]

    # Modified residue
    if residue_name in MODIFIED_AA_MAP:
        return MODIFIED_AA_MAP[residue_name]

    # Unknown
    return "X"


def get_backbone_atoms(chain: gemmi.Chain) -> ChainData:
    """
    Extract backbone atoms from chain.

    Returns coordinates for N, CA, C, O atoms for each residue.
    Missing atoms are marked in the mask.

    Args:
        chain: Gemmi Chain object

    Returns:
        ChainData with sequence, coordinates, and mask
    """
    sequence = []
    seq_indices = []
    coords_list = []
    mask_list = []
    residue_names = []

    for residue in chain:
        # Skip non-amino acid residues (water, ligands, etc.)
        if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
            # Also check our modified residue list
            if residue.name not in MODIFIED_AA_MAP:
                continue

        # Get 1-letter code
        aa1 = map_residue_to_aa(residue.name)
        sequence.append(aa1)
        seq_indices.append(AA_TO_IDX.get(aa1, AA_TO_IDX["X"]))
        residue_names.append(residue.name)

        # Get backbone atom coordinates
        res_coords = np.zeros((4, 3), dtype=np.float32)
        res_mask = np.zeros(4, dtype=bool)

        for i, atom_name in enumerate(BACKBONE_ATOMS):
            atom = resolve_altloc(residue, atom_name)
            if atom is not None:
                res_coords[i] = [atom.pos.x, atom.pos.y, atom.pos.z]
                res_mask[i] = True

        coords_list.append(res_coords)
        mask_list.append(res_mask)

    if len(sequence) == 0:
        # Return empty arrays with correct shapes
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


def write_pdb(
    coords: np.ndarray,
    sequence: list[str],
    chain_ids: np.ndarray,
    output_path: str | Path,
    atom_mask: np.ndarray | None = None,
) -> None:
    """
    Write coordinates to PDB file.

    Args:
        coords: [Natom, 3] backbone atom coordinates
        sequence: List of 1-letter AA codes
        chain_ids: [L] chain IDs (0 or 1)
        output_path: Path to write PDB file
        atom_mask: [Natom] optional mask for valid atoms
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chain_names = ["A", "B"]
    atom_idx = 0

    with open(output_path, "w") as f:
        global_atom_num = 1
        for res_idx, aa in enumerate(sequence):
            chain_id = chain_ids[res_idx]
            chain_name = chain_names[chain_id]

            for atom_i, atom_name in enumerate(BACKBONE_ATOMS):
                if atom_mask is not None and not atom_mask[atom_idx]:
                    atom_idx += 1
                    continue

                x, y, z = coords[atom_idx]
                # PDB ATOM record format
                line = (
                    f"ATOM  {global_atom_num:5d}  {atom_name:<3s} "
                    f"{'ALA':3s} {chain_name}{res_idx + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                    f"{atom_name[0]:>2s}\n"
                )
                f.write(line)
                global_atom_num += 1
                atom_idx += 1

        f.write("END\n")
