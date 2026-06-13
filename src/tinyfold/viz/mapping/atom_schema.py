"""Atom schema for structure interpretation."""

from dataclasses import dataclass, field


@dataclass
class AtomSchema:
    """Defines how to interpret atom types and write structure files.

    Default configuration is for backbone-only (N, CA, C, O).
    """

    # Atom type index -> atom name
    atom_type_to_name: dict[int, str] = field(
        default_factory=lambda: {0: "N", 1: "CA", 2: "C", 3: "O"}
    )

    # Atom type index -> element symbol
    atom_type_to_element: dict[int, str] = field(
        default_factory=lambda: {0: "N", 1: "C", 2: "C", 3: "O"}
    )

    # Number of atoms per residue
    atoms_per_residue: int = 4

    # Chain labels for output
    chain_labels: tuple[str, str] = ("A", "B")

    def get_atom_name(self, atom_type: int) -> str:
        """Get atom name for PDB output."""
        return self.atom_type_to_name.get(atom_type, "X")

    def get_element(self, atom_type: int) -> str:
        """Get element symbol for PDB output."""
        return self.atom_type_to_element.get(atom_type, "X")

    def get_chain_label(self, chain_id: int) -> str:
        """Get chain label (A or B) from chain ID."""
        return self.chain_labels[chain_id] if chain_id < len(self.chain_labels) else "X"


# Standard amino acid 3-letter codes.
# MUST match the data pipeline's encoding: constants.AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
# (AA_TO_IDX is 1-letter alphabetical). The previous table was 3-letter
# alphabetical, which mislabeled residue names in every PDB written from seq
# indices (e.g. idx 5 -> "GLN" instead of the correct "GLY").
AA_INDEX_TO_THREE = {
    0: "ALA", 1: "CYS", 2: "ASP", 3: "GLU", 4: "PHE",
    5: "GLY", 6: "HIS", 7: "ILE", 8: "LYS", 9: "LEU",
    10: "MET", 11: "ASN", 12: "PRO", 13: "GLN", 14: "ARG",
    15: "SER", 16: "THR", 17: "VAL", 18: "TRP", 19: "TYR",
    20: "UNK",  # Unknown
}


def seq_index_to_three(seq_idx: int) -> str:
    """Convert sequence index to 3-letter amino acid code."""
    return AA_INDEX_TO_THREE.get(seq_idx, "UNK")
