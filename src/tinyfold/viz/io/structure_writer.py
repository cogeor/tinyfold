"""PDB structure writing utilities."""

from pathlib import Path

import numpy as np

from tinyfold.viz.mapping.atom_schema import AtomSchema, seq_index_to_three


def coords_to_pdb_string(
    xyz: np.ndarray,
    atom_to_res: np.ndarray,
    atom_type: np.ndarray,
    chain_id_res: np.ndarray,
    res_idx: np.ndarray,
    seq: np.ndarray | None = None,
    atom_mask: np.ndarray | None = None,
    schema: AtomSchema | None = None,
) -> str:
    """Convert coordinates to PDB format string.

    Args:
        xyz: [N_atom, 3] atom coordinates
        atom_to_res: [N_atom] residue index per atom
        atom_type: [N_atom] atom type index
        chain_id_res: [L] chain ID per residue (0 or 1)
        res_idx: [L] within-chain residue index
        seq: [L] amino acid indices (optional)
        atom_mask: [N_atom] boolean mask for valid atoms (optional)
        schema: AtomSchema for naming (optional, uses default)

    Returns:
        PDB format string
    """
    if schema is None:
        schema = AtomSchema()

    if atom_mask is None:
        atom_mask = np.ones(len(xyz), dtype=bool)

    lines = []
    atom_serial = 1

    for i in range(len(xyz)):
        if not atom_mask[i]:
            continue

        x, y, z = xyz[i]
        res = atom_to_res[i]
        atype = atom_type[i]
        chain = chain_id_res[res]
        resnum = res_idx[res] + 1  # PDB uses 1-based

        atom_name = schema.get_atom_name(atype)
        element = schema.get_element(atype)
        chain_label = schema.get_chain_label(chain)

        if seq is not None:
            resname = seq_index_to_three(int(seq[res]))
        else:
            resname = "UNK"

        # Format atom name (left-justified for 1-2 char, right for 3-4)
        if len(atom_name) < 4:
            atom_name_fmt = f" {atom_name:<3}"
        else:
            atom_name_fmt = f"{atom_name:<4}"

        # PDB ATOM record format (columns 1-indexed):
        # 1-6: ATOM, 7-11: serial, 12: blank, 13-16: atom name
        # 17: altLoc, 18-20: resName, 21: blank, 22: chainID
        # 23-26: resSeq, 27: iCode, 28-30: blank, 31-54: xyz coords
        line = (
            f"ATOM  {atom_serial:5d} {atom_name_fmt} {resname:>3} "
            f"{chain_label}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2}"
        )
        lines.append(line)
        atom_serial += 1

    # Add TER records at chain breaks
    # Find where chain changes
    output_lines = []
    prev_chain = None
    for line in lines:
        chain = line[21]  # Chain ID position in PDB format
        if prev_chain is not None and chain != prev_chain:
            output_lines.append("TER")
        output_lines.append(line)
        prev_chain = chain

    output_lines.append("TER")
    output_lines.append("END")

    return "\n".join(output_lines)


def write_pdb(
    path: str | Path,
    xyz: np.ndarray,
    atom_to_res: np.ndarray,
    atom_type: np.ndarray,
    chain_id_res: np.ndarray,
    res_idx: np.ndarray,
    seq: np.ndarray | None = None,
    atom_mask: np.ndarray | None = None,
    schema: AtomSchema | None = None,
) -> None:
    """Write coordinates to PDB file.

    Args:
        path: Output file path
        xyz: [N_atom, 3] atom coordinates
        atom_to_res: [N_atom] residue index per atom
        atom_type: [N_atom] atom type index
        chain_id_res: [L] chain ID per residue
        res_idx: [L] within-chain residue index
        seq: [L] amino acid indices (optional)
        atom_mask: [N_atom] boolean mask (optional)
        schema: AtomSchema (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdb_string = coords_to_pdb_string(
        xyz=xyz,
        atom_to_res=atom_to_res,
        atom_type=atom_type,
        chain_id_res=chain_id_res,
        res_idx=res_idx,
        seq=seq,
        atom_mask=atom_mask,
        schema=schema,
    )

    with open(path, "w") as f:
        f.write(pdb_string)
