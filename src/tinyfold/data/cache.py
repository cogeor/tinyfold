"""Parquet caching for preprocessed samples."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sample_to_dict(
    sample_id: str,
    pdb_id: str,
    seq: np.ndarray,
    chain_id_res: np.ndarray,
    res_idx: np.ndarray,
    atom_coords: np.ndarray,
    atom_mask: np.ndarray,
    atom_to_res: np.ndarray,
    atom_type: np.ndarray,
    bonds_src: np.ndarray,
    bonds_dst: np.ndarray,
    bond_type: np.ndarray,
    iface_mask: np.ndarray,
    LA: int,
    LB: int,
) -> dict[str, Any]:
    """
    Convert sample arrays to a dictionary for Parquet storage.

    Arrays are stored as lists for Parquet compatibility.
    """
    return {
        "sample_id": sample_id,
        "pdb_id": pdb_id,
        "seq": seq.tolist(),
        "chain_id_res": chain_id_res.tolist(),
        "res_idx": res_idx.tolist(),
        "atom_coords": atom_coords.flatten().tolist(),
        "atom_mask": atom_mask.tolist(),
        "atom_to_res": atom_to_res.tolist(),
        "atom_type": atom_type.tolist(),
        "bonds_src": bonds_src.tolist(),
        "bonds_dst": bonds_dst.tolist(),
        "bond_type": bond_type.tolist(),
        "iface_mask": iface_mask.tolist(),
        "LA": LA,
        "LB": LB,
    }


def dict_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Parquet row back to numpy arrays.
    """
    L = row["LA"] + row["LB"]
    Natom = L * 4

    return {
        "sample_id": row["sample_id"],
        "pdb_id": row["pdb_id"],
        "seq": np.array(row["seq"], dtype=np.int64),
        "chain_id_res": np.array(row["chain_id_res"], dtype=np.int64),
        "res_idx": np.array(row["res_idx"], dtype=np.int64),
        "atom_coords": np.array(row["atom_coords"], dtype=np.float32).reshape(Natom, 3),
        "atom_mask": np.array(row["atom_mask"], dtype=bool),
        "atom_to_res": np.array(row["atom_to_res"], dtype=np.int64),
        "atom_type": np.array(row["atom_type"], dtype=np.int64),
        "bonds_src": np.array(row["bonds_src"], dtype=np.int64),
        "bonds_dst": np.array(row["bonds_dst"], dtype=np.int64),
        "bond_type": np.array(row["bond_type"], dtype=np.int64),
        "iface_mask": np.array(row["iface_mask"], dtype=bool),
        "LA": row["LA"],
        "LB": row["LB"],
    }


def write_parquet(
    samples: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Write samples to Parquet file.

    Args:
        samples: List of sample dictionaries
        output_path: Path to output Parquet file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define schema
    schema = pa.schema([
        ("sample_id", pa.string()),
        ("pdb_id", pa.string()),
        ("seq", pa.list_(pa.int64())),
        ("chain_id_res", pa.list_(pa.int64())),
        ("res_idx", pa.list_(pa.int64())),
        ("atom_coords", pa.list_(pa.float32())),
        ("atom_mask", pa.list_(pa.bool_())),
        ("atom_to_res", pa.list_(pa.int64())),
        ("atom_type", pa.list_(pa.int64())),
        ("bonds_src", pa.list_(pa.int64())),
        ("bonds_dst", pa.list_(pa.int64())),
        ("bond_type", pa.list_(pa.int64())),
        ("iface_mask", pa.list_(pa.bool_())),
        ("LA", pa.int64()),
        ("LB", pa.int64()),
    ])

    # Convert to columnar format
    columns = {field.name: [] for field in schema}
    for sample in samples:
        for key in columns:
            columns[key].append(sample[key])

    # Create table and write
    table = pa.table(columns, schema=schema)
    pq.write_table(table, output_path, compression="snappy")


def read_parquet(input_path: str | Path) -> list[dict[str, Any]]:
    """
    Read samples from Parquet file.

    Args:
        input_path: Path to Parquet file

    Returns:
        List of sample dictionaries with numpy arrays
    """
    table = pq.read_table(input_path)
    df = table.to_pandas()

    samples = []
    for _, row in df.iterrows():
        samples.append(dict_to_sample(row.to_dict()))

    return samples


def get_parquet_sample_ids(input_path: str | Path) -> list[str]:
    """
    Get list of sample IDs from Parquet file without loading all data.
    """
    table = pq.read_table(input_path, columns=["sample_id"])
    return table["sample_id"].to_pylist()


def generate_splits(
    sample_ids: list[str],
    fractions: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Generate random splits with custom names and fractions.

    Args:
        sample_ids: List of all sample IDs
        fractions: Dict mapping split name to fraction, e.g. {"train": 0.8, "val": 0.1, "test": 0.1}
                   If None, defaults to {"train": 0.8, "val": 0.1, "test": 0.1}
        seed: Random seed for reproducibility

    Returns:
        Dict mapping split name to list of sample IDs

    Example:
        # Default splits
        splits = generate_splits(sample_ids)
        # splits = {"train": [...], "val": [...], "test": [...]}

        # Custom named splits
        splits = generate_splits(sample_ids, {"split1": 0.5, "split2": 0.5})
        # splits = {"split1": [...], "split2": [...]}
    """
    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Validate fractions sum to ~1.0
    total = sum(fractions.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    ids = np.array(sample_ids)
    rng.shuffle(ids)

    n = len(ids)
    splits = {}
    start = 0

    for name, frac in fractions.items():
        count = int(n * frac)
        # Last split gets remaining samples to handle rounding
        if name == list(fractions.keys())[-1]:
            splits[name] = ids[start:].tolist()
        else:
            splits[name] = ids[start : start + count].tolist()
            start += count

    return splits


def write_split_files(
    output_dir: str | Path,
    splits: dict[str, list[str]],
) -> None:
    """
    Write split files (one sample_id per line).

    Args:
        output_dir: Directory to write split files
        splits: Dict mapping split name to list of sample IDs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in splits.items():
        with open(output_dir / f"{name}.txt", "w") as f:
            for sample_id in ids:
                f.write(sample_id + "\n")


def load_split(split_path: str | Path) -> list[str]:
    """Load sample IDs from a split file."""
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def write_stats(
    output_path: str | Path,
    stats: dict[str, Any],
) -> None:
    """Write dataset statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def write_errors(
    output_path: str | Path,
    errors: list[dict[str, Any]],
) -> None:
    """Write processing errors to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for error in errors:
            f.write(json.dumps(error) + "\n")
