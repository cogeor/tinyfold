"""DIPS-Plus dataset download and manifest creation.

DIPS-Plus is a curated dataset of ~42K binary protein complexes from PDB.
Source: https://zenodo.org/records/8140981
"""

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

# Zenodo record for DIPS-Plus
ZENODO_RECORD_ID = "8140981"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def get_zenodo_files() -> list[dict]:
    """Fetch file metadata from Zenodo record."""
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    record = response.json()
    return record.get("files", [])


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract tar.gz or zip archive."""
    if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def download_dips_plus(output_dir: str | Path, skip_existing: bool = True) -> Path:
    """
    Download DIPS-Plus dataset from Zenodo.

    Args:
        output_dir: Directory to store downloaded files
        skip_existing: Skip download if files already exist

    Returns:
        Path to the extracted dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Get file list from Zenodo
    print("Fetching DIPS-Plus file list from Zenodo...")
    files = get_zenodo_files()

    # Download each file
    for file_info in files:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]
        file_size = file_info["size"]

        dest_path = raw_dir / filename

        if skip_existing and dest_path.exists():
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename} ({file_size / 1e9:.2f} GB)...")
        download_file(download_url, dest_path)

    # Extract archives
    extracted_dir = raw_dir / "dips_plus"
    if not extracted_dir.exists():
        for archive in raw_dir.glob("*.tar.gz"):
            print(f"Extracting {archive.name}...")
            extract_archive(archive, raw_dir)
        for archive in raw_dir.glob("*.zip"):
            print(f"Extracting {archive.name}...")
            extract_archive(archive, raw_dir)

    return raw_dir


def find_structure_files(data_dir: Path) -> Iterator[Path]:
    """Find all PDB/mmCIF structure files in directory."""
    for ext in ["*.pdb", "*.cif", "*.pdb.gz", "*.cif.gz", "*.ent", "*.ent.gz"]:
        yield from data_dir.rglob(ext)


def find_dips_dill_files(data_dir: Path) -> Iterator[Path]:
    """Find all DIPS-Plus dill files in directory."""
    yield from data_dir.rglob("*.dill")


def parse_dips_filename(filepath: Path) -> dict | None:
    """
    Parse DIPS-style filename to extract PDB ID and chain info.

    DIPS filenames are typically: {pdb_id}_{chain1}_{chain2}.pdb
    or similar patterns.

    Returns:
        dict with pdb_id, chain_a, chain_b or None if can't parse
    """
    stem = filepath.stem
    if stem.endswith(".pdb"):
        stem = stem[:-4]

    parts = stem.split("_")

    if len(parts) >= 3:
        pdb_id = parts[0].lower()
        chain_a = parts[1]
        chain_b = parts[2]
        return {
            "pdb_id": pdb_id,
            "chain_a": chain_a,
            "chain_b": chain_b,
        }

    # Try other patterns
    if len(parts) == 2 and len(parts[0]) == 4:
        # Might be pdb_chains format
        pdb_id = parts[0].lower()
        chains = parts[1]
        if len(chains) >= 2:
            return {
                "pdb_id": pdb_id,
                "chain_a": chains[0],
                "chain_b": chains[1],
            }

    return None


def parse_dips_dill_filename(filepath: Path) -> dict | None:
    """
    Parse DIPS dill filename to extract metadata.

    Filenames are like: 10gs.pdb1_0.dill
    - 10gs: PDB ID
    - pdb1: model number
    - 0: pair index

    For DIPS-Plus, we use filename as the sample ID since chain info
    is embedded in the df0/df1 DataFrames, not the filename.

    Returns:
        dict with pdb_id, sample_id, or None if can't parse
    """
    stem = filepath.stem  # e.g., "10gs.pdb1_0"

    # Extract PDB ID (first 4 chars)
    parts = stem.split(".")
    if len(parts) >= 1:
        pdb_id = parts[0].lower()
        if len(pdb_id) == 4:
            return {
                "pdb_id": pdb_id,
                "sample_id": stem,  # Use full stem as unique ID
            }

    return None


def create_manifest_from_dill(
    data_dir: str | Path,
    output_path: str | Path,
) -> int:
    """
    Create manifest.jsonl from DIPS-Plus dill files.

    Args:
        data_dir: Directory containing .dill files
        output_path: Path to write manifest.jsonl

    Returns:
        Number of samples in manifest
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    manifest_entries = []
    skipped = 0

    print("Scanning for DIPS dill files...")
    dill_files = list(find_dips_dill_files(data_dir))
    print(f"Found {len(dill_files)} dill files")

    for filepath in tqdm(dill_files, desc="Building manifest"):
        parsed = parse_dips_dill_filename(filepath)

        if parsed is None:
            skipped += 1
            continue

        entry = {
            "sample_id": parsed["sample_id"],
            "pdb_id": parsed["pdb_id"],
            "path": str(filepath.relative_to(data_dir)),
            "source": "dips_plus",
            "format": "dill",
        }
        manifest_entries.append(entry)

    # Sort by sample_id for reproducibility
    manifest_entries.sort(key=lambda x: x["sample_id"])

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Created manifest with {len(manifest_entries)} samples ({skipped} skipped)")
    return len(manifest_entries)


def create_manifest(
    data_dir: str | Path,
    output_path: str | Path,
) -> int:
    """
    Create manifest.jsonl indexing all structure files.

    Args:
        data_dir: Directory containing structure files
        output_path: Path to write manifest.jsonl

    Returns:
        Number of samples in manifest
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    manifest_entries = []
    skipped = 0

    print("Scanning for structure files...")
    structure_files = list(find_structure_files(data_dir))
    print(f"Found {len(structure_files)} structure files")

    for filepath in tqdm(structure_files, desc="Building manifest"):
        parsed = parse_dips_filename(filepath)

        if parsed is None:
            skipped += 1
            continue

        sample_id = f"{parsed['pdb_id']}_{parsed['chain_a']}_{parsed['chain_b']}"

        entry = {
            "sample_id": sample_id,
            "pdb_id": parsed["pdb_id"],
            "chain_a": parsed["chain_a"],
            "chain_b": parsed["chain_b"],
            "path": str(filepath.relative_to(data_dir)),
            "source": "dips_plus",
        }
        manifest_entries.append(entry)

    # Sort by sample_id for reproducibility
    manifest_entries.sort(key=lambda x: x["sample_id"])

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Created manifest with {len(manifest_entries)} samples ({skipped} skipped)")
    return len(manifest_entries)


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load manifest from jsonl file."""
    manifest_path = Path(manifest_path)
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries
