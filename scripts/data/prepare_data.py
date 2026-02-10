#!/usr/bin/env python3
"""
Unified data preparation script for TinyFold.

Downloads DIPS-Plus dataset, processes structures, and caches to Parquet.

Usage:
    # Full pipeline
    python scripts/data/prepare_data.py --output-dir data/processed

    # Individual steps
    python scripts/data/prepare_data.py --output-dir data/processed --only download
    python scripts/data/prepare_data.py --output-dir data/processed --only preprocess
    python scripts/data/prepare_data.py --output-dir data/processed --only split

    # Use local PDB files instead of downloading
    python scripts/data/prepare_data.py --output-dir data/processed --input-dir path/to/structures
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tinyfold.constants import NUM_ATOM_TYPES
from tinyfold.data.cache import (
    generate_splits,
    sample_to_dict,
    write_errors,
    write_parquet,
    write_split_files,
    write_stats,
)
from tinyfold.data.parsing.dips_loader import (
    get_chains_from_dips_pair,
    load_dips_pair,
)
from tinyfold.data.parsing.structure_io import (
    extract_chain,
    get_backbone_atoms,
    load_structure,
)
from tinyfold.data.processing.atomization import atomize_chains, build_bonds
from tinyfold.data.processing.cleaning import clean_chain
from tinyfold.data.processing.filters import FilterReason, validate_sample
from tinyfold.data.processing.interface import compute_interface_mask
from tinyfold.data.sources.dips_plus import (
    create_manifest,
    create_manifest_from_dill,
    download_dips_plus,
    find_dips_dill_files,
    load_manifest,
)


def process_single_sample(args: tuple) -> dict[str, Any] | None:
    """
    Process a single sample from manifest entry.

    Args:
        args: Tuple of (manifest_entry, data_dir)

    Returns:
        Processed sample dict or error dict
    """
    entry, data_dir = args
    sample_id = entry["sample_id"]

    try:
        # Load structure
        structure_path = Path(data_dir) / entry["path"]
        if not structure_path.exists():
            return {"sample_id": sample_id, "error": "file_not_found", "path": str(structure_path)}

        # Check if this is a dill file (DIPS-Plus format)
        is_dill = entry.get("format") == "dill" or structure_path.suffix == ".dill"

        if is_dill:
            # Load DIPS-Plus dill file
            pair = load_dips_pair(structure_path)
            chain_data_a, chain_data_b = get_chains_from_dips_pair(pair)
        else:
            # Load PDB/mmCIF file
            structure = load_structure(structure_path)

            # Extract chains
            chain_a = extract_chain(structure, entry["chain_a"])
            chain_b = extract_chain(structure, entry["chain_b"])

            if chain_a is None:
                return {"sample_id": sample_id, "error": "chain_a_not_found", "chain": entry["chain_a"]}
            if chain_b is None:
                return {"sample_id": sample_id, "error": "chain_b_not_found", "chain": entry["chain_b"]}

            # Get backbone atoms
            chain_data_a = get_backbone_atoms(chain_a)
            chain_data_b = get_backbone_atoms(chain_b)

        # Clean chains
        seq_a, seq_idx_a, coords_a, mask_a = clean_chain(
            chain_data_a.sequence,
            chain_data_a.seq_indices,
            chain_data_a.coords,
            chain_data_a.mask,
            chain_data_a.residue_names,
        )
        seq_b, seq_idx_b, coords_b, mask_b = clean_chain(
            chain_data_b.sequence,
            chain_data_b.seq_indices,
            chain_data_b.coords,
            chain_data_b.mask,
            chain_data_b.residue_names,
        )

        LA = len(seq_a)
        LB = len(seq_b)

        if LA == 0 or LB == 0:
            return {"sample_id": sample_id, "error": "empty_chain", "LA": LA, "LB": LB}

        # Atomize
        atom_coords, atom_mask, atom_to_res, atom_type, chain_id_atom = atomize_chains(
            coords_a, mask_a, coords_b, mask_b
        )

        # Build bonds
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        # Validate
        result = validate_sample(
            LA=LA,
            LB=LB,
            coords_a=coords_a,
            mask_a=mask_a,
            coords_b=coords_b,
            mask_b=mask_b,
            atom_coords=atom_coords,
            atom_mask=atom_mask,
            bonds_src=bonds_src,
            bonds_dst=bonds_dst,
            bond_type=bond_type,
        )

        if not result.passed:
            return {
                "sample_id": sample_id,
                "error": result.reason.value,
                "details": result.details,
            }

        # Compute interface
        iface_a, iface_b = compute_interface_mask(coords_a, mask_a, coords_b, mask_b)
        iface_mask = np.concatenate([iface_a, iface_b])

        # Build concatenated sequence tensors
        seq = np.concatenate([seq_idx_a, seq_idx_b])
        chain_id_res = np.concatenate([
            np.zeros(LA, dtype=np.int64),
            np.ones(LB, dtype=np.int64),
        ])
        res_idx = np.concatenate([
            np.arange(LA, dtype=np.int64),
            np.arange(LB, dtype=np.int64),
        ])

        # Create sample dict
        sample = sample_to_dict(
            sample_id=sample_id,
            pdb_id=entry["pdb_id"],
            seq=seq,
            chain_id_res=chain_id_res,
            res_idx=res_idx,
            atom_coords=atom_coords,
            atom_mask=atom_mask,
            atom_to_res=atom_to_res,
            atom_type=atom_type,
            bonds_src=bonds_src,
            bonds_dst=bonds_dst,
            bond_type=bond_type,
            iface_mask=iface_mask,
            LA=LA,
            LB=LB,
        )
        sample["_success"] = True
        return sample

    except Exception as e:
        return {"sample_id": sample_id, "error": "exception", "message": str(e)}


def run_download(output_dir: Path, skip_existing: bool = True) -> Path:
    """Download DIPS-Plus dataset."""
    print("=" * 60)
    print("Step 1: Download DIPS-Plus from Zenodo")
    print("=" * 60)

    raw_dir = download_dips_plus(output_dir, skip_existing=skip_existing)
    return raw_dir


def run_manifest(data_dir: Path, output_dir: Path) -> Path:
    """Create manifest from structure files."""
    print("=" * 60)
    print("Step 2: Create manifest")
    print("=" * 60)

    manifest_path = output_dir / "manifest.jsonl"

    # Check if we have dill files (DIPS-Plus format)
    dill_files = list(find_dips_dill_files(data_dir))
    if dill_files:
        print(f"Found {len(dill_files)} DIPS dill files, using dill manifest creator")
        create_manifest_from_dill(data_dir, manifest_path)
    else:
        print("No dill files found, using PDB/mmCIF manifest creator")
        create_manifest(data_dir, manifest_path)

    return manifest_path


def run_preprocess(
    manifest_path: Path,
    data_dir: Path,
    output_dir: Path,
    num_workers: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Preprocess all samples from manifest."""
    print("=" * 60)
    print("Step 3: Preprocess structures")
    print("=" * 60)

    manifest = load_manifest(manifest_path)
    print(f"Loaded {len(manifest)} samples from manifest")

    # Prepare arguments for parallel processing
    args = [(entry, data_dir) for entry in manifest]

    # Process samples
    samples = []
    errors = []

    print(f"Processing with {num_workers} workers...")
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_sample, args),
            total=len(args),
            desc="Processing",
        ))

    for result in results:
        if result is None:
            continue
        if result.get("_success"):
            del result["_success"]
            samples.append(result)
        else:
            errors.append(result)

    elapsed = time.time() - start_time
    print(f"Processed {len(manifest)} samples in {elapsed:.1f}s")
    print(f"  Success: {len(samples)}")
    print(f"  Errors: {len(errors)}")

    # Write samples to Parquet
    if samples:
        parquet_path = output_dir / "samples.parquet"
        print(f"Writing {len(samples)} samples to {parquet_path}")
        write_parquet(samples, parquet_path)

    # Write errors
    if errors:
        errors_path = output_dir / "errors.jsonl"
        print(f"Writing {len(errors)} errors to {errors_path}")
        write_errors(errors_path, errors)

    # Compute and write stats
    stats = compute_stats(samples, errors)
    stats_path = output_dir / "stats.json"
    write_stats(stats_path, stats)
    print(f"Dataset stats written to {stats_path}")

    return samples, errors


def compute_stats(samples: list[dict], errors: list[dict]) -> dict[str, Any]:
    """Compute dataset statistics."""
    if not samples:
        return {"n_samples": 0, "n_errors": len(errors)}

    lengths_a = [s["LA"] for s in samples]
    lengths_b = [s["LB"] for s in samples]
    total_lengths = [s["LA"] + s["LB"] for s in samples]

    # Count error types
    error_counts = defaultdict(int)
    for e in errors:
        error_counts[e.get("error", "unknown")] += 1

    return {
        "n_samples": len(samples),
        "n_errors": len(errors),
        "error_counts": dict(error_counts),
        "length_stats": {
            "chain_a": {
                "min": int(np.min(lengths_a)),
                "max": int(np.max(lengths_a)),
                "mean": float(np.mean(lengths_a)),
                "median": float(np.median(lengths_a)),
            },
            "chain_b": {
                "min": int(np.min(lengths_b)),
                "max": int(np.max(lengths_b)),
                "mean": float(np.mean(lengths_b)),
                "median": float(np.median(lengths_b)),
            },
            "total": {
                "min": int(np.min(total_lengths)),
                "max": int(np.max(total_lengths)),
                "mean": float(np.mean(total_lengths)),
                "median": float(np.median(total_lengths)),
            },
        },
    }


def run_split(
    output_dir: Path,
    seed: int = 42,
    split_name: str | None = None,
    fractions: dict[str, float] | None = None,
) -> None:
    """
    Generate dataset splits.

    Args:
        output_dir: Output directory containing samples.parquet
        seed: Random seed for reproducibility
        split_name: Optional name prefix for split (e.g., "split1" creates split1_train.txt)
                    If None, creates default train.txt, val.txt, test.txt
        fractions: Optional custom fractions, e.g. {"train": 0.8, "val": 0.1, "test": 0.1}
    """
    print("=" * 60)
    print("Step 4: Generate splits")
    print("=" * 60)

    parquet_path = output_dir / "samples.parquet"
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found. Run preprocess first.")
        return

    # Load sample IDs
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=["sample_id"])
    sample_ids = table["sample_id"].to_pylist()

    print(f"Splitting {len(sample_ids)} samples...")

    # Generate splits
    splits = generate_splits(sample_ids, fractions=fractions, seed=seed)

    # Add name prefix if specified
    if split_name:
        splits = {f"{split_name}_{k}": v for k, v in splits.items()}

    for name, ids in splits.items():
        print(f"  {name}: {len(ids)}")

    splits_dir = output_dir / "splits"
    write_split_files(splits_dir, splits)
    print(f"Split files written to {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyFold data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory with structure files (skips download)",
    )
    parser.add_argument(
        "--only",
        choices=["download", "manifest", "preprocess", "split"],
        default=None,
        help="Run only a specific step",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Name prefix for splits (e.g., 'split1' creates split1_train.txt, split1_val.txt, split1_test.txt)",
    )
    parser.add_argument(
        "--split-fractions",
        type=str,
        default=None,
        help="Custom split fractions as JSON, e.g., '{\"train\": 0.8, \"val\": 0.2}'",
    )
    args = parser.parse_args()

    # Parse split fractions if provided
    split_fractions = None
    if args.split_fractions:
        split_fractions = json.loads(args.split_fractions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data directory
    if args.input_dir is not None:
        data_dir = args.input_dir
        manifest_path = output_dir / "manifest.jsonl"
    else:
        data_dir = output_dir / "raw"
        manifest_path = output_dir / "manifest.jsonl"

    # Run steps
    if args.only == "download":
        run_download(output_dir)
    elif args.only == "manifest":
        run_manifest(data_dir, output_dir)
    elif args.only == "preprocess":
        if not manifest_path.exists():
            print(f"Manifest not found at {manifest_path}. Creating...")
            run_manifest(data_dir, output_dir)
        run_preprocess(manifest_path, data_dir, output_dir, num_workers=args.workers)
    elif args.only == "split":
        run_split(output_dir, seed=args.seed, split_name=args.split_name, fractions=split_fractions)
    else:
        # Run full pipeline
        if args.input_dir is None:
            run_download(output_dir)
        run_manifest(data_dir, output_dir)
        run_preprocess(manifest_path, data_dir, output_dir, num_workers=args.workers)
        run_split(output_dir, seed=args.seed, split_name=args.split_name, fractions=split_fractions)

    print("\nDone!")


if __name__ == "__main__":
    main()
