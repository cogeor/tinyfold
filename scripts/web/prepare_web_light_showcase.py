#!/usr/bin/env python
"""Build web-light showcase JSON from assets predictions + parquet ground truth."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices


PREDICTIONS_PATH = Path("assets/showcase_predictions.json")
DATA_PATH = Path("data/processed/samples.parquet")
OUTPUT_PATH = Path("assets/showcase_samples.json")

ATOM_NAMES = ["N", "CA", "C", "O"]
AA_3LETTER = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR", "UNK",
]


def coords_to_pdb_string(
    xyz: np.ndarray,
    atom_to_res: np.ndarray,
    atom_types: np.ndarray,
    chain_res: np.ndarray,
    res_idx: np.ndarray,
    seq_res: np.ndarray,
) -> str:
    lines: list[str] = []
    atom_serial = 1
    prev_chain = None

    for i in range(len(xyz)):
        x, y, z = xyz[i]
        res = int(atom_to_res[i])
        atype = int(atom_types[i])
        chain = int(chain_res[res])
        resnum = int(res_idx[res]) + 1

        atom_name = ATOM_NAMES[atype] if 0 <= atype < 4 else "X"
        element = atom_name[0]
        chain_label = "A" if chain == 0 else "B"
        aa_idx = int(seq_res[res])
        resname = AA_3LETTER[aa_idx] if 0 <= aa_idx < len(AA_3LETTER) else "UNK"

        if prev_chain is not None and chain != prev_chain:
            lines.append("TER")
        prev_chain = chain

        atom_name_fmt = f" {atom_name:<3}" if len(atom_name) < 4 else f"{atom_name:<4}"
        line = (
            f"ATOM  {atom_serial:5d} {atom_name_fmt} {resname:>3} "
            f"{chain_label}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2}"
        )
        lines.append(line)
        atom_serial += 1

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload at {path}")
    return payload


def main() -> None:
    predictions = _load_json(PREDICTIONS_PATH)
    table = pq.read_table(DATA_PATH)

    split_cfg = DataSplitConfig(
        n_train=5000,
        n_test=1000,
        min_atoms=100,
        max_atoms=1600,
        seed=42,
    )
    train_indices, test_indices = get_train_test_indices(table, split_cfg)
    split_by_idx = {idx: "train" for idx in train_indices}
    split_by_idx.update({idx: "test" for idx in test_indices})

    id_to_idx = {
        table["sample_id"][i].as_py(): i
        for i in range(table.num_rows)
    }

    showcase_samples = []
    skipped = 0
    for sample_id, pred_entry in predictions.items():
        idx = id_to_idx.get(sample_id)
        if idx is None:
            skipped += 1
            continue

        gt_coords = np.array(table["atom_coords"][idx].as_py(), dtype=np.float32).reshape(-1, 3)
        pred_coords = np.array(pred_entry["coords"], dtype=np.float32).reshape(-1, 3)

        atom_types = np.array(table["atom_type"][idx].as_py(), dtype=np.int64)
        atom_to_res = np.array(table["atom_to_res"][idx].as_py(), dtype=np.int64)
        seq_res = np.array(table["seq"][idx].as_py(), dtype=np.int64)
        chain_res = np.array(table["chain_id_res"][idx].as_py(), dtype=np.int64)
        res_idx = np.array(table["res_idx"][idx].as_py(), dtype=np.int64)

        n_atoms = min(len(gt_coords), len(pred_coords), len(atom_types), len(atom_to_res))
        gt_pdb = coords_to_pdb_string(
            gt_coords[:n_atoms],
            atom_to_res[:n_atoms],
            atom_types[:n_atoms],
            chain_res,
            res_idx,
            seq_res,
        )
        pred_pdb = coords_to_pdb_string(
            pred_coords[:n_atoms],
            atom_to_res[:n_atoms],
            atom_types[:n_atoms],
            chain_res,
            res_idx,
            seq_res,
        )

        showcase_samples.append(
            {
                "sample_id": sample_id,
                "split": split_by_idx.get(idx, "unknown"),
                "n_atoms": int(n_atoms),
                "n_residues": int(len(seq_res)),
                "rmsd": float(pred_entry.get("rmsd", 0.0)),
                "inference_time": float(pred_entry.get("inference_time", 0.0)),
                "ground_truth_pdb": gt_pdb,
                "prediction_pdb": pred_pdb,
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_predictions": str(PREDICTIONS_PATH),
        "source_ground_truths": str(DATA_PATH),
        "samples": showcase_samples,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Wrote {len(showcase_samples)} showcase samples -> {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} prediction ids missing in parquet table")


if __name__ == "__main__":
    main()
