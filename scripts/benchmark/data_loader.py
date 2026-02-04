"""Standardized data loading for benchmark evaluation.

Provides unified sample format that works with all model types.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor


@dataclass
class BenchmarkSample:
    """Standardized sample format for all models.

    All coordinates are in Angstroms (not normalized).
    """

    sample_id: str

    # Ground truth coordinates (Angstroms)
    gt_atoms: np.ndarray  # [L, 4, 3] backbone atoms (N, CA, C, O)
    gt_ca: np.ndarray  # [L, 3] CA coordinates

    # Sequence features
    aa_seq: np.ndarray  # [L] amino acid indices (0-20)
    chain_ids: np.ndarray  # [L] chain IDs (0 or 1)
    res_idx: np.ndarray  # [L] residue indices

    # For atom-level models (af3_style)
    atom_types: np.ndarray  # [L*4] atom type indices (0-3)
    atom_to_res: np.ndarray  # [L*4] mapping to residue index

    # Normalization info (for models that need normalized input)
    centroid: np.ndarray  # [3] center of mass
    std: float  # coordinate std for normalization

    # Metadata
    n_residues: int
    n_atoms: int

    def to_device(self, device: torch.device) -> "BenchmarkSampleTensors":
        """Convert to device tensors for model inference."""
        return BenchmarkSampleTensors(
            sample_id=self.sample_id,
            gt_atoms=torch.tensor(self.gt_atoms, dtype=torch.float32, device=device),
            gt_ca=torch.tensor(self.gt_ca, dtype=torch.float32, device=device),
            aa_seq=torch.tensor(self.aa_seq, dtype=torch.long, device=device),
            chain_ids=torch.tensor(self.chain_ids, dtype=torch.long, device=device),
            res_idx=torch.tensor(self.res_idx, dtype=torch.long, device=device),
            atom_types=torch.tensor(self.atom_types, dtype=torch.long, device=device),
            atom_to_res=torch.tensor(self.atom_to_res, dtype=torch.long, device=device),
            centroid=self.centroid,
            std=self.std,
            n_residues=self.n_residues,
            n_atoms=self.n_atoms,
        )


@dataclass
class BenchmarkSampleTensors:
    """Tensor version of BenchmarkSample for model inference."""

    sample_id: str

    gt_atoms: Tensor  # [L, 4, 3]
    gt_ca: Tensor  # [L, 3]

    aa_seq: Tensor  # [L]
    chain_ids: Tensor  # [L]
    res_idx: Tensor  # [L]

    atom_types: Tensor  # [L*4]
    atom_to_res: Tensor  # [L*4]

    centroid: np.ndarray  # [3]
    std: float

    n_residues: int
    n_atoms: int

    @property
    def device(self) -> torch.device:
        return self.gt_atoms.device

    def get_normalized_atoms(self) -> Tensor:
        """Get normalized atom coordinates for model input."""
        centered = self.gt_atoms - torch.tensor(
            self.centroid, device=self.device, dtype=torch.float32
        )
        return centered / self.std

    def get_normalized_ca(self) -> Tensor:
        """Get normalized CA coordinates for residue-level models."""
        centered = self.gt_ca - torch.tensor(
            self.centroid, device=self.device, dtype=torch.float32
        )
        return centered / self.std

    def get_flat_atoms(self) -> Tensor:
        """Get flattened atom coordinates [L*4, 3] for atom-level models."""
        return self.gt_atoms.reshape(-1, 3)

    def get_flat_normalized_atoms(self) -> Tensor:
        """Get flattened normalized atoms [L*4, 3]."""
        return self.get_normalized_atoms().reshape(-1, 3)

    def get_atom_chain_ids(self) -> Tensor:
        """Get per-atom chain IDs [L*4]."""
        return self.chain_ids[self.atom_to_res]

    def get_atom_aa_seq(self) -> Tensor:
        """Get per-atom amino acid indices [L*4]."""
        return self.aa_seq[self.atom_to_res]


def load_benchmark_sample(table, idx: int) -> BenchmarkSample:
    """Load a sample in standardized format from parquet table.

    Args:
        table: PyArrow table from samples.parquet
        idx: Row index in table

    Returns:
        BenchmarkSample with all fields populated
    """
    # Load raw data
    coords = np.array(table["atom_coords"][idx].as_py(), dtype=np.float32)
    atom_types = np.array(table["atom_type"][idx].as_py(), dtype=np.int64)
    atom_to_res = np.array(table["atom_to_res"][idx].as_py(), dtype=np.int64)
    seq_res = np.array(table["seq"][idx].as_py(), dtype=np.int64)
    chain_res = np.array(table["chain_id_res"][idx].as_py(), dtype=np.int64)
    res_idx = np.array(table["res_idx"][idx].as_py(), dtype=np.int64)
    sample_id = table["sample_id"][idx].as_py()

    n_atoms = len(atom_types)
    n_residues = len(seq_res)
    coords = coords.reshape(n_atoms, 3)

    # Reshape to [L, 4, 3] for backbone atoms
    # Assumes atoms are ordered N, CA, C, O per residue
    gt_atoms = coords.reshape(n_residues, 4, 3)

    # Extract CA coordinates (atom_type == 1)
    gt_ca = gt_atoms[:, 1, :]  # [L, 3]

    # Compute normalization parameters from CA
    centroid = gt_ca.mean(axis=0)
    ca_centered = gt_ca - centroid
    std = float(ca_centered.std())

    return BenchmarkSample(
        sample_id=sample_id,
        gt_atoms=gt_atoms,
        gt_ca=gt_ca,
        aa_seq=seq_res,
        chain_ids=chain_res,
        res_idx=res_idx,
        atom_types=atom_types,
        atom_to_res=atom_to_res,
        centroid=centroid,
        std=std,
        n_residues=n_residues,
        n_atoms=n_atoms,
    )


def load_parquet_table(path: str):
    """Load parquet table from path."""
    return pq.read_table(path)


def get_benchmark_indices(
    table,
    split_path: Optional[str] = None,
    n_train: int = 5000,
    n_test: int = 1000,
    min_atoms: int = 100,
    max_atoms: int = 1600,
) -> tuple[list[int], list[int]]:
    """Get train/test indices for benchmarking.

    If split_path is provided, loads indices from saved split.
    Otherwise, computes indices using DataSplitConfig.

    Args:
        table: PyArrow table
        split_path: Path to split.json (optional)
        n_train: Number of training samples
        n_test: Number of test samples
        min_atoms: Minimum atoms filter
        max_atoms: Maximum atoms filter

    Returns:
        (train_indices, test_indices)
    """
    import json
    import sys
    from pathlib import Path

    # Add src to path for imports
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices

    if split_path is not None:
        # Load from saved split
        with open(split_path) as f:
            split_data = json.load(f)
        return split_data["train_indices"], split_data["test_indices"]

    # Compute fresh split
    config = DataSplitConfig(
        n_train=n_train,
        n_test=n_test,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
    )
    return get_train_test_indices(table, config)
