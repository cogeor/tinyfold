"""PyTorch Dataset for PPI data."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tinyfold.data.cache import dict_to_sample


class PPIDataset(Dataset):
    """
    PyTorch Dataset for protein-protein interaction data.

    Loads data from Parquet files and returns tensors.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        split_file: str | Path | None = None,
        esm_cache_dir: str | Path | None = None,
    ):
        """
        Initialize dataset.

        Args:
            parquet_path: Path to Parquet file with all samples
            split_file: Optional path to split file (one sample_id per line)
                       If None, use all samples in Parquet file
            esm_cache_dir: Optional directory of per-sample ESM-2 NPZ files
                       (one file per ``sample_id``). When set, every batch
                       returned by ``__getitem__`` includes an ``esm_embed``
                       tensor of shape ``[L, esm_dim]`` (float32). Phase D
                       training path goes through
                       ``tinyfold.training.data.load_sample`` rather than this
                       dataset; the arg is plumbed here for future use and to
                       keep the two loaders consistent.
        """
        self.parquet_path = Path(parquet_path)
        self.esm_cache_dir = Path(esm_cache_dir) if esm_cache_dir is not None else None

        # Load Parquet table
        self.table = pq.read_table(self.parquet_path)
        self.df = self.table.to_pandas()

        # Build sample_id -> row index mapping
        self.id_to_idx = {
            row["sample_id"]: idx for idx, row in self.df.iterrows()
        }

        # Filter to split if provided
        if split_file is not None:
            with open(split_file) as f:
                split_ids = [line.strip() for line in f if line.strip()]
            self.sample_ids = [sid for sid in split_ids if sid in self.id_to_idx]
        else:
            self.sample_ids = list(self.id_to_idx.keys())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample by index.

        Returns dict with torch tensors.
        """
        sample_id = self.sample_ids[idx]
        row_idx = self.id_to_idx[sample_id]
        row = self.df.iloc[row_idx].to_dict()

        # Convert to numpy arrays
        sample = dict_to_sample(row)

        # Convert to torch tensors
        out = {
            "sample_id": sample["sample_id"],
            "pdb_id": sample["pdb_id"],
            "seq": torch.from_numpy(sample["seq"]),
            "chain_id_res": torch.from_numpy(sample["chain_id_res"]),
            "res_idx": torch.from_numpy(sample["res_idx"]),
            "atom_coords": torch.from_numpy(sample["atom_coords"]),
            "atom_mask": torch.from_numpy(sample["atom_mask"]),
            "atom_to_res": torch.from_numpy(sample["atom_to_res"]),
            "atom_type": torch.from_numpy(sample["atom_type"]),
            "bonds_src": torch.from_numpy(sample["bonds_src"]),
            "bonds_dst": torch.from_numpy(sample["bonds_dst"]),
            "bond_type": torch.from_numpy(sample["bond_type"]),
            "iface_mask": torch.from_numpy(sample["iface_mask"]),
            "LA": sample["LA"],
            "LB": sample["LB"],
        }
        if self.esm_cache_dir is not None:
            cache_path = self.esm_cache_dir / f"{sample_id}.npz"
            if not cache_path.exists():
                raise ValueError(
                    f"ESM cache missing for {sample_id}: {cache_path}"
                )
            with np.load(cache_path, mmap_mode="r") as npz:
                emb_np = np.asarray(npz["embeddings"])
            n_res = int(sample["LA"]) + int(sample["LB"])
            if emb_np.shape[0] != n_res:
                raise ValueError(
                    f"ESM cache shape mismatch for {sample_id}: "
                    f"got {emb_np.shape[0]} residues, expected {n_res}"
                )
            out["esm_embed"] = torch.from_numpy(emb_np).float()
        return out

    def get_sample_by_id(self, sample_id: str) -> dict[str, Any] | None:
        """Get a sample by its ID.

        Returns None if sample_id is not in the current split.
        """
        if sample_id not in self.id_to_idx:
            return None
        # Check if sample is in current split (sample_ids is filtered list)
        try:
            idx = self.sample_ids.index(sample_id)
        except ValueError:
            # sample_id exists in parquet but not in current split
            return None
        return self[idx]
