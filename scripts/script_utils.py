"""Shared utilities for TinyFold training/evaluation scripts.

This module consolidates common patterns across scripts:
- Logger: Dual output to console and file
- Data loading: load_sample_raw, collate_batch
- Checkpoint utilities

Usage:
    from script_utils import Logger, load_sample_raw, collate_batch
"""

import os
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Any, Optional
import pyarrow.parquet as pq

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Paths and Config
# =============================================================================

def get_project_root() -> str:
    """Get the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_data_path() -> str:
    """Get the default data path."""
    return os.path.join(get_project_root(), "data/processed/samples.parquet")


def save_config(args, output_dir: str) -> str:
    """Save training config to JSON at startup.
    
    Args:
        args: Argument namespace or dict
        output_dir: Output directory
        
    Returns:
        Path to saved config file
    """
    config = vars(args) if hasattr(args, '__dict__') else dict(args)
    config['_saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config_path


# =============================================================================
# Visualization
# =============================================================================

def plot_prediction(pred, target, chain_ids, sample_id, rmse, output_path):
    """Plot prediction vs ground truth for a protein structure.
    
    Args:
        pred: Predicted coordinates [N, 3]
        target: Ground truth coordinates [N, 3]  
        chain_ids: Chain IDs [N]
        sample_id: Sample identifier string
        rmse: RMSE value to display
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(12, 5))

    pred = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
    target = target.cpu().numpy() if hasattr(target, 'cpu') else target
    chain_ids = chain_ids.cpu().numpy() if hasattr(chain_ids, 'cpu') else chain_ids

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    mask_a = chain_ids == 0
    mask_b = chain_ids == 1
    if mask_a.any():
        ax1.scatter(target[mask_a, 0], target[mask_a, 1], target[mask_a, 2], c='blue', s=10, alpha=0.7)
    if mask_b.any():
        ax1.scatter(target[mask_b, 0], target[mask_b, 1], target[mask_b, 2], c='red', s=10, alpha=0.7)
    ax1.set_title(f'{sample_id}\nGround Truth')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if mask_a.any():
        ax2.scatter(pred[mask_a, 0], pred[mask_a, 1], pred[mask_a, 2], c='cyan', s=10, alpha=0.7)
    if mask_b.any():
        ax2.scatter(pred[mask_b, 0], pred[mask_b, 1], pred[mask_b, 2], c='orange', s=10, alpha=0.7)
    ax2.set_title(f'Prediction\nRMSE: {rmse:.2f} Å')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Dual output to console and file with line buffering."""

    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to log file
        """
        self.log_path = log_path
        self.file = open(log_path, 'w', buffering=1)  # Line buffered

    def log(self, msg: str = ""):
        """Log message to both console and file."""
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Data Loading - Residue Level (for ResFold)
# =============================================================================

def load_sample_residue(table, i: int, normalize: bool = True) -> Dict[str, Any]:
    """Load sample at residue level (4 atoms per residue).

    This is the standard data loading for ResFold-style models that
    work with residue centroids and atom coordinates.

    Args:
        table: PyArrow table from samples.parquet
        i: Sample index
        normalize: If True, normalize coords to unit variance. If False, keep in Angstroms.

    Returns:
        Dict with:
            - coords: [N_atoms, 3] all atom coordinates
            - coords_res: [L, 4, 3] atoms per residue (N, CA, C, O)
            - centroids: [L, 3] residue centroids
            - atom_types: [N_atoms] atom type indices
            - atom_to_res: [N_atoms] residue index for each atom
            - aa_seq: [L] amino acid sequence
            - chain_ids: [L] chain IDs
            - res_idx: [L] residue indices
            - std: normalization factor
            - n_atoms: number of atoms
            - n_res: number of residues
            - sample_id: sample identifier
    """
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center coordinates
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    # Compute std (always, for reference)
    original_std = coords.std()

    # Optionally normalize to unit variance
    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)  # No scaling, coords stay in Angstroms

    # Compute residue centroids (mean of 4 backbone atoms)
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)  # [L, 3]

    # Residue-level features
    aa_seq = seq_res  # [L]
    chain_ids = chain_res  # [L]
    res_idx = torch.arange(n_res)  # [L]

    return {
        'coords': coords,  # [N_atoms, 3]
        'coords_res': coords_res,  # [L, 4, 3]
        'centroids': centroids,  # [L, 3]
        'atom_types': atom_types,  # [N_atoms]
        'atom_to_res': atom_to_res,  # [N_atoms]
        'aa_seq': aa_seq,  # [L]
        'chain_ids': chain_ids,  # [L]
        'res_idx': res_idx,  # [L]
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': table['sample_id'][i].as_py(),
    }


# Alias for backward compatibility
load_sample_raw = load_sample_residue


def collate_batch_residue(samples: List[Dict], device: torch.device) -> Dict[str, Any]:
    """Collate residue-level samples into a padded batch.

    Args:
        samples: List of sample dicts from load_sample_residue
        device: Target device

    Returns:
        Batched dict with padded tensors
    """
    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    # Residue-level tensors
    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors (for evaluation)
    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)

    stds = []

    for i, s in enumerate(samples):
        L = s['n_res']
        N = s['n_atoms']

        centroids[i, :L] = s['centroids']
        coords_res[i, :L] = s['coords_res']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True

        coords[i, :N] = s['coords']
        atom_types[i, :N] = s['atom_types']
        atom_to_res[i, :N] = s['atom_to_res']
        mask_atom[i, :N] = True

        stds.append(s['std'])

    return {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }


# Alias for backward compatibility
collate_batch = collate_batch_residue


# =============================================================================
# Data Loading - Atom Level (for train.py / af3_style)
# =============================================================================

def load_sample_atom(table, i: int, normalize: bool = True) -> Dict[str, Any]:
    """Load sample at atom level (flat atom coordinates).

    This is for models that work directly with atoms (like af3_style).

    Args:
        table: PyArrow table from samples.parquet
        i: Sample index
        normalize: If True, normalize coords to unit variance.

    Returns:
        Dict with atom-level data
    """
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    coords = coords.reshape(n_atoms, 3)
    aa_seq = seq_res[atom_to_res]
    chain_ids = chain_res[atom_to_res]

    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid
    std = coords.std()

    if normalize:
        coords = coords / std

    return {
        'coords': coords,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'std': std.item(),
        'n_atoms': n_atoms,
        'sample_id': table['sample_id'][i].as_py(),
    }


def collate_batch_atom(samples: List[Dict], device: torch.device) -> Dict[str, Any]:
    """Collate atom-level samples into a padded batch.

    Args:
        samples: List of sample dicts from load_sample_atom
        device: Target device

    Returns:
        Batched dict with padded tensors
    """
    B = len(samples)
    max_atoms = max(s['n_atoms'] for s in samples)

    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    aa_seq = torch.zeros(B, max_atoms, dtype=torch.long)
    chain_ids = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    stds = []

    for i, s in enumerate(samples):
        n = s['n_atoms']
        coords[i, :n] = s['coords']
        atom_types[i, :n] = s['atom_types']
        atom_to_res[i, :n] = s['atom_to_res']
        aa_seq[i, :n] = s['aa_seq']
        chain_ids[i, :n] = s['chain_ids']
        mask[i, :n] = True
        stds.append(s['std'])

    return {
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'mask': mask.to(device),
        'stds': stds,
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    step: int,
    train_loss: float = None,
    test_loss: float = None,
    args: dict = None,
    **extra_info
):
    """Save model checkpoint with metadata.

    Args:
        path: Save path
        model: Model to save
        step: Training step
        train_loss: Training loss (optional)
        test_loss: Test loss (optional)
        args: Training arguments (optional)
        **extra_info: Additional metadata
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
    }
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if test_loss is not None:
        checkpoint['test_loss'] = test_loss
    if args is not None:
        checkpoint['args'] = args if isinstance(args, dict) else vars(args)
    checkpoint.update(extra_info)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device = None) -> Dict[str, Any]:
    """Load checkpoint from file.

    Args:
        path: Checkpoint path
        device: Device to load to (default: auto-detect)

    Returns:
        Checkpoint dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(path, map_location=device, weights_only=False)
