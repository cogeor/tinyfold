"""Unified metrics computation for benchmark evaluation.

Computes standardized metrics across all model types using existing
implementations from tinyfold.model.losses and metrics.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from tinyfold.model.losses import (
    kabsch_align,
    compute_lddt,
    compute_ilddt,
    compute_interface_mask,
)
from tinyfold.model.metrics.dockq import compute_dockq


@dataclass
class BenchmarkMetrics:
    """Standardized metrics computed for all models."""

    # Sample identification
    sample_id: str
    n_residues: int
    n_atoms: int

    # RMSD-based (after Kabsch alignment, in Angstroms)
    rmsd_all_atoms: float  # All backbone atoms
    rmsd_ca: float  # CA only
    rmsd_interface: Optional[float]  # Interface residues only

    # lDDT-based (superposition-free, 0-1 scale)
    lddt: float  # Global lDDT
    ilddt: float  # Interface lDDT

    # Contact-based (0-1 scale)
    contact_precision: float  # Predicted contacts that are true
    contact_recall: float  # True contacts that are predicted
    contact_f1: float  # F1 score

    # DockQ (protein docking quality)
    dockq: Optional[float]  # Overall DockQ score (0-1)
    fnat: Optional[float]  # Fraction of native contacts
    irms: Optional[float]  # Interface RMSD
    lrms: Optional[float]  # Ligand RMSD

    # Timing
    inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def kabsch_align_np(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch alignment using numpy - align pred to target.

    Returns:
        (aligned_pred, rmsd)
    """
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    P = pred - pred_mean
    Q = target - target_mean

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.linalg.det(V @ U.T)
    if d < 0:
        V[:, 2] *= -1

    R = V @ U.T
    P_aligned = P @ R.T
    pred_aligned = P_aligned + target_mean

    diff = pred_aligned - target
    rmsd = float(np.sqrt((diff**2).sum(axis=-1).mean()))

    return pred_aligned, rmsd


def compute_contact_metrics(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    chain_ids: np.ndarray,
    threshold: float = 8.0,
) -> dict:
    """Compute contact prediction metrics.

    Args:
        pred_ca: [L, 3] predicted CA coordinates (Angstroms)
        gt_ca: [L, 3] ground truth CA coordinates (Angstroms)
        chain_ids: [L] chain IDs (0 or 1)
        threshold: Distance threshold for contact definition

    Returns:
        dict with precision, recall, f1
    """
    L = len(pred_ca)

    # Compute pairwise distances
    pred_dists = np.linalg.norm(pred_ca[:, None] - pred_ca[None, :], axis=-1)
    gt_dists = np.linalg.norm(gt_ca[:, None] - gt_ca[None, :], axis=-1)

    # Inter-chain mask
    chain_a = chain_ids == 0
    chain_b = chain_ids == 1
    inter_chain = (chain_a[:, None] & chain_b[None, :]) | (
        chain_b[:, None] & chain_a[None, :]
    )

    # Contact predictions (inter-chain only)
    pred_contacts = (pred_dists < threshold) & inter_chain
    gt_contacts = (gt_dists < threshold) & inter_chain

    # Metrics
    tp = (pred_contacts & gt_contacts).sum()
    fp = (pred_contacts & ~gt_contacts).sum()
    fn = (~pred_contacts & gt_contacts).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "contact_f1": float(f1),
    }


def compute_interface_rmsd(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    chain_ids: np.ndarray,
    threshold: float = 8.0,
) -> Optional[float]:
    """Compute RMSD for interface residues only.

    Args:
        pred_ca: [L, 3] predicted CA coordinates (Angstroms)
        gt_ca: [L, 3] ground truth CA coordinates (Angstroms)
        chain_ids: [L] chain IDs (0 or 1)
        threshold: Distance threshold for interface definition

    Returns:
        Interface RMSD or None if no interface residues
    """
    L = len(pred_ca)

    # Compute GT pairwise distances
    gt_dists = np.linalg.norm(gt_ca[:, None] - gt_ca[None, :], axis=-1)

    # Inter-chain mask
    chain_a = chain_ids == 0
    chain_b = chain_ids == 1
    inter_chain = (chain_a[:, None] & chain_b[None, :]) | (
        chain_b[:, None] & chain_a[None, :]
    )

    # Interface residues (any residue with inter-chain contact in GT)
    has_contact = (gt_dists < threshold) & inter_chain
    interface_mask = has_contact.any(axis=1)

    if not interface_mask.any():
        return None

    # RMSD on interface residues
    pred_interface = pred_ca[interface_mask]
    gt_interface = gt_ca[interface_mask]

    diff = pred_interface - gt_interface
    rmsd = float(np.sqrt((diff**2).sum(axis=-1).mean()))

    return rmsd


def compute_all_metrics(
    pred_atoms: np.ndarray,
    gt_atoms: np.ndarray,
    aa_seq: np.ndarray,
    chain_ids: np.ndarray,
    inference_time_ms: float,
    sample_id: str,
    compute_dockq_score: bool = True,
) -> BenchmarkMetrics:
    """Compute all standardized metrics for a single prediction.

    Args:
        pred_atoms: [L, 4, 3] predicted backbone atoms (Angstroms)
        gt_atoms: [L, 4, 3] ground truth backbone atoms (Angstroms)
        aa_seq: [L] amino acid indices
        chain_ids: [L] chain IDs
        inference_time_ms: Inference time in milliseconds
        sample_id: Sample identifier
        compute_dockq_score: Whether to compute DockQ (slower)

    Returns:
        BenchmarkMetrics with all metrics populated
    """
    L = len(aa_seq)
    n_atoms = L * 4

    # Extract CA coordinates
    pred_ca = pred_atoms[:, 1, :]  # [L, 3]
    gt_ca = gt_atoms[:, 1, :]  # [L, 3]

    # Flatten atoms for all-atom RMSD
    pred_flat = pred_atoms.reshape(-1, 3)  # [L*4, 3]
    gt_flat = gt_atoms.reshape(-1, 3)  # [L*4, 3]

    # Kabsch alignment and RMSD
    pred_aligned, rmsd_all = kabsch_align_np(pred_flat, gt_flat)
    pred_ca_aligned, rmsd_ca = kabsch_align_np(pred_ca, gt_ca)

    # Interface RMSD
    rmsd_interface = compute_interface_rmsd(pred_ca_aligned, gt_ca, chain_ids)

    # lDDT metrics (using aligned predictions, coord_scale=1.0 since already in Angstroms)
    pred_ca_tensor = torch.tensor(pred_ca_aligned, dtype=torch.float32)
    gt_ca_tensor = torch.tensor(gt_ca, dtype=torch.float32)
    chain_ids_tensor = torch.tensor(chain_ids, dtype=torch.long)

    lddt = compute_lddt(pred_ca_tensor, gt_ca_tensor, coord_scale=1.0).item()
    ilddt = compute_ilddt(
        pred_ca_tensor, gt_ca_tensor, chain_ids_tensor, coord_scale=1.0
    ).item()

    # Contact metrics
    contact_metrics = compute_contact_metrics(pred_ca_aligned, gt_ca, chain_ids)

    # DockQ (optional, slower)
    if compute_dockq_score:
        pred_atoms_tensor = torch.tensor(
            pred_aligned.reshape(L, 4, 3), dtype=torch.float32
        )
        gt_atoms_tensor = torch.tensor(gt_atoms, dtype=torch.float32)
        aa_seq_tensor = torch.tensor(aa_seq, dtype=torch.long)

        dockq_result = compute_dockq(
            pred_atoms_tensor,
            gt_atoms_tensor,
            aa_seq_tensor,
            chain_ids_tensor,
            std=1.0,  # Already in Angstroms
        )
    else:
        dockq_result = {"dockq": None, "fnat": None, "irms": None, "lrms": None}

    return BenchmarkMetrics(
        sample_id=sample_id,
        n_residues=L,
        n_atoms=n_atoms,
        rmsd_all_atoms=rmsd_all,
        rmsd_ca=rmsd_ca,
        rmsd_interface=rmsd_interface,
        lddt=lddt,
        ilddt=ilddt,
        contact_precision=contact_metrics["contact_precision"],
        contact_recall=contact_metrics["contact_recall"],
        contact_f1=contact_metrics["contact_f1"],
        dockq=dockq_result["dockq"],
        fnat=dockq_result["fnat"],
        irms=dockq_result["irms"],
        lrms=dockq_result["lrms"],
        inference_time_ms=inference_time_ms,
    )


def aggregate_metrics(metrics_list: list[BenchmarkMetrics]) -> dict:
    """Aggregate metrics from multiple samples.

    Args:
        metrics_list: List of BenchmarkMetrics

    Returns:
        dict with mean/std/median for each metric
    """
    if not metrics_list:
        return {}

    # Collect values for each metric
    metric_values = {
        "rmsd_all_atoms": [],
        "rmsd_ca": [],
        "rmsd_interface": [],
        "lddt": [],
        "ilddt": [],
        "contact_precision": [],
        "contact_recall": [],
        "contact_f1": [],
        "dockq": [],
        "fnat": [],
        "irms": [],
        "lrms": [],
        "inference_time_ms": [],
    }

    for m in metrics_list:
        metric_values["rmsd_all_atoms"].append(m.rmsd_all_atoms)
        metric_values["rmsd_ca"].append(m.rmsd_ca)
        if m.rmsd_interface is not None:
            metric_values["rmsd_interface"].append(m.rmsd_interface)
        metric_values["lddt"].append(m.lddt)
        metric_values["ilddt"].append(m.ilddt)
        metric_values["contact_precision"].append(m.contact_precision)
        metric_values["contact_recall"].append(m.contact_recall)
        metric_values["contact_f1"].append(m.contact_f1)
        if m.dockq is not None:
            metric_values["dockq"].append(m.dockq)
            metric_values["fnat"].append(m.fnat)
            metric_values["irms"].append(m.irms)
            metric_values["lrms"].append(m.lrms)
        metric_values["inference_time_ms"].append(m.inference_time_ms)

    # Compute aggregates
    result = {}
    for name, values in metric_values.items():
        if not values:
            result[name] = {"mean": None, "std": None, "median": None}
            continue

        arr = np.array(values)
        result[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }

    # DockQ quality thresholds
    dockq_vals = metric_values["dockq"]
    if dockq_vals:
        n_total = len(dockq_vals)
        result["dockq_acceptable_pct"] = 100 * sum(1 for d in dockq_vals if d >= 0.23) / n_total
        result["dockq_medium_pct"] = 100 * sum(1 for d in dockq_vals if d >= 0.49) / n_total
        result["dockq_high_pct"] = 100 * sum(1 for d in dockq_vals if d >= 0.80) / n_total
    else:
        result["dockq_acceptable_pct"] = None
        result["dockq_medium_pct"] = None
        result["dockq_high_pct"] = None

    result["n_samples"] = len(metrics_list)

    return result
