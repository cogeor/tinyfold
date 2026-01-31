"""Hierarchical clustering utilities for iterative atom assembly.

Determines the order in which to place atoms during iterative construction.
Uses a combination of spatial proximity, chain connectivity, and covalent bonds.
"""

from typing import List, Optional
import torch
from torch import Tensor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import numpy as np


# Backbone atom types: N=0, CA=1, C=2, O=3
BACKBONE_BONDS = {
    # Within residue: N-CA, CA-C, C-O
    (0, 1): 1.458,  # N-CA bond length (Angstroms)
    (1, 2): 1.524,  # CA-C bond length
    (2, 3): 1.231,  # C-O bond length
}
# Between residues: C(i) - N(i+1) peptide bond
PEPTIDE_BOND_LENGTH = 1.329


def compute_bond_connectivity(
    n_residues: int,
    chain_ids: Tensor,  # [L] chain ID per residue
) -> Tensor:
    """Compute bond connectivity matrix for backbone atoms.

    Each residue has 4 atoms: N(0), CA(1), C(2), O(3).
    - Intra-residue bonds: N-CA, CA-C, C-O
    - Inter-residue bonds: C(i)-N(i+1) if same chain

    Args:
        n_residues: Number of residues (L)
        chain_ids: Chain ID for each residue [L]

    Returns:
        connectivity: [N, N] adjacency matrix where N = L * 4
    """
    n_atoms = n_residues * 4
    connectivity = torch.zeros(n_atoms, n_atoms, dtype=torch.bool)

    for i in range(n_residues):
        base = i * 4
        # Intra-residue bonds
        connectivity[base + 0, base + 1] = True  # N-CA
        connectivity[base + 1, base + 0] = True
        connectivity[base + 1, base + 2] = True  # CA-C
        connectivity[base + 2, base + 1] = True
        connectivity[base + 2, base + 3] = True  # C-O
        connectivity[base + 3, base + 2] = True

        # Peptide bond to next residue (if same chain)
        if i < n_residues - 1 and chain_ids[i] == chain_ids[i + 1]:
            connectivity[base + 2, base + 4 + 0] = True  # C(i)-N(i+1)
            connectivity[base + 4 + 0, base + 2] = True

    return connectivity


def hierarchical_cluster_atoms(
    coords: Tensor,         # [N, 3] atom coordinates
    chain_ids: Tensor,      # [L] chain ID per residue (N = L * 4)
    n_clusters: int = 10,
    chain_weight: float = 2.0,  # Penalty for crossing chains
) -> Tensor:
    """Hierarchical clustering of atoms using proximity + chain info.

    Computes a modified distance matrix that penalizes cross-chain distances,
    then performs agglomerative clustering.

    Args:
        coords: Atom coordinates [N, 3]
        chain_ids: Chain ID per residue [L]
        n_clusters: Number of clusters to form
        chain_weight: Multiplier for cross-chain distances

    Returns:
        cluster_ids: [N] cluster assignment for each atom (0-indexed)
    """
    n_atoms = coords.shape[0]
    n_residues = n_atoms // 4
    device = coords.device

    # Edge case: if n_clusters >= n_atoms, each atom is its own cluster
    if n_clusters >= n_atoms:
        return torch.arange(n_atoms, device=device, dtype=torch.long)

    # Expand chain_ids to atom level [N]
    atom_chain_ids = chain_ids.repeat_interleave(4)

    # Compute pairwise distances
    coords_np = coords.detach().cpu().numpy()

    # Handle edge case of too few atoms
    if n_atoms < 2:
        return torch.zeros(n_atoms, device=device, dtype=torch.long)

    dist_condensed = pdist(coords_np)

    # Convert to square form for chain penalty
    dist_matrix = squareform(dist_condensed)

    # Apply chain penalty: increase distance for cross-chain pairs
    chain_np = atom_chain_ids.cpu().numpy()
    cross_chain = chain_np[:, None] != chain_np[None, :]
    dist_matrix[cross_chain] *= chain_weight

    # Back to condensed form
    dist_modified = squareform(dist_matrix)

    # Hierarchical clustering
    Z = linkage(dist_modified, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return torch.tensor(labels - 1, device=device, dtype=torch.long)  # 0-indexed


def select_next_atoms_to_place(
    coords_gt: Tensor,     # [N, 3] ground truth (for cluster selection)
    known_mask: Tensor,    # [N] bool, True = already placed
    k: int,                # number of atoms to select
    cluster_ids: Optional[Tensor] = None,   # [N] pre-computed cluster assignments
) -> Tensor:
    """Select K atoms to place next based on proximity to known atoms.

    Strategy:
    1. Find unknown atoms
    2. Score each by minimum distance to any known atom
    3. Select K closest atoms

    Args:
        coords_gt: Ground truth coordinates [N, 3]
        known_mask: Boolean mask for already-placed atoms [N]
        k: Number of atoms to select
        cluster_ids: Pre-computed cluster IDs [N] (optional, for future use)

    Returns:
        target_idx: Indices of atoms to predict [K]
    """
    n_atoms = coords_gt.shape[0]
    device = coords_gt.device

    # Edge case: if no atoms known yet, pick first k atoms
    if not known_mask.any():
        return torch.arange(min(k, n_atoms), device=device, dtype=torch.long)

    # Get known atom positions
    known_coords = coords_gt[known_mask]  # [n_known, 3]

    # For each unknown atom, compute min distance to known atoms
    unknown_idx = (~known_mask).nonzero(as_tuple=True)[0]

    if len(unknown_idx) == 0:
        return torch.tensor([], device=device, dtype=torch.long)

    unknown_coords = coords_gt[unknown_idx]  # [n_unknown, 3]

    # Pairwise distances: [n_unknown, n_known]
    dists = torch.cdist(unknown_coords, known_coords)
    min_dists = dists.min(dim=1).values  # [n_unknown]

    # Select k atoms with smallest min distance
    actual_k = min(k, len(unknown_idx))
    _, top_k_local = torch.topk(min_dists, k=actual_k, largest=False)
    target_idx = unknown_idx[top_k_local]

    return target_idx


def get_placement_order(
    coords_gt: Tensor,     # [N, 3]
    chain_ids: Tensor,     # [L]
    k_per_step: int = 4,
) -> List[Tensor]:
    """Pre-compute the full placement order for a structure.

    Returns a list of index tensors, each of length k, representing
    the order in which atoms should be placed.

    Useful for debugging and visualization.

    Args:
        coords_gt: Ground truth atom coordinates [N, 3]
        chain_ids: Chain ID per residue [L]
        k_per_step: Number of atoms to place per step

    Returns:
        order: List of index tensors, each of length <= k_per_step
    """
    n_atoms = coords_gt.shape[0]
    device = coords_gt.device
    known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

    # Compute clustering once
    n_clusters = max(1, n_atoms // k_per_step + 1)
    cluster_ids = hierarchical_cluster_atoms(
        coords_gt, chain_ids, n_clusters=n_clusters
    )

    order = []
    while known_mask.sum() < n_atoms:
        remaining = n_atoms - known_mask.sum().item()
        k = min(k_per_step, remaining)

        target_idx = select_next_atoms_to_place(
            coords_gt, known_mask, k, cluster_ids
        )

        if len(target_idx) == 0:
            break

        order.append(target_idx)
        known_mask[target_idx] = True

    return order


def select_next_residues_to_place(
    centroids: Tensor,     # [L, 3] residue centroids
    known_mask: Tensor,    # [L] bool, True = already placed
    k: int,                # number of residues to select
) -> Tensor:
    """Select K residues to place next based on proximity to known residues.

    Strategy:
    - If no residues known: select residue closest to geometric center (most central)
    - Otherwise: select residues closest to any known residue

    Args:
        centroids: Centroid coordinates [L, 3]
        known_mask: Boolean mask for already-placed residues [L]
        k: Number of residues to select

    Returns:
        target_idx: Indices of residues to predict [K]
    """
    L = centroids.shape[0]
    device = centroids.device

    # Edge case: if no residues known yet, pick most central residue
    if not known_mask.any():
        # Compute geometric center
        center = centroids.mean(dim=0, keepdim=True)  # [1, 3]
        dists_to_center = torch.cdist(centroids.unsqueeze(0), center).squeeze()  # [L]
        _, top_k = torch.topk(dists_to_center, k=min(k, L), largest=False)
        return top_k

    # Get known residue positions
    known_coords = centroids[known_mask]  # [n_known, 3]

    # For each unknown residue, compute min distance to known residues
    unknown_idx = (~known_mask).nonzero(as_tuple=True)[0]

    if len(unknown_idx) == 0:
        return torch.tensor([], device=device, dtype=torch.long)

    unknown_coords = centroids[unknown_idx]  # [n_unknown, 3]

    # Pairwise distances: [n_unknown, n_known]
    dists = torch.cdist(unknown_coords, known_coords)
    min_dists = dists.min(dim=1).values  # [n_unknown]

    # Select k residues with smallest min distance (closest to known)
    actual_k = min(k, len(unknown_idx))
    _, top_k_local = torch.topk(min_dists, k=actual_k, largest=False)
    target_idx = unknown_idx[top_k_local]

    return target_idx


def get_residue_placement_order(
    centroids: Tensor,     # [L, 3] residue centroids
    k_per_step: int = 1,
) -> List[Tensor]:
    """Pre-compute the full residue placement order starting from most central.

    Args:
        centroids: Centroid coordinates [L, 3]
        k_per_step: Residues to place per step

    Returns:
        order: List of index tensors
    """
    L = centroids.shape[0]
    device = centroids.device
    known_mask = torch.zeros(L, dtype=torch.bool, device=device)

    order = []
    while known_mask.sum() < L:
        remaining = L - known_mask.sum().item()
        k = min(k_per_step, remaining)

        target_idx = select_next_residues_to_place(centroids, known_mask, k)

        if len(target_idx) == 0:
            break

        order.append(target_idx)
        known_mask[target_idx] = True

    return order


def simulate_known_mask(
    coords_gt: Tensor,     # [N, 3]
    chain_ids: Tensor,     # [L]
    n_known: int,          # number of atoms to mark as known
    k_per_step: int = 4,
) -> Tensor:
    """Simulate a known_mask by running placement order up to n_known atoms.

    Used during training to create realistic partial structures.

    Args:
        coords_gt: Ground truth atom coordinates [N, 3]
        chain_ids: Chain ID per residue [L]
        n_known: Number of atoms to mark as known
        k_per_step: Atoms placed per step

    Returns:
        known_mask: Boolean mask [N] with n_known True values
    """
    n_atoms = coords_gt.shape[0]
    device = coords_gt.device
    known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

    if n_known <= 0:
        return known_mask

    # Compute clustering once
    n_clusters = max(1, n_atoms // k_per_step + 1)
    cluster_ids = hierarchical_cluster_atoms(
        coords_gt, chain_ids, n_clusters=n_clusters
    )

    placed_count = 0
    while placed_count < n_known:
        remaining_to_place = n_known - placed_count
        k = min(k_per_step, remaining_to_place, n_atoms - placed_count)

        if k <= 0:
            break

        target_idx = select_next_atoms_to_place(
            coords_gt, known_mask, k, cluster_ids
        )

        if len(target_idx) == 0:
            break

        known_mask[target_idx] = True
        placed_count += len(target_idx)

    return known_mask
