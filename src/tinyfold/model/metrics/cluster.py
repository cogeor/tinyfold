"""HDOCK-style greedy pose clustering over interface-CA RMSD.

Helpers in this module are CPU-only: torch.linalg ops over K<=40 small
tensors are dominated by CUDA sync overhead, so we keep everything on
the host. Callers are expected to `.cpu()` their inputs before invoking
these functions.
"""

from typing import List

import torch
from torch import Tensor


def interface_mask_from_gt(
    gt_ca: Tensor,
    chain_ids: Tensor,
    valid: Tensor,
    contact_cutoff: float = 8.0,
) -> Tensor:
    """Compute the interface-residue mask from the GT complex.

    A residue is "interface" if it belongs to chain A and lies within
    ``contact_cutoff`` of any chain-B CA in the GT, OR vice versa.

    Args:
        gt_ca: ``[L, 3]`` GT CA coordinates (Angstroms or normalized; the
            cutoff must match the units).
        chain_ids: ``[L]`` long/int tensor with values in {0, 1}.
        valid: ``[L]`` bool tensor; residues with ``valid=False`` cannot
            be in the interface.
        contact_cutoff: Pairwise CA-CA distance threshold (default 8 A,
            the standard contact definition).

    Returns:
        ``[L]`` bool tensor; True for interface residues.
    """
    assert gt_ca.dim() == 2 and gt_ca.shape[-1] == 3, "gt_ca must be [L, 3]"
    L = gt_ca.shape[0]
    assert chain_ids.shape == (L,) and valid.shape == (L,)

    chain_a = (chain_ids == 0) & valid
    chain_b = (chain_ids == 1) & valid
    if not chain_a.any() or not chain_b.any():
        return torch.zeros(L, dtype=torch.bool, device=gt_ca.device)

    ca_a = gt_ca[chain_a]
    ca_b = gt_ca[chain_b]
    # [Na, Nb] pairwise distances; min over the other chain gives nearest contact.
    dists = torch.cdist(ca_a, ca_b)
    a_min = dists.min(dim=1).values  # [Na]
    b_min = dists.min(dim=0).values  # [Nb]
    a_iface = a_min < contact_cutoff
    b_iface = b_min < contact_cutoff

    mask = torch.zeros(L, dtype=torch.bool, device=gt_ca.device)
    a_idx = torch.nonzero(chain_a, as_tuple=False).squeeze(-1)
    b_idx = torch.nonzero(chain_b, as_tuple=False).squeeze(-1)
    mask[a_idx[a_iface]] = True
    mask[b_idx[b_iface]] = True
    return mask


def pairwise_interface_rmsd(
    poses: Tensor,
    interface_mask: Tensor,
) -> Tensor:
    """Pairwise RMSD between K poses computed over interface residues only.

    No Kabsch alignment between samples — they are already in the GT frame
    because the sampler recenters every step.

    Args:
        poses: ``[K, L, 3]`` predicted CA coordinates for K samples.
        interface_mask: ``[L]`` bool tensor; columns to use.

    Returns:
        ``[K, K]`` symmetric matrix of pairwise RMSDs; diagonal is exactly 0.
    """
    assert poses.dim() == 3 and poses.shape[-1] == 3, "poses must be [K, L, 3]"
    L_iface = int(interface_mask.sum().item())
    if L_iface == 0:
        raise ValueError("interface_mask is empty; caller should fall back to all-True mask")

    iface_poses = poses[:, interface_mask, :]  # [K, L_iface, 3]
    # [K, 1, L_iface, 3] - [1, K, L_iface, 3] -> [K, K, L_iface, 3]
    diff = iface_poses.unsqueeze(1) - iface_poses.unsqueeze(0)
    sq = (diff * diff).sum(dim=-1)  # [K, K, L_iface]
    rmsd = torch.sqrt(sq.mean(dim=-1).clamp(min=0.0))  # [K, K]
    return rmsd


def cluster_poses(
    poses: Tensor,
    interface_mask: Tensor,
    radius: float,
) -> List[dict]:
    """Greedy nearest-neighbour clustering over interface RMSD.

    Algorithm (HDOCK convention):
      1. Visit poses in index order. For the first unassigned pose, open
         a new cluster centred on it; absorb any later unassigned pose
         within ``radius``.
      2. Repeat until every pose is assigned.
      3. Within each cluster, pick the representative = member with the
         smallest mean RMSD to the other members (closest-to-centroid).
         Singleton clusters use the single member as representative.
      4. Sort clusters by size desc, then by ``mean_intra_rmsd`` asc
         (more compact cluster wins ties).

    Args:
        poses: ``[K, L, 3]`` predicted CA coordinates.
        interface_mask: ``[L]`` bool mask. Must be non-empty.
        radius: Pairwise RMSD cutoff for cluster membership (Angstroms).

    Returns:
        List of cluster dicts, ordered by the ranking above. Each dict:
            ``{
                'members': List[int],
                'representative': int,
                'mean_intra_rmsd': float,
            }``
    """
    K = poses.shape[0]
    if K == 0:
        return []

    rmsd = pairwise_interface_rmsd(poses, interface_mask)
    assigned = torch.zeros(K, dtype=torch.bool)

    clusters: List[dict] = []
    for i in range(K):
        if assigned[i]:
            continue
        within = (rmsd[i] < radius) & (~assigned)
        # Centre is i, so it's always in its own cluster.
        within[i] = True
        members = torch.nonzero(within, as_tuple=False).squeeze(-1).tolist()
        assigned |= within

        # Representative: smallest mean RMSD to other members.
        if len(members) == 1:
            rep = members[0]
            mean_intra = 0.0
        else:
            member_idx = torch.tensor(members, dtype=torch.long)
            sub = rmsd[member_idx][:, member_idx]  # [m, m]
            # Exclude self in the mean; divide by (m-1).
            mean_to_others = (sub.sum(dim=1) - sub.diag()) / (len(members) - 1)
            rep_local = int(torch.argmin(mean_to_others).item())
            rep = members[rep_local]
            # Mean intra-cluster RMSD over the upper triangle.
            mean_intra = float(sub.sum().item() / (len(members) * (len(members) - 1)))

        clusters.append({
            "members": members,
            "representative": rep,
            "mean_intra_rmsd": mean_intra,
        })

    clusters.sort(key=lambda c: (-len(c["members"]), c["mean_intra_rmsd"]))
    return clusters
