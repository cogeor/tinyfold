"""Unit tests for the HDOCK-style pose clusterer in tinyfold.model.metrics.cluster.

Covers the three behavioural claims of Loop 02 Task 1:
  - Two well-separated modes produce two clusters.
  - Each cluster's representative is one of its own members.
  - Cluster ranking puts the larger / more compact cluster first.
"""


import pytest
import torch

from tinyfold.model.metrics.cluster import (
    cluster_poses,
    interface_mask_from_gt,
    pairwise_interface_rmsd,
)


def _build_two_chain_pose(n_per_chain: int = 3) -> torch.Tensor:
    """Two-chain CA layout: chain A near the origin, chain B near x=5 A."""
    chain_a = torch.tensor([[float(i), 0.0, 0.0] for i in range(n_per_chain)])
    chain_b = torch.tensor([[float(i), 5.0, 0.0] for i in range(n_per_chain)])
    return torch.cat([chain_a, chain_b], dim=0)  # [2 * n_per_chain, 3]


def test_interface_mask_from_gt_picks_close_residues() -> None:
    pose = _build_two_chain_pose(n_per_chain=3)
    chain_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    valid = torch.ones(6, dtype=torch.bool)
    mask = interface_mask_from_gt(pose, chain_ids, valid, contact_cutoff=8.0)
    # All 6 residues sit within 5-8 A of the other chain, so all are interface.
    assert mask.sum().item() == 6


def test_interface_mask_excludes_far_residues() -> None:
    # Move chain B 30 A away on the y axis -> no contacts -> empty mask.
    pose = _build_two_chain_pose(n_per_chain=3)
    pose[3:, 1] = 30.0
    chain_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    valid = torch.ones(6, dtype=torch.bool)
    mask = interface_mask_from_gt(pose, chain_ids, valid, contact_cutoff=8.0)
    assert mask.sum().item() == 0


def test_pairwise_interface_rmsd_diag_zero_and_symmetric() -> None:
    pose = _build_two_chain_pose(n_per_chain=3)
    poses = torch.stack([pose, pose + 0.5, pose + 2.0], dim=0)  # [3, 6, 3]
    iface = torch.ones(6, dtype=torch.bool)
    d = pairwise_interface_rmsd(poses, iface)
    assert d.shape == (3, 3)
    assert torch.allclose(d, d.t())
    assert torch.allclose(d.diag(), torch.zeros(3))


def _two_mode_poses() -> tuple:
    """Build 6 poses: 0..2 cluster around mode A, 3..5 cluster around mode B.

    Mode B differs from mode A by translating chain B residues by +20 A
    on the x axis — far beyond the 5 A cluster radius.
    """
    base = _build_two_chain_pose(n_per_chain=3)  # [6, 3]
    chain_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    valid = torch.ones(6, dtype=torch.bool)

    torch.manual_seed(0)
    poses = []
    for _ in range(3):
        jitter = torch.randn_like(base) * 0.05  # ~0.05 A jitter
        poses.append(base + jitter)
    for _ in range(3):
        moved = base.clone()
        moved[3:, 0] += 20.0  # translate chain B
        jitter = torch.randn_like(moved) * 0.05
        poses.append(moved + jitter)
    return torch.stack(poses, dim=0), chain_ids, valid, base


def test_cluster_poses_finds_two_modes() -> None:
    poses, chain_ids, valid, gt = _two_mode_poses()
    iface = interface_mask_from_gt(gt, chain_ids, valid, contact_cutoff=8.0)
    assert iface.any()

    clusters = cluster_poses(poses, iface, radius=5.0)
    assert len(clusters) == 2
    # Both clusters have size 3 -> ordering tiebreaks on compactness.
    sizes = sorted(len(c["members"]) for c in clusters)
    assert sizes == [3, 3]
    # Each cluster's members are a partition of {0..5} into the two halves.
    members_top = set(clusters[0]["members"])
    members_bottom = set(clusters[1]["members"])
    assert members_top | members_bottom == set(range(6))
    assert members_top in ({0, 1, 2}, {3, 4, 5})
    # Representative must be a member of its own cluster.
    for c in clusters:
        assert c["representative"] in c["members"]


def test_cluster_singleton_when_poses_far_apart() -> None:
    """Six poses spaced 50 A apart in chain-B x should produce 6 singletons."""
    base = _build_two_chain_pose(n_per_chain=3)
    chain_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    valid = torch.ones(6, dtype=torch.bool)
    iface = interface_mask_from_gt(base, chain_ids, valid, contact_cutoff=8.0)

    poses = []
    for k in range(6):
        moved = base.clone()
        moved[3:, 0] += 50.0 * k
        poses.append(moved)
    poses_t = torch.stack(poses, dim=0)
    clusters = cluster_poses(poses_t, iface, radius=5.0)
    assert len(clusters) == 6
    for c in clusters:
        assert len(c["members"]) == 1
        assert c["mean_intra_rmsd"] == 0.0
        assert c["representative"] == c["members"][0]


def test_cluster_single_pose() -> None:
    base = _build_two_chain_pose(n_per_chain=3)
    chain_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    valid = torch.ones(6, dtype=torch.bool)
    iface = interface_mask_from_gt(base, chain_ids, valid, contact_cutoff=8.0)

    poses = base.unsqueeze(0)  # [1, 6, 3]
    clusters = cluster_poses(poses, iface, radius=5.0)
    assert len(clusters) == 1
    assert clusters[0]["members"] == [0]
    assert clusters[0]["representative"] == 0


def test_pairwise_rmsd_raises_on_empty_mask() -> None:
    poses = torch.zeros(2, 6, 3)
    empty_mask = torch.zeros(6, dtype=torch.bool)
    with pytest.raises(ValueError):
        pairwise_interface_rmsd(poses, empty_mask)
