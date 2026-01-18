"""Edge building utilities for EGNN denoiser."""

import torch


def build_knn_edges(
    x: torch.Tensor,
    k: int = 16,
    exclude_self: bool = True,
) -> torch.Tensor:
    """Build k-nearest neighbor edges from coordinates.

    Uses PyTorch Geometric's approach: find k+1 neighbors, then filter self-loops.
    This is more robust than setting diagonal to inf.

    Args:
        x: [N, 3] coordinates
        k: Number of neighbors per node
        exclude_self: Whether to exclude self-loops
    Returns:
        edge_index: [2, E] where E = N * k
    """
    N = x.size(0)

    # Handle edge cases
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)

    # Compute pairwise distances
    dist = torch.cdist(x, x)  # [N, N]

    if exclude_self:
        # Set diagonal to large value to exclude self-loops
        dist = dist.clone()
        max_dist = dist.max().item() + 1.0
        dist.fill_diagonal_(max_dist)

        actual_k = min(k, N - 1)  # Can't have more than N-1 neighbors
        _, indices = dist.topk(actual_k, dim=1, largest=False)  # [N, k]

        # Build edge index
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, actual_k).flatten()
        dst = indices.flatten()
    else:
        actual_k = min(k, N)
        _, indices = dist.topk(actual_k, dim=1, largest=False)
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, actual_k).flatten()
        dst = indices.flatten()

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def merge_edges(
    bond_src: torch.Tensor,
    bond_dst: torch.Tensor,
    knn_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge bond edges and KNN edges.

    Args:
        bond_src: [E_bond] source indices for bond edges
        bond_dst: [E_bond] destination indices for bond edges
        knn_edges: [2, E_knn] KNN edge index
    Returns:
        edge_index: [2, E_total] merged edges
        edge_type: [E_total] 0 for bond, 1 for knn
    """
    device = bond_src.device
    n_bond = bond_src.size(0)
    n_knn = knn_edges.size(1)

    # Combine edge indices
    bond_edges = torch.stack([bond_src, bond_dst], dim=0)  # [2, E_bond]
    edge_index = torch.cat([bond_edges, knn_edges], dim=1)  # [2, E_total]

    # Create edge type indicator
    edge_type = torch.cat([
        torch.zeros(n_bond, dtype=torch.long, device=device),
        torch.ones(n_knn, dtype=torch.long, device=device),
    ])

    return edge_index, edge_type


def build_edge_attr(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    bond_type: torch.Tensor | None,
    pair_proj: torch.Tensor,
    atom_to_res: torch.Tensor,
    t_embed: torch.Tensor,
    n_bond_edges: int,
) -> torch.Tensor:
    """Build edge attributes from various sources.

    Args:
        edge_index: [2, E] edge indices
        edge_type: [E] edge type (0=bond, 1=knn)
        bond_type: [E_bond] bond type indices (or None)
        pair_proj: [L, L, c_pair] projected pair representation
        atom_to_res: [N_atom] residue index per atom
        t_embed: [c_time] timestep embedding
        n_bond_edges: Number of bond edges
    Returns:
        edge_attr: [E, c_edge] edge features
    """
    E = edge_index.size(1)
    device = edge_index.device

    src, dst = edge_index

    # Get residue indices for atoms
    res_src = atom_to_res[src]  # [E]
    res_dst = atom_to_res[dst]  # [E]

    # Get pair features
    pair_feat = pair_proj[res_src, res_dst]  # [E, c_pair]

    # Edge type embedding (2 types: bond, knn)
    edge_type_embed = torch.zeros(E, 2, device=device)
    edge_type_embed.scatter_(1, edge_type.unsqueeze(1), 1.0)  # One-hot

    # Bond type embedding for bond edges (0 for knn edges)
    if bond_type is not None and n_bond_edges > 0:
        # Create bond type one-hot (4 types: N-CA, CA-C, C-O, C-N peptide)
        bond_type_embed = torch.zeros(E, 4, device=device)
        bond_type_embed[:n_bond_edges].scatter_(
            1, bond_type.unsqueeze(1), 1.0
        )
    else:
        bond_type_embed = torch.zeros(E, 4, device=device)

    # Broadcast timestep embedding
    t_broadcast = t_embed.unsqueeze(0).expand(E, -1)  # [E, c_time]

    # Concatenate all features
    edge_attr = torch.cat([
        edge_type_embed,
        bond_type_embed,
        pair_feat,
        t_broadcast,
    ], dim=-1)

    return edge_attr
