"""EGNN (E(n) Equivariant Graph Neural Network) denoiser.

Equivariant GNN that updates node features and coordinates while
respecting rotations and translations.
"""

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    """Single EGNN layer with message passing on coordinates.

    Updates both node features h and coordinates x in an equivariant manner:
    - Feature update uses aggregated messages
    - Coordinate update uses weighted relative position vectors
    """

    def __init__(
        self,
        h_dim: int = 128,
        edge_dim: int = 48,
        coord_weight_clamp: float = 1.0,
    ):
        """
        Args:
            h_dim: Node feature dimension
            edge_dim: Edge feature dimension
            coord_weight_clamp: Clamp value for coordinate update weights
        """
        super().__init__()
        self.coord_weight_clamp = coord_weight_clamp

        # Message MLP: combines node features, edge features, distance
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * h_dim + edge_dim + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        # Node update MLP
        self.mlp_node = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        # Coordinate weight (scalar per edge)
        self.mlp_coord = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )

        self.ln = nn.LayerNorm(h_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, 3] coordinates
            h: [N, h_dim] node features
            edge_index: [2, E] source, dest
            edge_attr: [E, edge_dim] edge features
        Returns:
            x_new: [N, 3] updated coordinates
            h_new: [N, h_dim] updated features
        """
        src, dst = edge_index
        N = h.size(0)

        # Compute relative positions and squared distances
        r_ij = x[src] - x[dst]  # [E, 3]
        d2 = (r_ij ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Message computation
        h_src, h_dst = h[src], h[dst]
        msg_input = torch.cat([h_src, h_dst, edge_attr, d2], dim=-1)
        msg = self.mlp_msg(msg_input)  # [E, h_dim]

        # Aggregate messages (sum)
        agg = torch.zeros(N, h.size(1), device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), msg)

        # Update node features
        h_new = h + self.mlp_node(torch.cat([h, agg], dim=-1))
        h_new = self.ln(h_new)

        # Coordinate update (equivariant)
        w = self.mlp_coord(msg).squeeze(-1)  # [E]
        w = torch.clamp(w, -self.coord_weight_clamp, self.coord_weight_clamp)

        # Weighted sum of relative vectors
        coord_delta = torch.zeros(N, 3, device=x.device, dtype=x.dtype)
        weighted_r = w.unsqueeze(-1) * r_ij
        coord_delta.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), weighted_r)

        x_new = x + coord_delta

        return x_new, h_new


class EGNNDenoiser(nn.Module):
    """Stack of EGNN layers for noise prediction.

    Takes noisy coordinates and conditioning, outputs predicted noise.
    """

    def __init__(
        self,
        n_layers: int = 6,
        h_dim: int = 128,
        edge_dim: int = 48,
        coord_weight_clamp: float = 1.0,
    ):
        """
        Args:
            n_layers: Number of EGNN layers
            h_dim: Node feature dimension
            edge_dim: Edge feature dimension
            coord_weight_clamp: Clamp value for coordinate update weights
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(h_dim, edge_dim, coord_weight_clamp)
            for _ in range(n_layers)
        ])

        # Output head for noise prediction
        self.head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 3),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [N, 3] noisy coordinates
            h: [N, h_dim] atom conditioning features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge features
        Returns:
            eps_hat: [N, 3] predicted noise
        """
        x = x_t
        for layer in self.layers:
            x, h = layer(x, h, edge_index, edge_attr)

        # Use coordinate difference for equivariant output
        # The EGNN layers update x equivariantly, so (x - x_t) transforms correctly
        # under rotation: R(x - x_t) = Rx - Rx_t
        eps_hat = x - x_t
        return eps_hat
