"""Atom conditioner that lifts residue representations to atoms."""

import torch
import torch.nn as nn


class AtomConditioner(nn.Module):
    """Lifts residue-level representations to atom-level conditioning.

    Each atom gets:
    - Residue single features from its parent residue
    - Atom type embedding (N, CA, C, O)
    """

    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        c_a: int = 128,
        n_atom_types: int = 4,
        c_pair_proj: int = 32,
    ):
        """
        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            c_a: Atom hidden dimension
            n_atom_types: Number of atom types (N, CA, C, O)
            c_pair_proj: Projected pair dimension for edges
        """
        super().__init__()

        # Atom type embedding
        self.E_atom = nn.Embedding(n_atom_types, 16)

        # MLP to produce atom node features
        self.node_mlp = nn.Sequential(
            nn.Linear(c_s + 16, c_a),
            nn.SiLU(),
            nn.Linear(c_a, c_a),
        )

        # Pair projection for edge features
        self.pair_proj = nn.Linear(c_z, c_pair_proj)

        # Output dimension
        self.c_pair_proj = c_pair_proj

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        atom_to_res: torch.Tensor,
        atom_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
            atom_to_res: [N_atom] residue index per atom
            atom_type: [N_atom] atom type (0-3)
        Returns:
            h: [N_atom, c_a] atom node features
            pair_proj: [L, L, c_pair_proj] projected pair (for edges)
        """
        # Broadcast residue features to atoms
        s_atom = s[atom_to_res]  # [N_atom, c_s]

        # Atom type embedding
        a_embed = self.E_atom(atom_type)  # [N_atom, 16]

        # Combine and project
        h = self.node_mlp(torch.cat([s_atom, a_embed], dim=-1))

        # Project pair representation for edge features
        pair_proj = self.pair_proj(z)

        return h, pair_proj


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        """
        Args:
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to process sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [1] or scalar timestep
        Returns:
            embed: [dim] timestep embedding
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, dtype=torch.float32))
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )

        # Handle scalar or tensor input
        if t.dim() == 0:
            t = t.unsqueeze(0)

        args = t.float() * freqs
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimension
        if self.dim % 2:
            embed = torch.cat([embed, torch.zeros_like(embed[:1])], dim=-1)

        # Project through MLP
        embed = self.mlp(embed)

        return embed.squeeze(0) if embed.size(0) == 1 else embed
