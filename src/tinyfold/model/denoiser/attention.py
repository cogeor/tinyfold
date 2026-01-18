"""Attention-based denoiser (replaces EGNN).

Uses transformer attention with pair bias for coordinate prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDenoiserLayer(nn.Module):
    """Single attention layer with pair bias for coordinate denoising."""

    def __init__(
        self,
        h_dim: int = 128,
        c_z: int = 128,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads

        # Self-attention with pair bias
        self.ln1 = nn.LayerNorm(h_dim)
        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.o_proj = nn.Linear(h_dim, h_dim)

        # Pair bias projection (z -> attention bias)
        self.pair_bias = nn.Linear(c_z, n_heads)

        # FFN
        self.ln2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 4, h_dim),
            nn.Dropout(dropout),
        )

        # Coordinate update (predict delta per atom)
        self.coord_ln = nn.LayerNorm(h_dim)
        self.coord_proj = nn.Linear(h_dim, 3)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        atom_to_res: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [N, h_dim] atom features
            x: [N, 3] atom coordinates
            z: [L, L, c_z] pair representation
            atom_to_res: [N] residue index for each atom
            mask: [N] boolean mask
        Returns:
            h_new: [N, h_dim] updated features
            x_new: [N, 3] updated coordinates
        """
        N = h.size(0)

        # Self-attention with pair bias
        h_norm = self.ln1(h)
        q = self.q_proj(h_norm).view(N, self.n_heads, self.head_dim)
        k = self.k_proj(h_norm).view(N, self.n_heads, self.head_dim)
        v = self.v_proj(h_norm).view(N, self.n_heads, self.head_dim)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('ihd,jhd->ijh', q, k) * scale  # [N, N, heads]

        # Add pair bias (lookup from residue pair representation)
        res_i = atom_to_res.unsqueeze(1).expand(-1, N)  # [N, N]
        res_j = atom_to_res.unsqueeze(0).expand(N, -1)  # [N, N]
        z_ij = z[res_i, res_j]  # [N, N, c_z]
        pair_bias = self.pair_bias(z_ij)  # [N, N, n_heads]
        attn = attn + pair_bias

        # Mask
        if mask is not None:
            attn_mask = mask.unsqueeze(0).expand(N, -1)  # [N, N]
            attn = attn.masked_fill(~attn_mask.unsqueeze(-1), float('-inf'))

        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum('ijh,jhd->ihd', attn, v)
        out = out.reshape(N, self.h_dim)
        h = h + self.o_proj(out)

        # FFN
        h = h + self.ffn(self.ln2(h))

        # Coordinate update
        coord_delta = self.coord_proj(self.coord_ln(h))
        x = x + coord_delta

        return h, x


class AttentionDenoiser(nn.Module):
    """Stack of attention layers for coordinate denoising.

    Replaces EGNN with pure attention + pair bias from Pairformer.
    """

    def __init__(
        self,
        n_layers: int = 4,
        h_dim: int = 128,
        c_z: int = 128,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionDenoiserLayer(h_dim, c_z, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        z: torch.Tensor,
        atom_to_res: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [N, h_dim] atom conditioning features
            x_t: [N, 3] noisy coordinates
            z: [L, L, c_z] pair representation
            atom_to_res: [N] residue index for each atom
            mask: [N] boolean mask
        Returns:
            x0_pred: [N, 3] predicted clean coordinates
        """
        x = x_t.clone()
        for layer in self.layers:
            h, x = layer(h, x, z, atom_to_res, mask)
        return x
