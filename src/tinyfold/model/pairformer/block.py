"""Pairformer block combining all sub-layers.

Implements Algorithm 17 from AF3 supplement.
"""

import torch
import torch.nn as nn

from tinyfold.model.pairformer.attn_pair_bias import AttentionPairBias
from tinyfold.model.pairformer.dropout import DropoutColumnwise, DropoutRowwise
from tinyfold.model.pairformer.transition import Transition
from tinyfold.model.pairformer.triangle_attn import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from tinyfold.model.pairformer.triangle_mul import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairformerBlock(nn.Module):
    """Single Pairformer block.

    Order (per AF3 Algorithm 17):
    1. z += DropoutRowwise(TriangleMulOutgoing(z))
    2. z += DropoutRowwise(TriangleMulIncoming(z))
    3. z += DropoutRowwise(TriangleAttnStarting(z))
    4. z += DropoutColumnwise(TriangleAttnEnding(z))
    5. z += Transition(z)
    6. s += AttentionPairBias(s, z)
    7. s += Transition(s)
    """

    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        n_heads_single: int = 8,
        n_heads_tri: int = 4,
        c_tri_attn: int = 32,
        c_tri_mul: int = 128,
        transition_expansion: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 16,
    ):
        """
        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            n_heads_single: Attention heads for single stream
            n_heads_tri: Attention heads for triangle attention
            c_tri_attn: Per-head dimension for triangle attention
            c_tri_mul: Hidden dimension for triangle multiplication
            transition_expansion: MLP expansion factor
            dropout: Structured dropout probability
            chunk_size: Chunk size for triangle attention
        """
        super().__init__()

        # Triangle multiplication
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_tri_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_tri_mul)

        # Triangle attention
        self.tri_attn_start = TriangleAttentionStartingNode(
            c_z, n_heads_tri, c_tri_attn, chunk_size
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            c_z, n_heads_tri, c_tri_attn, chunk_size
        )

        # Pair transition
        self.transition_z = Transition(c_z, transition_expansion)

        # Single attention with pair bias
        self.attn_pair_bias = AttentionPairBias(c_s, c_z, n_heads_single)

        # Single transition
        self.transition_s = Transition(c_s, transition_expansion)

        # Structured dropout
        self.dropout_row = DropoutRowwise(dropout)
        self.dropout_col = DropoutColumnwise(dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        res_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
            res_mask: [L] boolean mask for valid residues
            pair_mask: [L, L] boolean mask for valid pairs
        Returns:
            s: [L, c_s] updated single representation
            z: [L, L, c_z] updated pair representation
        """
        # 1. Triangle multiplication outgoing
        z = z + self.dropout_row(self.tri_mul_out(z, pair_mask), pair_mask)

        # 2. Triangle multiplication incoming
        z = z + self.dropout_row(self.tri_mul_in(z, pair_mask), pair_mask)

        # 3. Triangle attention starting node
        z = z + self.dropout_row(self.tri_attn_start(z, pair_mask, res_mask), pair_mask)

        # 4. Triangle attention ending node
        z = z + self.dropout_col(self.tri_attn_end(z, pair_mask, res_mask), pair_mask)

        # 5. Pair transition
        z = z + self.transition_z(z)
        z = z * pair_mask.unsqueeze(-1)

        # 6. Single attention with pair bias
        s = s + self.attn_pair_bias(s, z, res_mask)

        # 7. Single transition
        s = s + self.transition_s(s)
        s = s * res_mask.unsqueeze(-1)

        return s, z
