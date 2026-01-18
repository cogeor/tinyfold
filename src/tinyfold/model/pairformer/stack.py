"""Pairformer stack with gradient checkpointing."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from tinyfold.model.pairformer.block import PairformerBlock


class PairformerStack(nn.Module):
    """Stack of Pairformer blocks with optional gradient checkpointing.

    Updates single (s) and pair (z) representations through multiple blocks.
    """

    def __init__(
        self,
        n_blocks: int = 12,
        c_s: int = 256,
        c_z: int = 128,
        n_heads_single: int = 8,
        n_heads_tri: int = 4,
        c_tri_attn: int = 32,
        c_tri_mul: int = 128,
        transition_expansion: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 16,
        use_checkpoint: bool = True,
    ):
        """
        Args:
            n_blocks: Number of Pairformer blocks
            c_s: Single representation dimension
            c_z: Pair representation dimension
            n_heads_single: Attention heads for single stream
            n_heads_tri: Attention heads for triangle attention
            c_tri_attn: Per-head dimension for triangle attention
            c_tri_mul: Hidden dimension for triangle multiplication
            transition_expansion: MLP expansion factor
            dropout: Structured dropout probability
            chunk_size: Chunk size for triangle attention
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            PairformerBlock(
                c_s=c_s,
                c_z=c_z,
                n_heads_single=n_heads_single,
                n_heads_tri=n_heads_tri,
                c_tri_attn=c_tri_attn,
                c_tri_mul=c_tri_mul,
                transition_expansion=transition_expansion,
                dropout=dropout,
                chunk_size=chunk_size,
            )
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
            res_mask: [L] boolean mask for valid residues
        Returns:
            s: [L, c_s] updated single representation
            z: [L, L, c_z] updated pair representation
        """
        # Compute pair mask from residue mask
        pair_mask = res_mask.unsqueeze(1) & res_mask.unsqueeze(0)  # [L, L]

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory
                s, z = checkpoint(
                    self._block_forward,
                    block,
                    s,
                    z,
                    res_mask,
                    pair_mask,
                    use_reentrant=False,
                )
            else:
                s, z = block(s, z, res_mask, pair_mask)

        return s, z

    def _block_forward(
        self,
        block: PairformerBlock,
        s: torch.Tensor,
        z: torch.Tensor,
        res_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for checkpointing."""
        return block(s, z, res_mask, pair_mask)
