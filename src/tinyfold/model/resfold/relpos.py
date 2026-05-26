"""Relative-position attention bias for the residue trunk and denoiser.

The current TinyFold trunk encodes position via absolute sinusoidal features
on ``res_idx``. The ``test_positional_invariance.py`` diagnostic in
``scripts/`` shows this leaks absolute position into the structural prediction:
shifting ``res_idx`` by 5 indices changes the predicted backbone by 5-12 A on
the SAME protein. Per-chain ``res_idx`` reset (already shipped) only fixes the
chain-B-offset half of the bug; the absolute-vs-relative half remains.

This module adds the AlphaFold-Multimer-style relative-position bias as an
ADDITIVE bias on the attention logits (``QK^T + bias``, before softmax). It
is complementary to — not a replacement for — the existing sinusoidal feature:
the sinusoidal gives the trunk a per-token positional anchor, the relpos bias
gives every attention head an inductive bias toward attending to nearby
residues regardless of where in the sequence they happen to live.

Combined with InterfaceCrop-style training, this is the v2 architectural fix
for the cliff (notes/phase_d_stratified_finding.md L170-205 flagged it as the
single highest-priority change; never built until now).

Bucketing:
- Offsets ``res_idx[i] - res_idx[j]`` are clipped to ``+-clip`` (default 32,
  AF-M convention) giving ``2*clip + 1 = 65`` buckets.
- A ``same_chain`` bit (i and j on the same chain or not) gives 2 buckets.
- Total table size: ``[2*clip+1, 2, n_heads]`` -> very small (e.g. 65*2*8 =
  1040 trainable scalars per layer at clip=32, n_heads=8). Trivial vs the
  ~12M model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RelposBias(nn.Module):
    """Learned additive attention bias from (res_idx, chain_ids).

    Returns a ``[B, n_heads, L, L]`` tensor suitable for adding to
    ``F.scaled_dot_product_attention``'s ``attn_mask`` argument when that
    argument is given a float (not bool) tensor.
    """

    def __init__(self, n_heads: int, clip: int = 32):
        super().__init__()
        if clip < 1:
            raise ValueError(f"clip must be >= 1, got {clip}")
        self.n_heads = n_heads
        self.clip = clip
        # Init zero so this layer starts as a no-op on top of the existing
        # absolute encoding. Training will move it.
        self.table = nn.Parameter(
            torch.zeros(2 * clip + 1, 2, n_heads)
        )

    def forward(self, res_idx: Tensor, chain_ids: Tensor) -> Tensor:
        """
        Args:
            res_idx: [B, L] long, GLOBAL residue indices (per-chain-reset).
            chain_ids: [B, L] long, 0 or 1.

        Returns:
            bias: [B, n_heads, L, L] float, added to QK^T before softmax.
        """
        # Offset diff in [-clip, +clip] -> bucket in [0, 2*clip].
        diff = res_idx.unsqueeze(2) - res_idx.unsqueeze(1)  # [B, L, L]
        diff = diff.clamp(-self.clip, self.clip) + self.clip  # [B, L, L]
        same = (chain_ids.unsqueeze(2) == chain_ids.unsqueeze(1)).long()  # [B, L, L]
        # Advanced indexing into the table: result [B, L, L, n_heads].
        bias = self.table[diff, same]
        # Permute to [B, n_heads, L, L] expected by SDPA.
        return bias.permute(0, 3, 1, 2).contiguous()
