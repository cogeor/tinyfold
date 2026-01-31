"""ResFold Stage 2 V2: Large Atom Refinement Network.

Single-step prediction of 4 backbone atom positions from:
1. Trunk embeddings from Stage 1 encoder (reused, not recomputed)
2. Residue centroid positions

Architecture:
- Input: trunk_tokens [B, L, c_token] + centroids [B, L, 3]
- Transformer layers (~15M params)
- Output: relative atom offsets [B, L, 4, 3]
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AtomRefinerV2(nn.Module):
    """Stage 2: Large atom refinement from trunk embeddings + centroids.

    Takes trunk embeddings from Stage 1 encoder (not recomputed),
    plus centroid positions, and predicts relative atom positions.

    Architecture:
    1. Combine trunk tokens + centroid embeddings (additive)
    2. Transformer layers (similar to Stage 1)
    3. Output 4 relative atom positions per residue

    Parameters (~15M with defaults):
    - c_token=256, n_layers=18: ~14.2M params
    """

    def __init__(
        self,
        c_token: int = 256,      # Match Stage 1
        n_layers: int = 18,      # For ~15M params
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers

        # Embed centroid positions
        self.centroid_embed = nn.Linear(3, c_token)

        # Transformer layers (same style as Stage 1 trunk)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token,
            nhead=n_heads,
            dim_feedforward=c_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(c_token)

        # Output: predict 4 atoms × 3 coords = 12 values per residue
        self.output_proj = nn.Linear(c_token, 4 * 3)

    def forward(
        self,
        trunk_tokens: Tensor,  # [B, L, c_token] from Stage 1 encoder
        centroids: Tensor,     # [B, L, 3] centroid positions
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict relative atom positions from trunk embeddings + centroids.

        Args:
            trunk_tokens: Embeddings from Stage 1's trunk encoder
            centroids: Residue centroid positions (possibly with noise)
            mask: Valid residue mask

        Returns:
            atom_offsets: [B, L, 4, 3] relative positions from centroid
        """
        B, L, _ = centroids.shape
        device = centroids.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Embed centroids
        centroid_emb = self.centroid_embed(centroids)  # [B, L, c_token]

        # Combine with trunk tokens (additive, like Stage 1 denoiser)
        tokens = trunk_tokens + centroid_emb  # [B, L, c_token]

        # Transformer
        attn_mask = ~mask
        tokens = self.transformer(tokens, src_key_padding_mask=attn_mask)
        tokens = self.final_norm(tokens)

        # Output: 4 atoms × 3 coords
        offsets = self.output_proj(tokens)  # [B, L, 12]
        offsets = offsets.view(B, L, 4, 3)  # [B, L, 4, 3]

        return offsets

    def forward_with_coords(
        self,
        trunk_tokens: Tensor,  # [B, L, c_token] from Stage 1 encoder
        centroids: Tensor,     # [B, L, 3] centroid positions
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict absolute atom positions (centroid + offset).

        Returns:
            atom_coords: [B, L, 4, 3] absolute atom positions
        """
        offsets = self.forward(trunk_tokens, centroids, mask)
        # Add centroid to get absolute positions
        atom_coords = centroids.unsqueeze(2) + offsets  # [B, L, 4, 3]
        return atom_coords

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'layers': self.n_layers,
            'c_token': self.c_token,
        }
