"""AtomRefinerV2 with Multi-Sample Centroid Conditioning.

Stage 2 that accepts K diffusion samples from Stage 1 and aggregates them
to produce atom predictions. This allows the model to leverage multiple
noisy predictions for more robust atom placement.

Architecture:
- Input: trunk_tokens [B, L, c_token] + centroids_samples [B, K, L, 3]
- Embed each sample, aggregate across K dimension
- Transformer layers for sequence processing
- Output: atom offsets [B, L, 4, 3]
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


AggregationType = Literal["learned", "mean", "attention"]


class AtomRefinerV2MultiSample(nn.Module):
    """Stage 2 with multi-sample centroid conditioning.

    Takes K centroid samples from Stage 1 diffusion and learns to aggregate
    them for robust atom prediction.

    Parameters (~5M with defaults):
    - c_token=256, n_layers=6: ~4.8M params
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_samples: int = 5,
        dropout: float = 0.0,
        aggregation: AggregationType = "learned",
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers
        self.n_samples = n_samples
        self.aggregation = aggregation

        # Embed centroid positions (same embedding for all K samples)
        self.centroid_embed = nn.Linear(3, c_token)

        # Sample aggregation
        if aggregation == "learned":
            # Learned weighted combination of K samples
            self.sample_weights = nn.Linear(n_samples, 1, bias=False)
            # Initialize to uniform weights
            nn.init.constant_(self.sample_weights.weight, 1.0 / n_samples)
        elif aggregation == "attention":
            # Cross-attention to aggregate samples
            self.sample_query = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)
            self.sample_attn = nn.MultiheadAttention(
                c_token, n_heads, dropout=dropout, batch_first=True
            )
        # "mean" aggregation needs no learnable parameters

        # Transformer layers (same style as AtomRefinerV2)
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

        # Output: predict 4 atoms x 3 coords = 12 values per residue
        self.output_proj = nn.Linear(c_token, 4 * 3)

    def _aggregate_samples(
        self,
        sample_emb: Tensor,  # [B, K, L, c_token]
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Aggregate K sample embeddings into single embedding per residue.

        Args:
            sample_emb: Embedded centroid samples [B, K, L, c_token]
            mask: Valid residue mask [B, L]

        Returns:
            agg_emb: Aggregated embeddings [B, L, c_token]
        """
        B, K, L, C = sample_emb.shape

        if self.aggregation == "mean":
            # Simple average across samples
            return sample_emb.mean(dim=1)  # [B, L, c_token]

        elif self.aggregation == "learned":
            # Learned weighted combination
            # [B, K, L, C] -> [B, L, C, K] -> apply weights -> [B, L, C]
            sample_emb_t = sample_emb.permute(0, 2, 3, 1)  # [B, L, C, K]
            agg_emb = self.sample_weights(sample_emb_t).squeeze(-1)  # [B, L, C]
            return agg_emb

        elif self.aggregation == "attention":
            # Cross-attention: query attends to K samples per residue
            # Reshape for attention: [B*L, K, C]
            sample_emb_flat = sample_emb.permute(0, 2, 1, 3).reshape(B * L, K, C)

            # Expand query for all residues
            query = self.sample_query.expand(B * L, 1, C)

            # Cross-attention
            agg_flat, _ = self.sample_attn(query, sample_emb_flat, sample_emb_flat)
            agg_emb = agg_flat.view(B, L, C)  # [B, L, c_token]
            return agg_emb

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(
        self,
        trunk_tokens: Tensor,       # [B, L, c_token] from Stage 1 encoder
        centroids_samples: Tensor,  # [B, K, L, 3] K centroid samples
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict atom positions from trunk embeddings + multi-sample centroids.

        Args:
            trunk_tokens: Embeddings from Stage 1's trunk encoder
            centroids_samples: K centroid samples from diffusion
            mask: Valid residue mask

        Returns:
            atom_coords: [B, L, 4, 3] absolute atom positions
        """
        B, K, L, _ = centroids_samples.shape
        device = centroids_samples.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Embed each centroid sample
        # [B, K, L, 3] -> [B, K, L, c_token]
        sample_emb = self.centroid_embed(centroids_samples)

        # Aggregate across K samples
        agg_emb = self._aggregate_samples(sample_emb, mask)  # [B, L, c_token]

        # Combine with trunk tokens (additive, like AtomRefinerV2)
        tokens = trunk_tokens + agg_emb  # [B, L, c_token]

        # Transformer
        attn_mask = ~mask
        tokens = self.transformer(tokens, src_key_padding_mask=attn_mask)
        tokens = self.final_norm(tokens)

        # Output: 4 atoms x 3 coords
        offsets = self.output_proj(tokens)  # [B, L, 12]
        offsets = offsets.view(B, L, 4, 3)  # [B, L, 4, 3]

        # Get mean centroid as anchor for atom positions
        mean_centroid = centroids_samples.mean(dim=1)  # [B, L, 3]
        atom_coords = mean_centroid.unsqueeze(2) + offsets  # [B, L, 4, 3]

        return atom_coords

    def forward_offsets(
        self,
        trunk_tokens: Tensor,
        centroids_samples: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return only offsets (for debugging/analysis)."""
        B, K, L, _ = centroids_samples.shape
        device = centroids_samples.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        sample_emb = self.centroid_embed(centroids_samples)
        agg_emb = self._aggregate_samples(sample_emb, mask)
        tokens = trunk_tokens + agg_emb

        attn_mask = ~mask
        tokens = self.transformer(tokens, src_key_padding_mask=attn_mask)
        tokens = self.final_norm(tokens)

        offsets = self.output_proj(tokens).view(B, L, 4, 3)
        return offsets

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'n_layers': self.n_layers,
            'c_token': self.c_token,
            'n_samples': self.n_samples,
            'aggregation': self.aggregation,
        }
