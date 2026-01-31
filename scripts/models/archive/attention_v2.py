"""AttentionDiffusionV2 - Original attention-based diffusion decoder.

This is the baseline model that operates on all atoms (4 per residue).
Kept for backwards compatibility and comparison.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseDecoder, sinusoidal_pos_enc


class AttentionDiffusionV2(BaseDecoder):
    """Attention-based diffusion decoder operating on all atoms.

    This model treats each atom as a token and uses a Transformer encoder
    to predict clean coordinates from noisy coordinates.

    Attributes:
        h_dim: Hidden dimension
        n_timesteps: Number of diffusion timesteps
    """

    def __init__(
        self,
        h_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_timesteps: int = 50,
        dropout: float = 0.0,
        n_aa_types: int = 21,
        n_chains: int = 2,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.n_timesteps = n_timesteps

        # Embeddings
        self.atom_type_embed = nn.Embedding(4, h_dim // 4)
        self.aa_embed = nn.Embedding(n_aa_types, h_dim)
        self.chain_embed = nn.Embedding(n_chains, h_dim // 4)
        self.time_embed = nn.Embedding(n_timesteps + 1, h_dim)  # +1 for t=T in linear_chain
        self.coord_proj = nn.Linear(3, h_dim)

        # Input projection
        input_dim = (h_dim // 4) + h_dim + (h_dim // 4) + h_dim + h_dim + h_dim
        self.input_proj = nn.Linear(input_dim, h_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=n_heads,
            dim_feedforward=h_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Output
        self.output_norm = nn.LayerNorm(h_dim)
        self.output_proj = nn.Linear(h_dim, 3)

    def forward(
        self,
        x_t: Tensor,
        atom_types: Tensor,
        atom_to_res: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        t: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """Predict clean coordinates from noisy coordinates.

        Args:
            x_t: Noisy coordinates [B, N_atoms, 3]
            atom_types: Atom types [B, N_atoms]
            atom_to_res: Residue index for each atom [B, N_atoms]
            aa_seq: Amino acid type for each atom [B, N_atoms]
            chain_ids: Chain ID for each atom [B, N_atoms]
            t: Diffusion timestep [B]
            mask: Valid atom mask [B, N_atoms]

        Returns:
            x0_pred: Predicted clean coordinates [B, N_atoms, 3]
        """
        B, N, _ = x_t.shape

        # Compute embeddings
        atom_emb = self.atom_type_embed(atom_types)
        aa_emb = self.aa_embed(aa_seq)
        chain_emb = self.chain_embed(chain_ids)
        res_emb = sinusoidal_pos_enc(atom_to_res, self.h_dim)
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)
        coord_emb = self.coord_proj(x_t)

        # Concatenate and project
        h = torch.cat([atom_emb, aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
        h = self.input_proj(h)

        # Apply transformer with attention mask for padding
        attn_mask = ~mask if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Output projection
        h = self.output_norm(h)
        return self.output_proj(h)
