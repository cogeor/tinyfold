"""HierarchicalDecoder - Residue-level decoder with atom offset prediction.

This decoder operates at the residue level (L tokens instead of 4L atoms),
then decodes atom offsets from residue representations. This provides:
1. 4x smaller attention matrix (L x L instead of 4L x 4L)
2. Built-in structural prior (atoms decoded from residue representation)
3. Better sample efficiency
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..base import BaseDecoder, sinusoidal_pos_enc


class HierarchicalDecoder(BaseDecoder):
    """Hierarchical decoder: residue-level transformer + atom offset MLP.

    Architecture:
        1. Extract CA positions from input (every 4th atom starting at index 1)
        2. Process at residue level with transformer
        3. Decode 4 atom offsets per residue with MLP
        4. Output = residue_position + atom_offsets

    This maintains the same interface as other decoders (atom-level input/output)
    but internally operates at residue level for efficiency.
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

        # Residue-level embeddings
        self.aa_embed = nn.Embedding(n_aa_types, h_dim)
        self.chain_embed = nn.Embedding(n_chains, h_dim // 4)
        self.time_embed = nn.Embedding(n_timesteps, h_dim)
        self.coord_proj = nn.Linear(3, h_dim)  # For CA position

        # Input projection for residue features
        # aa_emb (h_dim) + chain_emb (h_dim//4) + res_pos (h_dim) + time (h_dim) + coord (h_dim)
        input_dim = h_dim + (h_dim // 4) + h_dim + h_dim + h_dim
        self.input_proj = nn.Linear(input_dim, h_dim)

        # Residue-level transformer (operates on L tokens, not 4L)
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

        # Atom offset decoder: predicts 4 atom positions relative to residue center
        # Input: residue features (h_dim)
        # Output: 4 atoms x 3 coords = 12
        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 12),  # 4 atoms * 3 coords
        )

        # Learnable initial atom offsets (approximate backbone geometry)
        # These provide a good initialization for the atom positions
        self.register_buffer('init_offsets', torch.tensor([
            [-0.5, 0.0, 0.0],   # N offset from CA
            [0.0, 0.0, 0.0],    # CA (center)
            [0.5, 0.0, 0.0],    # C offset from CA
            [0.7, 0.4, 0.0],    # O offset from CA
        ]))

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

        Internally converts to residue level, processes, then converts back.

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
        B, N_atoms, _ = x_t.shape
        N_res = N_atoms // 4

        # Extract CA positions (atom_type=1, every 4th atom starting at 1)
        x_ca = x_t[:, 1::4, :]  # [B, N_res, 3]

        # Extract residue-level features (take from CA atoms)
        aa_res = aa_seq[:, 1::4]        # [B, N_res]
        chain_res = chain_ids[:, 1::4]  # [B, N_res]
        res_idx = atom_to_res[:, 1::4]  # [B, N_res]

        # Residue-level mask
        if mask is not None:
            mask_res = mask[:, 1::4]  # [B, N_res]
        else:
            mask_res = None

        # Compute residue-level embeddings
        aa_emb = self.aa_embed(aa_res)           # [B, N_res, h_dim]
        chain_emb = self.chain_embed(chain_res)  # [B, N_res, h_dim//4]
        res_emb = sinusoidal_pos_enc(res_idx, self.h_dim)  # [B, N_res, h_dim]
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N_res, -1)  # [B, N_res, h_dim]
        coord_emb = self.coord_proj(x_ca)        # [B, N_res, h_dim]

        # Concatenate and project
        h = torch.cat([aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
        h = self.input_proj(h)  # [B, N_res, h_dim]

        # Apply transformer at residue level
        attn_mask = ~mask_res if mask_res is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)  # [B, N_res, h_dim]

        # Decode atom offsets from residue features
        atom_offsets = self.atom_decoder(h)  # [B, N_res, 12]
        atom_offsets = atom_offsets.view(B, N_res, 4, 3)  # [B, N_res, 4, 3]

        # Add initial offset bias for better initialization
        atom_offsets = atom_offsets + self.init_offsets.unsqueeze(0).unsqueeze(0)

        # Predict residue center position (use CA with small correction)
        # The transformer output already incorporates x_ca information
        res_center = x_ca.unsqueeze(2)  # [B, N_res, 1, 3]

        # Final atom positions = residue center + learned offsets
        x0_pred = res_center + atom_offsets  # [B, N_res, 4, 3]
        x0_pred = x0_pred.view(B, N_atoms, 3)  # [B, N_atoms, 3]

        return x0_pred
