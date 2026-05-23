"""Iterative Atom Assembler: Predict positions for new atoms given known atoms.

This is Stage 3 of the ResFold E2E pipeline. Given a partial structure with
some atoms already placed, predict positions for the next K atoms.

Key differences from AtomRefinerV2MultiSample:
- Takes known_coords + known_mask instead of centroid samples
- Cross-attends from target atoms to known atoms
- Predicts relative positions (distances) rather than absolute coords
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IterativeAtomAssembler(nn.Module):
    """Predict positions for K new atoms given known atom positions.

    Architecture:
    1. Encode known atoms into context embeddings
    2. Create query embeddings for target atoms (from trunk_tokens)
    3. Cross-attention: target queries attend to known context
    4. Output: predicted coordinates for target atoms

    Parameters:
        c_token: Hidden dimension (matches Stage 1/2)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers

        # Embed known atom coordinates [3] -> [c_token]
        self.known_coord_embed = nn.Sequential(
            nn.Linear(3, c_token),
            nn.LayerNorm(c_token),
            nn.GELU(),
            nn.Linear(c_token, c_token),
        )

        # Embed target atom queries (will use trunk_tokens as base)
        # Additional embedding for "target" vs "known" distinction
        self.target_type_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)
        self.known_type_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)

        # Atom type embedding (N=0, CA=1, C=2, O=3)
        self.atom_type_embed = nn.Embedding(4, c_token)

        # Cross-attention layers: target attends to known
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                c_token, n_heads, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])

        # Self-attention layers for target refinement
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                c_token, n_heads, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])

        # FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_token),
                nn.Linear(c_token, c_token * 4),
                nn.GELU(),
                nn.Linear(c_token * 4, c_token),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        # Layer norms
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(c_token) for _ in range(n_layers)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(c_token) for _ in range(n_layers)
        ])

        # Output projection: predict 3D coordinates
        self.output_norm = nn.LayerNorm(c_token)
        self.output_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        trunk_tokens: Tensor,      # [B, L, c_token] from Stage 1 encoder
        known_coords: Tensor,      # [B, M, 3] coordinates of known atoms
        known_mask: Tensor,        # [B, M] bool, True = valid known atom
        target_atom_idx: Tensor,   # [B, K] indices of atoms to predict
        target_res_idx: Tensor,    # [B, K] residue index for each target atom
    ) -> Tensor:
        """Predict coordinates for target atoms.

        Args:
            trunk_tokens: Pre-computed sequence embeddings [B, L, c_token]
            known_coords: Coordinates of already-placed atoms [B, M, 3]
            known_mask: Mask for valid known atoms [B, M]
            target_atom_idx: Which atoms to predict [B, K]
            target_res_idx: Residue index for each target (for trunk lookup) [B, K]

        Returns:
            pred_coords: Predicted coordinates for target atoms [B, K, 3]
        """
        B, L, C = trunk_tokens.shape
        K = target_atom_idx.shape[1]
        M = known_coords.shape[1]
        device = trunk_tokens.device

        # === Build known context ===
        # Embed known coordinates
        known_emb = self.known_coord_embed(known_coords)  # [B, M, c_token]
        known_emb = known_emb + self.known_type_embed

        # === Build target queries ===
        # Gather trunk tokens for target residues
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        target_trunk = trunk_tokens[batch_idx, target_res_idx.clamp(0, L-1)]  # [B, K, C]

        # Add atom type embedding (atom_idx % 4 gives atom type: N=0, CA=1, C=2, O=3)
        atom_types = target_atom_idx % 4  # [B, K]
        atom_type_emb = self.atom_type_embed(atom_types)  # [B, K, C]

        target_queries = target_trunk + atom_type_emb + self.target_type_embed

        # === Transformer layers ===
        # Key padding mask for known atoms (True = ignore in attention)
        known_key_pad = ~known_mask  # [B, M]

        for i in range(self.n_layers):
            # Cross-attention: targets attend to known
            q = self.cross_norms[i](target_queries)
            k = v = known_emb

            # Handle empty known case
            if M > 0 and known_mask.any():
                attn_out, _ = self.cross_attn_layers[i](
                    q, k, v, key_padding_mask=known_key_pad
                )
                target_queries = target_queries + attn_out

            # Self-attention among targets
            q = self.self_norms[i](target_queries)
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            target_queries = target_queries + attn_out

            # FFN
            target_queries = target_queries + self.ffn_layers[i](target_queries)

        # === Output ===
        target_queries = self.output_norm(target_queries)
        pred_coords = self.output_proj(target_queries)  # [B, K, 3]

        return pred_coords

    def forward_with_anchor(
        self,
        trunk_tokens: Tensor,      # [B, L, c_token]
        known_coords: Tensor,      # [B, M, 3]
        known_mask: Tensor,        # [B, M]
        target_atom_idx: Tensor,   # [B, K]
        target_res_idx: Tensor,    # [B, K]
        anchor_coords: Tensor,     # [B, 3] anchor point (e.g., centroid of known)
    ) -> Tensor:
        """Predict coordinates relative to an anchor point.

        Same as forward() but adds anchor_coords to output.
        Useful when predicting in a local coordinate frame.
        """
        pred_offsets = self.forward(
            trunk_tokens, known_coords, known_mask,
            target_atom_idx, target_res_idx
        )
        return pred_offsets + anchor_coords.unsqueeze(1)

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'n_layers': self.n_layers,
            'c_token': self.c_token,
        }
