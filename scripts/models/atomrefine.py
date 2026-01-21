"""ResFold Stage 2: Atom Refinement Network.

One-shot prediction of 4 backbone atom positions from residue centroids.
Uses local atom attention (from AF3-style) for within-residue refinement.

Architecture:
1. Input embedding (centroid + sequence + chain + position)
2. Global transformer for inter-residue context
3. LocalAtomTransformer for within-residue atom prediction
4. Output: 4 atom offsets per residue
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import sinusoidal_pos_enc


# =============================================================================
# Local Atom Attention (copied from af3_style.py)
# =============================================================================

class LocalAtomAttention(nn.Module):
    """Local attention within each residue's atoms.

    Attention is masked to only attend within the same residue (4 atoms per residue).
    This keeps memory bounded: O(L * 4^2) instead of O((4L)^2).
    """

    def __init__(
        self,
        c_atom: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.n_heads = n_heads
        self.head_dim = c_atom // n_heads

        self.q_proj = nn.Linear(c_atom, c_atom)
        self.k_proj = nn.Linear(c_atom, c_atom)
        self.v_proj = nn.Linear(c_atom, c_atom)
        self.out_proj = nn.Linear(c_atom, c_atom)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,  # [B, L, 4, c_atom]
        mask: Optional[Tensor] = None,  # [B, L, 4] boolean
    ) -> Tensor:
        """Local attention within each residue."""
        B, L, n_atoms, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [B, L, n_heads, 4, head_dim]
        q = q.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)
        k = k.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)
        v = v.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)

        # Attention scores within each residue
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_mask = mask.unsqueeze(2).unsqueeze(3)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).reshape(B, L, n_atoms, self.c_atom)

        out = self.out_proj(out)

        if mask is not None:
            residue_mask = mask.any(dim=-1, keepdim=True).unsqueeze(-1)
            out = out * residue_mask.float()

        return out


class LocalAtomTransformerBlock(nn.Module):
    """Single block of local atom transformer."""

    def __init__(
        self,
        c_atom: int,
        n_heads: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(c_atom)
        self.attn = LocalAtomAttention(c_atom, n_heads, dropout)
        self.norm2 = nn.LayerNorm(c_atom)

        # FFN
        hidden = c_atom * expansion
        self.ffn = nn.Sequential(
            nn.Linear(c_atom, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c_atom),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Atom Refiner (Stage 2 Main Model)
# =============================================================================

class AtomRefiner(nn.Module):
    """Stage 2: Atom refinement from residue centroids.

    Given residue centroid positions, predicts 4 backbone atom positions
    (N, CA, C, O) for each residue using local atom attention.

    Architecture:
    1. Global context encoder (transformer on residue tokens)
    2. Broadcast to atoms + LocalAtomTransformer
    3. Output 4 atom offsets per residue
    """

    def __init__(
        self,
        c_token: int = 128,
        c_atom: int = 64,
        n_global_layers: int = 4,
        n_local_blocks: int = 3,
        n_heads: int = 4,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom

        # === INPUT EMBEDDING ===
        self.coord_proj = nn.Linear(3, c_token // 2)
        self.aa_embed = nn.Embedding(n_aa_types, c_token)
        self.chain_embed = nn.Embedding(n_chains, c_token // 4)

        # Input projection
        input_dim = (c_token // 2) + c_token + (c_token // 4) + c_token
        self.input_proj = nn.Linear(input_dim, c_token)

        # === GLOBAL CONTEXT ENCODER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token,
            nhead=n_heads,
            dim_feedforward=c_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_global_layers, enable_nested_tensor=False
        )
        self.global_norm = nn.LayerNorm(c_token)

        # === LOCAL ATOM REFINEMENT ===

        # Broadcast residue tokens to atoms
        self.broadcast_proj = nn.Linear(c_token, c_atom)

        # Atom type embedding (N=0, CA=1, C=2, O=3)
        self.atom_type_embed = nn.Embedding(4, c_atom)

        # Initial atom position embedding (offsets from centroid)
        # Standard backbone atom offsets (approximate)
        self.register_buffer('init_offsets', torch.tensor([
            [-1.458, 0.0, 0.0],   # N relative to centroid
            [0.0, 0.0, 0.0],      # CA (roughly at centroid)
            [1.524, 0.0, 0.0],    # C
            [2.4, 1.0, 0.0],      # O
        ], dtype=torch.float32) * 0.3)  # Scale down for normalized coords

        self.init_offset_proj = nn.Linear(3, c_atom)

        # Combine all atom features
        self.atom_combine = nn.Linear(c_atom * 3, c_atom)

        # Local atom transformer blocks
        self.local_blocks = nn.ModuleList([
            LocalAtomTransformerBlock(c_atom, n_heads, dropout=dropout)
            for _ in range(n_local_blocks)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(c_atom)
        self.output_proj = nn.Linear(c_atom, 3)

    def forward(
        self,
        centroids: Tensor,   # [B, L, 3] residue centroids
        aa_seq: Tensor,      # [B, L]
        chain_ids: Tensor,   # [B, L]
        res_idx: Tensor,     # [B, L]
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict atom positions from residue centroids.

        Returns:
            atom_coords: [B, L, 4, 3] backbone atom positions
        """
        B, L, _ = centroids.shape
        device = centroids.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # === INPUT EMBEDDING ===
        coord_emb = self.coord_proj(centroids)
        aa_emb = self.aa_embed(aa_seq)
        chain_emb = self.chain_embed(chain_ids)
        pos_emb = sinusoidal_pos_enc(res_idx, self.c_token)

        h = torch.cat([coord_emb, aa_emb, chain_emb, pos_emb], dim=-1)
        tokens = self.input_proj(h)

        # === GLOBAL CONTEXT ===
        attn_mask = ~mask
        tokens = self.global_encoder(tokens, src_key_padding_mask=attn_mask)
        tokens = self.global_norm(tokens)

        # === LOCAL ATOM REFINEMENT ===

        # Broadcast residue tokens to atoms: [B, L, c_token] -> [B, L, 4, c_atom]
        atom_tokens = self.broadcast_proj(tokens).unsqueeze(2).expand(-1, -1, 4, -1)

        # Atom type embeddings: [4, c_atom] -> [1, 1, 4, c_atom]
        atom_types = torch.arange(4, device=device)
        atom_type_emb = self.atom_type_embed(atom_types).view(1, 1, 4, -1)
        atom_type_emb = atom_type_emb.expand(B, L, -1, -1)

        # Initial offset embeddings
        init_offset_emb = self.init_offset_proj(self.init_offsets).view(1, 1, 4, -1)
        init_offset_emb = init_offset_emb.expand(B, L, -1, -1)

        # Combine
        atom_h = torch.cat([atom_tokens, atom_type_emb, init_offset_emb], dim=-1)
        atom_h = self.atom_combine(atom_h)  # [B, L, 4, c_atom]

        # Create atom mask from residue mask
        atom_mask = mask.unsqueeze(-1).expand(-1, -1, 4)  # [B, L, 4]

        # Local atom transformer
        for block in self.local_blocks:
            atom_h = block(atom_h, atom_mask)

        # Output: predict offsets from centroid
        atom_h = self.output_norm(atom_h)
        offsets = self.output_proj(atom_h)  # [B, L, 4, 3]

        # Add centroid to get absolute positions
        atom_coords = centroids.unsqueeze(2) + offsets

        return atom_coords

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
