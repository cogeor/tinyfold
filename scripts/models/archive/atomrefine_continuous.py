"""AtomRefinerContinuous: Stage 2 Atom Refinement from Noisy Centroids.

Single-step prediction of 4 backbone atom positions from noisy centroids.
No diffusion - direct regression with local atom attention.

Architecture:
1. Trunk encoder (sequence features only, like Stage 1)
2. Centroid embedding (noisy centroid positions)
3. Refinement transformer with local atom attention
4. Output: atom offsets from centroid

Target: ~5M parameters
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import sinusoidal_pos_enc


# =============================================================================
# Local Atom Attention (from atomrefine.py)
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
# Trunk Encoder (sequence features only)
# =============================================================================

class TrunkEncoder(nn.Module):
    """Residue-level encoder that processes sequence features only.

    Similar to Stage 1's ResidueEncoder but smaller for ~5M total params.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token

        # Residue embeddings
        self.aa_embed = nn.Embedding(n_aa_types, c_token)
        self.chain_embed = nn.Embedding(n_chains, c_token // 4)

        # Input projection: aa + chain + pos
        input_dim = c_token + (c_token // 4) + c_token
        self.input_proj = nn.Linear(input_dim, c_token)

        # Transformer
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

        self.output_norm = nn.LayerNorm(c_token)

    def forward(
        self,
        aa_seq: Tensor,          # [B, L]
        chain_ids: Tensor,       # [B, L]
        res_idx: Tensor,         # [B, L]
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Encode sequence features (no coordinates).

        Returns:
            tokens: [B, L, c_token]
        """
        # Embeddings
        aa_emb = self.aa_embed(aa_seq)
        chain_emb = self.chain_embed(chain_ids)
        res_emb = sinusoidal_pos_enc(res_idx, self.c_token)

        # Concatenate and project
        h = torch.cat([aa_emb, chain_emb, res_emb], dim=-1)
        h = self.input_proj(h)

        # Transformer
        attn_mask = ~mask if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        return self.output_norm(h)


# =============================================================================
# Main Model: AtomRefinerContinuous
# =============================================================================

class AtomRefinerContinuous(nn.Module):
    """Stage 2: Atom refinement from noisy centroids.

    Single forward pass (no diffusion) to predict 4 backbone atoms per residue.

    Architecture:
    1. Trunk encoder: sequence features -> token embeddings
    2. Centroid embedding: noisy centroids -> coordinate embeddings
    3. Refinement transformer: global context
    4. Local atom attention: within-residue atom refinement
    5. Output: atom offsets from centroid

    Target: ~5M parameters with default settings.
    """

    def __init__(
        self,
        c_token: int = 256,
        c_atom: int = 128,
        trunk_layers: int = 4,
        refine_layers: int = 4,
        local_atom_blocks: int = 2,
        n_heads: int = 8,
        n_atom_heads: int = 4,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom

        # === TRUNK ENCODER ===
        self.trunk = TrunkEncoder(
            c_token=c_token,
            n_layers=trunk_layers,
            n_heads=n_heads,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # === CENTROID EMBEDDING ===
        self.centroid_proj = nn.Linear(3, c_token)

        # === REFINEMENT TRANSFORMER ===
        # Takes trunk tokens + centroid embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token,
            nhead=n_heads,
            dim_feedforward=c_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.refine_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=refine_layers, enable_nested_tensor=False
        )
        self.refine_norm = nn.LayerNorm(c_token)

        # === LOCAL ATOM REFINEMENT ===
        # Broadcast residue tokens to atoms
        self.broadcast_proj = nn.Linear(c_token, c_atom)

        # Atom type embedding (N=0, CA=1, C=2, O=3)
        self.atom_type_embed = nn.Embedding(4, c_atom)

        # Initial atom position offsets (from centroid, approximate backbone geometry)
        # Scaled for normalized coordinates (divide by ~10)
        self.register_buffer('init_offsets', torch.tensor([
            [-0.146, 0.0, 0.0],    # N relative to centroid
            [0.0, 0.0, 0.0],       # CA (roughly at centroid)
            [0.152, 0.0, 0.0],     # C
            [0.24, 0.10, 0.0],     # O
        ], dtype=torch.float32))

        self.init_offset_proj = nn.Linear(3, c_atom)

        # Combine: broadcast tokens + atom type + init offset
        self.atom_combine = nn.Linear(c_atom * 3, c_atom)

        # Local atom transformer blocks
        self.local_blocks = nn.ModuleList([
            LocalAtomTransformerBlock(c_atom, n_atom_heads, dropout=dropout)
            for _ in range(local_atom_blocks)
        ])

        # === OUTPUT ===
        self.output_norm = nn.LayerNorm(c_atom)
        self.output_proj = nn.Linear(c_atom, 3)

    def forward(
        self,
        aa_seq: Tensor,          # [B, L]
        chain_ids: Tensor,       # [B, L]
        res_idx: Tensor,         # [B, L]
        centroids: Tensor,       # [B, L, 3] noisy centroids
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict atom positions from noisy centroids.

        Args:
            aa_seq: Amino acid indices [B, L]
            chain_ids: Chain IDs (0 or 1) [B, L]
            res_idx: Residue indices within chain [B, L]
            centroids: Noisy centroid positions [B, L, 3]
            mask: Valid residue mask [B, L]

        Returns:
            atom_coords: [B, L, 4, 3] backbone atom positions
        """
        B, L, _ = centroids.shape
        device = centroids.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # === TRUNK ENCODING ===
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)  # [B, L, c_token]

        # === CENTROID EMBEDDING ===
        centroid_emb = self.centroid_proj(centroids)  # [B, L, c_token]

        # === REFINEMENT TRANSFORMER ===
        tokens = trunk_tokens + centroid_emb  # [B, L, c_token]
        attn_mask = ~mask
        tokens = self.refine_transformer(tokens, src_key_padding_mask=attn_mask)
        tokens = self.refine_norm(tokens)  # [B, L, c_token]

        # === LOCAL ATOM REFINEMENT ===
        # Broadcast to atoms: [B, L, c_token] -> [B, L, 4, c_atom]
        atom_tokens = self.broadcast_proj(tokens).unsqueeze(2).expand(-1, -1, 4, -1)

        # Atom type embeddings
        atom_types = torch.arange(4, device=device)
        atom_type_emb = self.atom_type_embed(atom_types).view(1, 1, 4, -1)
        atom_type_emb = atom_type_emb.expand(B, L, -1, -1)

        # Initial offset embeddings
        init_offset_emb = self.init_offset_proj(self.init_offsets).view(1, 1, 4, -1)
        init_offset_emb = init_offset_emb.expand(B, L, -1, -1)

        # Combine all atom features
        atom_h = torch.cat([atom_tokens, atom_type_emb, init_offset_emb], dim=-1)
        atom_h = self.atom_combine(atom_h)  # [B, L, 4, c_atom]

        # Atom mask from residue mask
        atom_mask = mask.unsqueeze(-1).expand(-1, -1, 4)  # [B, L, 4]

        # Local atom transformer
        for block in self.local_blocks:
            atom_h = block(atom_h, atom_mask)

        # Output: predict offsets from centroid
        atom_h = self.output_norm(atom_h)
        offsets = self.output_proj(atom_h)  # [B, L, 4, 3]

        # Add centroid to get absolute positions
        atom_coords = centroids.unsqueeze(2) + offsets  # [B, L, 4, 3]

        return atom_coords

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        centroid_params = sum(p.numel() for p in self.centroid_proj.parameters())
        refine_params = sum(p.numel() for p in self.refine_transformer.parameters())
        refine_params += sum(p.numel() for p in self.refine_norm.parameters())
        local_params = sum(p.numel() for p in self.broadcast_proj.parameters())
        local_params += sum(p.numel() for p in self.atom_type_embed.parameters())
        local_params += sum(p.numel() for p in self.init_offset_proj.parameters())
        local_params += sum(p.numel() for p in self.atom_combine.parameters())
        for block in self.local_blocks:
            local_params += sum(p.numel() for p in block.parameters())
        output_params = sum(p.numel() for p in self.output_norm.parameters())
        output_params += sum(p.numel() for p in self.output_proj.parameters())

        total = trunk_params + centroid_params + refine_params + local_params + output_params

        return {
            'trunk': trunk_params,
            'centroid': centroid_params,
            'refine': refine_params,
            'local_atom': local_params,
            'output': output_params,
            'total': total,
        }
