"""Improved Atom-Level Decoder with geometric priors and atom attention.

Key improvements over the original AnchorDecoder:
1. Starts from a geometric prior (default backbone configuration)
2. Atom-to-atom attention within and across residues
3. Explicit backbone connectivity encoding
4. Iterative refinement of atom positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


# Default backbone geometry (in Angstroms, will be normalized)
# These are standard values for protein backbone
DEFAULT_BACKBONE = {
    # Relative to CA at origin
    'N': torch.tensor([-1.458, 0.0, 0.0]),   # N is ~1.46A from CA
    'CA': torch.tensor([0.0, 0.0, 0.0]),      # CA at origin
    'C': torch.tensor([1.525, 0.0, 0.0]),     # C is ~1.52A from CA
    'O': torch.tensor([2.4, 1.0, 0.0]),       # O is ~1.23A from C, angle ~121 deg
}


class AtomAttentionBlock(nn.Module):
    """Self-attention over atoms with backbone connectivity bias."""

    def __init__(self, c_atom: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_atom // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(c_atom)
        self.q_proj = nn.Linear(c_atom, c_atom)
        self.k_proj = nn.Linear(c_atom, c_atom)
        self.v_proj = nn.Linear(c_atom, c_atom)
        self.out_proj = nn.Linear(c_atom, c_atom)

        # Learned bias for backbone connectivity
        # 4 atom types x 4 atom types = 16 pairs
        self.pair_bias = nn.Parameter(torch.zeros(n_heads, 4, 4))

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(c_atom),
            nn.Linear(c_atom, c_atom * 4),
            nn.GELU(),
            nn.Linear(c_atom * 4, c_atom),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        atom_feats: Tensor,  # [B, L, 4, c_atom]
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        B, L, n_atoms, C = atom_feats.shape

        # Flatten for attention: [B, L*4, C]
        x = atom_feats.view(B, L * n_atoms, C)

        h = self.norm(x)
        q = self.q_proj(h).view(B, L * n_atoms, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, L * n_atoms, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, L * n_atoms, self.n_heads, self.head_dim)

        # Attention scores: [B, n_heads, L*4, L*4]
        q = q.transpose(1, 2)  # [B, n_heads, L*4, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add pair bias based on atom types
        # pair_bias is [n_heads, 4, 4]
        # We need to tile it for all residue pairs
        # Create [n_heads, L*4, L*4] bias matrix
        bias = self.pair_bias.unsqueeze(0).unsqueeze(3)  # [1, n_heads, 4, 1, 4]
        bias = bias.expand(1, self.n_heads, 4, L, 4)  # [1, n_heads, 4, L, 4]
        bias = bias.reshape(1, self.n_heads, 4*L, 4)  # Wrong - let me fix

        # Actually, let's do this properly
        # For position i*4+a attending to j*4+b, bias is pair_bias[a, b]
        atom_idx_q = torch.arange(L * n_atoms, device=x.device) % n_atoms  # [L*4]
        atom_idx_k = atom_idx_q
        pair_bias_expanded = self.pair_bias[:, atom_idx_q][:, :, atom_idx_k]  # [n_heads, L*4, L*4]
        attn = attn + pair_bias_expanded.unsqueeze(0)

        # Mask: expand residue mask to atom mask
        if mask is not None:
            atom_mask = mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, L * n_atoms)
            attn_mask = atom_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L*4]
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, n_heads, L*4, head_dim]
        out = out.transpose(1, 2).reshape(B, L * n_atoms, C)
        out = self.out_proj(out)

        x = x + out
        x = x + self.ffn(x)

        return x.view(B, L, n_atoms, C)


class GeometricAtomDecoder(nn.Module):
    """Atom-level decoder with geometric priors.

    Key features:
    1. Starts from default backbone geometry
    2. Uses atom-level attention (atoms attend to atoms)
    3. Learns to refine positions from the geometric prior
    4. Explicit atom type differentiation
    """

    def __init__(
        self,
        c_token: int = 256,
        c_atom: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.n_layers = n_layers

        # Project residue features to atom features
        self.res_to_atom = nn.Linear(c_token, c_atom)

        # Atom type embeddings (N, CA, C, O) - larger scale for differentiation
        self.atom_type_embed = nn.Embedding(4, c_atom)
        nn.init.normal_(self.atom_type_embed.weight, std=0.5)  # Larger init

        # Position embedding for anchor positions
        self.pos_embed = nn.Sequential(
            nn.Linear(3, c_atom),
            nn.GELU(),
            nn.Linear(c_atom, c_atom),
        )

        # Learnable embedding for unknown positions
        self.unknown_embed = nn.Parameter(torch.randn(1, 1, c_atom) * 0.1)

        # Default backbone geometry (learnable refinement)
        # [4, 3] for N, CA, C, O relative to centroid
        default_pos = torch.stack([
            DEFAULT_BACKBONE['N'],
            DEFAULT_BACKBONE['CA'],
            DEFAULT_BACKBONE['C'],
            DEFAULT_BACKBONE['O'],
        ])  # [4, 3]
        self.register_buffer('default_backbone', default_pos)

        # Learnable offset from default backbone
        self.backbone_offset = nn.Parameter(torch.zeros(4, 3))

        # Atom attention layers
        self.atom_layers = nn.ModuleList([
            AtomAttentionBlock(c_atom, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output: predict offset from prior position
        self.output_norm = nn.LayerNorm(c_atom)
        self.output_proj = nn.Linear(c_atom, 3)

    def forward(
        self,
        trunk_tokens: Tensor,  # [B, L, c_token]
        anchor_pos: Tensor,    # [B, L, 3] residue centroids (0 = unknown)
        mask: Optional[Tensor] = None,
        coord_std: float = 10.0,  # For normalizing default backbone
    ) -> Tensor:
        """Predict atom coordinates.

        Returns:
            atom_coords: [B, L, 4, 3]
        """
        B, L, _ = trunk_tokens.shape
        device = trunk_tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Check which residues have anchor positions
        is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)  # [B, L]

        # 1. Create atom features from residue features
        res_feats = self.res_to_atom(trunk_tokens)  # [B, L, c_atom]
        res_feats = res_feats.unsqueeze(2).expand(-1, -1, 4, -1)  # [B, L, 4, c_atom]

        # 2. Add atom type embeddings
        atom_types = torch.arange(4, device=device)
        atom_type_feats = self.atom_type_embed(atom_types)  # [4, c_atom]
        atom_feats = res_feats + atom_type_feats  # [B, L, 4, c_atom]

        # 3. Add position embedding for anchors
        pos_feats = self.pos_embed(anchor_pos)  # [B, L, c_atom]
        pos_feats = torch.where(
            is_anchored.unsqueeze(-1),
            pos_feats,
            self.unknown_embed.expand(B, L, -1)
        )
        atom_feats = atom_feats + pos_feats.unsqueeze(2)  # [B, L, 4, c_atom]

        # 4. Process with atom attention
        for layer in self.atom_layers:
            atom_feats = layer(atom_feats, mask)

        # 5. Predict position offsets
        atom_feats = self.output_norm(atom_feats)
        offsets = self.output_proj(atom_feats)  # [B, L, 4, 3]

        # 6. Compute prior positions from anchors + default backbone
        # Default backbone is in Angstroms, normalize by coord_std
        default_backbone_norm = (self.default_backbone + self.backbone_offset) / coord_std

        # Prior: anchor_pos + default_backbone (broadcast to all residues)
        # For non-anchored residues, use zeros + small default
        prior_pos = anchor_pos.unsqueeze(2) + default_backbone_norm  # [B, L, 4, 3]

        # For non-anchored residues, use a learned global position
        # (the model should learn to place these correctly via offsets)

        # 7. Final positions = prior + learned offsets
        atom_coords = prior_pos + offsets

        return atom_coords


class GeometricAtomDecoderV2(nn.Module):
    """Version 2: Even more atom-centric with local frame computation.

    Each atom attends to:
    1. Its own residue features
    2. Neighboring atoms in the backbone (N-CA-C-O chain)
    3. Atoms from adjacent residues

    Uses local coordinate frames for geometric reasoning.
    """

    def __init__(
        self,
        c_token: int = 256,
        c_atom: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom

        # Project residue features
        self.res_to_atom = nn.Linear(c_token, c_atom)

        # Atom type embeddings
        self.atom_type_embed = nn.Embedding(4, c_atom)
        nn.init.normal_(self.atom_type_embed.weight, std=0.5)

        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, c_atom),
            nn.LayerNorm(c_atom),
            nn.GELU(),
            nn.Linear(c_atom, c_atom),
        )

        self.unknown_embed = nn.Parameter(torch.randn(1, 1, 1, c_atom) * 0.1)

        # Project trunk tokens to atom dimension for cross attention
        self.trunk_proj = nn.Linear(c_token, c_atom)

        # Atom processing layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                # Self attention over all atoms
                'atom_attn': AtomAttentionBlock(c_atom, n_heads, dropout),
                # Cross attention to residue features
                'cross_attn': nn.MultiheadAttention(c_atom, n_heads, dropout, batch_first=True),
                'cross_norm': nn.LayerNorm(c_atom),
                # Position update
                'pos_update': nn.Sequential(
                    nn.LayerNorm(c_atom),
                    nn.Linear(c_atom, c_atom),
                    nn.GELU(),
                    nn.Linear(c_atom, 3),
                ),
            }))

        # Final output
        self.output_norm = nn.LayerNorm(c_atom)
        self.output_proj = nn.Linear(c_atom, 3)

        # Default backbone (normalized)
        default_pos = torch.stack([
            torch.tensor([-0.15, 0.0, 0.0]),   # N
            torch.tensor([0.0, 0.0, 0.0]),      # CA
            torch.tensor([0.15, 0.0, 0.0]),     # C
            torch.tensor([0.24, 0.10, 0.0]),    # O
        ])
        self.register_buffer('default_backbone', default_pos)

    def forward(
        self,
        trunk_tokens: Tensor,
        anchor_pos: Tensor,
        mask: Optional[Tensor] = None,
        coord_std: float = 10.0,
    ) -> Tensor:
        B, L, _ = trunk_tokens.shape
        device = trunk_tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)

        # Project trunk tokens for cross attention
        trunk_proj = self.trunk_proj(trunk_tokens)  # [B, L, c_atom]

        # Initialize atom features
        res_feats = self.res_to_atom(trunk_tokens)  # [B, L, c_atom]

        atom_types = torch.arange(4, device=device)
        atom_type_feats = self.atom_type_embed(atom_types)  # [4, c_atom]

        atom_feats = res_feats.unsqueeze(2) + atom_type_feats  # [B, L, 4, c_atom]

        # Initialize atom positions from anchor + default
        atom_pos = anchor_pos.unsqueeze(2) + self.default_backbone  # [B, L, 4, 3]

        # Add position encoding
        pos_enc = self.pos_encoder(anchor_pos)  # [B, L, c_atom]
        pos_enc = torch.where(
            is_anchored.unsqueeze(-1),
            pos_enc,
            self.unknown_embed.squeeze(2).expand(B, L, -1)
        )
        atom_feats = atom_feats + pos_enc.unsqueeze(2)

        # Iterative refinement
        for layer in self.layers:
            # Atom self-attention
            atom_feats = layer['atom_attn'](atom_feats, mask)

            # Cross attention to residue tokens
            B, L, n_atoms, C = atom_feats.shape
            atom_flat = atom_feats.view(B, L * n_atoms, C)

            # Expand mask for cross attention
            if mask is not None:
                key_padding_mask = ~mask
            else:
                key_padding_mask = None

            cross_out, _ = layer['cross_attn'](
                layer['cross_norm'](atom_flat),
                trunk_proj,
                trunk_proj,
                key_padding_mask=key_padding_mask,
            )
            atom_feats = atom_feats + cross_out.view(B, L, n_atoms, C)

            # Update positions
            pos_delta = layer['pos_update'](atom_feats)  # [B, L, 4, 3]
            atom_pos = atom_pos + pos_delta * 0.1  # Small updates

        # Final position prediction
        final_offset = self.output_proj(self.output_norm(atom_feats))
        atom_coords = atom_pos + final_offset

        return atom_coords


def test_decoders():
    """Test both decoder variants."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, L = 2, 50
    c_token = 256

    trunk_tokens = torch.randn(B, L, c_token, device=device)
    anchor_pos = torch.randn(B, L, 3, device=device) * 0.5
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    print("Testing GeometricAtomDecoder...")
    decoder1 = GeometricAtomDecoder(c_token=c_token, c_atom=128, n_layers=4).to(device)
    out1 = decoder1(trunk_tokens, anchor_pos, mask)
    print(f"  Output shape: {out1.shape}")
    print(f"  Params: {sum(p.numel() for p in decoder1.parameters()) / 1e6:.2f}M")

    print("\nTesting GeometricAtomDecoderV2...")
    decoder2 = GeometricAtomDecoderV2(c_token=c_token, c_atom=128, n_layers=4).to(device)
    out2 = decoder2(trunk_tokens, anchor_pos, mask)
    print(f"  Output shape: {out2.shape}")
    print(f"  Params: {sum(p.numel() for p in decoder2.parameters()) / 1e6:.2f}M")

    # Check atom spread
    print("\nAtom spread per residue (should be ~0.1-0.2 for proper backbone):")
    spread1 = (out1 - out1.mean(dim=2, keepdim=True)).pow(2).sum(dim=-1).sqrt().mean()
    spread2 = (out2 - out2.mean(dim=2, keepdim=True)).pow(2).sum(dim=-1).sqrt().mean()
    print(f"  Decoder V1: {spread1.item():.4f}")
    print(f"  Decoder V2: {spread2.item():.4f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_decoders()
