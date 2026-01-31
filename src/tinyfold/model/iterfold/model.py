"""IterFold: Iterative anchor-conditioned protein structure prediction.

Alternative to diffusion-based ResFold that directly predicts atom positions
via iterative masked prediction. Instead of denoising random noise, we condition
on known residue positions (anchors) and predict unknown ones.

Architecture:
1. ResidueEncoder (Trunk): Runs ONCE to produce conditioning embeddings (reused from resfold.py)
2. AnchorDecoder: Takes trunk_tokens + anchor_pos, predicts atom positions

Key insight: anchor_pos tensor encodes both positions AND mask:
- Non-zero values = known anchor positions (ground truth centroids)
- Zero values = unknown positions (to predict)

The residual connection `atom_coords = offsets + anchor_pos.unsqueeze(2)` ensures:
- Anchored residues have low reconstruction error (offset from GT is small)
- Model focuses learning on unknown residues
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .resfold import ResidueEncoder
from .clustering import select_next_residues_to_place
from .atom_decoder import GeometricAtomDecoder
from .frame_decoder import FrameDecoder


class FlashSelfAttentionBlock(nn.Module):
    """Self-attention block with Flash Attention (via scaled_dot_product_attention)."""

    def __init__(self, c_token: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_token // n_heads

        self.norm1 = nn.LayerNorm(c_token)
        self.q_proj = nn.Linear(c_token, c_token)
        self.k_proj = nn.Linear(c_token, c_token)
        self.v_proj = nn.Linear(c_token, c_token)
        self.out_proj = nn.Linear(c_token, c_token)

        self.norm2 = nn.LayerNorm(c_token)
        self.ffn = nn.Sequential(
            nn.Linear(c_token, c_token * 4),
            nn.GELU(),
            nn.Linear(c_token * 4, c_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L, C = x.shape

        # Self-attention with pre-norm
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention via SDPA
        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, C)
        x = x + self.dropout(self.out_proj(out))

        # FFN with pre-norm
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class FlashCrossAttention(nn.Module):
    """Cross-attention with Flash Attention."""

    def __init__(self, c_token: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_token // n_heads

        self.q_proj = nn.Linear(c_token, c_token)
        self.k_proj = nn.Linear(c_token, c_token)
        self.v_proj = nn.Linear(c_token, c_token)
        self.out_proj = nn.Linear(c_token, c_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_input: Tensor, kv_input: Tensor, kv_mask: Optional[Tensor] = None) -> Tensor:
        B, Lq, C = q_input.shape
        Lkv = kv_input.shape[1]

        q = self.q_proj(q_input).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)

        # Mask: [B, Lkv] -> [B, 1, 1, Lkv]
        attn_mask = kv_mask.unsqueeze(1).unsqueeze(2) if kv_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Lq, C)
        return self.dropout(self.out_proj(out))


class AnchorDecoder(nn.Module):
    """Decodes atom positions conditioned on known anchor positions (CENTROID-BASED).

    Uses Flash Attention (scaled_dot_product_attention) for speed.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        n_atom_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers
        self.n_atom_layers = n_atom_layers
        self.n_heads = n_heads

        # Position embedding: [B, L, 3] -> [B, L, c_token]
        self.pos_embed = nn.Sequential(
            nn.Linear(3, c_token),
            nn.LayerNorm(c_token),
            nn.GELU(),
            nn.Linear(c_token, c_token),
        )

        # Learnable embedding for unknown (zero) positions
        self.unknown_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)

        # Main transformer with Flash Attention
        self.blocks = nn.ModuleList([
            FlashSelfAttentionBlock(c_token, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Atom query embeddings [1, 1, 4, c_token] for N, CA, C, O
        # Larger init to ensure atoms differentiate
        self.atom_type_embed = nn.Parameter(torch.randn(1, 1, 4, c_token) * 0.1)

        # Atom refinement with Flash Cross-Attention
        self.cross_attn_layers = nn.ModuleList([
            FlashCrossAttention(c_token, n_heads, dropout)
            for _ in range(n_atom_layers)
        ])

        self.cross_norms = nn.ModuleList([nn.LayerNorm(c_token) for _ in range(n_atom_layers)])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_token),
                nn.Linear(c_token, c_token * 4),
                nn.GELU(),
                nn.Linear(c_token * 4, c_token),
                nn.Dropout(dropout),
            )
            for _ in range(n_atom_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(c_token)
        self.output_proj = nn.Linear(c_token, 3)


class AtomAnchorDecoder(nn.Module):
    """Decodes atom positions conditioned on known anchor ATOM positions (ATOM-BASED).

    Instead of conditioning on residue centroids, this decoder conditions on
    actual atom positions (N, CA, C, O) for visible residues.

    Input: anchor_atoms [B, L, 4, 3] - atom positions (zeros for masked residues)
    Output: atom_coords [B, L, 4, 3] - predicted atom coordinates
    """

    def __init__(
        self,
        c_token: int = 256,
        c_atom: int = 64,
        n_layers: int = 12,
        n_heads: int = 8,
        n_atom_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.n_layers = n_layers
        self.n_atom_layers = n_atom_layers
        self.n_heads = n_heads

        # Atom-level position embedding: [B, L, 4, 3] -> [B, L, 4, c_atom]
        self.atom_pos_embed = nn.Sequential(
            nn.Linear(3, c_atom),
            nn.LayerNorm(c_atom),
            nn.GELU(),
            nn.Linear(c_atom, c_atom),
        )

        # Atom type embedding for input conditioning (N, CA, C, O)
        self.atom_type_embed_input = nn.Parameter(torch.randn(4, c_atom) * 0.02)

        # Aggregate atom embeddings to residue level: [B, L, 4*c_atom] -> [B, L, c_token]
        self.atom_to_res = nn.Sequential(
            nn.Linear(4 * c_atom, c_token),
            nn.LayerNorm(c_token),
            nn.GELU(),
            nn.Linear(c_token, c_token),
        )

        # Learnable embedding for unknown (zero) positions
        self.unknown_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)

        # Main transformer with Flash Attention
        self.blocks = nn.ModuleList([
            FlashSelfAttentionBlock(c_token, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Atom query embeddings [1, 1, 4, c_token] for N, CA, C, O
        self.atom_type_embed = nn.Parameter(torch.randn(1, 1, 4, c_token) * 0.1)

        # Atom refinement with Flash Cross-Attention
        self.cross_attn_layers = nn.ModuleList([
            FlashCrossAttention(c_token, n_heads, dropout)
            for _ in range(n_atom_layers)
        ])

        self.cross_norms = nn.ModuleList([nn.LayerNorm(c_token) for _ in range(n_atom_layers)])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_token),
                nn.Linear(c_token, c_token * 4),
                nn.GELU(),
                nn.Linear(c_token * 4, c_token),
                nn.Dropout(dropout),
            )
            for _ in range(n_atom_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(c_token)
        self.output_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        trunk_tokens: Tensor,    # [B, L, c_token]
        anchor_atoms: Tensor,    # [B, L, 4, 3] atom positions (0 = unknown residue)
        mask: Optional[Tensor] = None,  # [B, L] bool, True = valid residue
    ) -> Tensor:
        """Predict atom coordinates conditioned on anchor atom positions.

        Args:
            trunk_tokens: Sequence embeddings from trunk encoder [B, L, c_token]
            anchor_atoms: Anchor atom positions (all zeros for masked residues) [B, L, 4, 3]
            mask: Valid residue mask [B, L] (for padding only)

        Returns:
            atom_coords: Predicted atom coordinates [B, L, 4, 3]
        """
        B, L, _, _ = anchor_atoms.shape
        device = trunk_tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Detect anchored residues: any atom has non-zero position
        is_anchored = (anchor_atoms.abs().sum(dim=(-1, -2)) > 1e-6)  # [B, L]

        # Embed each atom position: [B, L, 4, 3] -> [B, L, 4, c_atom]
        atom_pos_feat = self.atom_pos_embed(anchor_atoms)

        # Add atom type embedding (broadcast over B, L)
        atom_pos_feat = atom_pos_feat + self.atom_type_embed_input  # [B, L, 4, c_atom]

        # Aggregate to residue level: concatenate and project
        atom_pos_flat = atom_pos_feat.view(B, L, -1)  # [B, L, 4*c_atom]
        pos_feat = self.atom_to_res(atom_pos_flat)    # [B, L, c_token]

        # Replace unknown (masked) residues with learnable embedding
        pos_feat = torch.where(
            is_anchored.unsqueeze(-1),
            pos_feat,
            self.unknown_embed.expand(B, L, -1)
        )

        # Combine with trunk tokens (additive conditioning)
        h = trunk_tokens + pos_feat  # [B, L, c_token]

        # Transformer processing with Flash Attention
        for block in self.blocks:
            h = block(h, mask)  # [B, L, c_token]

        # Create atom queries: [B, L, 4, c_token]
        atom_queries = h.unsqueeze(2) + self.atom_type_embed
        atom_queries = atom_queries.view(B, L * 4, self.c_token)

        # Refine with Flash cross-attention to residues
        for i in range(self.n_atom_layers):
            q = self.cross_norms[i](atom_queries)
            attn_out = self.cross_attn_layers[i](q, h, mask)
            atom_queries = atom_queries + attn_out
            atom_queries = atom_queries + self.ffn_layers[i](atom_queries)

        # Project to coordinates (absolute, no residual)
        atom_queries = self.output_norm(atom_queries)
        atom_coords = self.output_proj(atom_queries.view(B, L, 4, -1))  # [B, L, 4, 3]

        return atom_coords


class IterFold(nn.Module):
    """IterFold: Iterative anchor-conditioned structure prediction.

    Combines:
    - ResidueEncoder (trunk): Sequence features -> trunk_tokens
    - AnchorDecoder or GeometricAtomDecoder: trunk_tokens + anchor_pos -> atom_coords

    Training:
    - Random subset of residues are anchored (GT centroids or atoms provided)
    - Model predicts all atoms, loss on all atoms (anchored ones easier)

    Inference:
    - Start with seed anchor (central residue or known position)
    - Iteratively predict -> fix closest residues -> update anchors -> repeat
    """

    def __init__(
        self,
        c_token: int = 256,
        trunk_layers: int = 9,
        trunk_heads: int = 8,
        decoder_layers: int = 12,
        decoder_heads: int = 8,
        n_atom_layers: int = 8,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
        use_geometric_decoder: bool = False,
        use_frame_decoder: bool = False,
        use_atom_anchor_decoder: bool = False,
        c_atom: int = 128,
    ):
        super().__init__()
        self.c_token = c_token
        self.use_geometric_decoder = use_geometric_decoder
        self.use_frame_decoder = use_frame_decoder
        self.use_atom_anchor_decoder = use_atom_anchor_decoder

        # Trunk encoder (from resfold.py)
        self.trunk = ResidueEncoder(
            c_token=c_token,
            n_layers=trunk_layers,
            n_heads=trunk_heads,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # Decoder choice
        if use_atom_anchor_decoder:
            # Atom-based anchor decoder (conditions on atom positions, not centroids)
            self.decoder = AtomAnchorDecoder(
                c_token=c_token,
                c_atom=c_atom,
                n_layers=decoder_layers,
                n_heads=decoder_heads,
                n_atom_layers=n_atom_layers,
                dropout=dropout,
            )
        elif use_frame_decoder:
            # Frame-based decoder (centroid + rotation)
            self.decoder = FrameDecoder(
                c_token=c_token,
                c_hidden=c_atom * 2,
                n_layers=decoder_layers,
                n_heads=decoder_heads,
                dropout=dropout,
            )
        elif use_geometric_decoder:
            # Geometric atom decoder with priors
            self.decoder = GeometricAtomDecoder(
                c_token=c_token,
                c_atom=c_atom,
                n_layers=decoder_layers,
                n_heads=decoder_heads,
                dropout=dropout,
            )
        else:
            # Original anchor decoder (centroid-based)
            self.decoder = AnchorDecoder(
                c_token=c_token,
                n_layers=decoder_layers,
                n_heads=decoder_heads,
                n_atom_layers=n_atom_layers,
                dropout=dropout,
            )

    def forward(
        self,
        aa_seq: Tensor,        # [B, L]
        chain_ids: Tensor,     # [B, L]
        res_idx: Tensor,       # [B, L]
        anchor_input: Tensor,  # [B, L, 3] or [B, L, 4, 3] depending on decoder
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Full forward pass.

        Args:
            aa_seq: Amino acid indices [B, L]
            chain_ids: Chain IDs [B, L]
            res_idx: Residue indices [B, L]
            anchor_input: Anchor positions/atoms (0 = unknown)
                - [B, L, 3] for centroid-based decoders
                - [B, L, 4, 3] for atom-based decoder
            mask: Valid residue mask [B, L]

        Returns:
            atom_coords: Predicted atom coordinates [B, L, 4, 3]
        """
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)
        return self.decoder(trunk_tokens, anchor_input, mask)

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Get trunk embeddings (for caching during inference)."""
        return self.trunk(aa_seq, chain_ids, res_idx, mask)

    @torch.no_grad()
    def sample_iterative(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Optional[Tensor] = None,
        n_iter: int = 10,
        k_per_iter: Optional[int] = None,
        seed_idx: Optional[int] = None,
        seed_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Iterative inference starting from seed anchor.

        Process:
        1. Start with seed residue(s) anchored
        2. Predict all atoms conditioned on current anchors
        3. Select K closest unknown residues and anchor them
        4. Repeat until all residues anchored
        5. Final prediction with all anchors

        Args:
            aa_seq, chain_ids, res_idx: Sequence features [B, L]
            mask: Valid residue mask [B, L]
            n_iter: Number of iterations
            k_per_iter: Residues to anchor per iteration (default: L // n_iter)
            seed_idx: Index of seed residue (default: central residue)
            seed_pos: Initial position for seed [B, 3] (required if no GT available)

        Returns:
            atom_coords: Final predicted coordinates [B, L, 4, 3]
        """
        B, L = aa_seq.shape
        device = aa_seq.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        k_per_iter = k_per_iter or max(1, L // n_iter)

        # Get trunk tokens (once)
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)

        # Initialize anchor_pos: all zeros
        anchor_pos = torch.zeros(B, L, 3, device=device)

        # Seed with central residue if no seed provided
        if seed_idx is None:
            seed_idx = L // 2

        if seed_pos is not None:
            anchor_pos[:, seed_idx] = seed_pos
        else:
            # Will need to do a blind prediction first
            # Predict with all zeros, use predicted centroid as seed
            pred_atoms = self.decoder(trunk_tokens, anchor_pos, mask)
            pred_centroids = pred_atoms.mean(dim=2)  # [B, L, 3]
            anchor_pos[:, seed_idx] = pred_centroids[:, seed_idx]

        for _ in range(n_iter):
            # Predict all atoms conditioned on current anchors
            pred_atoms = self.decoder(trunk_tokens, anchor_pos, mask)
            pred_centroids = pred_atoms.mean(dim=2)  # [B, L, 3]

            # Select next residues to anchor (closest to current anchors)
            for b in range(B):
                is_anchored = (anchor_pos[b].abs().sum(dim=-1) > 1e-6)

                if is_anchored.all():
                    # All anchored - still update centroids to latest predictions
                    anchor_pos[b] = pred_centroids[b]
                    continue

                next_res = select_next_residues_to_place(
                    pred_centroids[b], is_anchored, k_per_iter
                )

                # Mark new residues as anchored
                new_anchored = is_anchored.clone()
                new_anchored[next_res] = True

                # Update ALL anchored centroids to latest predictions
                anchor_pos[b] = torch.where(
                    new_anchored.unsqueeze(-1),
                    pred_centroids[b],
                    anchor_pos[b]
                )

        # Final prediction with all anchors
        final_atoms = self.decoder(trunk_tokens, anchor_pos, mask)
        return final_atoms

    def count_parameters(self) -> dict:
        """Count parameters in trunk vs decoder."""
        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = trunk_params + decoder_params

        return {
            'trunk': trunk_params,
            'decoder': decoder_params,
            'total': total,
            'trunk_pct': 100 * trunk_params / total,
            'decoder_pct': 100 * decoder_params / total,
        }


def test_iterfold():
    """Quick test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IterFold(
        c_token=256,
        trunk_layers=6,
        decoder_layers=6,
        n_atom_layers=4,
    ).to(device)

    params = model.count_parameters()
    print(f"IterFold: {params['total']/1e6:.2f}M params")
    print(f"  Trunk: {params['trunk']/1e6:.2f}M ({params['trunk_pct']:.1f}%)")
    print(f"  Decoder: {params['decoder']/1e6:.2f}M ({params['decoder_pct']:.1f}%)")

    # Test forward
    B, L = 2, 50
    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    chain_ids[:, L//2:] = 1
    res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Anchor 30% of residues
    anchor_mask = torch.rand(B, L, device=device) < 0.3
    gt_centroids = torch.randn(B, L, 3, device=device)
    anchor_pos = gt_centroids * anchor_mask.unsqueeze(-1).float()

    atoms = model(aa_seq, chain_ids, res_idx, anchor_pos, mask)
    print(f"Output shape: {atoms.shape}")  # [B, L, 4, 3]

    # Test iterative inference
    atoms_iter = model.sample_iterative(
        aa_seq, chain_ids, res_idx, mask,
        n_iter=5, k_per_iter=10
    )
    print(f"Iterative output shape: {atoms_iter.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_iterfold()
