"""AF3-Style Decoder - AlphaFold3-inspired architecture with separate trunk and denoiser.

Architecture follows AF3's design philosophy:
1. ResidueEncoder (Trunk): Residue-level transformer that runs ONCE to produce conditioning
2. Denoiser: Runs at EACH diffusion step
   - AtomAttentionEncoder: Local atom attention, aggregates atoms → tokens
   - DiffusionTransformer: Global token-level transformer with AdaLN conditioning
   - AtomAttentionDecoder: Broadcasts tokens → atoms, outputs coordinate updates

Key differences from AF3:
- No pair embeddings (z), only single/token embeddings (s)
- Smaller scale for tractable training

The denoiser conditions on current noisy coordinates x_t at each step,
which is essential for diffusion models to work properly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseDecoder, sinusoidal_pos_enc


# =============================================================================
# Building Blocks
# =============================================================================

class AdaLN(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning.

    Modulates normalized activations with learned scale and shift
    predicted from the conditioning signal.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)  # scale and shift

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: [*, dim] input
            cond: [*, cond_dim] conditioning (e.g., timestep embedding)
        Returns:
            [*, dim] normalized and modulated output
        """
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class SwiGLU(nn.Module):
    """SwiGLU feedforward block."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# =============================================================================
# Atom Attention (Local, within-residue)
# =============================================================================

class LocalAtomAttention(nn.Module):
    """Local attention within each residue's atoms.

    This implements sequence-local atom attention as described in AF3.
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
        """Local attention within each residue.

        Args:
            x: Atom features [B, L, 4, c_atom]
            mask: Valid atom mask [B, L, 4]
        Returns:
            Updated atom features [B, L, 4, c_atom]
        """
        B, L, n_atoms, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, L, 4, c_atom]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [B, L, n_heads, 4, head_dim]
        q = q.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)
        k = k.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)
        v = v.view(B, L, n_atoms, self.n_heads, self.head_dim).transpose(2, 3)
        # Now: [B, L, n_heads, 4, head_dim]

        # Attention scores within each residue
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, L, n_heads, 4, 4]

        if mask is not None:
            # Expand mask for attention: [B, L, 1, 1, 4]
            attn_mask = mask.unsqueeze(2).unsqueeze(3)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # Handle fully-masked residues: softmax([-inf,...]) = NaN → 0
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, L, n_heads, 4, head_dim]
        out = out.transpose(2, 3).reshape(B, L, n_atoms, self.c_atom)

        out = self.out_proj(out)

        # Zero out fully-masked residues
        if mask is not None:
            residue_mask = mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # [B, L, 1, 1]
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
        self.ffn = SwiGLU(c_atom, expansion, dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# AtomAttentionEncoder
# =============================================================================

class AtomAttentionEncoder(nn.Module):
    """Encodes atoms to token representations.

    Process:
    1. Embed atom positions and types
    2. Apply local atom attention (within each residue)
    3. Aggregate atoms → tokens (mean pooling over 4 atoms per residue)
    4. Save skip connections for decoder
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_token: int = 256,
        n_blocks: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_token = c_token

        # Atom position embedding (linear projection of 3D coords)
        self.pos_embed = nn.Linear(3, c_atom)

        # Atom type embedding (N=0, CA=1, C=2, O=3)
        self.atom_type_embed = nn.Embedding(4, c_atom)

        # Input projection
        self.input_proj = nn.Linear(c_atom * 2, c_atom)

        # Local atom transformer blocks
        self.blocks = nn.ModuleList([
            LocalAtomTransformerBlock(c_atom, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Aggregation: atoms → tokens
        self.aggregate_proj = nn.Linear(c_atom, c_token)

    def forward(
        self,
        atom_coords: Tensor,     # [B, L, 4, 3] noisy atom positions
        atom_types: Tensor,      # [B, L, 4] atom types
        trunk_tokens: Tensor,    # [B, L, c_token] trunk conditioning (added to aggregated)
        mask: Optional[Tensor] = None,  # [B, L, 4] valid atom mask
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode atoms to tokens.

        Returns:
            tokens: [B, L, c_token] token representations
            skip_states: list of [B, L, 4, c_atom] for decoder skip connections
        """
        B, L, n_atoms, _ = atom_coords.shape

        # Embed atoms
        pos_emb = self.pos_embed(atom_coords)  # [B, L, 4, c_atom]
        type_emb = self.atom_type_embed(atom_types)  # [B, L, 4, c_atom]

        h = self.input_proj(torch.cat([pos_emb, type_emb], dim=-1))  # [B, L, 4, c_atom]

        # Apply local attention blocks, saving skip states
        skip_states = []
        for block in self.blocks:
            h = block(h, mask)
            skip_states.append(h)

        # Aggregate atoms → tokens (mean pool over atoms in each residue)
        if mask is not None:
            # Masked mean pooling
            mask_float = mask.float().unsqueeze(-1)  # [B, L, 4, 1]
            h_sum = (h * mask_float).sum(dim=2)  # [B, L, c_atom]
            h_count = mask_float.sum(dim=2).clamp(min=1)  # [B, L, 1]
            h_mean = h_sum / h_count
        else:
            h_mean = h.mean(dim=2)  # [B, L, c_atom]

        tokens = self.aggregate_proj(h_mean)  # [B, L, c_token]

        # Add trunk conditioning
        tokens = tokens + trunk_tokens

        return tokens, skip_states


# =============================================================================
# DiffusionTransformer (Global, token-level)
# =============================================================================

class DiffusionTransformerBlock(nn.Module):
    """Single block of the diffusion transformer with AdaLN conditioning."""

    def __init__(
        self,
        c_token: int = 256,
        n_heads: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_heads = n_heads
        self.head_dim = c_token // n_heads

        # AdaLN for attention
        self.adaln_attn = AdaLN(c_token, c_token)

        # Multi-head self-attention
        self.q_proj = nn.Linear(c_token, c_token)
        self.k_proj = nn.Linear(c_token, c_token)
        self.v_proj = nn.Linear(c_token, c_token)
        self.out_proj = nn.Linear(c_token, c_token)

        # AdaLN for FFN
        self.adaln_ffn = AdaLN(c_token, c_token)

        # Feedforward
        self.ffn = SwiGLU(c_token, expansion, dropout)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,           # [B, L, c_token]
        cond: Tensor,        # [B, L, c_token] timestep conditioning
        mask: Optional[Tensor] = None,  # [B, L] valid token mask
    ) -> Tensor:
        B, L, _ = x.shape

        # AdaLN + Attention
        h = self.adaln_attn(x, cond)

        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_heads, L, head_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, n_heads, L, L]

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # Handle edge case where all tokens are masked
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, n_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, self.c_token)
        out = self.out_proj(out)

        x = x + out

        # AdaLN + FFN
        x = x + self.ffn(self.adaln_ffn(x, cond))

        # Zero out masked positions
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        return x


class DiffusionTransformer(nn.Module):
    """Global token-level transformer for diffusion.

    Operates on token representations with AdaLN conditioning from timestep.
    This is the main "brain" of the denoiser.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_blocks: int = 12,
        n_heads: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(c_token, n_heads, expansion, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(c_token)

    def forward(
        self,
        tokens: Tensor,      # [B, L, c_token]
        time_cond: Tensor,   # [B, L, c_token] timestep conditioning
        mask: Optional[Tensor] = None,  # [B, L] valid token mask
    ) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, time_cond, mask)
        return self.final_norm(tokens)


# =============================================================================
# AtomAttentionDecoder
# =============================================================================

class AtomAttentionDecoder(nn.Module):
    """Decodes tokens back to atom coordinate updates.

    Process:
    1. Broadcast tokens → atoms
    2. Add skip connections from encoder
    3. Apply local atom attention
    4. Project to 3D coordinate updates
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_token: int = 256,
        n_blocks: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.n_blocks = n_blocks

        # Broadcast: tokens → atoms
        self.broadcast_proj = nn.Linear(c_token, c_atom)

        # Skip connection projections (one per encoder block)
        self.skip_projs = nn.ModuleList([
            nn.Linear(c_atom, c_atom)
            for _ in range(n_blocks)
        ])

        # Local atom transformer blocks
        self.blocks = nn.ModuleList([
            LocalAtomTransformerBlock(c_atom, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output projection to 3D coordinates
        self.output_norm = nn.LayerNorm(c_atom)
        self.output_proj = nn.Linear(c_atom, 3)

    def forward(
        self,
        tokens: Tensor,          # [B, L, c_token]
        skip_states: list[Tensor],  # list of [B, L, 4, c_atom]
        atom_types: Tensor,      # [B, L, 4] for positional info
        mask: Optional[Tensor] = None,  # [B, L, 4]
    ) -> Tensor:
        """Decode tokens to atom coordinate updates.

        Returns:
            coord_updates: [B, L, 4, 3] per-atom coordinate updates
        """
        B, L, _ = tokens.shape

        # Broadcast tokens to atoms: [B, L, c_token] → [B, L, 4, c_atom]
        h = self.broadcast_proj(tokens).unsqueeze(2).expand(-1, -1, 4, -1)

        # Apply blocks with skip connections (reverse order)
        for i, (block, skip_proj) in enumerate(zip(self.blocks, self.skip_projs)):
            # Add skip connection from encoder (reverse order)
            skip_idx = self.n_blocks - 1 - i
            skip = skip_proj(skip_states[skip_idx])
            h = h + skip
            h = block(h, mask)

        # Output projection
        h = self.output_norm(h)
        coord_updates = self.output_proj(h)  # [B, L, 4, 3]

        return coord_updates


# =============================================================================
# ResidueEncoder (Trunk)
# =============================================================================

class ResidueEncoder(nn.Module):
    """Residue-level encoder (trunk) that runs ONCE per sample.

    Produces token embeddings that condition the denoiser.
    Uses regular attention over ~300 residue tokens.
    Each token contains aggregated atom information + positional encoding.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 6,
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

        # Atom info aggregation (mean of 4 atoms' positions projected)
        self.atom_info_proj = nn.Linear(3 * 4, c_token // 2)  # 4 atoms * 3 coords

        # Input projection
        # aa_emb (c_token) + chain_emb (c_token//4) + res_pos (c_token) + atom_info (c_token//2)
        input_dim = c_token + (c_token // 4) + c_token + (c_token // 2)
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
        atom_coords: Tensor,     # [B, L, 4, 3]
        aa_seq: Tensor,          # [B, L]
        chain_ids: Tensor,       # [B, L]
        res_idx: Tensor,         # [B, L]
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Encode residue-level features.

        Returns:
            tokens: [B, L, c_token] conditioning for denoiser
        """
        B, L, _, _ = atom_coords.shape

        # Embeddings
        aa_emb = self.aa_embed(aa_seq)  # [B, L, c_token]
        chain_emb = self.chain_embed(chain_ids)  # [B, L, c_token//4]
        res_emb = sinusoidal_pos_enc(res_idx, self.c_token)  # [B, L, c_token]

        # Aggregate atom positions into residue info
        atom_info = atom_coords.view(B, L, -1)  # [B, L, 12]
        atom_emb = self.atom_info_proj(atom_info)  # [B, L, c_token//2]

        # Concatenate and project
        h = torch.cat([aa_emb, chain_emb, res_emb, atom_emb], dim=-1)
        h = self.input_proj(h)  # [B, L, c_token]

        # Apply transformer
        attn_mask = ~mask if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        return self.output_norm(h)


# =============================================================================
# Main Model: AF3StyleDecoder
# =============================================================================

class AF3StyleDecoder(BaseDecoder):
    """AlphaFold3-style decoder with separate trunk and denoiser.

    Architecture:
        1. ResidueEncoder (Trunk): Runs ONCE to produce conditioning embeddings
        2. Denoiser (runs at EACH diffusion step):
           - AtomAttentionEncoder: atoms → tokens with local attention
           - DiffusionTransformer: global token attention with AdaLN
           - AtomAttentionDecoder: tokens → atom coordinate updates

    The denoiser conditions on:
        - Current noisy coordinates x_t (injected via AtomAttentionEncoder)
        - Trunk embeddings (conditioning signal)
        - Timestep (AdaLN modulation)

    This is the "proper" way to do diffusion on atomic structures:
    the denoiser sees the current noisy geometry at each step.
    """

    def __init__(
        self,
        # Token/hidden dimensions
        c_token: int = 256,
        c_atom: int = 128,
        # Trunk config (residue encoder - runs ONCE)
        trunk_layers: int = 9,
        trunk_heads: int = 8,
        # Denoiser config (runs EACH diffusion step)
        denoiser_blocks: int = 7,  # DiffusionTransformer blocks
        denoiser_heads: int = 8,
        atom_attn_blocks: int = 3,  # AtomAttention encoder/decoder blocks
        atom_attn_heads: int = 4,
        # Common
        n_timesteps: int = 50,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.n_timesteps = n_timesteps

        # === TRUNK (runs once) ===
        self.trunk = ResidueEncoder(
            c_token=c_token,
            n_layers=trunk_layers,
            n_heads=trunk_heads,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # === DENOISER (runs each step) ===

        # Timestep embedding for conditioning
        self.time_embed = nn.Embedding(n_timesteps, c_token)

        # AtomAttentionEncoder
        self.atom_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_token=c_token,
            n_blocks=atom_attn_blocks,
            n_heads=atom_attn_heads,
            dropout=dropout,
        )

        # DiffusionTransformer
        self.diff_transformer = DiffusionTransformer(
            c_token=c_token,
            n_blocks=denoiser_blocks,
            n_heads=denoiser_heads,
            dropout=dropout,
        )

        # AtomAttentionDecoder
        self.atom_decoder = AtomAttentionDecoder(
            c_atom=c_atom,
            c_token=c_token,
            n_blocks=atom_attn_blocks,
            n_heads=atom_attn_heads,
            dropout=dropout,
        )

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
        B, N_atoms, _ = x_t.shape
        N_res = N_atoms // 4
        device = x_t.device

        # Reshape to residue structure: [B, L, 4, 3]
        x_res = x_t.view(B, N_res, 4, 3)
        atom_types_res = atom_types.view(B, N_res, 4)

        # Extract residue-level info (from CA atoms)
        aa_res = aa_seq.view(B, N_res, 4)[:, :, 1]       # [B, L]
        chain_res = chain_ids.view(B, N_res, 4)[:, :, 1]  # [B, L]
        res_idx = atom_to_res.view(B, N_res, 4)[:, :, 1]  # [B, L]

        # Masks
        if mask is not None:
            mask_res = mask.view(B, N_res, 4)  # [B, L, 4]
            mask_token = mask_res[:, :, 1]     # [B, L]
        else:
            mask_res = torch.ones(B, N_res, 4, dtype=torch.bool, device=device)
            mask_token = torch.ones(B, N_res, dtype=torch.bool, device=device)

        # === TRUNK (once) ===
        trunk_tokens = self.trunk(x_res, aa_res, chain_res, res_idx, mask_token)  # [B, L, c_token]

        # === DENOISER ===

        # Timestep conditioning
        time_cond = self.time_embed(t).unsqueeze(1).expand(-1, N_res, -1)  # [B, L, c_token]

        # AtomAttentionEncoder: atoms → tokens
        tokens, skip_states = self.atom_encoder(
            x_res, atom_types_res, trunk_tokens, mask_res
        )  # [B, L, c_token], list of [B, L, 4, c_atom]

        # DiffusionTransformer: global token attention
        tokens = self.diff_transformer(tokens, time_cond, mask_token)  # [B, L, c_token]

        # AtomAttentionDecoder: tokens → atom updates (refinements)
        coord_updates = self.atom_decoder(
            tokens, skip_states, atom_types_res, mask_res
        )  # [B, L, 4, 3]

        # Residual connection with sqrt noise-level scaling
        # sqrt scaling is less aggressive - allows learning at low timesteps
        # t=1: scale=0.14, t=25: scale=0.71, t=50: scale=1.0
        noise_scale = (t.float() / self.n_timesteps).sqrt().view(-1, 1, 1)  # [B, 1, 1]
        x0_pred = x_t + noise_scale * coord_updates.view(B, N_atoms, 3)

        return x0_pred

    def forward_flow(
        self,
        x_t: Tensor,
        atom_types: Tensor,
        atom_to_res: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        t: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """Forward for linear_flow - directly predicts x_t (no residual/scaling).

        The model directly outputs the next step coordinates.
        This is simpler and avoids gradient issues from small step scales.
        """
        B, N_atoms, _ = x_t.shape
        N_res = N_atoms // 4
        device = x_t.device

        # Reshape to residue structure: [B, L, 4, 3]
        x_res = x_t.view(B, N_res, 4, 3)
        atom_types_res = atom_types.view(B, N_res, 4)

        # Extract residue-level info (from CA atoms)
        aa_res = aa_seq.view(B, N_res, 4)[:, :, 1]
        chain_res = chain_ids.view(B, N_res, 4)[:, :, 1]
        res_idx = atom_to_res.view(B, N_res, 4)[:, :, 1]

        # Masks
        if mask is not None:
            mask_res = mask.view(B, N_res, 4)
            mask_token = mask_res[:, :, 1]
        else:
            mask_res = torch.ones(B, N_res, 4, dtype=torch.bool, device=device)
            mask_token = torch.ones(B, N_res, dtype=torch.bool, device=device)

        # === TRUNK (once) ===
        trunk_tokens = self.trunk(x_res, aa_res, chain_res, res_idx, mask_token)

        # === DENOISER ===
        time_cond = self.time_embed(t).unsqueeze(1).expand(-1, N_res, -1)
        tokens, skip_states = self.atom_encoder(x_res, atom_types_res, trunk_tokens, mask_res)
        tokens = self.diff_transformer(tokens, time_cond, mask_token)
        coord_updates = self.atom_decoder(tokens, skip_states, atom_types_res, mask_res)

        # Direct prediction of x_t (next step) - no residual, no scaling
        x_next = coord_updates.view(B, N_atoms, 3)

        return x_next

    def count_parameters(self) -> dict:
        """Count parameters in trunk vs denoiser for verification."""
        trunk_params = sum(p.numel() for p in self.trunk.parameters())

        denoiser_params = (
            sum(p.numel() for p in self.time_embed.parameters()) +
            sum(p.numel() for p in self.atom_encoder.parameters()) +
            sum(p.numel() for p in self.diff_transformer.parameters()) +
            sum(p.numel() for p in self.atom_decoder.parameters())
        )

        total = trunk_params + denoiser_params

        return {
            'trunk': trunk_params,
            'denoiser': denoiser_params,
            'total': total,
            'trunk_pct': 100 * trunk_params / total,
            'denoiser_pct': 100 * denoiser_params / total,
        }
