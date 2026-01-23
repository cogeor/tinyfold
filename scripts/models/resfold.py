"""ResFold Stage 1: Residue-Level Diffusion Model.

Diffuses on residue centroids (L points) instead of all atoms (4L points).
This is 4x more efficient and matches biological intuition: backbone topology
is the hard problem, local bond geometry is well-constrained.

Architecture:
1. ResidueEncoder (Trunk): Runs ONCE to produce conditioning embeddings
2. ResidueDiffusionTransformer (Denoiser): Runs at EACH diffusion step
   - Takes noisy residue centroids x_t
   - Produces predicted clean centroids x0_pred
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseDecoder, sinusoidal_pos_enc


# =============================================================================
# Building Blocks (copied from af3_style.py for independence)
# =============================================================================

class AdaLN(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
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
# Residue Encoder (Trunk)
# =============================================================================

class ResidueEncoder(nn.Module):
    """Residue-level encoder (trunk) that runs ONCE per sample.

    Produces token embeddings that condition the denoiser.

    IMPORTANT: This trunk processes ONLY sequence/token features (aa_seq, chain_ids, res_idx).
    It does NOT take coordinates as input. This enables the trunk-once optimization
    where trunk runs once and denoiser runs multiple times with different noisy coords.
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

        # Residue embeddings (sequence-only, no coordinates!)
        self.aa_embed = nn.Embedding(n_aa_types, c_token)
        self.chain_embed = nn.Embedding(n_chains, c_token // 4)

        # Input projection
        # aa_emb (c_token) + chain_emb (c_token//4) + res_pos (c_token)
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
        """Encode residue-level sequence features (NO coordinates).

        Returns:
            tokens: [B, L, c_token] conditioning for denoiser
        """
        B, L = aa_seq.shape

        # Embeddings (sequence-only)
        aa_emb = self.aa_embed(aa_seq)  # [B, L, c_token]
        chain_emb = self.chain_embed(chain_ids)  # [B, L, c_token//4]
        res_emb = sinusoidal_pos_enc(res_idx, self.c_token)  # [B, L, c_token]

        # Concatenate and project
        h = torch.cat([aa_emb, chain_emb, res_emb], dim=-1)
        h = self.input_proj(h)  # [B, L, c_token]

        # Apply transformer
        attn_mask = ~mask if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        return self.output_norm(h)


# =============================================================================
# Diffusion Transformer Block
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

        # Use scaled_dot_product_attention (FlashAttention when available)
        # Convert mask to attention mask format: [B, 1, 1, L] for broadcasting
        attn_mask = None
        if mask is not None:
            # SDPA expects True = attend, False = mask out (opposite of key_padding_mask)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, L, self.c_token)
        out = self.out_proj(out)

        x = x + out

        # AdaLN + FFN
        x = x + self.ffn(self.adaln_ffn(x, cond))

        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        return x


class DiffusionTransformer(nn.Module):
    """Global token-level transformer for diffusion."""

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
        tokens: Tensor,
        time_cond: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, time_cond, mask)
        return self.final_norm(tokens)


# =============================================================================
# Residue Denoiser (Stage 1 Main Model)
# =============================================================================

class ResidueDenoiser(BaseDecoder):
    """Stage 1: Residue-level diffusion model.

    Predicts clean residue centroids from noisy centroids.

    Architecture:
    1. ResidueEncoder (Trunk): Runs ONCE per sample
    2. DiffusionTransformer (Denoiser): Runs at EACH diffusion step
       - coord_embed(x_t) -> add trunk conditioning -> transformer -> output coord_delta
       - x0_pred = x_t + scale(t) * coord_delta
    """

    def __init__(
        self,
        c_token: int = 256,
        trunk_layers: int = 9,
        trunk_heads: int = 8,
        denoiser_blocks: int = 7,
        denoiser_heads: int = 8,
        n_timesteps: int = 50,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
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

        # Timestep embedding (discrete, for backward compatibility)
        self.time_embed = nn.Embedding(n_timesteps, c_token)

        # Continuous sigma embedding (AF3-style Fourier features)
        # Uses sinusoidal encoding of log(sigma/sigma_data)/4
        self.sigma_data = 1.0  # For normalized coordinates
        self.sigma_embed = nn.Sequential(
            nn.Linear(c_token, c_token),
            nn.SiLU(),
            nn.Linear(c_token, c_token),
        )

        # Coordinate embedding for noisy input (same dim for additive)
        self.coord_embed = nn.Linear(3, c_token)

        # Self-conditioning embedding (for x0_prev from previous iteration)
        # This allows the model to refine its own predictions during inference
        self.self_cond_embed = nn.Linear(3, c_token)

        # Diffusion transformer
        self.diff_transformer = DiffusionTransformer(
            c_token=c_token,
            n_blocks=denoiser_blocks,
            n_heads=denoiser_heads,
            dropout=dropout,
        )

        # Output projection to 3D coordinates
        self.output_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        x_t: Tensor,         # [B, L, 3] noisy residue centroids
        aa_seq: Tensor,      # [B, L]
        chain_ids: Tensor,   # [B, L]
        res_idx: Tensor,     # [B, L]
        t: Tensor,           # [B] timestep
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict clean centroids x0 from noisy input (x0 prediction).

        Returns:
            x0_pred: [B, L, 3] predicted clean centroids
        """
        B, L, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # === TRUNK (once, sequence-only) ===
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)

        # === DENOISER ===

        # Embed noisy coordinates
        coord_emb = self.coord_embed(x_t)  # [B, L, c_token]

        # Additive conditioning (like AF3)
        tokens = coord_emb + trunk_tokens  # [B, L, c_token]

        # Timestep conditioning
        time_cond = self.time_embed(t).unsqueeze(1).expand(-1, L, -1)

        # Diffusion transformer
        tokens = self.diff_transformer(tokens, time_cond, mask)

        # Output: predict clean centroids x0 directly
        x0_pred = self.output_proj(tokens)  # [B, L, 3]

        return x0_pred

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        """Embed continuous sigma using Fourier features (AF3-style).

        Uses c_noise = log(sigma/sigma_data) / 4 as input to sinusoidal encoding.

        Args:
            sigma: [B] noise levels

        Returns:
            sigma_emb: [B, c_token] sigma embeddings
        """
        # AF3-style noise encoding: c_noise = log(sigma/sigma_data) / 4
        c_noise = torch.log(sigma / self.sigma_data + 1e-8) / 4.0  # [B]

        # Sinusoidal encoding (same as timestep but continuous)
        half_dim = self.c_token // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb_scale)
        emb = c_noise.unsqueeze(-1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, c_token]

        # Project through MLP
        return self.sigma_embed(emb)  # [B, c_token]

    def forward_sigma(
        self,
        x_t: Tensor,         # [B, L, 3] noisy residue centroids
        aa_seq: Tensor,      # [B, L]
        chain_ids: Tensor,   # [B, L]
        res_idx: Tensor,     # [B, L]
        sigma: Tensor,       # [B] continuous noise level
        mask: Optional[Tensor] = None,  # [B, L]
        x0_prev: Optional[Tensor] = None,  # [B, L, 3] previous x0 prediction (self-conditioning)
    ) -> Tensor:
        """Predict clean centroids x0 from noisy input using continuous sigma.

        This is the AF3-style forward pass with continuous noise levels.
        Supports self-conditioning: if x0_prev is provided, it's used to help
        the model refine predictions.

        Args:
            x_t: Noisy centroids
            aa_seq, chain_ids, res_idx: Sequence features
            sigma: Continuous noise level (NOT discrete timestep)
            mask: Valid residue mask
            x0_prev: Previous x0 prediction for self-conditioning (optional)

        Returns:
            x0_pred: [B, L, 3] predicted clean centroids
        """
        B, L, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # === TRUNK (once, sequence-only) ===
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)

        # === DENOISER ===

        # Embed noisy coordinates
        coord_emb = self.coord_embed(x_t)  # [B, L, c_token]

        # Additive conditioning (like AF3)
        tokens = coord_emb + trunk_tokens  # [B, L, c_token]

        # Self-conditioning: add embedding of previous prediction
        if x0_prev is not None:
            self_cond_emb = self.self_cond_embed(x0_prev)  # [B, L, c_token]
            tokens = tokens + self_cond_emb

        # Continuous sigma conditioning (instead of discrete timestep)
        sigma_cond = self._embed_sigma(sigma).unsqueeze(1).expand(-1, L, -1)

        # Diffusion transformer
        tokens = self.diff_transformer(tokens, sigma_cond, mask)

        # Output: predict clean centroids x0 directly
        x0_pred = self.output_proj(tokens)  # [B, L, 3]

        return x0_pred

    def forward_sigma_with_trunk(
        self,
        x_t: Tensor,         # [B, L, 3] noisy residue centroids
        trunk_tokens: Tensor,  # [B, L, c_token] pre-computed trunk embeddings
        sigma: Tensor,       # [B] continuous noise level
        mask: Optional[Tensor] = None,  # [B, L]
        x0_prev: Optional[Tensor] = None,  # [B, L, 3] previous x0 prediction (self-conditioning)
    ) -> Tensor:
        """Predict x0 using pre-computed trunk and continuous sigma.

        Combines AF3's trunk-once optimization with continuous sigma conditioning.
        Supports self-conditioning for improved inference.

        Args:
            x_t: Noisy centroids (possibly augmented)
            trunk_tokens: Pre-computed trunk embeddings
            sigma: Continuous noise level
            mask: Valid residue mask
            x0_prev: Previous x0 prediction for self-conditioning (optional)

        Returns:
            x0_pred: [B, L, 3] predicted clean centroids
        """
        B, L, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Embed noisy coordinates
        coord_emb = self.coord_embed(x_t)  # [B, L, c_token]

        # Additive conditioning with pre-computed trunk tokens
        tokens = coord_emb + trunk_tokens  # [B, L, c_token]

        # Self-conditioning: add embedding of previous prediction
        if x0_prev is not None:
            self_cond_emb = self.self_cond_embed(x0_prev)  # [B, L, c_token]
            tokens = tokens + self_cond_emb

        # Continuous sigma conditioning
        sigma_cond = self._embed_sigma(sigma).unsqueeze(1).expand(-1, L, -1)

        # Diffusion transformer
        tokens = self.diff_transformer(tokens, sigma_cond, mask)

        # Output: predict clean centroids x0 directly
        x0_pred = self.output_proj(tokens)  # [B, L, 3]

        return x0_pred

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,      # [B, L]
        chain_ids: Tensor,   # [B, L]
        res_idx: Tensor,     # [B, L]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute trunk embeddings from sequence features (for Stage 2 or multi-copy).

        The trunk processes ONLY sequence features, enabling trunk-once optimization.

        Returns:
            trunk_tokens: [B, L, c_token] embeddings
        """
        return self.trunk(aa_seq, chain_ids, res_idx, mask)

    def forward_with_trunk(
        self,
        x_t: Tensor,         # [B, L, 3] noisy residue centroids
        trunk_tokens: Tensor,  # [B, L, c_token] pre-computed trunk embeddings
        t: Tensor,           # [B] timestep
        mask: Optional[Tensor] = None,  # [B, L]
    ) -> Tensor:
        """Predict x0 using pre-computed trunk tokens (for efficient multi-copy training).

        This is the AF3-style training optimization: trunk runs ONCE on clean coords,
        then denoiser runs on multiple noisy copies with different augmentations.

        Args:
            x_t: Noisy centroids (possibly augmented)
            trunk_tokens: Pre-computed trunk embeddings from clean centroids
            t: Timestep indices
            mask: Valid residue mask

        Returns:
            x0_pred: [B, L, 3] predicted clean centroids
        """
        B, L, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Embed noisy coordinates
        coord_emb = self.coord_embed(x_t)  # [B, L, c_token]

        # Additive conditioning with pre-computed trunk tokens
        tokens = coord_emb + trunk_tokens  # [B, L, c_token]

        # Timestep conditioning
        time_cond = self.time_embed(t).unsqueeze(1).expand(-1, L, -1)

        # Diffusion transformer
        tokens = self.diff_transformer(tokens, time_cond, mask)

        # Output: predict clean centroids x0 directly
        x0_pred = self.output_proj(tokens)  # [B, L, 3]

        return x0_pred

    def count_parameters(self) -> dict:
        """Count parameters in trunk vs denoiser."""
        trunk_params = sum(p.numel() for p in self.trunk.parameters())

        denoiser_params = (
            sum(p.numel() for p in self.time_embed.parameters()) +
            sum(p.numel() for p in self.coord_embed.parameters()) +
            sum(p.numel() for p in self.diff_transformer.parameters()) +
            sum(p.numel() for p in self.output_proj.parameters())
        )

        total = trunk_params + denoiser_params

        return {
            'trunk': trunk_params,
            'denoiser': denoiser_params,
            'total': total,
            'trunk_pct': 100 * trunk_params / total,
            'denoiser_pct': 100 * denoiser_params / total,
        }
