"""ResFold One-Step: centroid diffusion + atom head, end-to-end.

A single network that keeps residue-centroid diffusion as the headline science
target (4x fewer tokens than atom-level), but adds a small atom head reading the
denoiser tokens so the model also produces PDB-renderable backbone atoms in one
forward pass. Trained end-to-end with both centroid and atom losses.

Architecture:
1. ResidueEncoder (Trunk): sequence-only, runs ONCE per sample.
2. DiffusionTransformer (Denoiser): runs each step; conditioned on sigma; emits
   denoiser_tokens [B, L, c_token] used by BOTH output heads.
3. Centroid head: Linear(c_token -> 3) on denoiser_tokens -> x0_centroids.
4. Atom head: a 2-layer transformer on denoiser_tokens -> 4 atom offsets per
   residue. Atoms = x0_centroids[..., None, :] + atom_offsets.

The two heads are parallel (not cascaded). The atom head sees the SAME denoiser
tokens the centroid head sees; this forces the trunk + denoiser to learn features
useful for both jobs, which acts as a regularizer.

Loss combination (assembled in the training script):
    L = mse(centroid_pred, centroid_gt, Kabsch)
      + alpha * mse(atom_pred, atom_gt, Kabsch)
      + beta * geometry_losses(atom_pred)
      + gamma * distance_consistency(centroid_pred)

The training script applies a warmup on alpha so the centroid head trains alone
for the first ~500 steps; this prevents a noisy atom head from poisoning the
trunk early.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseDecoder
from .denoiser import DiffusionTransformer, ResidueEncoder


class AtomHead(nn.Module):
    """Small shallow transformer that maps denoiser tokens to backbone-atom offsets.

    A token-level transformer over residues (not over atoms) followed by a Linear
    that produces 12 outputs per residue, reshaped to [L, 4, 3]. Atom positions
    are computed downstream as centroid + offset, so the head is predicting
    relative geometry to the centroid rather than absolute coordinates.

    Defaults (c_token=128, n_layers=2, n_heads=4) give roughly 0.4M params, which
    keeps the Phase A configuration at ~2M total.
    """

    def __init__(
        self,
        c_token: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token,
            nhead=n_heads,
            dim_feedforward=c_token * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(c_token)
        self.proj = nn.Linear(c_token, 4 * 3)

        # Small init on the output projection so atom offsets start near zero
        # (so atoms ≈ centroid at init; geometry losses then nudge them apart).
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Predict 4 backbone-atom offsets per residue.

        Args:
            tokens: [B, L, c_token] denoiser tokens.
            mask:   [B, L] valid-residue mask (True = real residue).

        Returns:
            offsets: [B, L, 4, 3] offsets relative to the predicted centroid.
        """
        B, L, _ = tokens.shape
        attn_mask = ~mask if mask is not None else None
        x = self.transformer(tokens, src_key_padding_mask=attn_mask)
        x = self.norm(x)
        offsets = self.proj(x).view(B, L, 4, 3)
        if mask is not None:
            offsets = offsets * mask.unsqueeze(-1).unsqueeze(-1).float()
        return offsets


class ResFoldOneStep(BaseDecoder):
    """One-step ResFold: residue-centroid diffusion with a co-trained atom head.

    Returns both centroid predictions [B, L, 3] and full backbone atom predictions
    [B, L, 4, 3] from a single forward pass. The diffusion target is centroids
    only; atoms come from a parallel head that does not feed back into the
    diffusion loop.

    The discrete-timestep `forward()` is kept for backward compatibility with
    samplers that still use `t` indices, but training uses `forward_sigma()` with
    continuous noise levels (AF3-style).
    """

    def __init__(
        self,
        c_token: int = 128,
        trunk_layers: int = 4,
        trunk_heads: int = 8,
        denoiser_blocks: int = 4,
        denoiser_heads: int = 8,
        atom_head_layers: int = 2,
        atom_head_heads: int = 4,
        n_timesteps: int = 50,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_timesteps = n_timesteps
        self.sigma_data = 1.0

        # === TRUNK (sequence-only, runs once) ===
        self.trunk = ResidueEncoder(
            c_token=c_token,
            n_layers=trunk_layers,
            n_heads=trunk_heads,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # === DENOISER (per-step) ===
        self.coord_embed = nn.Linear(3, c_token)
        self.self_cond_embed = nn.Linear(3, c_token)
        self.time_embed = nn.Embedding(n_timesteps, c_token)
        self.sigma_embed = nn.Sequential(
            nn.Linear(c_token, c_token),
            nn.SiLU(),
            nn.Linear(c_token, c_token),
        )
        self.diff_transformer = DiffusionTransformer(
            c_token=c_token,
            n_blocks=denoiser_blocks,
            n_heads=denoiser_heads,
            dropout=dropout,
        )

        # === HEADS (parallel) ===
        self.centroid_proj = nn.Linear(c_token, 3)
        self.atom_head = AtomHead(
            c_token=c_token,
            n_layers=atom_head_layers,
            n_heads=atom_head_heads,
            dropout=dropout,
        )

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        """AF3-style continuous-sigma embedding via sinusoidal Fourier features."""
        c_noise = torch.log(sigma / self.sigma_data + 1e-8) / 4.0
        half_dim = self.c_token // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb_scale)
        emb = c_noise.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.sigma_embed(emb)

    def _denoiser_tokens(
        self,
        x_t: Tensor,
        trunk_tokens: Tensor,
        cond: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the diffusion transformer and return final tokens [B, L, c_token]."""
        L = x_t.shape[1]
        tokens = self.coord_embed(x_t) + trunk_tokens
        if x0_prev is not None:
            tokens = tokens + self.self_cond_embed(x0_prev)
        cond_per_token = cond.unsqueeze(1).expand(-1, L, -1)
        return self.diff_transformer(tokens, cond_per_token, mask)

    def _heads(
        self, denoiser_tokens: Tensor, mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Compute centroid + atom predictions from shared denoiser tokens."""
        centroid_pred = self.centroid_proj(denoiser_tokens)  # [B, L, 3]
        atom_offsets = self.atom_head(denoiser_tokens, mask)  # [B, L, 4, 3]
        atoms_pred = centroid_pred.unsqueeze(2) + atom_offsets  # [B, L, 4, 3]
        return centroid_pred, atoms_pred

    def forward_sigma(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Continuous-sigma forward pass. Returns (centroid_pred, atoms_pred)."""
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)
        cond = self._embed_sigma(sigma)
        denoiser_tokens = self._denoiser_tokens(x_t, trunk_tokens, cond, mask, x0_prev)
        return self._heads(denoiser_tokens, mask)

    def forward_sigma_with_trunk(
        self,
        x_t: Tensor,
        trunk_tokens: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Continuous-sigma forward with precomputed trunk tokens."""
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        cond = self._embed_sigma(sigma)
        denoiser_tokens = self._denoiser_tokens(x_t, trunk_tokens, cond, mask, x0_prev)
        return self._heads(denoiser_tokens, mask)

    def forward(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Discrete-timestep forward (for backwards compatibility)."""
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask)
        cond = self.time_embed(t)
        denoiser_tokens = self._denoiser_tokens(x_t, trunk_tokens, cond, mask, None)
        return self._heads(denoiser_tokens, mask)

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.trunk(aa_seq, chain_ids, res_idx, mask)

    def count_parameters(self) -> dict:
        trunk = sum(p.numel() for p in self.trunk.parameters())
        denoiser = (
            sum(p.numel() for p in self.coord_embed.parameters())
            + sum(p.numel() for p in self.self_cond_embed.parameters())
            + sum(p.numel() for p in self.time_embed.parameters())
            + sum(p.numel() for p in self.sigma_embed.parameters())
            + sum(p.numel() for p in self.diff_transformer.parameters())
            + sum(p.numel() for p in self.centroid_proj.parameters())
        )
        atom_head = sum(p.numel() for p in self.atom_head.parameters())
        total = trunk + denoiser + atom_head
        return {
            "trunk": trunk,
            "denoiser": denoiser,
            "atom_head": atom_head,
            "total": total,
            "trunk_pct": 100 * trunk / total,
            "denoiser_pct": 100 * denoiser / total,
            "atom_head_pct": 100 * atom_head / total,
        }
