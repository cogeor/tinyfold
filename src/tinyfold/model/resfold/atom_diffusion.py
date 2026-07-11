"""Atom-diffusion stage: iterative refinement of backbone atoms.

Replaces the one-shot atom REGRESSION head (which plateaus: overfitting a single
complex, atoms floor at ~0.9 A / DockQ 0.75 for both free-offset and frame heads)
with an EDM DIFFUSION over per-residue backbone offsets ``delta = atoms - centroid``.

Why this should break the ceiling: a diffusion denoiser conditions on the CURRENT
atom state (where atoms are right now) and refines iteratively, so it can fix
inter-atom / interface geometry that a single feed-forward map cannot.

Design (see notes/2026-07-11-crop-diffusion-2stage-SPEC.md):
- Centroids stay a separate (cheap, global) diffusion; this stage runs on TOP,
  conditioned on the centroid stage's per-residue tokens (which encode the
  denoised centroid layout) -> atoms are anchored to the global centroid
  scaffold, NOT to neighbours (drift-free, the AF3/Boltz principle). No crops at
  our sizes; run full.
- End-to-end: atoms = centroid_pred + delta_pred, loss on absolute atoms, so
  gradients flow atom -> centroid -> trunk.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AtomDiffusionHead(nn.Module):
    """EDM denoiser for per-residue backbone offsets, conditioned on tokens.

    Diffuses ``delta [B, L, 4, 3]`` (offset of the 4 backbone atoms from the
    residue centroid). A shallow residue-level transformer maps
    ``(noised delta, conditioning tokens, sigma)`` -> denoised delta.

    ``sigma_data`` is the std of the offset distribution in the model's
    normalized coordinate units (backbone atoms sit ~1.5 A from the centroid;
    at global_scale~=11 that is ~0.14).
    """

    def __init__(self, c_token: int = 128, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.0, sigma_data: float = 0.15):
        super().__init__()
        self.c_token = c_token
        self.sigma_data = float(sigma_data)

        self.coord_in = nn.Linear(4 * 3, c_token)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(c_token, c_token), nn.SiLU(), nn.Linear(c_token, c_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token, nhead=n_heads, dim_feedforward=c_token * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c_token)
        self.proj = nn.Linear(c_token, 4 * 3)
        # Small output init so F starts near zero (delta ~= c_skip * delta_t).
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def edm_coefficients(self, sigma: Tensor):
        """Karras (EDM) preconditioning; sigma [B] -> c_skip,c_out,c_in [B,1,1,1], c_noise [B]."""
        sd = self.sigma_data
        s2 = sigma * sigma
        denom = s2 + sd * sd
        view = (-1, 1, 1, 1)
        c_skip = (sd * sd / denom).view(*view)
        c_out = (sigma * sd / torch.sqrt(denom)).view(*view)
        c_in = (1.0 / torch.sqrt(denom)).view(*view)
        c_noise = 0.25 * torch.log(sigma + 1e-8)
        return c_skip, c_out, c_in, c_noise

    def _sigma_embed(self, c_noise: Tensor) -> Tensor:
        half = self.c_token // 2
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=c_noise.device) * -scale)
        emb = c_noise.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.sigma_mlp(emb)  # [B, c_token]

    def forward(self, delta_t: Tensor, tokens: Tensor, sigma: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """delta_t [B,L,4,3], tokens [B,L,c], sigma [B] -> denoised delta_0 [B,L,4,3]."""
        B, L, _, _ = delta_t.shape
        c_skip, c_out, c_in, c_noise = self.edm_coefficients(sigma)
        x = (c_in * delta_t).reshape(B, L, 12)
        h = self.coord_in(x) + tokens + self._sigma_embed(c_noise).unsqueeze(1)
        attn_mask = ~mask if mask is not None else None
        h = self.norm(self.transformer(h, src_key_padding_mask=attn_mask))
        F = self.proj(h).reshape(B, L, 4, 3)
        delta0 = c_skip * delta_t + c_out * F
        if mask is not None:
            delta0 = delta0 * mask.unsqueeze(-1).unsqueeze(-1).float()
        return delta0
