"""Stage-3 sidechain diffusion: pack sidechains onto a FROZEN backbone.

Design: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md.

This is the third EDM diffusion of the same kind we already run twice (centroids,
then backbone offsets), and it is deliberately the FREE-OFFSET variant (§4B) --
the prototype that reuses :class:`AtomDiffusionHead` almost verbatim (4 -> 10
slots, sigma_data 0.15 -> ~0.35). The torsion/torus variant (§4A) is deferred
behind the overfit gate (§11): build the cheap thing, prove a small head can
overfit one structure's sidechains, and only then pay for a wrapped noiser.

WHY THE LOCAL FRAME (§4)
------------------------
Because the backbone is FIXED, every sidechain atom can be expressed in its
residue's own N-CA-C frame. That makes the whole stage rotation/translation
invariant FOR FREE: rotate the complex and the local coordinates do not move. It
is the clean answer to the frame-dependent-offset caveat that dogged the
backbone offset head (which needed explicit aug_R bookkeeping) -- here rotation
augmentation simply works, with nothing to bookkeep.

WHY JOINT, NOT PER-RESIDUE-INDEPENDENT (§5)
-------------------------------------------
All residues are denoised jointly under one shared sigma, each attending to the
others' CURRENT sidechain estimate, so packing and clashes can resolve during
the reverse process. Independent per-residue packing gives no inter-sidechain
clash resolution and is explicitly rejected by the spec. This first cut uses a
DENSE residue transformer (the same coupling mechanism AtomDiffusionHead uses);
the spec's sparse CB-CB neighbour graph is the efficiency refinement, worth
doing once the gate passes and L grows.

Stages 1-2 are untouched and the backbone is frozen (callers pass detached
coords), per §2's "never perturbs the working backbone path".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from tinyfold.atom14 import NUM_ATOM14
from tinyfold.retrieval.template_features import local_frames

# atom14 slots 0-3 are the backbone; 4..13 are the sidechain heavy atoms.
NUM_SIDECHAIN_SLOTS = NUM_ATOM14 - 4


def sidechain_to_local(atom14: Tensor, eps: float = 1e-6) -> Tensor:
    """Global sidechain coords -> each residue's local backbone frame.

    Args:
        atom14: ``[B, L, 14, 3]`` (slots 0-3 = N, CA, C, O).

    Returns:
        ``[B, L, 10, 3]`` sidechain positions relative to CA, expressed in the
        residue's N-CA-C frame. Invariant to global rotation AND translation.
    """
    R = local_frames(atom14[..., :4, :], eps=eps)          # [B, L, 3, 3] columns = axes
    ca = atom14[..., 1:2, :]                               # [B, L, 1, 3]
    rel = atom14[..., 4:, :] - ca                          # [B, L, 10, 3]
    # Express in the local frame: R^T @ v  (einsum over the axis dim).
    return torch.einsum("...mk,...am->...ak", R, rel)


def sidechain_to_global(local: Tensor, backbone: Tensor, eps: float = 1e-6) -> Tensor:
    """Inverse of :func:`sidechain_to_local`.

    Args:
        local:    ``[B, L, 10, 3]`` local-frame sidechain coords.
        backbone: ``[B, L, 4, 3]`` the FROZEN N, CA, C, O.

    Returns:
        ``[B, L, 10, 3]`` global sidechain coords.
    """
    R = local_frames(backbone, eps=eps)                    # [B, L, 3, 3]
    ca = backbone[..., 1:2, :]                             # [B, L, 1, 3]
    # R @ v, then translate back to CA.
    return torch.einsum("...km,...am->...ak", R, local) + ca


def assemble_atom14(backbone: Tensor, sidechain_global: Tensor) -> Tensor:
    """Splice a frozen backbone and predicted sidechains into ``[B, L, 14, 3]``."""
    return torch.cat([backbone, sidechain_global], dim=-2)


class SidechainDiffusionHead(nn.Module):
    """EDM denoiser for local-frame sidechain offsets, conditioned on identity.

    Mirrors :class:`~tinyfold.model.resfold.atom_diffusion.AtomDiffusionHead`
    (Karras preconditioning, sigma embedding, residue transformer), operating on
    ``[B, L, 10, 3]`` local-frame coordinates instead of ``[B, L, 4, 3]`` global
    offsets.

    Conditioning per residue: noised sidechain state + sigma + RESIDUE IDENTITY
    (an embedding -- without it the head cannot know whether it is packing a GLY
    or a TRP) + optionally the trunk token.

    ``sigma_data`` is the std of local-frame sidechain coordinates in normalized
    units: sidechains reach ~6 A from CA (ARG/TRP), so at global_scale ~= 11 the
    spread is ~0.3-0.5.
    """

    def __init__(
        self,
        c_token: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        sigma_data: float = 0.35,
        n_restypes: int = 21,
        use_tokens: bool = True,
    ):
        super().__init__()
        self.c_token = c_token
        self.sigma_data = float(sigma_data)
        self.use_tokens = bool(use_tokens)

        self.coord_in = nn.Linear(NUM_SIDECHAIN_SLOTS * 3, c_token)
        self.restype_emb = nn.Embedding(n_restypes, c_token)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(c_token, c_token), nn.SiLU(), nn.Linear(c_token, c_token)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token, nhead=n_heads, dim_feedforward=c_token * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(c_token)
        self.proj = nn.Linear(c_token, NUM_SIDECHAIN_SLOTS * 3)
        # Small output init so F starts near zero (x0 ~= c_skip * x_t).
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

    def forward(
        self,
        x_t: Tensor,                 # [B, L, 10, 3] noised LOCAL-frame sidechain coords
        aatype: Tensor,              # [B, L] long residue types
        sigma: Tensor,               # [B]
        tokens: Tensor | None = None,  # [B, L, c_token] trunk tokens
        mask: Tensor | None = None,    # [B, L] bool, True = valid residue
    ) -> Tensor:
        """Denoise to ``x_0`` ``[B, L, 10, 3]`` (local frame)."""
        B, L = x_t.shape[:2]
        c_skip, c_out, c_in, c_noise = self.edm_coefficients(sigma)

        h = self.coord_in((c_in * x_t).reshape(B, L, NUM_SIDECHAIN_SLOTS * 3))
        h = h + self.restype_emb(aatype)
        h = h + self._sigma_embed(c_noise).unsqueeze(1)
        if self.use_tokens and tokens is not None:
            h = h + tokens

        attn_mask = ~mask if mask is not None else None
        h = self.norm(self.transformer(h, src_key_padding_mask=attn_mask))
        F = self.proj(h).reshape(B, L, NUM_SIDECHAIN_SLOTS, 3)

        x0 = c_skip * x_t + c_out * F
        if mask is not None:
            x0 = x0 * mask.unsqueeze(-1).unsqueeze(-1).to(x0.dtype)
        return x0
