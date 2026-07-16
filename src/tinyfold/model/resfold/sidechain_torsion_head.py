"""Stage-3 sidechain packing by TORSION diffusion (variant A, B3).

Design: notes/2026-07-14-sidechain-diffusion-stage3-SPEC.md §4A, §6.

The free-offset prototype (variant B) floored at ~2.0 A sampled: its Cartesian
denoiser regressed toward a per-residue MEAN at mid sigma and the Euler ODE
amplified that bias (the S5 finding). This head diffuses the chi torsions on the
torus instead. Two properties change the failure mode:

  * the target is BOUNDED and periodic (chi in (-pi, pi]) -- there is no unbounded
    Cartesian direction for the ODE to run away along;
  * chi is fed and predicted as (cos, sin) UNIT vectors, so the representation is
    naturally normalized and a low-sigma noised chi pins chi0 tightly.

Formulation (wrapped-normal / x0-prediction, DiffPack-style):
  * forward noise: chi_t = wrap(chi0 + sigma * eps), eps ~ N(0, I) per angle;
  * the head predicts chi0 directly as a unit 2-vector per chi (atan2 -> angle);
  * loss (B4) is the symmetry-corrected wrapped angular error, chi-masked.

Conditioning mirrors the free-offset head: restype identity + sigma + sinusoidal
position (a residue transformer is otherwise permutation-invariant -> per-restype
mean floor) + optional frozen-backbone geometry, denoised JOINTLY so packing can
resolve clashes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from tinyfold.atom14 import NUM_CHI


def wrap_angle(a: Tensor) -> Tensor:
    """Wrap radians to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class SidechainTorsionHead(nn.Module):
    """Wrapped-diffusion denoiser over chi1..chi4, conditioned on identity.

    Predicts clean chi0 from noised chi_t. chi is carried as (cos, sin) so the
    network never sees the +-pi wraparound discontinuity, and the output is a unit
    2-vector per chi (normalized, then atan2 to an angle by the caller/loss).

    ``sigma`` is the angular noise scale (radians). No EDM c_skip/c_out
    preconditioning: those are for unbounded Cartesian targets; here the (cos,sin)
    input is already bounded and the head predicts chi0 directly.
    """

    def __init__(
        self,
        c_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        n_restypes: int = 21,
        use_tokens: bool = True,
        backbone_cond: bool = True,
    ):
        super().__init__()
        self.c_token = c_token
        self.use_tokens = bool(use_tokens)

        # chi_t as (cos, sin) per chi -> 2 * NUM_CHI features.
        self.chi_in = nn.Linear(2 * NUM_CHI, c_token)
        self.restype_emb = nn.Embedding(n_restypes, c_token)

        self.backbone_cond = bool(backbone_cond)
        if self.backbone_cond:
            self.bb_enc = nn.Sequential(
                nn.Linear(4 * 3, c_token), nn.SiLU(), nn.Linear(c_token, c_token)
            )
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
        # Predict chi0 as a (cos, sin) pair per chi.
        self.proj = nn.Linear(c_token, 2 * NUM_CHI)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def _sigma_embed(self, sigma: Tensor) -> Tensor:
        # Condition on log-sigma (the noise decades matter, not the raw scale).
        c_noise = torch.log(sigma.clamp(min=1e-6))
        half = self.c_token // 2
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=sigma.device) * -scale)
        emb = c_noise.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.sigma_mlp(emb)  # [B, c_token]

    def _pos_embed(self, length: int, device) -> Tensor:
        half = self.c_token // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        pos = torch.arange(length, device=device).unsqueeze(-1).float()
        ang = pos * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(
        self,
        chi_t: Tensor,               # [B, L, 4] noised chi (radians)
        aatype: Tensor,              # [B, L] long
        sigma: Tensor,               # [B]
        tokens: Tensor | None = None,        # [B, L, c_token]
        mask: Tensor | None = None,          # [B, L] bool
        backbone_feats: Tensor | None = None,  # [B, L, 4, 3] per-complex-centered
        return_vec: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Denoise to chi0. Returns ``chi0 [B,L,4]`` (angles), or ``(chi0, vec)``
        with ``vec [B,L,4,2]`` the unit (cos,sin) prediction when ``return_vec``.
        """
        B, L = chi_t.shape[:2]
        feats = torch.stack([torch.cos(chi_t), torch.sin(chi_t)], dim=-1)   # [B,L,4,2]
        h = self.chi_in(feats.reshape(B, L, 2 * NUM_CHI))
        h = h + self.restype_emb(aatype)
        h = h + self._sigma_embed(sigma).unsqueeze(1)
        h = h + self._pos_embed(L, chi_t.device).unsqueeze(0)
        if self.backbone_cond and backbone_feats is not None:
            h = h + self.bb_enc(backbone_feats.reshape(B, L, 4 * 3))
        if self.use_tokens and tokens is not None:
            h = h + tokens

        attn_mask = ~mask if mask is not None else None
        h = self.norm(self.transformer(h, src_key_padding_mask=attn_mask))
        vec = self.proj(h).reshape(B, L, NUM_CHI, 2)                        # [B,L,4,2]
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)                 # unit circle
        chi0 = torch.atan2(vec[..., 1], vec[..., 0])                       # [B,L,4]
        if mask is not None:
            chi0 = chi0 * mask.unsqueeze(-1).to(chi0.dtype)
        if return_vec:
            return chi0, vec
        return chi0
