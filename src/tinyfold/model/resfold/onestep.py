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
        aa_embed: str = "learned",
        esm_dim: Optional[int] = None,
        confidence_head: bool = False,
        sigma_data: float = 1.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_timesteps = n_timesteps
        # sigma_data is the EDM preconditioning constant; should match the
        # std of the data distribution in the units the model trains in.
        # 1.0 is correct when coords are per-sample-normalized to unit std;
        # ~16.0 (Boltz/AF3 convention) is correct when coords are in raw
        # Angstroms and only centered.
        self.sigma_data = float(sigma_data)
        self.aa_embed_mode = aa_embed

        # === TRUNK (sequence-only, runs once) ===
        self.trunk = ResidueEncoder(
            c_token=c_token,
            n_layers=trunk_layers,
            n_heads=trunk_heads,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
            aa_embed=aa_embed,
            esm_dim=esm_dim,
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

        # Optional per-target confidence head (Loop 06). Disabled by default so
        # legacy training runs stay byte-identical. When enabled, returns a
        # scalar predicted lDDT per target from mean-pooled denoiser tokens.
        if confidence_head:
            from .confidence_head import ConfidenceHead
            self.confidence_head = ConfidenceHead(
                c_token=c_token,
                dropout=dropout,
            )
        else:
            self.confidence_head = None

    def _edm_coefficients(self, sigma: Tensor):
        """Karras (EDM) preconditioning coefficients.

        Returns c_skip, c_out, c_in, c_noise — each shaped [B, 1, 1] so they
        broadcast over (L, 3). c_noise is shaped [B] (input to embedder).

        At sigma -> 0:  c_skip -> 1, c_out -> 0  (output = x_t + tiny)
        At sigma -> inf: c_skip -> 0, c_out -> sigma_data (output dominated by F)
        """
        sd = self.sigma_data
        s2 = sigma * sigma
        denom = s2 + sd * sd
        c_skip = (sd * sd / denom).view(-1, 1, 1)
        c_out = (sigma * sd / torch.sqrt(denom)).view(-1, 1, 1)
        c_in = (1.0 / torch.sqrt(denom)).view(-1, 1, 1)
        c_noise = 0.25 * torch.log(sigma + 1e-8)  # [B]
        return c_skip, c_out, c_in, c_noise

    def _embed_c_noise(self, c_noise: Tensor) -> Tensor:
        """Sinusoidal Fourier embedding of EDM's c_noise = 0.25 * log(sigma)."""
        half_dim = self.c_token // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=c_noise.device) * -emb_scale)
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

    def _heads_edm(
        self,
        denoiser_tokens: Tensor,
        x_t: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """EDM-blended centroid + atom predictions from shared denoiser tokens.

        F_centroid is the raw projection; the EDM blend gives the final centroid.
        Atom offsets are emitted in the centroid's frame (atoms = centroid + offset),
        so the atom head does NOT itself need EDM blending — its output is a delta.
        """
        F_centroid = self.centroid_proj(denoiser_tokens)              # [B, L, 3]
        centroid_pred = c_skip * x_t + c_out * F_centroid             # [B, L, 3]
        atom_offsets = self.atom_head(denoiser_tokens, mask)          # [B, L, 4, 3]
        atoms_pred = centroid_pred.unsqueeze(2) + atom_offsets        # [B, L, 4, 3]
        return centroid_pred, atoms_pred

    def _predict_confidence(
        self,
        denoiser_tokens: Tensor,
        mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Run the optional confidence head; return ``None`` when disabled.

        Returned tensor (when present) has shape ``[B]`` and lives in ``[0, 1]``
        (predicted lDDT). Pooling uses the residue mask via ``ConfidenceHead``.
        """
        if self.confidence_head is None:
            return None
        return self.confidence_head(denoiser_tokens, mask)

    def forward_sigma(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
        esm_embed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Continuous-sigma forward (EDM-preconditioned).

        Returns ``(centroid_pred, atoms_pred, pred_lddt_or_None)``. The third
        slot is the per-target predicted lDDT in ``[0, 1]`` when the model was
        constructed with ``confidence_head=True``, else ``None``.

        ``esm_embed`` (``[B, L, esm_dim]``) is required when the model was
        constructed with ``aa_embed in {"esm2_35M", "esm2_150M"}``. In the
        default ``aa_embed="learned"`` mode it is ignored.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)
        cond = self._embed_c_noise(c_noise)
        denoiser_tokens = self._denoiser_tokens(c_in * x_t, trunk_tokens, cond, mask, x0_prev)
        centroid_pred, atoms_pred = self._heads_edm(
            denoiser_tokens, x_t, c_skip, c_out, mask
        )
        pred_lddt = self._predict_confidence(denoiser_tokens, mask)
        return centroid_pred, atoms_pred, pred_lddt

    def forward_sigma_with_trunk(
        self,
        x_t: Tensor,
        trunk_tokens: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Continuous-sigma forward with precomputed trunk tokens (EDM-preconditioned).

        Returns ``(centroid_pred, atoms_pred, pred_lddt_or_None)`` — same
        contract as :meth:`forward_sigma`.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        cond = self._embed_c_noise(c_noise)
        denoiser_tokens = self._denoiser_tokens(c_in * x_t, trunk_tokens, cond, mask, x0_prev)
        centroid_pred, atoms_pred = self._heads_edm(
            denoiser_tokens, x_t, c_skip, c_out, mask
        )
        pred_lddt = self._predict_confidence(denoiser_tokens, mask)
        return centroid_pred, atoms_pred, pred_lddt

    def forward(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
        esm_embed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Discrete-timestep forward (legacy path).

        No EDM here — the discrete timestep -> sigma mapping is not defined in this
        model. Caller should prefer `forward_sigma` for new code. Output is the
        raw projection (centroid_pred = F_centroid); atoms = centroid + offsets.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)
        cond = self.time_embed(t)
        denoiser_tokens = self._denoiser_tokens(x_t, trunk_tokens, cond, mask, None)
        F_centroid = self.centroid_proj(denoiser_tokens)
        atom_offsets = self.atom_head(denoiser_tokens, mask)
        atoms_pred = F_centroid.unsqueeze(2) + atom_offsets
        return F_centroid, atoms_pred

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Optional[Tensor] = None,
        esm_embed: Optional[Tensor] = None,
    ) -> Tensor:
        return self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)

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
        confidence_head = (
            sum(p.numel() for p in self.confidence_head.parameters())
            if self.confidence_head is not None
            else 0
        )
        total = trunk + denoiser + atom_head + confidence_head
        return {
            "trunk": trunk,
            "denoiser": denoiser,
            "atom_head": atom_head,
            "confidence_head": confidence_head,
            "total": total,
            "trunk_pct": 100 * trunk / total,
            "denoiser_pct": 100 * denoiser / total,
            "atom_head_pct": 100 * atom_head / total,
            "confidence_head_pct": 100 * confidence_head / total,
        }
