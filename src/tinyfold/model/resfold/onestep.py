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
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseDecoder
from .denoiser import DiffusionTransformer, ResidueEncoder


class ModelOutput(NamedTuple):
    """Return type of ``forward_sigma`` / ``forward_sigma_with_trunk``.

    A 3-tuple, so every existing ``centroid, atoms, pred_lddt = model.forward_sigma(...)``
    caller keeps working unchanged, but it also exposes named accessors
    (``out.centroid_pred`` etc.). ``pred_lddt`` is ``None`` unless the model was
    built with ``confidence_head=True``.
    """

    centroid_pred: Tensor
    atoms_pred: Tensor
    pred_lddt: Tensor | None


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

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
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


# Idealized backbone geometry (N, CA, C, O), CA-centred, in ANGSTROMS. Standard
# bond lengths/angles; O placed off C in the peptide plane (psi averaged out).
_IDEAL_BACKBONE_A = torch.tensor([
    [-0.525,  1.363,  0.000],   # N
    [ 0.000,  0.000,  0.000],   # CA
    [ 1.526,  0.000,  0.000],   # C
    [ 2.130,  1.128,  0.000],   # O (approx)
], dtype=torch.float32)


def _rot6d_to_matrix(x: Tensor) -> Tensor:
    """[..., 6] -> [..., 3, 3] rotation via Gram-Schmidt (Zhou et al. 2019)."""
    a1, a2 = x[..., :3], x[..., 3:]
    e1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (e1 * a2).sum(-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(a2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)  # columns = axes


class FrameAtomHead(nn.Module):
    """Frame-based backbone placement (AF3/IPA-style encoding).

    Instead of predicting 12 free global-frame offsets, predict a per-residue
    rigid frame (6D rotation + 3D translation) and place a LEARNABLE idealized
    backbone template into it. This guarantees near-rigid backbone geometry and
    only asks the network to learn orientation + position -- far more precise and
    learnable than free offsets. Returns offsets relative to the centroid, so it
    is a drop-in replacement for :class:`AtomHead`.

    The template is a learnable ``[4, 3]`` parameter (init to ideal backbone /
    ``template_scale`` so it starts near-physical in the model's NORMALIZED
    coordinate units) and is mean-centred at use so ``mean(atoms) ~= centroid``.
    """

    def __init__(self, c_token: int = 128, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.0, template_scale: float = 11.0):
        super().__init__()
        self.c_token = c_token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_token, nhead=n_heads, dim_feedforward=c_token * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c_token)
        # 6 (rotation) + 3 (CA translation vs centroid) per residue.
        self.proj = nn.Linear(c_token, 9)
        nn.init.zeros_(self.proj.weight)
        # Init rotation to identity (6D = first two columns of I) so at start the
        # template is placed un-rotated; translation starts at zero.
        bias = torch.zeros(9)
        bias[0] = 1.0  # e1 = x
        bias[4] = 1.0  # e2 = y
        self.proj.bias.data.copy_(bias)
        self.template = nn.Parameter(_IDEAL_BACKBONE_A / float(template_scale))

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        B, L, _ = tokens.shape
        attn_mask = ~mask if mask is not None else None
        x = self.norm(self.transformer(tokens, src_key_padding_mask=attn_mask))
        out = self.proj(x)                                  # [B, L, 9]
        R = _rot6d_to_matrix(out[..., :6])                  # [B, L, 3, 3]
        t = out[..., 6:]                                    # [B, L, 3] CA vs centroid
        tmpl = self.template - self.template.mean(0, keepdim=True)  # mean-centred [4,3]
        # atoms_local = R @ tmpl^T  -> [B, L, 4, 3]; offset from centroid = that + t
        placed = torch.einsum("blij,kj->blki", R, tmpl)     # [B, L, 4, 3]
        offsets = placed + t.unsqueeze(2)                   # relative to centroid
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
        esm_dim: int | None = None,
        confidence_head: bool = False,
        sigma_data: float = 1.0,
        relpos_bias: bool = False,
        relpos_clip: int = 32,
        pair_repr: bool = False,
        c_pair: int = 64,
        pair_layers: int = 3,
        pair_hidden: int = 64,
        template_cond: bool = False,
        template_rbf: int = 32,
        template_d_max: float = 4.0,
        msa_cond: bool = False,
        grad_checkpoint: bool = False,
        pair_to_single: bool = False,
        frame_atom_head: bool = False,
        global_scale: float = 11.0,
        atom_diffusion: bool = False,
        atom_sigma_data: float = 0.15,
        atom_sigma_min: float = 0.002,
        atom_sigma_max: float = 1.0,
        sidechain_diffusion: bool = False,
        sc_head_layers: int = 2,
        sc_head_heads: int = 4,
        sc_neighbor_graph: bool = False,
        sc_neighbor_radius: float = 10.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_timesteps = n_timesteps
        self.template_cond_enabled = bool(template_cond)
        self.msa_cond_enabled = bool(msa_cond)
        self.atom_diffusion = bool(atom_diffusion)
        self.atom_sigma_min = float(atom_sigma_min)
        self.atom_sigma_max = float(atom_sigma_max)
        self.sidechain_diffusion = bool(sidechain_diffusion)
        # sigma_data is the EDM preconditioning constant; should match the
        # std of the data distribution in the units the model trains in.
        # 1.0 is correct when coords are per-sample-normalized to unit std;
        # ~16.0 (Boltz/AF3 convention) is correct when coords are in raw
        # Angstroms and only centered.
        self.sigma_data = float(sigma_data)
        self.aa_embed_mode = aa_embed
        self.relpos_bias_enabled = bool(relpos_bias)

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
            relpos_bias=relpos_bias,
            relpos_clip=relpos_clip,
            pair_repr=pair_repr,
            c_pair=c_pair,
            pair_layers=pair_layers,
            pair_hidden=pair_hidden,
            template_cond=template_cond,
            template_rbf=template_rbf,
            template_d_max=template_d_max,
            msa_cond=msa_cond,
            grad_checkpoint=grad_checkpoint,
            pair_to_single=pair_to_single,
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
            relpos_bias=relpos_bias,
            relpos_clip=relpos_clip,
        )

        # === HEADS (parallel) ===
        self.centroid_proj = nn.Linear(c_token, 3)
        if frame_atom_head:
            self.atom_head = FrameAtomHead(
                c_token=c_token,
                n_layers=atom_head_layers,
                n_heads=atom_head_heads,
                dropout=dropout,
                template_scale=global_scale,
            )
        else:
            self.atom_head = AtomHead(
                c_token=c_token,
                n_layers=atom_head_layers,
                n_heads=atom_head_heads,
                dropout=dropout,
            )

        # Optional atom-DIFFUSION stage (replaces the one-shot atom head at
        # eval; co-trained). Diffuses per-residue backbone offsets conditioned on
        # the denoiser tokens (which encode the centroid layout) + current atom
        # state. See atom_diffusion.py.
        if self.atom_diffusion:
            from .atom_diffusion import AtomDiffusionHead
            self.atom_diff_head = AtomDiffusionHead(
                c_token=c_token,
                n_layers=atom_head_layers,
                n_heads=atom_head_heads,
                dropout=dropout,
                sigma_data=atom_sigma_data,
            )
        else:
            self.atom_diff_head = None

        # Optional THIRD diffusion stage: sidechain packing by torsion (chi)
        # diffusion, conditioned on the denoiser tokens + the predicted backbone.
        # Sits on top of the atom-diffusion backbone; default off keeps every
        # existing run byte-identical. See sidechain_torsion_head.py + the
        # 2026-07-17 third-stage SPEC.
        if self.sidechain_diffusion:
            from .sidechain_torsion_head import SidechainTorsionHead
            self.sc_head = SidechainTorsionHead(
                c_token=c_token,
                n_layers=sc_head_layers,
                n_heads=sc_head_heads,
                dropout=dropout,
                use_tokens=True,
                backbone_cond=True,
                neighbor_graph=sc_neighbor_graph,
                neighbor_radius=sc_neighbor_radius,
            )
        else:
            self.sc_head = None

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
        mask: Tensor | None = None,
        x0_prev: Tensor | None = None,
        res_idx: Tensor | None = None,
        chain_ids: Tensor | None = None,
    ) -> Tensor:
        """Run the diffusion transformer and return final tokens [B, L, c_token].

        ``res_idx`` and ``chain_ids`` are forwarded to ``diff_transformer`` so
        its (optional) relpos bias can index into them. Required when the
        model was built with ``relpos_bias=True``; ignored otherwise.
        """
        L = x_t.shape[1]
        tokens = self.coord_embed(x_t) + trunk_tokens
        if x0_prev is not None:
            tokens = tokens + self.self_cond_embed(x0_prev)
        cond_per_token = cond.unsqueeze(1).expand(-1, L, -1)
        return self.diff_transformer(
            tokens, cond_per_token, mask,
            res_idx=res_idx, chain_ids=chain_ids,
        )

    def _heads_edm(
        self,
        denoiser_tokens: Tensor,
        x_t: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
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
        mask: Tensor | None,
    ) -> Tensor | None:
        """Run the optional confidence head; return ``None`` when disabled.

        Returned tensor (when present) has shape ``[B]`` and lives in ``[0, 1]``
        (predicted lDDT). Pooling uses the residue mask via ``ConfidenceHead``.
        """
        if self.confidence_head is None:
            return None
        return self.confidence_head(denoiser_tokens, mask)

    def _run_trunk_recycled(self, n_recycle: int, **trunk_kwargs) -> Tensor:
        """Run the sequence trunk with ``n_recycle`` recycling passes (C1).

        Each pass feeds the previous pass's token (and pair, when the pair track
        is on) representation back in. Only the FINAL pass is differentiated --
        all earlier passes run under ``torch.no_grad()`` -- so peak memory does
        not grow with ``n_recycle``. ``n_recycle=0`` runs the trunk exactly once
        with no fed-back state, which is bitwise-identical to the pre-recycling
        path (the recycle projection is zero-init and never even invoked).
        """
        recycle_tokens: Tensor | None = None
        recycle_pair: Tensor | None = None
        for i in range(n_recycle + 1):
            if i < n_recycle:
                with torch.no_grad():
                    tokens, pair_rep = self.trunk(
                        **trunk_kwargs, recycle_tokens=recycle_tokens,
                        recycle_pair=recycle_pair, return_pair=True,
                    )
                recycle_tokens = tokens.detach()
                recycle_pair = pair_rep.detach() if pair_rep is not None else None
            else:
                tokens, _ = self.trunk(
                    **trunk_kwargs, recycle_tokens=recycle_tokens,
                    recycle_pair=recycle_pair, return_pair=True,
                )
        return tokens

    def forward_sigma(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        sigma: Tensor,
        mask: Tensor | None = None,
        x0_prev: Tensor | None = None,
        esm_embed: Tensor | None = None,
        template_coords_res: Tensor | None = None,
        template_mask: Tensor | None = None,
        template_frame_id: Tensor | None = None,
        msa_feats: Tensor | None = None,
        n_recycle: int = 0,
    ) -> ModelOutput:
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
        trunk_tokens = self._run_trunk_recycled(
            n_recycle,
            aa_seq=aa_seq, chain_ids=chain_ids, res_idx=res_idx, mask=mask,
            esm_embed=esm_embed,
            template_coords_res=template_coords_res,
            template_mask=template_mask,
            template_frame_id=template_frame_id,
            msa_feats=msa_feats,
        )
        # The trunk is sequence/template-only; the noised-coord denoise + heads
        # live in the shared _with_trunk core.
        return self.forward_sigma_with_trunk(
            x_t, trunk_tokens, sigma, mask, x0_prev,
            res_idx=res_idx, chain_ids=chain_ids,
        )

    def centroid_tokens(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        sigma: Tensor,
        mask: Tensor | None = None,
        x0_prev: Tensor | None = None,
        esm_embed: Tensor | None = None,
        template_coords_res: Tensor | None = None,
        template_mask: Tensor | None = None,
        template_frame_id: Tensor | None = None,
        msa_feats: Tensor | None = None,
        n_recycle: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Centroid forward that also returns the denoiser tokens.

        For the atom-diffusion stage: returns ``(centroid_pred, denoiser_tokens,
        pred_lddt)``. The atom stage conditions on ``denoiser_tokens`` (which
        encode the denoised centroid layout).
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        trunk_tokens = self._run_trunk_recycled(
            n_recycle,
            aa_seq=aa_seq, chain_ids=chain_ids, res_idx=res_idx, mask=mask,
            esm_embed=esm_embed,
            template_coords_res=template_coords_res,
            template_mask=template_mask,
            template_frame_id=template_frame_id,
            msa_feats=msa_feats,
        )
        return self.centroid_tokens_with_trunk(
            x_t, trunk_tokens, sigma, mask, x0_prev,
            res_idx=res_idx, chain_ids=chain_ids,
        )

    def denoise_atoms(
        self,
        delta_t: Tensor,
        denoiser_tokens: Tensor,
        sigma_a: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """One atom-diffusion denoise step: noised offsets -> denoised offsets.

        ``delta`` are per-residue backbone offsets from the centroid. Conditions
        on ``denoiser_tokens`` from :meth:`centroid_tokens`.
        """
        assert self.atom_diff_head is not None, "atom_diffusion=False"
        return self.atom_diff_head(delta_t, denoiser_tokens, sigma_a, mask)

    def denoise_chi(
        self,
        chi_t: Tensor,
        denoiser_tokens: Tensor,
        sigma_c: Tensor,
        backbone_feats: Tensor,
        aatype: Tensor,
        mask: Tensor | None = None,
        ca_pos: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One sidechain-diffusion denoise step: noised chi -> clean chi0.

        Third diffusion stage. ``chi_t [B,L,4]`` are noised chi torsions;
        conditions on ``denoiser_tokens`` (centroid stage) and ``backbone_feats``
        ([B,L,4,3], the PREDICTED backbone, per-residue-centered/normalized).
        Returns ``(chi0 [B,L,4], vec [B,L,4,2])`` -- vec is the unit (cos,sin)
        used by the symmetry-corrected torsion loss.
        """
        assert self.sc_head is not None, "sidechain_diffusion=False"
        return self.sc_head(
            chi_t, aatype, sigma_c, tokens=denoiser_tokens, mask=mask,
            backbone_feats=backbone_feats, ca_pos=ca_pos, return_vec=True,
        )

    def centroid_tokens_with_trunk(
        self,
        x_t: Tensor,
        trunk_tokens: Tensor,
        sigma: Tensor,
        mask: Tensor | None = None,
        x0_prev: Tensor | None = None,
        res_idx: Tensor | None = None,
        chain_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Like :meth:`centroid_tokens` but with precomputed trunk tokens.

        Lets the atom-diffusion sampler reuse a single trunk pass across K
        centroid samples (the trunk is sequence/template-only, coord-independent)
        and still expose the denoiser tokens the atom stage conditions on. Returns
        ``(centroid_pred, denoiser_tokens, pred_lddt_or_None)``.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        cond = self._embed_c_noise(c_noise)
        denoiser_tokens = self._denoiser_tokens(
            c_in * x_t, trunk_tokens, cond, mask, x0_prev,
            res_idx=res_idx, chain_ids=chain_ids,
        )
        F_centroid = self.centroid_proj(denoiser_tokens)
        centroid_pred = c_skip * x_t + c_out * F_centroid
        pred_lddt = self._predict_confidence(denoiser_tokens, mask)
        return centroid_pred, denoiser_tokens, pred_lddt

    def forward_sigma_with_trunk(
        self,
        x_t: Tensor,
        trunk_tokens: Tensor,
        sigma: Tensor,
        mask: Tensor | None = None,
        x0_prev: Tensor | None = None,
        res_idx: Tensor | None = None,
        chain_ids: Tensor | None = None,
    ) -> ModelOutput:
        """Continuous-sigma forward with precomputed trunk tokens (EDM-preconditioned).

        Returns ``(centroid_pred, atoms_pred, pred_lddt_or_None)`` — same
        contract as :meth:`forward_sigma`.

        ``res_idx`` and ``chain_ids`` are forwarded to the denoiser so its
        (optional) relpos bias can index into them. They are required when the
        model was built with ``relpos_bias=True`` (the trunk tokens are already
        computed, but the denoiser still needs the indices for its own bias);
        ignored otherwise.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        if self.relpos_bias_enabled and (res_idx is None or chain_ids is None):
            raise ValueError(
                "forward_sigma_with_trunk requires res_idx and chain_ids when "
                "the model was built with relpos_bias=True."
            )
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        cond = self._embed_c_noise(c_noise)
        denoiser_tokens = self._denoiser_tokens(
            c_in * x_t, trunk_tokens, cond, mask, x0_prev,
            res_idx=res_idx, chain_ids=chain_ids,
        )
        centroid_pred, atoms_pred = self._heads_edm(
            denoiser_tokens, x_t, c_skip, c_out, mask
        )
        pred_lddt = self._predict_confidence(denoiser_tokens, mask)
        return ModelOutput(centroid_pred, atoms_pred, pred_lddt)

    def forward(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        t: Tensor,
        mask: Tensor | None = None,
        esm_embed: Tensor | None = None,
        template_coords_res: Tensor | None = None,
        template_mask: Tensor | None = None,
        template_frame_id: Tensor | None = None,
        msa_feats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Discrete-timestep forward (legacy path).

        No EDM here — the discrete timestep -> sigma mapping is not defined in this
        model. Caller should prefer `forward_sigma` for new code. Output is the
        raw projection (centroid_pred = F_centroid); atoms = centroid + offsets.
        """
        B, L, _ = x_t.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=x_t.device)
        trunk_tokens = self.trunk(
            aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed,
            template_coords_res=template_coords_res,
            template_mask=template_mask,
            template_frame_id=template_frame_id,
            msa_feats=msa_feats,
        )
        cond = self.time_embed(t)
        denoiser_tokens = self._denoiser_tokens(
            x_t, trunk_tokens, cond, mask, None,
            res_idx=res_idx, chain_ids=chain_ids,
        )
        F_centroid = self.centroid_proj(denoiser_tokens)
        atom_offsets = self.atom_head(denoiser_tokens, mask)
        atoms_pred = F_centroid.unsqueeze(2) + atom_offsets
        return F_centroid, atoms_pred

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Tensor | None = None,
        esm_embed: Tensor | None = None,
        template_coords_res: Tensor | None = None,
        template_mask: Tensor | None = None,
        template_frame_id: Tensor | None = None,
        msa_feats: Tensor | None = None,
        n_recycle: int = 0,
    ) -> Tensor:
        return self._run_trunk_recycled(
            n_recycle,
            aa_seq=aa_seq, chain_ids=chain_ids, res_idx=res_idx, mask=mask,
            esm_embed=esm_embed,
            template_coords_res=template_coords_res,
            template_mask=template_mask,
            template_frame_id=template_frame_id,
            msa_feats=msa_feats,
        )

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
        atom_diff = (
            sum(p.numel() for p in self.atom_diff_head.parameters())
            if self.atom_diff_head is not None
            else 0
        )
        confidence_head = (
            sum(p.numel() for p in self.confidence_head.parameters())
            if self.confidence_head is not None
            else 0
        )
        sc_head = (
            sum(p.numel() for p in self.sc_head.parameters())
            if self.sc_head is not None
            else 0
        )
        total = trunk + denoiser + atom_head + atom_diff + confidence_head + sc_head
        return {
            "trunk": trunk,
            "denoiser": denoiser,
            "atom_head": atom_head,
            "atom_diff": atom_diff,
            "confidence_head": confidence_head,
            "sc_head": sc_head,
            "total": total,
            "trunk_pct": 100 * trunk / total,
            "denoiser_pct": 100 * denoiser / total,
            "atom_head_pct": 100 * atom_head / total,
            "atom_diff_pct": 100 * atom_diff / total,
            "confidence_head_pct": 100 * confidence_head / total,
            "sc_head_pct": 100 * sc_head / total,
        }
