"""Minimal pair-representation track for the ResFold trunk (Phase H).

Motivation
----------
Every variant tried so far (wider data, ESM, multi-sample + ranking, sampler
steps, interface cropping) leaves the >=200-residue interface metric pinned at
the random-docking floor (DockQ ~0.01-0.05). The one architectural lever never
pulled is an explicit **pair representation** -- the (i, j) channel the whole
AF3/Boltz/Protenix family uses to reason about which inter-chain residues are in
contact. The active ``ResFoldOneStep`` trunk is single-track: it can attend
across the concatenated chains but has no place to *store* pairwise interface
geometry.

This module adds a deliberately small pair track (Pairmixer-minimal: triangle
multiplication + transition, NO triangle attention -- arXiv:2510.18870 shows
triangle attention is the removable part) that:

1. initialises a pair tensor ``z[b, i, j]`` from the single representation
   (outer sum of two linear projections) plus a learned relative-position /
   same-chain embedding;
2. refines it with ``n_layers`` of (TriMulOutgoing + TriMulIncoming +
   PairTransition), each residual and pair-masked;
3. projects ``z`` down to a per-head additive attention bias
   ``[B, n_heads, L, L]`` consumed by the trunk's attention exactly the way
   :class:`~tinyfold.model.resfold.relpos.RelposBias` already is.

The track lives in the trunk, which runs ONCE per sample, so its O(L^2) cost is
paid once -- not per diffusion step. Combined with interface cropping to
``crop_size`` residues, the L^2 tensors stay affordable on a single GPU.

At ``c_pair=64``, ``c_hidden=64``, ``n_layers=3`` this adds ~0.24M params
(~2% of the 12M model). The clean A/B is Phase G (relpos, no pair) vs Phase H
(relpos + pair) on the leakage-controlled cliff bins, read out in DockQ.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from ...msa.features import msa_feat_dim
from ...retrieval.template_features import (
    build_template_pair_features,
    template_feat_dim,
)


class TriangleMultiplication(nn.Module):
    """Batched triangle multiplicative update (AF3 Alg. 12/13).

    Operates on ``z: [B, L, L, c_z]`` (batched rewrite of the unbatched module
    in ``tinyfold/model/pairformer/triangle_mul.py``). ``mode`` selects the
    edge direction:

    * ``"outgoing"``: ``m[i,j] = sum_k a[i,k] * b[j,k]``  (einsum ``bikc,bjkc->bijc``)
    * ``"incoming"``: ``m[i,j] = sum_k a[k,i] * b[k,j]``  (einsum ``bkic,bkjc->bijc``)
    """

    def __init__(self, c_z: int, c_hidden: int, mode: Literal["outgoing", "incoming"]):
        super().__init__()
        if mode not in ("outgoing", "incoming"):
            raise ValueError(f"mode must be 'outgoing' or 'incoming', got {mode!r}")
        self.mode = mode
        self.ln_in = nn.LayerNorm(c_z)
        self.ln_out = nn.LayerNorm(c_hidden)
        self.proj_a = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_a = nn.Linear(c_z, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_b = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_out = nn.Linear(c_z, c_z, bias=False)
        self.proj_out = nn.Linear(c_hidden, c_z, bias=False)
        # Zero-init the output projection so the whole block starts as a no-op
        # residual; the rest of the trunk trains undisturbed at step 0.
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        """z: [B, L, L, c_z]; pair_mask: [B, L, L] bool -> dz: [B, L, L, c_z]."""
        z_ln = self.ln_in(z)
        m_exp = pair_mask.unsqueeze(-1).to(z.dtype)  # [B, L, L, 1]
        a = torch.sigmoid(self.gate_a(z_ln)) * self.proj_a(z_ln) * m_exp
        b = torch.sigmoid(self.gate_b(z_ln)) * self.proj_b(z_ln) * m_exp
        # Contract over k as a batched matmul (cuBLAS) rather than a literal
        # einsum: 'bikc,bjkc->bijc' makes torch materialise a [B,L,L,L,c]
        # intermediate (tens of GB, ~80x slower). Folding the channel c into the
        # batch dim turns each triangle update into a [B*c, L, L] @ [B*c, L, L]
        # matmul -- the standard AF trick.
        a_p = a.permute(0, 3, 1, 2)  # [B, c, L_i, L_k]
        b_p = b.permute(0, 3, 1, 2)  # [B, c, L_j, L_k]
        if self.mode == "outgoing":
            # m[i,j] = sum_k a[i,k] * b[j,k]  ->  a_p @ b_p^T
            m = torch.matmul(a_p, b_p.transpose(-2, -1))  # [B, c, L_i, L_j]
        else:
            # m[i,j] = sum_k a[k,i] * b[k,j]  ->  a_p^T @ b_p
            m = torch.matmul(a_p.transpose(-2, -1), b_p)  # [B, c, L_i, L_j]
        m = m.permute(0, 2, 3, 1).contiguous()  # [B, L_i, L_j, c]
        g = torch.sigmoid(self.gate_out(z_ln))
        dz = g * self.proj_out(self.ln_out(m))
        return dz * m_exp


class PairTransition(nn.Module):
    """Position-wise FFN over the pair channel (AF3 Alg. 15)."""

    def __init__(self, c_z: int, expansion: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(c_z)
        self.lin1 = nn.Linear(c_z, c_z * expansion)
        self.lin2 = nn.Linear(c_z * expansion, c_z)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        dz = self.lin2(F.relu(self.lin1(self.ln(z))))
        return dz * pair_mask.unsqueeze(-1).to(z.dtype)


class PairTrack(nn.Module):
    """Single -> pair -> per-head attention bias.

    Built once per ``ResidueEncoder``; called once per sample in the trunk.

    Args:
        c_token:     single-representation width (trunk token dim).
        n_heads:     number of trunk attention heads (bias output width).
        c_pair:      pair-channel width.
        n_layers:    number of (TriMulOut + TriMulIn + Transition) blocks.
        c_hidden:    triangle hidden width.
        relpos_clip: +- clip for the relative-position bucket embedding.
        transition_expansion: FFN expansion in :class:`PairTransition`.
    """

    def __init__(
        self,
        c_token: int,
        n_heads: int,
        c_pair: int = 64,
        n_layers: int = 3,
        c_hidden: int = 64,
        relpos_clip: int = 32,
        transition_expansion: int = 2,
        template_cond: bool = False,
        template_rbf: int = 32,
        template_d_max: float = 4.0,
        msa_cond: bool = False,
        grad_checkpoint: bool = False,
        pair_to_single: bool = False,
    ):
        super().__init__()
        if c_pair < 1 or n_layers < 1 or c_hidden < 1:
            raise ValueError("c_pair, n_layers, c_hidden must all be >= 1")
        self.c_pair = c_pair
        self.n_heads = n_heads
        self.relpos_clip = int(relpos_clip)
        # Recompute the O(L^2) triangle activations in backward instead of
        # storing them (frees the pair track's dominant memory cost).
        self.grad_checkpoint = bool(grad_checkpoint)

        # Pair init: outer sum of two projections of the single rep.
        self.left = nn.Linear(c_token, c_pair)
        self.right = nn.Linear(c_token, c_pair)
        # Relative-position + same-chain bucket embedding: (2*clip+1) offset
        # buckets x 2 (same / cross chain). Same bucketing as RelposBias.
        n_buckets = (2 * self.relpos_clip + 1) * 2
        self.relpos_emb = nn.Embedding(n_buckets, c_pair)

        # Template conditioning (C7): embed AF3-style relative pair features
        # (built here from per-residue template coords) into the pair channel.
        # Zero-init the projection so a template-conditioned model starts
        # byte-equivalent to its no-template arm and learns to use the template
        # gradually. When disabled this is a pure no-op.
        self.template_cond = bool(template_cond)
        self.template_rbf = int(template_rbf)
        self.template_d_max = float(template_d_max)
        if self.template_cond:
            self.template_proj = nn.Linear(template_feat_dim(self.template_rbf), c_pair)
            nn.init.zeros_(self.template_proj.weight)
            nn.init.zeros_(self.template_proj.bias)
        else:
            self.template_proj = None

        # Coevolution conditioning: the MIRROR of the template path above.
        # Templates and coevolution are complementary priors on the SAME pair
        # bus (spec §3): templates rescue homolog-rich but MSA-shallow targets,
        # coevolution rescues MSA-deep but analog-poor ones. They are additive
        # (z += template_proj(...) + msa_proj(...)) and, being zero-init, each
        # is independently ablatable and a byte-exact no-op at step 0.
        self.msa_cond = bool(msa_cond)
        if self.msa_cond:
            self.msa_proj = nn.Linear(msa_feat_dim(), c_pair)
            nn.init.zeros_(self.msa_proj.weight)
            nn.init.zeros_(self.msa_proj.bias)
        else:
            self.msa_proj = None

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TriangleMultiplication(c_pair, c_hidden, "outgoing"),
                        TriangleMultiplication(c_pair, c_hidden, "incoming"),
                        PairTransition(c_pair, transition_expansion),
                    ]
                )
                for _ in range(n_layers)
            ]
        )

        self.out_ln = nn.LayerNorm(c_pair)
        self.to_bias = nn.Linear(c_pair, n_heads)
        # Zero-init the bias head: the pair track starts as a no-op additive
        # bias, so a Phase H run begins byte-equivalent to its Phase G arm and
        # the optimiser introduces the pair signal gradually.
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

        # Pair -> single injection (higher-bandwidth conditioning): the attention
        # bias above only carries n_heads scalars per (i,j) into the ATTENTION
        # LOGITS -- too lossy to reconstruct a precise interface from the
        # template. This head projects each token's pair row back into the token
        # CONTENT, so the template's (cross-chain) geometry directly shapes the
        # single representation. Zero-init -> no-op at start.
        self.pair_to_single = bool(pair_to_single)
        if self.pair_to_single:
            self.single_ln = nn.LayerNorm(c_pair)
            self.to_single = nn.Linear(c_pair, c_token)
            nn.init.zeros_(self.to_single.weight)
            nn.init.zeros_(self.to_single.bias)

    def _run_block(self, block, z: Tensor, pair_mask: Tensor) -> Tensor:
        """One (TriMulOut + TriMulIn + Transition) residual block, pair-masked."""
        tri_out, tri_in, transition = block
        z = z + tri_out(z, pair_mask)
        z = z + tri_in(z, pair_mask)
        z = z + transition(z, pair_mask)
        return z * pair_mask.unsqueeze(-1).to(z.dtype)

    def _relpos_bucket(self, res_idx: Tensor, chain_ids: Tensor) -> Tensor:
        """[B, L] indices -> [B, L, L] long bucket ids for ``relpos_emb``."""
        clip = self.relpos_clip
        diff = res_idx.unsqueeze(2) - res_idx.unsqueeze(1)  # [B, L, L]
        diff = diff.clamp(-clip, clip) + clip                # [0, 2*clip]
        same = (chain_ids.unsqueeze(2) == chain_ids.unsqueeze(1)).long()  # [B, L, L]
        return diff * 2 + same

    def forward(
        self,
        s: Tensor,          # [B, L, c_token] single representation (post input_proj)
        res_idx: Tensor,    # [B, L] long, per-chain-reset global indices
        chain_ids: Tensor,  # [B, L] long
        mask: Tensor,       # [B, L] bool, True = valid residue
        template_coords_res: Tensor = None,  # [B, L, 4, 3] template backbone (normalized units)
        template_mask: Tensor = None,        # [B, L] bool coverage
        template_frame_id: Tensor = None,    # [B, L] long rigid-group id
        msa_feats: Tensor = None,            # [B, L, L, F_msa] coevolution pair features
        recycle_pair: Tensor = None,         # [B, L, L, c_pair] normed+projected prev-pass pair rep
    ):
        """Return ``(attn_bias [B, n_heads, L, L], single_update or None, z)``.

        ``z`` is the final pair representation ``[B, L, L, c_pair]``, returned so
        the trunk can feed it back into the next recycling pass (C1).

        ``recycle_pair`` is the previous pass's pair rep, already LayerNorm'd and
        projected by the trunk; it is added into the freshly built pair channel.
        ``None`` (the first/only pass) makes this a no-op.

        ``single_update`` is ``[B, L, c_token]`` when ``pair_to_single`` is
        enabled (the pair rep projected back into the token content), else None.

        When ``template_cond`` is enabled and ``template_coords_res`` is passed,
        AF3-style relative pair features are built from the template coords and
        added into the pair channel before the triangle blocks. If templates are
        enabled but not supplied for this call, the template term is skipped
        (equivalent to a fully-uncovered template).

        ``msa_feats`` is the same story for coevolution: precomputed
        ``[B, L, L, F_msa]`` APC-corrected couplings added into the same pair
        channel. Unlike templates these are NOT built here -- they need the MSA,
        so they are cached offline (see :mod:`tinyfold.msa.features`). Omitting
        them is equivalent to a zero-depth MSA.
        """
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, L, L]

        # Init pair from single (outer sum) + relpos/same-chain embedding.
        z = self.left(s).unsqueeze(2) + self.right(s).unsqueeze(1)  # [B, L, L, c_pair]
        z = z + self.relpos_emb(self._relpos_bucket(res_idx, chain_ids))

        # Template term (C7): relative pair features -> zero-init projection.
        if self.template_proj is not None and template_coords_res is not None:
            if template_mask is None:
                template_mask = mask
            if template_frame_id is None:
                template_frame_id = torch.zeros_like(chain_ids)
            tmpl_feats, _ = build_template_pair_features(
                template_coords_res,
                template_mask,
                template_frame_id,
                n_rbf=self.template_rbf,
                d_max=self.template_d_max,
            )
            z = z + self.template_proj(tmpl_feats)

        # Coevolution term: additive on the same bus, independent of templates.
        if self.msa_proj is not None and msa_feats is not None:
            z = z + self.msa_proj(msa_feats.to(z.dtype))

        # Recycling term (C1): the previous pass's pair rep, already normed and
        # zero-init-projected by the trunk. None on the first/only pass -> no-op.
        if recycle_pair is not None:
            z = z + recycle_pair.to(z.dtype)

        z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        for block in self.blocks:
            if self.grad_checkpoint and self.training and z.requires_grad:
                z = torch.utils.checkpoint.checkpoint(
                    self._run_block, block, z, pair_mask, use_reentrant=False
                )
            else:
                z = self._run_block(block, z, pair_mask)

        bias = self.to_bias(self.out_ln(z))            # [B, L, L, n_heads]
        bias = bias.permute(0, 3, 1, 2).contiguous()   # [B, n_heads, L, L]

        single_update = None
        if self.pair_to_single:
            # Masked mean over j of the pair row -> per-token geometric summary.
            zf = self.to_single(self.single_ln(z))     # [B, L, L, c_token]
            m = pair_mask.unsqueeze(-1).to(zf.dtype)    # [B, L, L, 1]
            denom = m.sum(2).clamp(min=1.0)             # [B, L, 1]
            single_update = (zf * m).sum(2) / denom     # [B, L, c_token]

        return bias, single_update, z
