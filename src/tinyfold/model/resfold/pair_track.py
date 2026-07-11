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
from torch import Tensor

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
    ):
        super().__init__()
        if c_pair < 1 or n_layers < 1 or c_hidden < 1:
            raise ValueError("c_pair, n_layers, c_hidden must all be >= 1")
        self.c_pair = c_pair
        self.n_heads = n_heads
        self.relpos_clip = int(relpos_clip)

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
    ) -> Tensor:
        """Return per-head additive attention bias ``[B, n_heads, L, L]``.

        When ``template_cond`` is enabled and ``template_coords_res`` is passed,
        AF3-style relative pair features are built from the template coords and
        added into the pair channel before the triangle blocks. If templates are
        enabled but not supplied for this call, the template term is skipped
        (equivalent to a fully-uncovered template).
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

        z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        for tri_out, tri_in, transition in self.blocks:
            z = z + tri_out(z, pair_mask)
            z = z + tri_in(z, pair_mask)
            z = z + transition(z, pair_mask)
            z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        bias = self.to_bias(self.out_ln(z))            # [B, L, L, n_heads]
        return bias.permute(0, 3, 1, 2).contiguous()   # [B, n_heads, L, L]
