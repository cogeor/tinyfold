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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseDecoder, sinusoidal_pos_enc
from .pair_track import PairTrack
from .relpos import RelposBias

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
# Trunk encoder layer with optional attention bias
# =============================================================================


class TrunkEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that accepts an attention bias.

    Mirrors ``nn.TransformerEncoderLayer(norm_first=True)`` but uses SDPA so a
    per-(batch, head, i, j) bias from ``RelposBias`` can be added to the
    attention logits before softmax. We use this instead of the stock PyTorch
    layer ONLY when ``relpos_bias=True``; the legacy code path (no bias) keeps
    the original ``nn.TransformerEncoderLayer`` for byte-identical
    checkpoint compatibility with Phase D/F runs.

    NOTE: Parameter layout differs from ``nn.MultiheadAttention``
    (separate q/k/v projections instead of bundled ``in_proj_weight``).
    Phase G is a fresh retrain, so this is fine.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        # Match nn.TransformerEncoderLayer's GELU FFN structure so the
        # behaviour is otherwise identical when attn_bias=None.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Tensor | None = None,  # [B, L] bool, True = pad
        attn_bias: Tensor | None = None,             # [B, n_heads, L, L]
    ) -> Tensor:
        B, L, _ = x.shape

        # Pre-norm attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Combine padding mask + bias into a single float attn_mask for SDPA.
        # SDPA contract: float mask is added to attention logits before softmax
        # (so -inf masks out, 0 is no-op).
        if src_key_padding_mask is not None or attn_bias is not None:
            attn_mask = torch.zeros(B, self.n_heads, L, L, device=x.device, dtype=q.dtype)
            if attn_bias is not None:
                attn_mask = attn_mask + attn_bias.to(q.dtype)
            if src_key_padding_mask is not None:
                # Mask out columns where key is padded.
                neg_inf = torch.finfo(q.dtype).min
                pad_cols = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                attn_mask = attn_mask.masked_fill(pad_cols, neg_inf)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        x = x + self.out_proj(out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TrunkEncoder(nn.Module):
    """Stack of ``TrunkEncoderLayer`` that broadcasts ``attn_bias`` to all layers."""

    def __init__(self, layer_factory, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(n_layers)])

    def forward(self, x, src_key_padding_mask=None, attn_bias=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, attn_bias=attn_bias)
        return x


# =============================================================================
# Residue Encoder (Trunk)
# =============================================================================

# ESM-2 hidden dims for the variants exposed via `aa_embed=...`. Kept here
# (and mirrored in ``scripts/prepare_esm2_embeddings.py`` /
# ``scripts/train_resfold.py``) so the model construction does not have to
# import either script.
ESM_DIMS = {
    "esm2_35M": 480,
    "esm2_150M": 640,
}


class ResidueEncoder(nn.Module):
    """Residue-level encoder (trunk) that runs ONCE per sample.

    Produces token embeddings that condition the denoiser.

    IMPORTANT: This trunk processes ONLY sequence/token features (aa_seq, chain_ids, res_idx).
    It does NOT take coordinates as input. This enables the trunk-once optimization
    where trunk runs once and denoiser runs multiple times with different noisy coords.

    AA-representation modes (``aa_embed``):
        * ``"learned"`` (default): the historical ``nn.Embedding(n_aa_types,
          c_token)`` lookup. Trainable from scratch; loss curve and parameter
          count are byte-identical to pre-Loop-05.
        * ``"esm2_35M"`` / ``"esm2_150M"``: frozen ESM-2 features supplied per
          forward via ``esm_embed=[B, L, esm_dim]`` (loaded from the parquet
          cache built by ``scripts/prepare_esm2_embeddings.py``). Projected
          through ``nn.Linear(esm_dim -> c_token)``; the chain + sinusoidal
          positional features are unchanged.
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
        aa_embed: str = "learned",
        esm_dim: int | None = None,
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
    ):
        super().__init__()
        self.c_token = c_token
        self.aa_embed_mode = aa_embed
        self.relpos_bias_enabled = bool(relpos_bias)
        self.pair_repr_enabled = bool(pair_repr)
        self.template_cond_enabled = bool(template_cond)
        self.msa_cond_enabled = bool(msa_cond)
        self.n_heads = n_heads
        if self.template_cond_enabled and not self.pair_repr_enabled:
            raise ValueError(
                "template_cond=True requires pair_repr=True (templates are "
                "injected into the pair track)."
            )
        if self.msa_cond_enabled and not self.pair_repr_enabled:
            raise ValueError(
                "msa_cond=True requires pair_repr=True (coevolution features are "
                "injected into the pair track)."
            )

        if aa_embed == "learned":
            # Bit-for-bit identical to the historical path.
            self.aa_embed = nn.Embedding(n_aa_types, c_token)
            self.esm_proj = None
            self.esm_dim = None
        elif aa_embed in ESM_DIMS:
            # Resolve esm_dim from the small lookup table unless overridden.
            resolved = esm_dim if esm_dim is not None else ESM_DIMS[aa_embed]
            self.esm_dim = resolved
            # No AA-lookup embedding in ESM mode. Set to None so a future
            # accidental ``self.aa_embed(aa_seq)`` raises immediately.
            self.aa_embed = None
            self.esm_proj = nn.Linear(resolved, c_token)
            nn.init.xavier_uniform_(self.esm_proj.weight)
            nn.init.zeros_(self.esm_proj.bias)
        else:
            raise ValueError(
                f"Unknown aa_embed={aa_embed!r}; expected 'learned' or one of "
                f"{sorted(ESM_DIMS.keys())}"
            )

        self.chain_embed = nn.Embedding(n_chains, c_token // 4)

        # Input projection
        # aa_emb (c_token) + chain_emb (c_token//4) + res_pos (c_token)
        input_dim = c_token + (c_token // 4) + c_token
        self.input_proj = nn.Linear(input_dim, c_token)

        # Transformer. Two backends:
        # - no bias (legacy / Phase D-F path): nn.TransformerEncoder.
        #   Byte-identical to historical runs; old checkpoints load cleanly.
        # - bias enabled (Phase G+): custom TrunkEncoder that accepts an
        #   additive [B, n_heads, L, L] attention bias. The bias is the sum of
        #   the (optional) RelposBias and the (optional) PairTrack output.
        self.bias_backend = self.relpos_bias_enabled or self.pair_repr_enabled
        if self.bias_backend:
            self.transformer = TrunkEncoder(
                lambda: TrunkEncoderLayer(
                    d_model=c_token,
                    n_heads=n_heads,
                    dim_feedforward=c_token * 4,
                    dropout=dropout,
                ),
                n_layers=n_layers,
            )
            self.relpos = RelposBias(n_heads=n_heads, clip=relpos_clip) if self.relpos_bias_enabled else None
            self.pair_track = (
                PairTrack(
                    c_token=c_token,
                    n_heads=n_heads,
                    c_pair=c_pair,
                    n_layers=pair_layers,
                    c_hidden=pair_hidden,
                    relpos_clip=relpos_clip,
                    template_cond=self.template_cond_enabled,
                    template_rbf=template_rbf,
                    template_d_max=template_d_max,
                    msa_cond=self.msa_cond_enabled,
                    grad_checkpoint=grad_checkpoint,
                    pair_to_single=pair_to_single,
                )
                if self.pair_repr_enabled
                else None
            )
        else:
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
            self.relpos = None
            self.pair_track = None

        self.output_norm = nn.LayerNorm(c_token)

        # === Recycling (C1) ===
        # The previous trunk pass's token (and, with the pair track, pair) rep is
        # LayerNorm'd, projected, and added into the next pass's rep. The
        # projection is zero-initialised so a fresh model starts with recycling as
        # a literal no-op -- n_recycle=0 is bitwise-identical to the pre-recycling
        # forward. Construction is wrapped in a save/restore of the global RNG
        # state so adding these parameters does not perturb the initialisation of
        # any weight created after the trunk: an existing from-scratch run stays
        # byte-identical with recycling left off.
        _rng_state = torch.random.get_rng_state()
        self.recycle_norm = nn.LayerNorm(c_token)
        self.recycle_proj = nn.Linear(c_token, c_token)
        nn.init.zeros_(self.recycle_proj.weight)
        nn.init.zeros_(self.recycle_proj.bias)
        if self.pair_repr_enabled:
            self.recycle_pair_norm = nn.LayerNorm(c_pair)
            self.recycle_pair_proj = nn.Linear(c_pair, c_pair)
            nn.init.zeros_(self.recycle_pair_proj.weight)
            nn.init.zeros_(self.recycle_pair_proj.bias)
        else:
            self.recycle_pair_norm = None
            self.recycle_pair_proj = None
        torch.random.set_rng_state(_rng_state)

    def forward(
        self,
        aa_seq: Tensor,          # [B, L]
        chain_ids: Tensor,       # [B, L]
        res_idx: Tensor,         # [B, L]
        mask: Tensor | None = None,  # [B, L]
        esm_embed: Tensor | None = None,  # [B, L, esm_dim], required in ESM mode
        template_coords_res: Tensor | None = None,  # [B, L, 4, 3]
        template_mask: Tensor | None = None,        # [B, L]
        template_frame_id: Tensor | None = None,    # [B, L]
        msa_feats: Tensor | None = None,            # [B, L, L, F_msa]
        recycle_tokens: Tensor | None = None,       # [B, L, c_token] prev pass
        recycle_pair: Tensor | None = None,         # [B, L, L, c_pair] prev pass
        return_pair: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Encode residue-level sequence features (NO coordinates).

        When ``aa_embed_mode == "learned"`` the original lookup is used (and
        ``esm_embed`` is ignored). When in ESM mode, the caller MUST pass
        ``esm_embed`` (a per-residue cached ESM-2 feature tensor); we project
        it through ``self.esm_proj`` instead of running an embedding lookup.
        The integer ``aa_seq`` is still accepted (it determines the [B, L]
        shape) but its values are ignored in ESM mode.

        Recycling (C1): when ``recycle_tokens`` (and, with the pair track,
        ``recycle_pair``) from a previous trunk pass is supplied, it is fed back
        in through a zero-initialised projection. ``return_pair=True`` additionally
        returns the pair rep (or ``None`` when the pair track is off) so the caller
        can feed it into the next pass.

        Returns:
            tokens: [B, L, c_token] conditioning for denoiser, or
            (tokens, pair_rep) when ``return_pair=True``.
        """
        B, L = aa_seq.shape

        # Sequence / ESM feature
        if self.aa_embed_mode == "learned":
            aa_emb = self.aa_embed(aa_seq)  # [B, L, c_token]
        else:
            assert esm_embed is not None, (
                f"ResidueEncoder in ESM mode (aa_embed={self.aa_embed_mode!r}) "
                "requires batch['esm_embed']; got None."
            )
            # Cast fp16 -> fp32 defensively (dataloader already returns fp32;
            # this keeps the path safe if a future caller passes fp16 directly).
            aa_emb = self.esm_proj(esm_embed.float())  # [B, L, c_token]
        chain_emb = self.chain_embed(chain_ids)  # [B, L, c_token//4]
        res_emb = sinusoidal_pos_enc(res_idx, self.c_token)  # [B, L, c_token]

        # Concatenate and project
        h = torch.cat([aa_emb, chain_emb, res_emb], dim=-1)
        h = self.input_proj(h)  # [B, L, c_token]

        # Recycling: add the previous pass's token rep (zero-init proj -> no-op
        # until trained; recycle_tokens is None on the first/only pass).
        if recycle_tokens is not None:
            h = h + self.recycle_proj(self.recycle_norm(recycle_tokens))
        pair_rep: Tensor | None = None

        # Apply transformer
        attn_mask = ~mask if mask is not None else None
        if self.bias_backend:
            attn_bias = None  # [B, n_heads, L, L], built additively below
            if self.relpos is not None:
                attn_bias = self.relpos(res_idx, chain_ids)
            if self.pair_track is not None:
                valid = mask if mask is not None else torch.ones(
                    B, L, dtype=torch.bool, device=h.device
                )
                pair_recycle_add = (
                    self.recycle_pair_proj(self.recycle_pair_norm(recycle_pair))
                    if recycle_pair is not None else None
                )
                pair_bias, single_update, pair_rep = self.pair_track(
                    h, res_idx, chain_ids, valid,
                    template_coords_res=template_coords_res,
                    template_mask=template_mask,
                    template_frame_id=template_frame_id,
                    msa_feats=msa_feats,
                    recycle_pair=pair_recycle_add,
                )
                attn_bias = pair_bias if attn_bias is None else attn_bias + pair_bias
                # Higher-bandwidth pair->single injection (zero-init -> no-op at
                # start): the pair rep flows into the token CONTENT, not just the
                # attention logits.
                if single_update is not None:
                    h = h + single_update
            h = self.transformer(h, src_key_padding_mask=attn_mask, attn_bias=attn_bias)
        else:
            h = self.transformer(h, src_key_padding_mask=attn_mask)

        tokens = self.output_norm(h)
        if return_pair:
            return tokens, pair_rep
        return tokens


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
        mask: Tensor | None = None,  # [B, L] valid token mask
        attn_bias: Tensor | None = None,  # [B, n_heads, L, L] additive bias
    ) -> Tensor:
        B, L, _ = x.shape

        # AdaLN + Attention
        h = self.adaln_attn(x, cond)

        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA mask. Three sub-cases:
        # 1) Neither mask nor bias: attn_mask=None (FlashAttention fast path).
        # 2) Mask only: bool [B, 1, 1, L] — legacy fast path preserved.
        # 3) Bias (with or without mask): float [B, n_heads, L, L]. Padding
        #    is encoded as -inf in the same float tensor so SDPA sees one
        #    combined argument.
        if attn_bias is None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
        else:
            attn_mask = attn_bias.to(q.dtype)
            if mask is not None:
                neg_inf = torch.finfo(q.dtype).min
                # mask: True = valid; we want -inf where key is INvalid.
                pad_cols = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                attn_mask = attn_mask.masked_fill(pad_cols, neg_inf)

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
        relpos_bias: bool = False,
        relpos_clip: int = 32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(c_token, n_heads, expansion, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(c_token)
        # Single shared bias module across blocks (AF-M convention): the
        # relative-position relationship is a property of the input pair,
        # not of the block. Saves params vs per-block bias.
        self.relpos = RelposBias(n_heads=n_heads, clip=relpos_clip) if relpos_bias else None

    def forward(
        self,
        tokens: Tensor,
        time_cond: Tensor,
        mask: Tensor | None = None,
        res_idx: Tensor | None = None,    # [B, L] long, required when relpos is enabled
        chain_ids: Tensor | None = None,  # [B, L] long, required when relpos is enabled
    ) -> Tensor:
        if self.relpos is not None:
            assert res_idx is not None and chain_ids is not None, (
                "DiffusionTransformer was built with relpos_bias=True; "
                "forward() requires res_idx and chain_ids."
            )
            attn_bias = self.relpos(res_idx, chain_ids)  # [B, n_heads, L, L]
        else:
            attn_bias = None
        for block in self.blocks:
            tokens = block(tokens, time_cond, mask, attn_bias=attn_bias)
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
        aa_embed: str = "learned",
        esm_dim: int | None = None,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_timesteps = n_timesteps
        self.aa_embed_mode = aa_embed

        # === TRUNK (runs once) ===
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
        mask: Tensor | None = None,  # [B, L]
        esm_embed: Tensor | None = None,  # [B, L, esm_dim], required in ESM mode
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
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)

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
        mask: Tensor | None = None,  # [B, L]
        x0_prev: Tensor | None = None,  # [B, L, 3] previous x0 prediction (self-conditioning)
        esm_embed: Tensor | None = None,  # [B, L, esm_dim], required in ESM mode
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
            esm_embed: Cached frozen ESM-2 embeddings (required in ESM mode)

        Returns:
            x0_pred: [B, L, 3] predicted clean centroids
        """
        B, L, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # === TRUNK (once, sequence-only) ===
        trunk_tokens = self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)

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
        mask: Tensor | None = None,  # [B, L]
        x0_prev: Tensor | None = None,  # [B, L, 3] previous x0 prediction (self-conditioning)
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
        mask: Tensor | None = None,
        esm_embed: Tensor | None = None,
    ) -> Tensor:
        """Compute trunk embeddings from sequence features (for Stage 2 or multi-copy).

        The trunk processes ONLY sequence features, enabling trunk-once optimization.

        Returns:
            trunk_tokens: [B, L, c_token] embeddings
        """
        return self.trunk(aa_seq, chain_ids, res_idx, mask, esm_embed=esm_embed)

    def forward_with_trunk(
        self,
        x_t: Tensor,         # [B, L, 3] noisy residue centroids
        trunk_tokens: Tensor,  # [B, L, c_token] pre-computed trunk embeddings
        t: Tensor,           # [B] timestep
        mask: Tensor | None = None,  # [B, L]
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
