"""PairformerDecoder - Uses existing Pairformer trunk for diffusion decoding.

Fully vectorized GPU implementation using torch.vmap for batching.
"""

import sys
import os

# Add src to path to import tinyfold.model.pairformer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from torch import Tensor
from functools import partial

from .base import BaseDecoder, sinusoidal_pos_enc
from tinyfold.model.pairformer import PairformerStack


class PairformerDecoder(BaseDecoder):
    """Pairformer-based diffusion decoder.

    Operates at residue level with pair representation, then decodes atoms.
    Fully vectorized on GPU using torch.vmap.

    Architecture:
        1. Extract residue features from atom input (vectorized)
        2. Initialize single (s) and pair (z) representations (vectorized)
        3. Run PairformerStack (vmap over batch)
        4. Decode atom positions from residue features (vectorized)
    """

    def __init__(
        self,
        h_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        n_timesteps: int = 50,
        dropout: float = 0.0,
        n_aa_types: int = 21,
        n_chains: int = 2,
        c_z: int = 64,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.c_z = c_z
        self.n_timesteps = n_timesteps

        # Residue-level embeddings for single representation
        self.aa_embed = nn.Embedding(n_aa_types, h_dim)
        self.chain_embed = nn.Embedding(n_chains, h_dim // 4)
        self.time_embed = nn.Embedding(n_timesteps, h_dim)
        self.coord_proj = nn.Linear(3, h_dim)

        # Single input projection
        single_input_dim = h_dim + (h_dim // 4) + h_dim + h_dim + h_dim
        self.single_proj = nn.Linear(single_input_dim, h_dim)

        # Pair initialization
        self.rel_pos_embed = nn.Embedding(65, c_z)  # -32 to +32 relative position
        self.chain_pair_embed = nn.Embedding(2, c_z)  # same chain or different
        self.dist_proj = nn.Linear(1, c_z)
        self.pair_proj = nn.Linear(c_z * 3, c_z)

        # Pairformer stack
        self.pairformer = PairformerStack(
            n_blocks=n_layers,
            c_s=h_dim,
            c_z=c_z,
            n_heads_single=n_heads,
            n_heads_tri=max(2, n_heads // 2),
            c_tri_attn=max(16, h_dim // 8),
            c_tri_mul=max(32, h_dim // 2),
            transition_expansion=4,
            dropout=dropout,
            chunk_size=16,
            use_checkpoint=False,  # Disable for vmap compatibility
        )

        # Atom decoder: residue features -> 4 atom offsets
        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 12),  # 4 atoms * 3 coords
        )

        # Learnable initial atom offsets
        self.register_buffer('init_offsets', torch.tensor([
            [-0.5, 0.0, 0.0],   # N
            [0.0, 0.0, 0.0],    # CA
            [0.5, 0.0, 0.0],    # C
            [0.7, 0.4, 0.0],    # O
        ]))

    def _init_single_batched(
        self,
        x_ca: Tensor,      # [B, L, 3]
        aa_res: Tensor,    # [B, L]
        chain_res: Tensor, # [B, L]
        res_idx: Tensor,   # [B, L]
        t: Tensor,         # [B]
    ) -> Tensor:
        """Initialize single representation - fully batched.

        Returns:
            s: [B, L, h_dim]
        """
        B, L, _ = x_ca.shape

        aa_emb = self.aa_embed(aa_res)          # [B, L, h_dim]
        chain_emb = self.chain_embed(chain_res)  # [B, L, h_dim//4]
        coord_emb = self.coord_proj(x_ca)        # [B, L, h_dim]
        time_emb = self.time_embed(t)            # [B, h_dim]
        time_emb = time_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, h_dim]

        # Sinusoidal position encoding (batched)
        res_emb = sinusoidal_pos_enc(res_idx.reshape(-1), self.h_dim)  # [B*L, h_dim]
        res_emb = res_emb.view(B, L, self.h_dim)  # [B, L, h_dim]

        s = torch.cat([aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
        s = self.single_proj(s)  # [B, L, h_dim]

        return s

    def _init_pairs_batched(
        self,
        res_idx: Tensor,    # [B, L]
        chain_res: Tensor,  # [B, L]
        ca_coords: Tensor,  # [B, L, 3]
    ) -> Tensor:
        """Initialize pair representation - fully batched.

        Returns:
            z: [B, L, L, c_z]
        """
        B, L, _ = ca_coords.shape

        # Relative position: [B, L, L]
        rel_pos = res_idx.unsqueeze(2) - res_idx.unsqueeze(1)  # [B, L, L]
        rel_pos = rel_pos.clamp(-32, 32) + 32
        rel_pos_emb = self.rel_pos_embed(rel_pos)  # [B, L, L, c_z]

        # Same chain indicator: [B, L, L]
        same_chain = (chain_res.unsqueeze(2) == chain_res.unsqueeze(1)).long()
        chain_emb = self.chain_pair_embed(same_chain)  # [B, L, L, c_z]

        # Distance features: [B, L, L]
        dist = torch.cdist(ca_coords, ca_coords)  # [B, L, L]
        dist_emb = self.dist_proj(dist.unsqueeze(-1))  # [B, L, L, c_z]

        # Combine
        z = torch.cat([rel_pos_emb, chain_emb, dist_emb], dim=-1)
        z = self.pair_proj(z)  # [B, L, L, c_z]

        return z

    def _pairformer_single(
        self,
        s: Tensor,        # [L, h_dim]
        z: Tensor,        # [L, L, c_z]
        res_mask: Tensor, # [L]
    ) -> tuple[Tensor, Tensor]:
        """Run pairformer on single sample (for vmap)."""
        return self.pairformer(s, z, res_mask)

    def forward(
        self,
        x_t: Tensor,
        atom_types: Tensor,
        atom_to_res: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        t: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """Predict clean coordinates from noisy coordinates.

        Uses vectorized pre/post processing with sequential pairformer calls.
        This avoids OOM from vmap while keeping good GPU utilization.

        Args:
            x_t: Noisy coordinates [B, N_atoms, 3]
            atom_types: Atom types [B, N_atoms]
            atom_to_res: Residue index for each atom [B, N_atoms]
            aa_seq: Amino acid type for each atom [B, N_atoms]
            chain_ids: Chain ID for each atom [B, N_atoms]
            t: Diffusion timestep [B]
            mask: Valid atom mask [B, N_atoms]

        Returns:
            x0_pred: Predicted clean coordinates [B, N_atoms, 3]
        """
        B, N_atoms, _ = x_t.shape
        N_res = N_atoms // 4
        device = x_t.device

        # Extract residue-level data from CA atoms (index 1 in each residue)
        # Vectorized: reshape to [B, L, 4, ...] then select CA
        x_ca = x_t.view(B, N_res, 4, 3)[:, :, 1, :]      # [B, L, 3]
        aa_res = aa_seq.view(B, N_res, 4)[:, :, 1]       # [B, L]
        chain_res = chain_ids.view(B, N_res, 4)[:, :, 1] # [B, L]
        res_idx = atom_to_res.view(B, N_res, 4)[:, :, 1] # [B, L]

        if mask is not None:
            res_mask = mask.view(B, N_res, 4)[:, :, 1]   # [B, L]
        else:
            res_mask = torch.ones(B, N_res, dtype=torch.bool, device=device)

        # Build single representation (fully batched)
        s = self._init_single_batched(x_ca, aa_res, chain_res, res_idx, t)  # [B, L, h_dim]

        # Build pair representation (fully batched)
        z = self._init_pairs_batched(res_idx, chain_res, x_ca)  # [B, L, L, c_z]

        # Run pairformer sequentially per sample (avoids OOM from vmap)
        # Pre-allocate output tensors
        s_out = torch.empty_like(s)
        for b in range(B):
            s_out[b], _ = self.pairformer(s[b], z[b], res_mask[b])

        # Decode atoms (fully batched)
        atom_offsets = self.atom_decoder(s_out)  # [B, L, 12]
        atom_offsets = atom_offsets.view(B, N_res, 4, 3)  # [B, L, 4, 3]
        atom_offsets = atom_offsets + self.init_offsets  # broadcast [4, 3]

        # Final positions = CA + offsets
        x0_pred = x_ca.unsqueeze(2) + atom_offsets  # [B, L, 4, 3]
        x0_pred = x0_pred.view(B, N_atoms, 3)  # [B, N_atoms, 3]

        return x0_pred
