"""PairformerDecoder - Uses existing Pairformer trunk for diffusion decoding.

Wraps the pairformer from src/tinyfold/model/pairformer/ to work as a
diffusion decoder with the standard BaseDecoder interface.
"""

import sys
import os

# Add src to path to import tinyfold.model.pairformer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseDecoder, sinusoidal_pos_enc
from tinyfold.model.pairformer import PairformerStack


class PairformerDecoder(BaseDecoder):
    """Pairformer-based diffusion decoder.

    Operates at residue level with pair representation, then decodes atoms.

    Architecture:
        1. Extract residue features from atom input
        2. Initialize single (s) and pair (z) representations
        3. Run PairformerStack
        4. Decode atom positions from residue features
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
        """
        Args:
            h_dim: Single representation dimension (c_s)
            n_heads: Attention heads for single stream
            n_layers: Number of pairformer blocks
            n_timesteps: Number of diffusion timesteps
            dropout: Dropout probability
            n_aa_types: Number of amino acid types
            n_chains: Number of chains (2 for binary PPI)
            c_z: Pair representation dimension
        """
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
        # aa (h_dim) + chain (h_dim//4) + pos (h_dim) + time (h_dim) + coord (h_dim)
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
            use_checkpoint=True,
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

    def _init_pairs(
        self,
        s: Tensor,
        res_idx: Tensor,
        chain_res: Tensor,
        ca_coords: Tensor,
    ) -> Tensor:
        """Initialize pair representation.

        Args:
            s: Single repr [L, c_s]
            res_idx: Residue indices [L]
            chain_res: Chain IDs [L]
            ca_coords: CA coordinates [L, 3]

        Returns:
            z: Pair repr [L, L, c_z]
        """
        L = s.shape[0]
        device = s.device

        # Relative position: clamp to [-32, 32] -> [0, 64]
        rel_pos = res_idx.unsqueeze(1) - res_idx.unsqueeze(0)
        rel_pos = rel_pos.clamp(-32, 32) + 32
        rel_pos_emb = self.rel_pos_embed(rel_pos)  # [L, L, c_z]

        # Same chain indicator
        same_chain = (chain_res.unsqueeze(1) == chain_res.unsqueeze(0)).long()
        chain_emb = self.chain_pair_embed(same_chain)  # [L, L, c_z]

        # Distance features
        dist = torch.cdist(ca_coords, ca_coords)  # [L, L]
        dist_emb = self.dist_proj(dist.unsqueeze(-1))  # [L, L, c_z]

        # Combine
        z = torch.cat([rel_pos_emb, chain_emb, dist_emb], dim=-1)
        z = self.pair_proj(z)

        return z

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

        # Process each sample (pairformer operates on single samples)
        outputs = []
        for b in range(B):
            # Extract residue-level data (from CA atoms)
            x_ca = x_t[b, 1::4, :]  # [L, 3]
            aa_res = aa_seq[b, 1::4]  # [L]
            chain_res = chain_ids[b, 1::4]  # [L]
            res_idx = atom_to_res[b, 1::4]  # [L]

            if mask is not None:
                res_mask = mask[b, 1::4]  # [L]
            else:
                res_mask = torch.ones(N_res, dtype=torch.bool, device=device)

            # Build single representation
            aa_emb = self.aa_embed(aa_res)
            chain_emb = self.chain_embed(chain_res)
            res_emb = sinusoidal_pos_enc(res_idx, self.h_dim)
            time_emb = self.time_embed(t[b:b+1]).expand(N_res, -1)
            coord_emb = self.coord_proj(x_ca)

            s = torch.cat([aa_emb, chain_emb, res_emb, time_emb, coord_emb], dim=-1)
            s = self.single_proj(s)  # [L, h_dim]

            # Initialize pair representation
            z = self._init_pairs(s, res_idx, chain_res, x_ca)  # [L, L, c_z]

            # Run pairformer
            s, z = self.pairformer(s, z, res_mask)  # [L, h_dim], [L, L, c_z]

            # Decode atoms
            atom_offsets = self.atom_decoder(s)  # [L, 12]
            atom_offsets = atom_offsets.view(N_res, 4, 3)
            atom_offsets = atom_offsets + self.init_offsets.unsqueeze(0)

            # Final positions = CA + offsets
            x0_pred = x_ca.unsqueeze(1) + atom_offsets  # [L, 4, 3]
            x0_pred = x0_pred.view(N_atoms, 3)

            outputs.append(x0_pred)

        return torch.stack(outputs, dim=0)  # [B, N_atoms, 3]
