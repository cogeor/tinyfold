"""Base class for diffusion decoders."""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module, ABC):
    """Base class for all diffusion decoders.

    All decoders must implement the forward method with this signature.
    """

    @abstractmethod
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
            atom_types: Atom types (0=N, 1=CA, 2=C, 3=O) [B, N_atoms]
            atom_to_res: Residue index for each atom [B, N_atoms]
            aa_seq: Amino acid type for each atom [B, N_atoms]
            chain_ids: Chain ID for each atom (0 or 1) [B, N_atoms]
            t: Diffusion timestep [B]
            mask: Valid atom mask [B, N_atoms]

        Returns:
            x0_pred: Predicted clean coordinates [B, N_atoms, 3]
        """
        raise NotImplementedError


def sinusoidal_pos_enc(positions: Tensor, dim: int) -> Tensor:
    """Sinusoidal positional encoding.

    Args:
        positions: Position indices [B, N] or [N]
        dim: Embedding dimension

    Returns:
        Positional encoding [B, N, dim] or [N, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
    emb = positions.float().unsqueeze(-1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
