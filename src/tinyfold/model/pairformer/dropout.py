"""Structured dropout for pair representations."""

import torch
import torch.nn as nn


class DropoutRowwise(nn.Module):
    """Dropout with mask shared across columns (j dimension).

    For pair tensor z[i,j,c], dropout mask is [L,1,C] broadcast to [L,L,C].
    This preserves row-wise statistics.
    """

    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [L, L, C] pair tensor
            pair_mask: [L, L] optional mask for valid pairs
        Returns:
            Dropped out tensor, same shape
        """
        if not self.training or self.p == 0:
            return x

        L, _, C = x.shape
        device = x.device

        # Mask shape [L, 1, C] - shared across j
        keep_prob = 1 - self.p
        keep = torch.bernoulli(torch.full((L, 1, C), keep_prob, device=device, dtype=x.dtype))
        x = x * keep / keep_prob

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1)

        return x


class DropoutColumnwise(nn.Module):
    """Dropout with mask shared across rows (i dimension).

    For pair tensor z[i,j,c], dropout mask is [1,L,C] broadcast to [L,L,C].
    This preserves column-wise statistics.
    """

    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [L, L, C] pair tensor
            pair_mask: [L, L] optional mask for valid pairs
        Returns:
            Dropped out tensor, same shape
        """
        if not self.training or self.p == 0:
            return x

        L, _, C = x.shape
        device = x.device

        # Mask shape [1, L, C] - shared across i
        keep_prob = 1 - self.p
        keep = torch.bernoulli(torch.full((1, L, C), keep_prob, device=device, dtype=x.dtype))
        x = x * keep / keep_prob

        if pair_mask is not None:
            x = x * pair_mask.unsqueeze(-1)

        return x
