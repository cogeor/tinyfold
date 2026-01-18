"""Triangle multiplicative updates for pair representation.

Implements Algorithms 12 & 13 from AF3 supplement.
"""

import torch
import torch.nn as nn


class TriangleMultiplicationOutgoing(nn.Module):
    """Triangle multiplication with outgoing edges.

    Computes: m[i,j] = sum_k a[i,k] * b[j,k]
    Then applies gating and projection.

    Updates pair (i,j) using triangles i-k-j where edges go out from i and j.
    """

    def __init__(self, c_z: int = 128, c_hidden: int = 128):
        """
        Args:
            c_z: Pair representation dimension
            c_hidden: Hidden dimension for triangle computation
        """
        super().__init__()
        self.ln_in = nn.LayerNorm(c_z)
        self.ln_out = nn.LayerNorm(c_hidden)

        # Gated projections for a and b
        self.proj_a = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_a = nn.Linear(c_z, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_b = nn.Linear(c_z, c_hidden, bias=False)

        # Output gate and projection
        self.gate_out = nn.Linear(c_z, c_z, bias=False)
        self.proj_out = nn.Linear(c_hidden, c_z, bias=False)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [L, L, c_z] pair representation
            pair_mask: [L, L] mask for valid pairs
        Returns:
            dz: [L, L, c_z] update to pair representation
        """
        z_ln = self.ln_in(z)

        # Gated projections: a, b shape [L, L, c_hidden]
        a = torch.sigmoid(self.gate_a(z_ln)) * self.proj_a(z_ln)
        b = torch.sigmoid(self.gate_b(z_ln)) * self.proj_b(z_ln)

        # Mask out invalid pairs
        mask = pair_mask.unsqueeze(-1)  # [L, L, 1]
        a = a * mask
        b = b * mask

        # Triangle update: m[i,j] = sum_k a[i,k] * b[j,k]
        # einsum: 'ikc,jkc->ijc'
        m = torch.einsum("ikc,jkc->ijc", a, b)

        # Layer norm and output projection
        m_ln = self.ln_out(m)

        # Output gating
        g = torch.sigmoid(self.gate_out(z_ln))  # [L, L, c_z]
        dz = g * self.proj_out(m_ln)

        return dz * pair_mask.unsqueeze(-1)


class TriangleMultiplicationIncoming(nn.Module):
    """Triangle multiplication with incoming edges.

    Computes: m[i,j] = sum_k a[k,i] * b[k,j]
    Then applies gating and projection.

    Updates pair (i,j) using triangles k-i-j where edges go into i and j.
    """

    def __init__(self, c_z: int = 128, c_hidden: int = 128):
        """
        Args:
            c_z: Pair representation dimension
            c_hidden: Hidden dimension for triangle computation
        """
        super().__init__()
        self.ln_in = nn.LayerNorm(c_z)
        self.ln_out = nn.LayerNorm(c_hidden)

        # Gated projections
        self.proj_a = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_a = nn.Linear(c_z, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_z, c_hidden, bias=False)
        self.gate_b = nn.Linear(c_z, c_hidden, bias=False)

        # Output
        self.gate_out = nn.Linear(c_z, c_z, bias=False)
        self.proj_out = nn.Linear(c_hidden, c_z, bias=False)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [L, L, c_z] pair representation
            pair_mask: [L, L] mask for valid pairs
        Returns:
            dz: [L, L, c_z] update to pair representation
        """
        z_ln = self.ln_in(z)

        # Gated projections
        a = torch.sigmoid(self.gate_a(z_ln)) * self.proj_a(z_ln)
        b = torch.sigmoid(self.gate_b(z_ln)) * self.proj_b(z_ln)

        # Mask
        mask = pair_mask.unsqueeze(-1)
        a = a * mask
        b = b * mask

        # Incoming: m[i,j] = sum_k a[k,i] * b[k,j]
        # einsum: 'kic,kjc->ijc'
        m = torch.einsum("kic,kjc->ijc", a, b)

        m_ln = self.ln_out(m)
        g = torch.sigmoid(self.gate_out(z_ln))
        dz = g * self.proj_out(m_ln)

        return dz * pair_mask.unsqueeze(-1)
