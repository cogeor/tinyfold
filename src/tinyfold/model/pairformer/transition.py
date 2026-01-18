"""SwiGLU-style transition layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    """SwiGLU-style gated MLP transition.

    Applies: output = Linear(SiLU(Linear_a(x)) * Linear_b(x))

    This is Algorithm 11 from AF3 supplement.
    """

    def __init__(self, c: int, expansion: int = 4):
        """
        Args:
            c: Input/output channel dimension
            expansion: Hidden layer expansion factor
        """
        super().__init__()
        c_hidden = expansion * c

        self.ln = nn.LayerNorm(c)
        self.proj_a = nn.Linear(c, c_hidden, bias=False)
        self.proj_b = nn.Linear(c, c_hidden, bias=False)
        self.proj_out = nn.Linear(c_hidden, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., C] input tensor
        Returns:
            dx: [..., C] update tensor (to be added as residual)
        """
        h = self.ln(x)
        a = self.proj_a(h)
        b = self.proj_b(h)
        return self.proj_out(F.silu(a) * b)
