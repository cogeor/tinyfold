"""Self-attention on single representation with pair bias.

Implements the single-stream attention used in Pairformer (Algorithm 24 simplified).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPairBias(nn.Module):
    """Self-attention on single representation with bias from pair.

    For each position i, attends over positions j with:
    - Q, K, V from single representation s
    - Attention bias b[i,j] from pair representation z
    - Output gating g from s
    """

    def __init__(self, c_s: int = 256, c_z: int = 128, n_heads: int = 8):
        """
        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln_s = nn.LayerNorm(c_s)
        self.ln_z = nn.LayerNorm(c_z)

        # Q has bias, K/V do not (per AF3)
        self.proj_q = nn.Linear(c_s, c_s, bias=True)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)

        # Pair bias projection
        self.proj_b = nn.Linear(c_z, n_heads, bias=False)

        # Output gating
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.proj_out = nn.Linear(c_s, c_s, bias=False)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
            res_mask: [L] boolean mask for valid residues
        Returns:
            ds: [L, c_s] update to single representation
        """
        L, c_s = s.shape
        H, d = self.n_heads, self.head_dim

        s_ln = self.ln_s(s)
        z_ln = self.ln_z(z)

        # Project Q, K, V and reshape to [L, H, d]
        Q = self.proj_q(s_ln).view(L, H, d)
        K = self.proj_k(s_ln).view(L, H, d)
        V = self.proj_v(s_ln).view(L, H, d)
        G = torch.sigmoid(self.proj_g(s_ln)).view(L, H, d)

        # Pair bias: [L, L, H]
        B = self.proj_b(z_ln)

        # Attention logits: [L, L, H]
        # Q[i,h,d] @ K[j,h,d] -> [i,j,h]
        logits = torch.einsum("ihd,jhd->ijh", Q, K) * self.scale
        logits = logits + B

        # Mask invalid positions (j)
        mask = res_mask.view(1, L, 1)  # [1, L, 1]
        logits = logits.masked_fill(~mask, float("-inf"))

        # Softmax over j (with NaN guard for fully masked rows)
        attn = F.softmax(logits, dim=1)  # [L, L, H]
        attn = torch.nan_to_num(attn, nan=0.0)  # Replace NaN from all-masked softmax

        # Apply attention to values
        # attn[i,j,h] @ V[j,h,d] -> [i,h,d]
        out = torch.einsum("ijh,jhd->ihd", attn, V)

        # Apply gating
        out = G * out

        # Reshape and project out
        out = out.reshape(L, c_s)
        ds = self.proj_out(out)

        # Mask output for invalid residues
        ds = ds * res_mask.unsqueeze(-1)

        return ds
