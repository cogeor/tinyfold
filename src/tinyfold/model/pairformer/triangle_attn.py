"""Triangle attention for pair representation.

Implements Algorithms 14 & 15 from AF3 supplement with chunking for memory efficiency.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangleAttentionStartingNode(nn.Module):
    """Triangle attention around the starting node.

    For each pair (i,j), attends over k with:
    - Queries from z[i,j]
    - Keys/Values from z[i,k]
    - Bias from z[j,k]

    Chunked over i dimension to avoid O(L^3) memory.
    """

    def __init__(
        self,
        c_z: int = 128,
        n_heads: int = 4,
        c_attn: int = 32,
        chunk_size: int = 16,
    ):
        """
        Args:
            c_z: Pair representation dimension
            n_heads: Number of attention heads
            c_attn: Per-head attention dimension
            chunk_size: Chunk size for i dimension (memory control)
        """
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = c_attn
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(c_attn)

        self.ln = nn.LayerNorm(c_z)

        # Projections
        self.proj_q = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_k = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_v = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_b = nn.Linear(c_z, n_heads, bias=False)
        self.proj_g = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_out = nn.Linear(n_heads * c_attn, c_z, bias=False)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: [L, L, c_z] pair representation
            pair_mask: [L, L] mask for valid pairs
            res_mask: [L] mask for valid residues
        Returns:
            dz: [L, L, c_z] update to pair representation
        """
        L, _, c_z = z.shape
        H, c = self.n_heads, self.c_attn

        z_ln = self.ln(z)

        # Project to Q, K, V, bias, gate
        Q = self.proj_q(z_ln).view(L, L, H, c)  # [L, L, H, c]
        K = self.proj_k(z_ln).view(L, L, H, c)
        V = self.proj_v(z_ln).view(L, L, H, c)
        B = self.proj_b(z_ln)  # [L, L, H]
        G = torch.sigmoid(self.proj_g(z_ln)).view(L, L, H, c)

        # Process in chunks over i to avoid O(L^3) memory
        outputs = []
        for i_start in range(0, L, self.chunk_size):
            i_end = min(i_start + self.chunk_size, L)
            chunk_out = self._process_chunk(Q, K, V, B, G, res_mask, i_start, i_end)
            outputs.append(chunk_out)

        dz = torch.cat(outputs, dim=0)  # [L, L, c_z]
        return dz * pair_mask.unsqueeze(-1)

    def _process_chunk(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        B: torch.Tensor,
        G: torch.Tensor,
        res_mask: torch.Tensor,
        i_start: int,
        i_end: int,
    ) -> torch.Tensor:
        """Process a chunk of i indices.

        For starting node: Q from (i,j), K from (i,k), bias from (j,k)
        """
        L = Q.size(0)
        H, c = self.n_heads, self.c_attn
        I = i_end - i_start

        # Extract chunk
        Q_chunk = Q[i_start:i_end]  # [I, L, H, c]
        K_chunk = K[i_start:i_end]  # [I, L, H, c] - keys at (i,k)
        V_chunk = V[i_start:i_end]  # [I, L, H, c]
        G_chunk = G[i_start:i_end]  # [I, L, H, c]

        # Compute attention logits: Q[i,j] @ K[i,k] for all j,k
        # Reshape for batched matmul
        Q_t = Q_chunk.permute(0, 2, 1, 3)  # [I, H, L(j), c]
        K_t = K_chunk.permute(0, 2, 3, 1)  # [I, H, c, L(k)]

        logits = torch.matmul(Q_t, K_t) * self.scale  # [I, H, L(j), L(k)]

        # Add bias B[j,k] (same for all i in chunk)
        B_t = B.permute(2, 0, 1).unsqueeze(0)  # [1, H, L(j), L(k)]
        logits = logits + B_t

        # Mask: k must be valid
        k_mask = res_mask.view(1, 1, 1, L)  # [1, 1, 1, L]
        logits = logits.masked_fill(~k_mask, float("-inf"))

        # Softmax over k
        attn = F.softmax(logits, dim=-1)  # [I, H, L(j), L(k)]

        # Apply to values
        V_t = V_chunk.permute(0, 2, 1, 3)  # [I, H, L(k), c]
        out = torch.matmul(attn, V_t)  # [I, H, L(j), c]

        # Apply gating and project out
        out = out.permute(0, 2, 1, 3)  # [I, L, H, c]
        out = G_chunk * out
        out = out.reshape(I, L, H * c)
        out = self.proj_out(out)  # [I, L, c_z]

        return out


class TriangleAttentionEndingNode(nn.Module):
    """Triangle attention around the ending node.

    For each pair (i,j), attends over k with:
    - Queries from z[i,j]
    - Keys/Values from z[k,j]
    - Bias from z[k,i]

    Chunked over j dimension to avoid O(L^3) memory.
    """

    def __init__(
        self,
        c_z: int = 128,
        n_heads: int = 4,
        c_attn: int = 32,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = c_attn
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(c_attn)

        self.ln = nn.LayerNorm(c_z)

        self.proj_q = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_k = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_v = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_b = nn.Linear(c_z, n_heads, bias=False)
        self.proj_g = nn.Linear(c_z, n_heads * c_attn, bias=False)
        self.proj_out = nn.Linear(n_heads * c_attn, c_z, bias=False)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: [L, L, c_z] pair representation
            pair_mask: [L, L] mask for valid pairs
            res_mask: [L] mask for valid residues
        Returns:
            dz: [L, L, c_z] update to pair representation
        """
        L, _, c_z = z.shape
        H, c = self.n_heads, self.c_attn

        z_ln = self.ln(z)

        # Project to Q, K, V, bias, gate
        Q = self.proj_q(z_ln).view(L, L, H, c)
        K = self.proj_k(z_ln).view(L, L, H, c)
        V = self.proj_v(z_ln).view(L, L, H, c)
        B = self.proj_b(z_ln)  # [L, L, H]
        G = torch.sigmoid(self.proj_g(z_ln)).view(L, L, H, c)

        # Process in chunks over j
        outputs = []
        for j_start in range(0, L, self.chunk_size):
            j_end = min(j_start + self.chunk_size, L)
            chunk_out = self._process_chunk(Q, K, V, B, G, res_mask, j_start, j_end)
            outputs.append(chunk_out)

        # Concatenate along j dimension (dim=1)
        dz = torch.cat(outputs, dim=1)  # [L, L, c_z]
        return dz * pair_mask.unsqueeze(-1)

    def _process_chunk(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        B: torch.Tensor,
        G: torch.Tensor,
        res_mask: torch.Tensor,
        j_start: int,
        j_end: int,
    ) -> torch.Tensor:
        """Process a chunk of j indices.

        For ending node: Q from (i,j), K from (k,j), bias from (k,i)
        """
        L = Q.size(0)
        H, c = self.n_heads, self.c_attn
        J = j_end - j_start

        # Extract chunk
        Q_chunk = Q[:, j_start:j_end]  # [L(i), J, H, c]
        K_chunk = K[:, j_start:j_end]  # [L(k), J, H, c] - keys at (k,j)
        V_chunk = V[:, j_start:j_end]  # [L(k), J, H, c]
        G_chunk = G[:, j_start:j_end]  # [L(i), J, H, c]

        # For ending node attention: attend over k for fixed j
        # Q[i,j] @ K[k,j] -> logits[i,k] per j
        # Reshape: treat j as batch
        Q_t = Q_chunk.permute(1, 2, 0, 3)  # [J, H, L(i), c]
        K_t = K_chunk.permute(1, 2, 3, 0)  # [J, H, c, L(k)]

        logits = torch.matmul(Q_t, K_t) * self.scale  # [J, H, L(i), L(k)]

        # Bias B[k,i] - need to transpose
        # B is [L, L, H] with B[k,i]
        B_t = B.permute(2, 1, 0).unsqueeze(0)  # [1, H, L(i), L(k)]
        logits = logits + B_t

        # Mask: k must be valid
        k_mask = res_mask.view(1, 1, 1, L)
        logits = logits.masked_fill(~k_mask, float("-inf"))

        # Softmax over k
        attn = F.softmax(logits, dim=-1)  # [J, H, L(i), L(k)]

        # Apply to values
        V_t = V_chunk.permute(1, 2, 0, 3)  # [J, H, L(k), c]
        out = torch.matmul(attn, V_t)  # [J, H, L(i), c]

        # Reshape back: [J, H, L(i), c] -> [L(i), J, H, c]
        out = out.permute(2, 0, 1, 3)  # [L(i), J, H, c]
        out = G_chunk * out
        out = out.reshape(L, J, H * c)
        out = self.proj_out(out)  # [L(i), J, c_z]

        return out
