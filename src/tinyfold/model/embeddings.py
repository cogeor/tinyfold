"""Token embedder for single and pair representations."""

import torch
import torch.nn as nn


class TokenEmbedder(nn.Module):
    """Embeds sequence tokens into single and pair representations.

    Single representation: s[i] = E_aa[seq_i] + E_pos[res_idx_i] + E_chain[chain_i]
    Pair representation: p[i,j] = proj(cat(u_i, u_j, u_i*u_j)) + E_chainpair + E_relpos
    """

    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        max_seq_len: int = 1024,
        n_aa: int = 21,
    ):
        """
        Args:
            c_s: Single embedding dimension
            c_z: Pair embedding dimension
            max_seq_len: Maximum sequence length
            n_aa: Number of amino acid types (20 + X)
        """
        super().__init__()

        # Single embeddings
        self.E_aa = nn.Embedding(n_aa, c_s)
        self.E_pos = nn.Embedding(max_seq_len, c_s)
        self.E_chain = nn.Embedding(2, c_s)

        # Pair embeddings
        # Chain pair: AA=0, AB=1, BA=2, BB=3
        self.E_chainpair = nn.Embedding(4, c_z)
        # Relative position: -32 to +32 (65 buckets) + 1 for cross-chain
        self.E_relpos = nn.Embedding(66, c_z)

        # Pair projection from single features
        self.pair_proj = nn.Linear(3 * c_s, c_z)
        self.ln = nn.LayerNorm(c_s)

    def forward(
        self,
        seq: torch.Tensor,
        chain_id_res: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq: [L] int - amino acid indices (0-20)
            chain_id_res: [L] int - chain ID (0 or 1)
            res_idx: [L] int - within-chain residue index
        Returns:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
        """
        L = seq.size(0)

        # Single embedding: sum of AA, position, and chain embeddings
        s = self.E_aa(seq) + self.E_pos(res_idx) + self.E_chain(chain_id_res)

        # Pair initialization from single features
        u = self.ln(s)  # [L, c_s]
        u_i = u.unsqueeze(1).expand(-1, L, -1)  # [L, L, c_s]
        u_j = u.unsqueeze(0).expand(L, -1, -1)  # [L, L, c_s]
        cat_ij = torch.cat([u_i, u_j, u_i * u_j], dim=-1)  # [L, L, 3*c_s]

        # Chain pair indices: 0=AA, 1=AB, 2=BA, 3=BB
        chain_pair = chain_id_res.unsqueeze(1) * 2 + chain_id_res.unsqueeze(0)

        # Relative position (clipped, special for cross-chain)
        relpos = self._compute_relpos(res_idx, chain_id_res)

        # Combine all pair features
        z = self.pair_proj(cat_ij) + self.E_chainpair(chain_pair) + self.E_relpos(relpos)

        return s, z

    def _compute_relpos(
        self,
        res_idx: torch.Tensor,
        chain_id_res: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative position with cross-chain handling.

        For same-chain pairs: relative position clipped to [-32, 32] -> indices [0, 64]
        For cross-chain pairs: special bucket 65
        """
        L = res_idx.size(0)
        same_chain = chain_id_res.unsqueeze(1) == chain_id_res.unsqueeze(0)

        # Compute raw difference
        diff = res_idx.unsqueeze(1) - res_idx.unsqueeze(0)  # [L, L]

        # Clip to [-32, 32] -> indices [0, 64]
        clipped = torch.clamp(diff, -32, 32) + 32

        # Cross-chain uses special bucket 65
        relpos = torch.where(same_chain, clipped, torch.full_like(clipped, 65))

        return relpos
