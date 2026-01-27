"""ResFold Atom Assembler: Predict atom positions from residue tokens + centroid samples.

This is Stage 2 of the ResFold pipeline. Takes:
- Residue tokens from Stage 1 trunk encoder
- K=5 centroid diffusion samples per residue

Outputs:
- 4 atom positions (N, CA, C, O) per residue

Key design:
- Concatenates K centroid samples as additional spatial features
- Supports masked training (predict subset of atoms)
- Iterative inference: predict atoms incrementally, starting from most central residues
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .clustering import (
    select_next_residues_to_place,
    get_residue_placement_order,
    select_next_atoms_to_place,
)


class ResFoldAssembler(nn.Module):
    """Predict atom positions from residue tokens + centroid samples.

    Architecture:
    1. Embed K centroid samples per residue -> spatial features
    2. Concatenate with trunk tokens
    3. Transformer refinement
    4. Output: 4 atom positions per residue

    Args:
        c_token: Hidden dimension (must match Stage 1)
        n_samples: Number of centroid samples (K=5)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        c_token: int = 256,
        n_samples: int = 5,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_samples = n_samples
        self.n_layers = n_layers

        # Embed centroid samples: [B, K, L, 3] -> [B, L, K*c_embed]
        # Each sample gets its own embedding, then concatenated
        c_embed = c_token // n_samples  # e.g., 256 // 5 = 51
        self.c_embed = c_embed

        self.centroid_embed = nn.Sequential(
            nn.Linear(3, c_embed),
            nn.LayerNorm(c_embed),
            nn.GELU(),
            nn.Linear(c_embed, c_embed),
        )

        # Project concatenated centroids to c_token
        self.centroid_proj = nn.Linear(n_samples * c_embed, c_token)

        # Combine trunk tokens + centroid features
        self.combine_proj = nn.Linear(c_token * 2, c_token)

        # Atom type queries (N=0, CA=1, C=2, O=3)
        self.atom_queries = nn.Parameter(torch.randn(1, 1, 4, c_token) * 0.02)

        # Transformer layers for refinement
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(c_token, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(c_token, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_token),
                nn.Linear(c_token, c_token * 4),
                nn.GELU(),
                nn.Linear(c_token * 4, c_token),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.self_norms = nn.ModuleList([nn.LayerNorm(c_token) for _ in range(n_layers)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(c_token) for _ in range(n_layers)])

        # Output: predict offset from centroid mean for each atom
        self.output_norm = nn.LayerNorm(c_token)
        self.output_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        trunk_tokens: Tensor,       # [B, L, c_token] from Stage 1 trunk
        centroid_samples: Tensor,   # [B, K, L, 3] K centroid samples per residue
        mask: Optional[Tensor] = None,  # [B, L] residue mask
        atom_mask: Optional[Tensor] = None,  # [B, L*4] atom mask for training
    ) -> Tensor:
        """Predict atom positions.

        Args:
            trunk_tokens: Residue embeddings from Stage 1 [B, L, c_token]
            centroid_samples: K centroid diffusion samples [B, K, L, 3]
            mask: Valid residue mask [B, L]
            atom_mask: Optional mask for which atoms to predict [B, L*4]

        Returns:
            atom_coords: Predicted atom coordinates [B, L, 4, 3]
        """
        B, K, L, _ = centroid_samples.shape
        device = trunk_tokens.device

        # === Embed centroid samples ===
        # [B, K, L, 3] -> [B, K, L, c_embed]
        centroid_emb = self.centroid_embed(centroid_samples)

        # Reshape to [B, L, K*c_embed] - concatenate samples per residue
        centroid_emb = centroid_emb.permute(0, 2, 1, 3)  # [B, L, K, c_embed]
        centroid_emb = centroid_emb.reshape(B, L, -1)     # [B, L, K*c_embed]

        # Project to c_token
        centroid_feat = self.centroid_proj(centroid_emb)  # [B, L, c_token]

        # === Combine with trunk tokens ===
        combined = torch.cat([trunk_tokens, centroid_feat], dim=-1)  # [B, L, 2*c_token]
        residue_feat = self.combine_proj(combined)  # [B, L, c_token]

        # === Create atom queries ===
        # Expand atom queries for each residue: [B, L, 4, c_token]
        atom_queries = self.atom_queries.expand(B, L, -1, -1)

        # Add residue features to atom queries
        atom_queries = atom_queries + residue_feat.unsqueeze(2)  # [B, L, 4, c_token]

        # Flatten for transformer: [B, L*4, c_token]
        atom_queries = atom_queries.reshape(B, L * 4, self.c_token)

        # === Transformer layers ===
        # Key padding mask for residues
        if mask is not None:
            # Expand mask to atom level
            res_key_mask = ~mask  # [B, L], True = ignore
        else:
            res_key_mask = None

        for i in range(self.n_layers):
            # Self-attention among atoms
            q = self.self_norms[i](atom_queries)
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            atom_queries = atom_queries + attn_out

            # Cross-attention to residue features
            q = self.cross_norms[i](atom_queries)
            attn_out, _ = self.cross_attn_layers[i](
                q, residue_feat, residue_feat,
                key_padding_mask=res_key_mask
            )
            atom_queries = atom_queries + attn_out

            # FFN
            atom_queries = atom_queries + self.ffn_layers[i](atom_queries)

        # === Output ===
        atom_queries = self.output_norm(atom_queries)
        atom_offsets = self.output_proj(atom_queries)  # [B, L*4, 3]

        # Reshape to [B, L, 4, 3]
        atom_offsets = atom_offsets.reshape(B, L, 4, 3)

        # Add mean centroid as anchor
        centroid_mean = centroid_samples.mean(dim=1)  # [B, L, 3]
        atom_coords = atom_offsets + centroid_mean.unsqueeze(2)  # [B, L, 4, 3]

        return atom_coords

    def forward_masked(
        self,
        trunk_tokens: Tensor,       # [B, L, c_token]
        centroid_samples: Tensor,   # [B, K, L, 3]
        known_atoms: Tensor,        # [B, L, 4, 3] known atom positions (use GT for known)
        known_mask: Tensor,         # [B, L, 4] bool, True = atom is known
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict atoms with some already known (for iterative training).

        During training, we can mask random atoms and provide GT for the rest.
        The model learns to predict masked atoms conditioned on known ones.

        Args:
            trunk_tokens: Residue embeddings [B, L, c_token]
            centroid_samples: Centroid samples [B, K, L, 3]
            known_atoms: Ground truth atoms (used where known_mask=True) [B, L, 4, 3]
            known_mask: Which atoms are known [B, L, 4]
            mask: Valid residue mask [B, L]

        Returns:
            atom_coords: Predicted coordinates [B, L, 4, 3]
                         (known positions unchanged, unknown predicted)
        """
        B, K, L, _ = centroid_samples.shape
        device = trunk_tokens.device

        # === Embed centroid samples ===
        centroid_emb = self.centroid_embed(centroid_samples)
        centroid_emb = centroid_emb.permute(0, 2, 1, 3).reshape(B, L, -1)
        centroid_feat = self.centroid_proj(centroid_emb)

        # === Combine with trunk tokens ===
        combined = torch.cat([trunk_tokens, centroid_feat], dim=-1)
        residue_feat = self.combine_proj(combined)

        # === Create atom queries with known positions ===
        atom_queries = self.atom_queries.expand(B, L, -1, -1).clone()
        atom_queries = atom_queries + residue_feat.unsqueeze(2)

        # For known atoms, add positional info from known_atoms
        # Embed known positions
        known_pos_emb = self.centroid_embed(known_atoms)  # [B, L, 4, c_embed]
        # Pad to c_token
        known_pos_emb = F.pad(known_pos_emb, (0, self.c_token - self.c_embed))

        # Add known position embedding where mask is True
        atom_queries = atom_queries + known_pos_emb * known_mask.unsqueeze(-1).float()

        # Flatten
        atom_queries = atom_queries.reshape(B, L * 4, self.c_token)

        # === Transformer layers ===
        if mask is not None:
            res_key_mask = ~mask
        else:
            res_key_mask = None

        for i in range(self.n_layers):
            q = self.self_norms[i](atom_queries)
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            atom_queries = atom_queries + attn_out

            q = self.cross_norms[i](atom_queries)
            attn_out, _ = self.cross_attn_layers[i](
                q, residue_feat, residue_feat,
                key_padding_mask=res_key_mask
            )
            atom_queries = atom_queries + attn_out

            atom_queries = atom_queries + self.ffn_layers[i](atom_queries)

        # === Output ===
        atom_queries = self.output_norm(atom_queries)
        atom_offsets = self.output_proj(atom_queries).reshape(B, L, 4, 3)

        centroid_mean = centroid_samples.mean(dim=1)
        pred_coords = atom_offsets + centroid_mean.unsqueeze(2)

        # Blend: use known_atoms where known, pred where not
        atom_coords = torch.where(
            known_mask.unsqueeze(-1),
            known_atoms,
            pred_coords
        )

        return atom_coords

    @torch.no_grad()
    def sample_iterative(
        self,
        trunk_tokens: Tensor,       # [B, L, c_token]
        centroid_samples: Tensor,   # [B, K, L, 3]
        mask: Optional[Tensor] = None,
        k_per_step: int = 4,        # Atoms to fix per iteration
        n_iterations: Optional[int] = None,  # If None, auto-compute
        update_centroids: bool = True,  # Update centroids when residue fully fixed
    ) -> Tensor:
        """Iterative inference with centroid updates at atom level.

        Process:
        1. Start from atoms closest to geometric center
        2. Predict all atoms using current centroids
        3. Fix K atoms closest to already-fixed atoms
        4. When all 4 atoms of a residue are fixed, update its centroid
        5. Repeat until all atoms fixed

        Args:
            trunk_tokens: Residue embeddings [B, L, c_token]
            centroid_samples: Centroid samples [B, K, L, 3]
            mask: Valid residue mask [B, L]
            k_per_step: Number of atoms to fix per iteration
            n_iterations: Number of iterations (default: ceil(n_atoms / k_per_step))
            update_centroids: Update centroids when residue is fully fixed

        Returns:
            atom_coords: Final predicted coordinates [B, L, 4, 3]
        """
        B, K, L, _ = centroid_samples.shape
        device = trunk_tokens.device
        n_atoms = L * 4

        if n_iterations is None:
            n_iterations = (n_atoms + k_per_step - 1) // k_per_step

        # Clone centroids for updates
        centroids = centroid_samples.clone()

        # Initialize: all atoms unknown
        known_atoms = torch.zeros(B, L, 4, 3, device=device)
        known_mask = torch.zeros(B, L, 4, dtype=torch.bool, device=device)

        # Track which atoms have been fixed (flat indexing)
        fixed_mask_flat = torch.zeros(B, n_atoms, dtype=torch.bool, device=device)

        # Track which residues have had centroids updated
        centroid_updated = torch.zeros(B, L, dtype=torch.bool, device=device)

        for iteration in range(n_iterations):
            # Predict all atoms conditioned on known
            pred_atoms = self.forward_masked(
                trunk_tokens, centroids,
                known_atoms, known_mask, mask
            )  # [B, L, 4, 3]

            # Determine how many atoms to fix this iteration
            n_remaining = (~fixed_mask_flat).sum(dim=1)  # [B]
            n_to_fix = min(k_per_step, n_remaining.min().item())

            if n_to_fix == 0:
                break

            pred_flat = pred_atoms.view(B, n_atoms, 3)

            for b in range(B):
                if fixed_mask_flat[b].all():
                    continue

                # Use clustering-based selection: closest to already-fixed atoms
                # Use predicted positions as reference for spatial ordering
                fix_indices = select_next_atoms_to_place(
                    pred_flat[b],           # Current predicted positions
                    fixed_mask_flat[b],     # Already-fixed mask
                    k_per_step,
                )

                # Mark as fixed and update known_atoms
                fixed_mask_flat[b, fix_indices] = True

                for idx in fix_indices:
                    res_idx = idx // 4
                    atom_idx = idx % 4
                    known_atoms[b, res_idx, atom_idx] = pred_flat[b, idx]
                    known_mask[b, res_idx, atom_idx] = True

                # Update centroids for fully-fixed residues
                if update_centroids:
                    for res_idx in range(L):
                        if centroid_updated[b, res_idx]:
                            continue
                        # Check if all 4 atoms of this residue are fixed
                        if known_mask[b, res_idx].all():
                            # Update centroid = mean of 4 atoms
                            pred_centroid = known_atoms[b, res_idx].mean(dim=0)
                            centroids[b, :, res_idx] = pred_centroid
                            centroid_updated[b, res_idx] = True

        # Final prediction with fully updated centroids
        final_atoms = self.forward_masked(
            trunk_tokens, centroids,
            known_atoms, known_mask, mask
        )

        return final_atoms

    @torch.no_grad()
    def sample_iterative_residue(
        self,
        trunk_tokens: Tensor,       # [B, L, c_token]
        centroid_samples: Tensor,   # [B, K, L, 3]
        mask: Optional[Tensor] = None,
        k_residues_per_step: int = 1,  # Residues to fix per iteration
        update_centroids: bool = True,  # Update centroids from predicted atoms
        centroid_blend: float = 1.0,    # How much to blend (1.0 = full replacement)
    ) -> Tensor:
        """Iterative inference at residue level with centroid updates.

        Process:
        1. Start from most central residue (closest to geometric center)
        2. Predict all atoms using current centroid samples
        3. Fix K residues closest to already-fixed residues
        4. Update centroid samples for fixed residues -> mean of predicted atoms
        5. Repeat with improved centroids

        Args:
            trunk_tokens: Residue embeddings [B, L, c_token]
            centroid_samples: Centroid samples [B, K, L, 3]
            mask: Valid residue mask [B, L]
            k_residues_per_step: Residues to fix per iteration
            update_centroids: Whether to update centroids from predictions
            centroid_blend: Blend factor (1.0 = replace, 0.5 = average old/new)

        Returns:
            atom_coords: Final predicted coordinates [B, L, 4, 3]
        """
        B, K, L, _ = centroid_samples.shape
        device = trunk_tokens.device

        n_iterations = (L + k_residues_per_step - 1) // k_residues_per_step

        # Clone centroid samples so we can update them
        centroids = centroid_samples.clone()

        # Initialize
        known_atoms = torch.zeros(B, L, 4, 3, device=device)
        known_mask = torch.zeros(B, L, 4, dtype=torch.bool, device=device)
        fixed_residues = torch.zeros(B, L, dtype=torch.bool, device=device)

        for iteration in range(n_iterations):
            # Predict using current centroids
            pred_atoms = self.forward_masked(
                trunk_tokens, centroids,
                known_atoms, known_mask, mask
            )

            # Select residues to fix using clustering - most central first
            for b in range(B):
                if fixed_residues[b].all():
                    continue

                # Get current centroids for ordering
                # Use the mean of K centroid samples as the reference positions
                current_centroids = centroids[b].mean(dim=0)  # [L, 3]

                # Select next residues using spatial proximity
                fix_res = select_next_residues_to_place(
                    current_centroids,
                    fixed_residues[b],
                    k_residues_per_step
                )

                # Fix all 4 atoms of selected residues
                for res_idx in fix_res:
                    known_atoms[b, res_idx] = pred_atoms[b, res_idx]
                    known_mask[b, res_idx] = True
                    fixed_residues[b, res_idx] = True

                    # Update centroids: new centroid = mean of predicted atoms
                    if update_centroids:
                        pred_centroid = pred_atoms[b, res_idx].mean(dim=0)  # [3]
                        # Update all K samples for this residue
                        if centroid_blend >= 1.0:
                            centroids[b, :, res_idx] = pred_centroid
                        else:
                            old = centroids[b, :, res_idx]
                            centroids[b, :, res_idx] = (1 - centroid_blend) * old + centroid_blend * pred_centroid

        # Final pass with fully updated centroids
        final_atoms = self.forward_masked(
            trunk_tokens, centroids,
            known_atoms, known_mask, mask
        )

        return final_atoms

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'n_layers': self.n_layers,
            'c_token': self.c_token,
            'n_samples': self.n_samples,
        }


def test_resfold_assembler():
    """Quick test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResFoldAssembler(
        c_token=256,
        n_samples=5,
        n_layers=6,
        n_heads=8,
    ).to(device)

    params = model.count_parameters()
    print(f"ResFoldAssembler: {params['total']/1e6:.2f}M params")

    # Test forward
    B, L, K = 2, 50, 5
    trunk_tokens = torch.randn(B, L, 256, device=device)
    centroid_samples = torch.randn(B, K, L, 3, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    atoms = model(trunk_tokens, centroid_samples, mask)
    print(f"Output shape: {atoms.shape}")  # [B, L, 4, 3]

    # Test masked forward
    known_atoms = torch.randn(B, L, 4, 3, device=device)
    known_mask = torch.rand(B, L, 4, device=device) > 0.5

    atoms_masked = model.forward_masked(
        trunk_tokens, centroid_samples, known_atoms, known_mask, mask
    )
    print(f"Masked output shape: {atoms_masked.shape}")

    # Test iterative inference (atom-level)
    atoms_iter = model.sample_iterative(
        trunk_tokens, centroid_samples, mask,
        k_per_step=8, n_iterations=None
    )
    print(f"Iterative output shape: {atoms_iter.shape}")

    # Test iterative inference (residue-level)
    atoms_iter_res = model.sample_iterative_residue(
        trunk_tokens, centroid_samples, mask,
        k_residues_per_step=5
    )
    print(f"Iterative (residue) output shape: {atoms_iter_res.shape}")

    # Verify known positions are preserved
    diff = (atoms_masked - known_atoms).abs()
    known_diff = (diff * known_mask.unsqueeze(-1).float()).sum()
    print(f"Known positions preserved: {known_diff.item() < 1e-5}")

    print("All tests passed!")


if __name__ == "__main__":
    test_resfold_assembler()
