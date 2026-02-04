"""ResFold Pipeline: Two-Stage PPI Structure Prediction."""

from typing import Optional, Literal
import torch
import torch.nn as nn
from torch import Tensor

from .denoiser import ResidueDenoiser

TrainingMode = Literal['stage1_only', 'stage2_only', 'end_to_end']

class ResFoldPipeline(nn.Module):
    def __init__(
        self,
        c_token_s1: int = 256,
        trunk_layers: int = 9,
        trunk_heads: int = 8,
        denoiser_blocks: int = 7,
        denoiser_heads: int = 8,
        n_timesteps: int = 50,
        c_token_s2: int = 256,
        s2_layers: int = 18,
        s2_heads: int = 8,
        n_aa_types: int = 21,
        n_chains: int = 2,
        dropout: float = 0.0,
        stage1_only: bool = False,  # Lightweight mode: skip Stage 2
    ):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.c_token_s1 = c_token_s1
        self._stage1_only = stage1_only

        self.stage1 = ResidueDenoiser(
            c_token=c_token_s1,
            trunk_layers=trunk_layers,
            trunk_heads=trunk_heads,
            denoiser_blocks=denoiser_blocks,
            denoiser_heads=denoiser_heads,
            n_timesteps=n_timesteps,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # Only create Stage 2 if needed (saves ~14M params for stage1_only training)
        self.stage2 = None
        if not stage1_only:
            from .atomrefine_v2 import AtomRefinerV2
            self.stage2 = AtomRefinerV2(
                c_token=c_token_s2,
                n_layers=s2_layers,
                n_heads=s2_heads,
                dropout=dropout,
            )

    def forward_stage1(self, x_t, aa_seq, chain_ids, res_idx, t, mask=None):
        return self.stage1(x_t, aa_seq, chain_ids, res_idx, t, mask)

    def get_trunk_tokens(self, aa_seq, chain_ids, res_idx, mask=None):
        """Get trunk tokens from sequence features (no coordinates)."""
        return self.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

    def forward_stage2(self, centroids, aa_seq, chain_ids, res_idx, mask=None, trunk_tokens=None):
        if self.stage2 is None:
            raise RuntimeError("Stage 2 not available (model created with stage1_only=True)")
        if trunk_tokens is None:
            trunk_tokens = self.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)
        offsets = self.stage2(trunk_tokens, centroids, mask)
        atom_coords = centroids.unsqueeze(2) + offsets
        return atom_coords

    def forward(self, x_t, aa_seq, chain_ids, res_idx, t, mask=None, mode='end_to_end', gt_centroids=None, centroid_noise=0.0):
        result = {}
        # Trunk tokens are sequence-only (computed once)
        trunk_tokens = self.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

        if mode == 'stage2_only':
            assert gt_centroids is not None
            centroids_input = gt_centroids
            if centroid_noise > 0:
                centroids_input = gt_centroids + centroid_noise * torch.randn_like(gt_centroids)
            atoms_pred = self.forward_stage2(centroids_input, aa_seq, chain_ids, res_idx, mask, trunk_tokens=trunk_tokens)
            result['centroids_pred'] = gt_centroids
            result['atoms_pred'] = atoms_pred
        elif mode == 'stage1_only':
            centroids_pred = self.forward_stage1(x_t, aa_seq, chain_ids, res_idx, t, mask)
            result['centroids_pred'] = centroids_pred
            result['atoms_pred'] = None
        else:
            centroids_pred = self.forward_stage1(x_t, aa_seq, chain_ids, res_idx, t, mask)
            atoms_pred = self.forward_stage2(centroids_pred, aa_seq, chain_ids, res_idx, mask, trunk_tokens=trunk_tokens)
            result['centroids_pred'] = centroids_pred
            result['atoms_pred'] = atoms_pred
        return result

    @torch.no_grad()
    def sample(self, aa_seq, chain_ids, res_idx, noiser, mask=None, clamp_val=3.0):
        B, L = aa_seq.shape
        device = aa_seq.device
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)
        x = torch.randn(B, L, 3, device=device)
        for t in reversed(range(noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x0_pred = self.forward_stage1(x, aa_seq, chain_ids, res_idx, t_batch, mask)
            x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
            if t > 0:
                ab_t = noiser.alpha_bar[t]
                ab_prev = noiser.alpha_bar[t - 1]
                beta = noiser.betas[t]
                alpha = noiser.alphas[t]
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred
        centroids = x
        atoms = self.forward_stage2(centroids, aa_seq, chain_ids, res_idx, mask)
        return atoms.view(B, L * 4, 3)

    def set_training_mode(self, mode):
        if mode == 'stage1_only':
            for p in self.stage1.parameters(): p.requires_grad = True
            if self.stage2 is not None:
                for p in self.stage2.parameters(): p.requires_grad = False
        elif mode == 'stage2_only':
            if self.stage2 is None:
                raise RuntimeError("Stage 2 not available (model created with stage1_only=True)")
            for p in self.stage1.parameters(): p.requires_grad = False
            for p in self.stage2.parameters(): p.requires_grad = True
        else:
            for p in self.parameters(): p.requires_grad = True

    def count_parameters(self):
        s1 = sum(p.numel() for p in self.stage1.parameters())
        s2 = sum(p.numel() for p in self.stage2.parameters()) if self.stage2 is not None else 0
        total = s1 + s2
        if total == 0:
            return {'stage1': s1, 'stage2': s2, 'total': total, 'stage1_pct': 0, 'stage2_pct': 0}
        return {'stage1': s1, 'stage2': s2, 'total': total, 'stage1_pct': 100*s1/total, 'stage2_pct': 100*s2/total}