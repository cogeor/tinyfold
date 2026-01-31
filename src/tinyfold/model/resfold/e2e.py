"""ResFold End-to-End: Two-Phase Training with Multi-Sample Diffusion.

Combines Stage 1 (residue diffusion) with Stage 2 (multi-sample atom refinement)
for end-to-end training where gradients flow through both stages.

Architecture:
1. ResidueEncoder (Trunk): Runs ONCE per sample
2. ResidueDenoiser: Generates K diffusion samples
3. AtomRefinerV2MultiSample: Aggregates K samples -> atom predictions

Training phases:
- Phase 1: Train Stage 1 only (standard diffusion training)
- Phase 2: Train both stages E2E with multi-sample conditioning
"""

from typing import Optional, Literal, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from .denoiser import ResidueDenoiser
from .atomrefine_multi_sample import AtomRefinerV2MultiSample


TrainingMode = Literal["stage1_only", "stage2_e2e", "end_to_end"]


class ResFoldE2E(nn.Module):
    """End-to-end ResFold with multi-sample diffusion conditioning.

    Model sizes:
    - Stage 1: ~15M params (ResidueDenoiser)
    - Stage 2: ~5M params (AtomRefinerV2MultiSample)
    - Total: ~20M params
    """

    def __init__(
        self,
        # Stage 1 config
        c_token: int = 256,
        trunk_layers: int = 9,
        trunk_heads: int = 8,
        denoiser_blocks: int = 7,
        denoiser_heads: int = 8,
        n_timesteps: int = 50,
        n_aa_types: int = 21,
        n_chains: int = 2,
        # Stage 2 config
        s2_layers: int = 6,
        s2_heads: int = 8,
        n_samples: int = 5,
        s2_aggregation: str = "learned",
        # Shared
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps

        # Stage 1: Residue-level diffusion
        self.stage1 = ResidueDenoiser(
            c_token=c_token,
            trunk_layers=trunk_layers,
            trunk_heads=trunk_heads,
            denoiser_blocks=denoiser_blocks,
            denoiser_heads=denoiser_heads,
            n_timesteps=n_timesteps,
            n_aa_types=n_aa_types,
            n_chains=n_chains,
            dropout=dropout,
        )

        # Stage 2: Multi-sample atom refinement
        self.stage2 = AtomRefinerV2MultiSample(
            c_token=c_token,
            n_layers=s2_layers,
            n_heads=s2_heads,
            n_samples=n_samples,
            dropout=dropout,
            aggregation=s2_aggregation,
        )

    def get_trunk_tokens(
        self,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Get trunk tokens from sequence features (no coordinates)."""
        return self.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

    def forward_stage1(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Stage 1 forward with discrete timestep."""
        return self.stage1(x_t, aa_seq, chain_ids, res_idx, t, mask)

    def forward_stage1_sigma(
        self,
        x_t: Tensor,
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> Tensor:
        """Stage 1 forward with continuous sigma."""
        return self.stage1.forward_sigma(
            x_t, aa_seq, chain_ids, res_idx, sigma, mask, x0_prev
        )

    def forward_e2e(
        self,
        gt_centroids: Tensor,   # [B, L, 3]
        aa_seq: Tensor,         # [B, L]
        chain_ids: Tensor,      # [B, L]
        res_idx: Tensor,        # [B, L]
        mask: Tensor,           # [B, L]
        noiser,                 # VENoiser with sample_sigma method
        self_cond_prob: float = 0.0,
        stratified_sigma: bool = True,
    ) -> Dict[str, Tensor]:
        """End-to-end forward pass generating K diffusion samples.

        For each of K samples:
        1. Sample sigma from noise schedule
        2. Create noisy centroids: x_t = gt_centroids + sigma * noise
        3. Predict clean centroids: x0_pred = denoiser(x_t, trunk, sigma)

        Then pass all K predictions to Stage 2 for atom prediction.

        Args:
            gt_centroids: Ground truth centroids for training
            aa_seq, chain_ids, res_idx: Sequence features
            mask: Valid residue mask
            noiser: VENoiser with sample_sigma_* methods
            self_cond_prob: Probability of self-conditioning
            stratified_sigma: Use stratified sigma sampling

        Returns:
            dict with:
                - centroids_samples: [B, K, L, 3] K centroid predictions
                - atoms_pred: [B, L, 4, 3] atom predictions
                - sigmas: [B, K] sigma values used for each sample
        """
        B, L, _ = gt_centroids.shape
        device = gt_centroids.device

        # 1. Run trunk ONCE (sequence-only, no coordinates)
        trunk_tokens = self.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

        # 2. Generate K diffusion samples
        centroids_samples = []
        sigmas_list = []

        for k in range(self.n_samples):
            # Sample sigma for this diffusion sample
            if stratified_sigma:
                sigma = noiser.sample_sigma_stratified(B, device)
            else:
                sigma = noiser.sample_sigma_af3(B, device)
            sigmas_list.append(sigma)

            # Create noisy centroids: x_t = x0 + sigma * noise
            noise = torch.randn_like(gt_centroids)
            x_t = gt_centroids + sigma.view(-1, 1, 1) * noise

            # Self-conditioning (optional)
            x0_prev = None
            if self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
                with torch.no_grad():
                    x0_prev = self.stage1.forward_sigma_with_trunk(
                        x_t, trunk_tokens, sigma, mask, x0_prev=None
                    ).detach()

            # Predict clean centroids using pre-computed trunk
            x0_pred = self.stage1.forward_sigma_with_trunk(
                x_t, trunk_tokens, sigma, mask, x0_prev=x0_prev
            )
            centroids_samples.append(x0_pred)

        # Stack samples: [B, K, L, 3]
        centroids_stack = torch.stack(centroids_samples, dim=1)
        sigmas_stack = torch.stack(sigmas_list, dim=1)  # [B, K]

        # 3. Stage 2: atoms from trunk + multi-sample centroids
        atoms_pred = self.stage2(trunk_tokens, centroids_stack, mask)

        return {
            'centroids_samples': centroids_stack,  # [B, K, L, 3]
            'atoms_pred': atoms_pred,              # [B, L, 4, 3]
            'sigmas': sigmas_stack,                # [B, K]
            'trunk_tokens': trunk_tokens,          # [B, L, c_token]
        }

    def forward_e2e_with_given_samples(
        self,
        centroids_samples: Tensor,  # [B, K, L, 3]
        aa_seq: Tensor,
        chain_ids: Tensor,
        res_idx: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Stage 2 forward with pre-computed centroid samples.

        Useful for inference when samples are generated via full diffusion.

        Returns:
            atoms_pred: [B, L, 4, 3]
        """
        trunk_tokens = self.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)
        return self.stage2(trunk_tokens, centroids_samples, mask)

    def set_training_mode(self, mode: TrainingMode):
        """Set which parameters are trainable.

        Args:
            mode: One of:
                - 'stage1_only': Train only Stage 1 (diffusion)
                - 'stage2_e2e': Train Stage 2, optionally fine-tune Stage 1
                - 'end_to_end': Train both stages
        """
        if mode == "stage1_only":
            for p in self.stage1.parameters():
                p.requires_grad = True
            for p in self.stage2.parameters():
                p.requires_grad = False

        elif mode == "stage2_e2e":
            # Stage 2 always trainable
            for p in self.stage2.parameters():
                p.requires_grad = True
            # Stage 1: trainable for E2E gradient flow
            for p in self.stage1.parameters():
                p.requires_grad = True

        elif mode == "end_to_end":
            for p in self.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def freeze_stage1(self):
        """Freeze Stage 1 parameters (for Stage 2-only training)."""
        for p in self.stage1.parameters():
            p.requires_grad = False

    def unfreeze_stage1(self):
        """Unfreeze Stage 1 parameters."""
        for p in self.stage1.parameters():
            p.requires_grad = True

    def load_stage1_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device,
        strict: bool = False,
    ):
        """Load Stage 1 weights from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load to
            strict: If True, raise error on missing/unexpected keys
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Filter to Stage 1 keys only
        stage1_state = {}
        for key, value in state_dict.items():
            if key.startswith('stage1.'):
                stage1_state[key] = value
            elif not key.startswith('stage2.'):
                # Handle case where checkpoint has stage1 params without prefix
                stage1_state[f'stage1.{key}'] = value

        missing, unexpected = self.load_state_dict(stage1_state, strict=False)

        return {
            'step': ckpt.get('step', 0),
            'missing_keys': len(missing),
            'unexpected_keys': len(unexpected),
        }

    def count_parameters(self) -> Dict[str, Any]:
        """Count parameters in each stage."""
        s1_params = sum(p.numel() for p in self.stage1.parameters())
        s2_params = sum(p.numel() for p in self.stage2.parameters())
        total = s1_params + s2_params

        s1_trainable = sum(p.numel() for p in self.stage1.parameters() if p.requires_grad)
        s2_trainable = sum(p.numel() for p in self.stage2.parameters() if p.requires_grad)

        return {
            'stage1': s1_params,
            'stage2': s2_params,
            'total': total,
            'stage1_pct': 100 * s1_params / total if total > 0 else 0,
            'stage2_pct': 100 * s2_params / total if total > 0 else 0,
            'stage1_trainable': s1_trainable,
            'stage2_trainable': s2_trainable,
            'total_trainable': s1_trainable + s2_trainable,
        }


@torch.no_grad()
def sample_e2e(
    model: ResFoldE2E,
    aa_seq: Tensor,
    chain_ids: Tensor,
    res_idx: Tensor,
    noiser,
    mask: Optional[Tensor] = None,
    n_samples: int = 5,
    clamp_val: float = 3.0,
    self_cond: bool = True,
    align_per_step: bool = True,
    recenter: bool = True,
) -> Dict[str, Tensor]:
    """Full E2E sampling: diffusion -> multi-sample -> atoms.

    Runs K independent diffusion trajectories from noise, then passes
    all K centroid predictions to Stage 2.

    Args:
        model: ResFoldE2E model
        aa_seq, chain_ids, res_idx: Sequence features
        noiser: VENoiser with sigma schedule
        mask: Valid residue mask
        n_samples: Number of diffusion samples (K)
        clamp_val: Clamp predictions to [-clamp_val, clamp_val]
        self_cond: Use self-conditioning during sampling
        align_per_step: Kabsch-align predictions each step
        recenter: Recenter coordinates each step

    Returns:
        dict with centroids_samples [B, K, L, 3] and atoms_pred [B, L, 4, 3]
    """
    from tinyfold.model.diffusion.utils import kabsch_align_to_target

    B, L = aa_seq.shape
    device = aa_seq.device

    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Get sigma schedule
    sigmas = noiser.sigmas.to(device)

    # Run trunk once
    trunk_tokens = model.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

    # Generate K independent diffusion samples
    all_centroids = []

    for k in range(n_samples):
        # Start from noise at highest sigma
        x = sigmas[0] * torch.randn(B, L, 3, device=device)
        x0_prev = None

        # Euler sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_batch = sigma.expand(B)

            # Predict x0
            x0_pred = model.stage1.forward_sigma_with_trunk(
                x, trunk_tokens, sigma_batch, mask,
                x0_prev=x0_prev if self_cond else None
            )
            x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

            # Kabsch alignment
            if align_per_step:
                x0_pred = kabsch_align_to_target(x0_pred, x, mask)

            # Self-conditioning: store for next iteration
            x0_prev = x0_pred.detach()

            # Euler step
            d = (x - x0_pred) / sigma
            dt = sigma_next - sigma
            x = x + d * dt

            # Recenter
            if recenter:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
                x = x - centroid

        all_centroids.append(x)

    # Stack: [B, K, L, 3]
    centroids_stack = torch.stack(all_centroids, dim=1)

    # Stage 2: atoms from multi-sample centroids
    atoms_pred = model.stage2(trunk_tokens, centroids_stack, mask)

    return {
        'centroids_samples': centroids_stack,
        'atoms_pred': atoms_pred,
        'mean_centroids': centroids_stack.mean(dim=1),
    }
