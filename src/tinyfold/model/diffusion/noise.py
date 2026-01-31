"""Noise types for diffusion models.

Separates the concept of "what noise looks like" from the schedule.
- GaussianNoise: Standard DDPM noise
- LinearChainNoise: Interpolation toward extended chain
- LinearChainFlow: Iterative refinement from extended chain
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Optional


def generate_extended_chain(
    n_atoms: int,
    atom_to_res: Tensor,
    atom_type: Tensor,
    chain_ids: Tensor,
    device: torch.device,
    apply_rotation: bool = False,
) -> Tensor:
    """Generate extended chain coordinates (all residues in straight lines).

    Fully vectorized - no Python loops for GPU efficiency.

    Args:
        n_atoms: Number of atoms
        atom_to_res: Residue index for each atom [N]
        atom_type: Atom type (0=N, 1=CA, 2=C, 3=O) [N]
        chain_ids: Chain ID for each atom [N]
        device: Device to create tensor on
        apply_rotation: If True, apply random rotation (default False for deterministic)

    Returns:
        x_linear: Extended chain coordinates [N, 3]
    """
    CA_CA_DIST = 3.8
    CHAIN_OFFSET = 20.0

    # Atom offsets from CA [4, 3]
    ATOM_OFFSETS = torch.tensor([
        [-0.4, 0.0, 0.0],   # N
        [0.0, 0.0, 0.0],    # CA
        [0.4, 0.0, 0.0],    # C
        [0.6, 0.3, 0.0],    # O
    ], device=device)

    # Get chain ID per residue (first atom of each residue)
    n_res = atom_to_res.max().item() + 1
    chain_id_res = torch.zeros(n_res, dtype=chain_ids.dtype, device=device)
    chain_id_res.scatter_(0, atom_to_res, chain_ids)

    # Compute position index within each chain (vectorized cumsum)
    chain_a_mask = (chain_id_res == 0)
    chain_b_mask = (chain_id_res == 1)

    # Cumsum gives position within chain
    pos_in_chain = torch.zeros(n_res, device=device)
    pos_in_chain[chain_a_mask] = torch.arange(chain_a_mask.sum(), device=device, dtype=torch.float32)
    pos_in_chain[chain_b_mask] = torch.arange(chain_b_mask.sum(), device=device, dtype=torch.float32)

    # Build CA positions vectorized [n_res, 3]
    ca_positions = torch.zeros(n_res, 3, device=device)
    ca_positions[:, 0] = pos_in_chain * CA_CA_DIST  # x = position * spacing
    ca_positions[:, 1] = chain_id_res.float() * CHAIN_OFFSET  # y = chain offset

    # Center
    ca_positions = ca_positions - ca_positions.mean(dim=0, keepdim=True)

    # Build atom positions: CA[res] + offset[atom_type] (fully vectorized)
    x_linear = ca_positions[atom_to_res] + ATOM_OFFSETS[atom_type] * CA_CA_DIST

    # Optional random rotation
    if apply_rotation:
        R = _random_rotation_matrix(device)
        x_linear = x_linear @ R.T

    return x_linear


def _random_rotation_matrix(device: torch.device) -> Tensor:
    """Generate a random 3x3 rotation matrix."""
    q = torch.randn(4, device=device)
    q = q / q.norm()

    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], device=device)

    return R


class GaussianNoise:
    """Standard Gaussian noise diffusion."""

    def __init__(self, schedule):
        """
        Args:
            schedule: Schedule object with alpha_bar, sqrt_alpha_bar, etc.
        """
        self.schedule = schedule

    @property
    def T(self):
        return self.schedule.T

    @property
    def alpha_bar(self):
        return self.schedule.alpha_bar

    @property
    def alphas(self):
        return self.schedule.alphas

    @property
    def betas(self):
        return self.schedule.betas

    def to(self, device):
        self.schedule = self.schedule.to(device)
        return self

    def add_noise(self, x0: Tensor, t: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Add Gaussian noise to x0.

        Args:
            x0: Clean coordinates [B, N, 3]
            t: Timesteps [B]
            **kwargs: Ignored (for API compatibility)

        Returns:
            x_t: Noisy coordinates [B, N, 3]
            noise: The Gaussian noise that was added [B, N, 3]
        """
        noise = torch.randn_like(x0)
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        return x_t, noise


class LinearChainNoise:
    """Diffusion toward extended chain structure.

    Instead of pure Gaussian noise, the "fully noised" state is an extended
    chain (residues in straight lines). Small Gaussian noise is added on top.
    """

    def __init__(self, schedule, noise_scale: float = 0.1, rotation: bool = False):
        """
        Args:
            schedule: The alpha_bar schedule to use
            noise_scale: Scale of additional Gaussian noise (0 = pure interpolation)
            rotation: If True, apply random rotation to extended chain
        """
        self.schedule = schedule
        self.noise_scale = noise_scale
        self.rotation = rotation

    @property
    def T(self):
        return self.schedule.T

    @property
    def alpha_bar(self):
        return self.schedule.alpha_bar

    @property
    def alphas(self):
        return self.schedule.alphas

    @property
    def betas(self):
        return self.schedule.betas

    def to(self, device):
        self.schedule = self.schedule.to(device)
        return self

    def add_noise(
        self,
        x0: Tensor,
        t: Tensor,
        atom_to_res: Tensor,
        atom_type: Tensor,
        chain_ids: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Interpolate toward extended chain + add small Gaussian noise.

        Args:
            x0: Clean coordinates [B, N, 3]
            t: Timesteps [B]
            atom_to_res: Residue indices [B, N]
            atom_type: Atom types [B, N]
            chain_ids: Chain IDs [B, N]

        Returns:
            x_t: Noisy coordinates [B, N, 3]
            x_linear: Extended chain target [B, N, 3]
        """
        B, N, _ = x0.shape
        device = x0.device

        # Generate extended chain for each sample in batch
        x_linear = torch.zeros_like(x0)
        for b in range(B):
            x_linear[b] = generate_extended_chain(
                N, atom_to_res[b], atom_type[b], chain_ids[b], device,
                apply_rotation=self.rotation
            )

        # Normalize extended chain to match x0 scale
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_linear_std = x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x_linear = x_linear / x_linear_std

        # Interpolate: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * x_linear
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        x_interp = sqrt_ab * x0 + sqrt_one_minus_ab * x_linear

        # Add small Gaussian noise for stochasticity
        if self.noise_scale > 0:
            noise = torch.randn_like(x0) * self.noise_scale
            x_t = x_interp + noise
        else:
            x_t = x_interp

        return x_t, x_linear

    def reverse_step(
        self,
        x_t: Tensor,
        x0_pred: Tensor,
        t: int,
        x_linear: Tensor,
    ) -> Tensor:
        """Reverse diffusion step for linear chain noise.

        Args:
            x_t: Current noisy state [B, N, 3]
            x0_pred: Model's prediction of x0 [B, N, 3]
            t: Current timestep (scalar)
            x_linear: Extended chain coordinates [B, N, 3]

        Returns:
            x_{t-1}: Previous state [B, N, 3]
        """
        if t == 0:
            return x0_pred

        sqrt_ab_prev = self.schedule.sqrt_alpha_bar[t - 1]
        sqrt_one_minus_ab_prev = self.schedule.sqrt_one_minus_alpha_bar[t - 1]

        x_prev = sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev * x_linear

        if self.noise_scale > 0:
            x_prev = x_prev + torch.randn_like(x_prev) * self.noise_scale

        return x_prev


class LinearChainFlow:
    """Iterative refinement flow from extended chain to folded structure.

    Simple approach:
    - t=0: x_linear (fully extended)
    - t=T: x0 (fully folded)
    - x_t = alpha * x0 + (1-alpha) * x_linear
    """

    def __init__(self, schedule, noise_scale: float = 0.1, rotation: bool = False):
        """
        Args:
            schedule: Used for T and alpha_bar
            noise_scale: Small noise added to input during training
            rotation: If True, apply random rotation to extended chain
        """
        self.schedule = schedule
        self.noise_scale = noise_scale
        self.rotation = rotation

    @property
    def T(self):
        return self.schedule.T

    @property
    def alpha_bar(self):
        return self.schedule.alpha_bar

    @property
    def alphas(self):
        return self.schedule.alphas

    @property
    def betas(self):
        return self.schedule.betas

    def to(self, device):
        self.schedule = self.schedule.to(device)
        return self

    def add_noise(
        self,
        x0: Tensor,
        t: Tensor,
        atom_to_res: Tensor,
        atom_type: Tensor,
        chain_ids: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Create noisy input at timestep t for x0 prediction.

        Args:
            x0: Clean/folded coordinates [B, N, 3]
            t: Timesteps [B] (0 to T-1)
            atom_to_res: Residue indices [B, N]
            atom_type: Atom types [B, N]
            chain_ids: Chain IDs [B, N]

        Returns:
            x_t: Noisy coordinates at timestep t [B, N, 3]
            x0: Target (clean coordinates) [B, N, 3]
        """
        B, N, _ = x0.shape
        device = x0.device

        # Generate extended chain for each sample
        x_linear = torch.zeros_like(x0)
        for b in range(B):
            x_linear[b] = generate_extended_chain(
                N, atom_to_res[b], atom_type[b], chain_ids[b], device,
                apply_rotation=self.rotation
            )

        # Normalize
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_linear_std = x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x_linear = x_linear / x_linear_std

        # Store for inference
        self._last_x_linear = x_linear

        # Interpolation using alpha_bar
        alpha = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        one_minus_alpha = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        x_t = alpha * x0 + one_minus_alpha * x_linear

        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale


# =============================================================================
# VE Noiser (Variance-Exploding, AF3-style)
# =============================================================================

class VENoiser:
    """Variance-Exploding noise process (AF3/EDM style).

    VE diffusion: x_t = x0 + sigma * eps (additive noise)

    Unlike VP (variance-preserving) which blends x0 with noise,
    VE simply adds noise of increasing magnitude.

    This is the noise process used in AlphaFold3.
    """

    def __init__(
        self,
        schedule,
        sigma_data: float = 1.0,
    ):
        """
        Args:
            schedule: KarrasSchedule with sigma values
            sigma_data: Standard deviation of data (1.0 for normalized coords)
        """
        self.schedule = schedule
        self.sigma_data = sigma_data

    @property
    def T(self):
        return self.schedule.T

    @property
    def sigmas(self):
        return self.schedule.sigmas

    @property
    def alpha_bar(self):
        return self.schedule.alpha_bar

    @property
    def betas(self):
        return self.schedule.betas

    @property
    def alphas(self):
        return self.schedule.alphas

    def to(self, device):
        self.schedule = self.schedule.to(device)
        return self

    def add_noise(
        self,
        x0: Tensor,
        t: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Add VE noise: x_t = x0 + sigma[t] * eps

        Args:
            x0: Clean coordinates [B, N, 3]
            t: Timestep indices [B] (used to look up sigma)
            **kwargs: Ignored (for API compatibility)

        Returns:
            x_t: Noisy coordinates [B, N, 3]
            noise: The noise that was added [B, N, 3]
        """
        noise = torch.randn_like(x0)
        sigma = self.schedule.sigmas[t].view(-1, 1, 1)  # [B, 1, 1]
        x_t = x0 + sigma * noise
        return x_t, noise

    def add_noise_continuous(
        self,
        x0: Tensor,
        sigma: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Add VE noise with continuous sigma (for training).

        Args:
            x0: Clean coordinates [B, N, 3]
            sigma: Noise level [B]

        Returns:
            x_t: Noisy coordinates [B, N, 3]
            noise: The noise that was added [B, N, 3]
        """
        noise = torch.randn_like(x0)
        sigma_view = sigma.view(-1, 1, 1)
        x_t = x0 + sigma_view * noise
        return x_t, noise

    def sample_sigma(self, batch_size: int, device) -> Tensor:
        """Sample sigma for training (log-normal, like AF3/EDM).

        Returns sigma in [sigma_min, sigma_max] with log-uniform density.
        """
        log_sigma = torch.randn(batch_size, device=device) * 1.2  # P_mean=0, P_std=1.2
        sigma = self.sigma_data * torch.exp(log_sigma)
        # Clamp to valid range
        sigma = sigma.clamp(self.schedule.sigma_min, self.schedule.sigma_max)
        return sigma

    def sample_sigma_af3(self, batch_size: int, device) -> Tensor:
        """Sample sigma for training using AF3's exact distribution.

        AF3 uses: σ = σ_data * exp(-1.2 + 1.5 * N(0,1))

        This distribution:
        - Is centered at σ_data * exp(-1.2) ≈ 0.3 * σ_data
        - Has wider spread than standard log-normal (std=1.5 vs 1.2)
        - Biases toward lower noise levels (the -1.2 shift)

        For normalized coords (σ_data=1.0), typical range is ~[0.01, 10].
        """
        log_sigma = -1.2 + 1.5 * torch.randn(batch_size, device=device)
        sigma = self.sigma_data * torch.exp(log_sigma)
        # Clamp to valid range
        sigma = sigma.clamp(self.schedule.sigma_min, self.schedule.sigma_max)
        return sigma

    def sample_sigma_stratified(self, batch_size: int, device) -> Tensor:
        """Sample sigma with stratified buckets for better high-sigma coverage.

        Problem: AF3's log-normal barely samples high sigma (>6 only ~2% of time),
        but inference starts at sigma_max (10.0). This causes poor denoising.

        Solution: Stratified sampling across log-space buckets:
        - Bucket 1: [sigma_min, 0.1] - fine detail denoising (25%)
        - Bucket 2: [0.1, 1.0] - medium noise (25%)
        - Bucket 3: [1.0, 5.0] - high noise (25%)
        - Bucket 4: [5.0, sigma_max] - very high noise (25%)

        This ensures the model trains adequately at all noise levels.
        """
        sigma_min = self.schedule.sigma_min
        sigma_max = self.schedule.sigma_max

        # Define bucket boundaries in log-space
        buckets = [
            (sigma_min, 0.1),
            (0.1, 1.0),
            (1.0, 5.0),
            (5.0, sigma_max),
        ]

        # Randomly assign each sample to a bucket
        bucket_idx = torch.randint(0, len(buckets), (batch_size,), device=device)

        # Sample uniformly in log-space within each bucket
        sigma = torch.zeros(batch_size, device=device)
        for i, (lo, hi) in enumerate(buckets):
            mask = bucket_idx == i
            n_in_bucket = mask.sum().item()
            if n_in_bucket > 0:
                # Uniform in log-space
                log_lo, log_hi = math.log(lo), math.log(hi)
                log_sigma = torch.rand(n_in_bucket, device=device) * (log_hi - log_lo) + log_lo
                sigma[mask] = torch.exp(log_sigma)

        return sigma
