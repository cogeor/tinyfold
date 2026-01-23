"""Diffusion components for TinyFold.

Separates two concepts:
- Schedule: How alpha_bar changes with timestep (e.g., cosine, linear)
- Noise type: What the "fully noised" state looks like (gaussian, linear_chain)

Usage:
    schedule = CosineSchedule(T=50)
    noiser = GaussianNoise(schedule)  # or LinearChainNoise(schedule)
    x_t, target = noiser.add_noise(x0, t, ...)
"""

import math
import torch
from torch import Tensor


# =============================================================================
# Schedules - define how alpha_bar changes with timestep
# =============================================================================

class CosineSchedule:
    """Cosine schedule for alpha_bar (from Nichol & Dhariwal 2021)."""

    def __init__(self, T: int = 50, s: float = 0.008):
        self.T = T
        self.s = s

        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1]])
        self.betas = 1 - self.alphas

    def to(self, device):
        for attr in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'alphas', 'betas']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


class LinearSchedule:
    """Linear alpha_bar schedule. Directly interpolates alpha_bar from 1 to 0."""

    def __init__(self, T: int = 50):
        self.T = T

        # Direct linear interpolation of alpha_bar
        t = torch.arange(T + 1, dtype=torch.float32)
        alpha_bar = 1 - t / T
        alpha_bar = alpha_bar.clamp(min=1e-6)  # Avoid exactly 0

        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        # Derive alphas and betas from alpha_bar
        self.alphas = torch.cat([torch.ones(1), alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-6)])
        self.betas = 1 - self.alphas

    def to(self, device):
        for attr in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'alphas', 'betas']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


# =============================================================================
# Timestep Curriculum - start easy, increase difficulty
# =============================================================================

class TimestepCurriculum:
    """Curriculum for diffusion timesteps - start with easy denoising, increase difficulty."""

    def __init__(self, T: int, warmup_steps: int, schedule: str = "linear"):
        """
        Args:
            T: Maximum timestep (from noiser)
            warmup_steps: Training steps to reach full T
            schedule: "linear" or "cosine" progression
        """
        self.T = T
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def get_max_t(self, step: int) -> int:
        """Get maximum timestep for current training step."""
        if step >= self.warmup_steps:
            return self.T
        progress = step / self.warmup_steps
        if self.schedule == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))
        # At least t_max=1 to avoid empty range
        return max(1, int(progress * self.T))

    def sample(self, batch_size: int, step: int, device) -> torch.Tensor:
        """Sample timesteps with curriculum constraint."""
        t_max = self.get_max_t(step)
        return torch.randint(0, t_max, (batch_size,), device=device)


# =============================================================================
# Noise types - define what "fully noised" looks like
# =============================================================================

class GaussianNoise:
    """Standard Gaussian noise diffusion."""

    def __init__(self, schedule: CosineSchedule):
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

    def add_noise(self, x0: Tensor, t: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
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

    def __init__(self, schedule: CosineSchedule, noise_scale: float = 0.1, rotation: bool = False):
        """
        Args:
            schedule: The alpha_bar schedule to use
            noise_scale: Scale of additional Gaussian noise (0 = pure interpolation)
            rotation: If True, apply random rotation to extended chain (default False)
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
    ) -> tuple[Tensor, Tensor]:
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

        # Normalize extended chain to match x0 scale (zero mean, unit std)
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

        Instead of DDPM's Gaussian noise addition, we interpolate between
        predicted x0 and the extended chain.

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

        # x_{t-1} = sqrt(ab_{t-1}) * x0_pred + sqrt(1-ab_{t-1}) * x_linear
        sqrt_ab_prev = self.schedule.sqrt_alpha_bar[t - 1]
        sqrt_one_minus_ab_prev = self.schedule.sqrt_one_minus_alpha_bar[t - 1]

        x_prev = sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev * x_linear

        # Add small noise for stochasticity (matching forward process)
        if self.noise_scale > 0:
            x_prev = x_prev + torch.randn_like(x_prev) * self.noise_scale

        return x_prev


class LinearChainFlow:
    """Iterative refinement flow from extended chain to folded structure.

    Simple approach:
    - t=0: x_linear (fully extended)
    - t=T: x0 (fully folded)
    - x_t = (t/T) * x0 + (1 - t/T) * x_linear (linear interpolation)

    Training:
    - Sample t from 1 to T
    - Input: x_{t-1} (less folded)
    - Target: x_t (more folded)
    - Model learns to take one refinement step

    Inference:
    - Start from x_linear
    - Run T steps: x_t = model(x_{t-1}, t)
    - Final output is the folded structure
    """

    def __init__(self, schedule: CosineSchedule, noise_scale: float = 0.1, rotation: bool = False):
        """
        Args:
            schedule: Used only for T (number of steps)
            noise_scale: Small noise added to input during training
            rotation: If True, apply random rotation to extended chain (default False)
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
    ) -> tuple[Tensor, Tensor]:
        """Create noisy input at timestep t for x0 prediction.

        Like gaussian diffusion but interpolates with extended chain instead of noise.
        Model always predicts x0 from x_t.

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

        # Normalize extended chain (same scale as x0)
        x_linear = x_linear - x_linear.mean(dim=1, keepdim=True)
        x_linear_std = x_linear.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x_linear = x_linear / x_linear_std

        # Store x_linear for inference (accessed via last call)
        self._last_x_linear = x_linear

        # Interpolation: t=0 is x_linear, t=T is x0
        # Use same alpha_bar as gaussian for consistency
        alpha = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)  # [B, 1, 1]
        one_minus_alpha = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        # x_t = alpha * x0 + (1-alpha) * x_linear
        x_t = alpha * x0 + one_minus_alpha * x_linear

        # Add small noise for robustness
        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale

        return x_t, x0  # Return x0 as target (model predicts x0)

    def reverse_step(
        self,
        x_t: Tensor,
        velocity_pred: Tensor,
        t: int,
        x_linear: Tensor,  # Not used, but kept for API compatibility
    ) -> Tensor:
        """Take one step from x_t toward x_0 using predicted velocity.

        Args:
            x_t: Current state [B, N, 3]
            velocity_pred: Predicted velocity v = x0 - x_linear [B, N, 3]
            t: Current timestep
            x_linear: Not used (API compatibility)

        Returns:
            x_{t-1}: Next state closer to x0
        """
        if t == 0:
            # At t=0, we're already at x0
            return x_t

        # Step size: we need to move 1/T of the total distance per step
        # x_{t-1} = x_t + velocity_pred / T
        step_size = 1.0 / self.T
        x_prev = x_t + velocity_pred * step_size

        # Add small noise for stochasticity
        if self.noise_scale > 0:
            x_prev = x_prev + torch.randn_like(x_prev) * self.noise_scale * 0.5

        return x_prev


# =============================================================================
# Helper functions
# =============================================================================

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
    # For each residue, count how many residues of same chain came before
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

    # Optional random rotation (disabled by default for deterministic behavior)
    if apply_rotation:
        R = random_rotation_matrix(device)
        x_linear = x_linear @ R.T

    return x_linear


def random_rotation_matrix(device: torch.device) -> Tensor:
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


def kabsch_align_to_target(
    pred: Tensor,
    target: Tensor,
    mask: Tensor = None,
) -> Tensor:
    """Kabsch-align pred INTO target's coordinate frame.

    This is the key fix for diffusion sampling drift (Boltz-1 style).
    Unlike standard kabsch_align() which returns both tensors centered,
    this returns pred transformed to match target's frame exactly.

    The problem: during diffusion sampling, the model predicts x0 in a
    potentially different rigid frame than the current x_t. If we naively
    interpolate (coef1 * x0_pred + coef2 * x_t), the result is warped garbage.

    The fix: before interpolation, Kabsch-align x0_pred to x_t's frame.

    Args:
        pred: Predicted coordinates [B, N, 3] (e.g., x0_pred from denoiser)
        target: Target frame coordinates [B, N, 3] (e.g., current x_t)
        mask: Optional mask for valid positions [B, N]

    Returns:
        pred_aligned: Prediction aligned to target's frame [B, N, 3]
    """
    B = pred.shape[0]
    device = pred.device

    # Compute centroids
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()  # [B, N, 1]
        n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        pred_mean = (pred * mask_exp).sum(dim=1, keepdim=True) / n_valid
        target_mean = (target * mask_exp).sum(dim=1, keepdim=True) / n_valid
    else:
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

    # Center both
    pred_c = pred - pred_mean
    target_c = target - target_mean

    if mask is not None:
        pred_c = pred_c * mask_exp
        target_c = target_c * mask_exp

    # SVD for optimal rotation: find R that minimizes ||target_c - pred_c @ R^T||
    H = torch.bmm(pred_c.transpose(1, 2), target_c)  # [B, 3, 3]
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case (ensure proper rotation, not reflection)
    d = torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    D[:, 2, 2] = d

    # Optimal rotation matrix
    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # [B, 3, 3]

    # Apply rotation to centered pred, then translate to target's frame
    pred_rotated = torch.bmm(pred_c, R.transpose(1, 2))  # [B, N, 3]
    pred_aligned = pred_rotated + target_mean  # Translate to target's centroid

    return pred_aligned


# =============================================================================
# Karras Schedule (EDM-style continuous sigma)
# =============================================================================

class KarrasSchedule:
    """Karras/EDM-style continuous sigma schedule.

    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., 2022). Uses continuous sigma values instead of discrete timesteps.

    Schedule: sigma[i] = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    For normalized coordinates (std=1), use sigma_min=0.002, sigma_max=10.0.
    """

    def __init__(
        self,
        n_steps: int = 200,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
    ):
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        # Build schedule: decreasing from sigma_max to sigma_min
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        inv_rho = 1.0 / rho
        sigmas = (sigma_max ** inv_rho + steps * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
        self.sigmas = sigmas  # [n_steps+1], from high to low

        # For compatibility with VP-style code, create fake alpha_bar
        # VP: x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
        # VE: x_t = x0 + sigma * eps
        # Mapping: sigma^2 = (1 - ab) / ab  =>  ab = 1 / (1 + sigma^2)
        self.alpha_bar = 1.0 / (1.0 + sigmas ** 2)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        # Fake betas/alphas for interface compatibility
        self.betas = torch.zeros(n_steps + 1)
        self.alphas = torch.ones(n_steps + 1)

    @property
    def T(self):
        return self.n_steps

    def to(self, device):
        for attr in ['sigmas', 'alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'betas', 'alphas']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def sigma_to_t(self, sigma: Tensor) -> Tensor:
        """Convert sigma to equivalent discrete timestep index."""
        # Find nearest sigma in schedule
        dists = (self.sigmas.unsqueeze(0) - sigma.unsqueeze(1)).abs()
        return dists.argmin(dim=1)


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
        schedule: KarrasSchedule,
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
    ) -> tuple[Tensor, Tensor]:
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
    ) -> tuple[Tensor, Tensor]:
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

    def loss_weight(self, sigma: Tensor) -> Tensor:
        """AF3/EDM-style loss weighting.

        Weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

        This gives higher weight to intermediate noise levels.
        """
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2 + 1e-8)

    def c_skip(self, sigma: Tensor) -> Tensor:
        """Skip connection coefficient for EDM preconditioning."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Tensor) -> Tensor:
        """Output scaling coefficient for EDM preconditioning."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: Tensor) -> Tensor:
        """Input scaling coefficient for EDM preconditioning."""
        return 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: Tensor) -> Tensor:
        """Noise level encoding for model input."""
        return torch.log(sigma / self.sigma_data) / 4.0


# =============================================================================
# Factory functions
# =============================================================================

_SCHEDULES = {
    "cosine": CosineSchedule,
    "linear": LinearSchedule,
    "karras": KarrasSchedule,
}

_NOISE_TYPES = {
    "gaussian": GaussianNoise,
    "linear_chain": LinearChainNoise,
    "linear_flow": LinearChainFlow,
    "ve": VENoiser,
}


def list_schedules() -> list[str]:
    """Return list of available schedule names."""
    return list(_SCHEDULES.keys())


def list_noise_types() -> list[str]:
    """Return list of available noise type names."""
    return list(_NOISE_TYPES.keys())


def create_schedule(name: str, **kwargs) -> CosineSchedule:
    """Create a schedule by name."""
    if name not in _SCHEDULES:
        raise ValueError(f"Unknown schedule: {name}. Available: {list(_SCHEDULES.keys())}")
    return _SCHEDULES[name](**kwargs)


def create_noiser(noise_type: str, schedule: CosineSchedule, **kwargs):
    """Create a noiser by name.

    Args:
        noise_type: "gaussian" or "linear_chain"
        schedule: The schedule to use
        **kwargs: Noise-type specific args (e.g., noise_scale for linear_chain)
    """
    if noise_type not in _NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(_NOISE_TYPES.keys())}")
    return _NOISE_TYPES[noise_type](schedule, **kwargs)
