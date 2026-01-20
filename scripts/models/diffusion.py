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

    def __init__(self, schedule: CosineSchedule, noise_scale: float = 0.1):
        """
        Args:
            schedule: The alpha_bar schedule to use
            noise_scale: Scale of additional Gaussian noise (0 = pure interpolation)
        """
        self.schedule = schedule
        self.noise_scale = noise_scale

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
                N, atom_to_res[b], atom_type[b], chain_ids[b], device
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

    def __init__(self, schedule: CosineSchedule, noise_scale: float = 0.1):
        """
        Args:
            schedule: Used only for T (number of steps)
            noise_scale: Small noise added to input during training
        """
        self.schedule = schedule
        self.noise_scale = noise_scale

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
                N, atom_to_res[b], atom_type[b], chain_ids[b], device
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
) -> Tensor:
    """Generate extended chain coordinates (all residues in straight lines).

    Fully vectorized - no Python loops for GPU efficiency.

    Args:
        n_atoms: Number of atoms
        atom_to_res: Residue index for each atom [N]
        atom_type: Atom type (0=N, 1=CA, 2=C, 3=O) [N]
        chain_ids: Chain ID for each atom [N]
        device: Device to create tensor on

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

    # Random rotation
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


# =============================================================================
# Factory functions
# =============================================================================

_SCHEDULES = {
    "cosine": CosineSchedule,
    "linear": LinearSchedule,
}

_NOISE_TYPES = {
    "gaussian": GaussianNoise,
    "linear_chain": LinearChainNoise,
    "linear_flow": LinearChainFlow,
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
