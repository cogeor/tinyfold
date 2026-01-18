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
}

_NOISE_TYPES = {
    "gaussian": GaussianNoise,
    "linear_chain": LinearChainNoise,
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
