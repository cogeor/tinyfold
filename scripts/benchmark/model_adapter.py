"""Model adapters for unified benchmark inference.

Each adapter wraps a specific model type and provides a standardized
predict_atoms() interface that returns [L, 4, 3] atom coordinates.
"""

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from .data_loader import BenchmarkSampleTensors

# Add paths for imports
scripts_path = Path(__file__).parent.parent
src_path = scripts_path.parent / "src"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models import create_model, create_schedule, create_noiser


@dataclass
class PredictionResult:
    """Result from model prediction."""

    atoms_pred: np.ndarray  # [L, 4, 3] in Angstroms
    inference_time_ms: float
    metadata: dict  # Model-specific info


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for benchmark-compatible model adapters."""

    model_type: str
    checkpoint_path: str

    def load(self, device: torch.device) -> dict:
        """Load model and return metadata."""
        ...

    def predict_atoms(
        self, sample: BenchmarkSampleTensors
    ) -> PredictionResult:
        """Run inference and return atoms + timing."""
        ...

    def get_config(self) -> dict:
        """Return model configuration for logging."""
        ...


class BaseAdapter(ABC):
    """Base class for model adapters."""

    model_type: str = "base"

    def __init__(self, checkpoint_path: str, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = None
        self.config = kwargs

    @abstractmethod
    def load(self, device: torch.device) -> dict:
        """Load model checkpoint."""
        pass

    @abstractmethod
    def predict_atoms(self, sample: BenchmarkSampleTensors) -> PredictionResult:
        """Run inference."""
        pass

    def get_config(self) -> dict:
        """Return configuration."""
        return {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            **self.config,
        }


class ResFoldAdapter(BaseAdapter):
    """Adapter for two-stage ResFold (stage1 + stage2)."""

    model_type = "resfold"

    def __init__(
        self,
        checkpoint_path: str,
        stage2_checkpoint: Optional[str] = None,
        n_timesteps: int = 50,
        clamp_val: float = 3.0,
    ):
        super().__init__(
            checkpoint_path,
            stage2_checkpoint=stage2_checkpoint,
            n_timesteps=n_timesteps,
            clamp_val=clamp_val,
        )
        self.stage2_checkpoint = stage2_checkpoint
        self.n_timesteps = n_timesteps
        self.clamp_val = clamp_val
        self.noiser = None

    def load(self, device: torch.device) -> dict:
        """Load ResFold model(s) from checkpoint."""
        self.device = device

        checkpoint = torch.load(
            self.checkpoint_path, map_location=device, weights_only=False
        )

        # Infer n_timesteps
        for k in checkpoint["model_state_dict"]:
            if "time_embed.weight" in k:
                self.n_timesteps = checkpoint["model_state_dict"][k].shape[0]
                break

        self.model = create_model("resfold", n_timesteps=self.n_timesteps)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(device)
        self.model.eval()

        schedule = create_schedule("cosine", T=self.n_timesteps)
        self.noiser = create_noiser("gaussian", schedule)

        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "n_timesteps": self.n_timesteps,
            "total_params": total_params,
            "has_stage2": self.model.stage2 is not None,
        }

    @torch.no_grad()
    def predict_atoms(self, sample: BenchmarkSampleTensors) -> PredictionResult:
        """Run two-stage ResFold inference."""
        device = self.device
        B, L = 1, sample.n_residues

        aa_seq = sample.aa_seq.unsqueeze(0)
        chain_ids = sample.chain_ids.unsqueeze(0)
        res_idx = sample.res_idx.unsqueeze(0)
        mask = torch.ones(1, L, dtype=torch.bool, device=device)

        start_time = time.time()

        # Stage 1: DDPM sampling for centroids
        x = torch.randn(B, L, 3, device=device)

        for t in reversed(range(self.noiser.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x0_pred = self.model.forward_stage1(x, aa_seq, chain_ids, res_idx, t_batch, mask)
            x0_pred = torch.clamp(x0_pred, -self.clamp_val, self.clamp_val)

            if t > 0:
                ab_t = self.noiser.alpha_bar[t]
                ab_prev = self.noiser.alpha_bar[t - 1]
                beta = self.noiser.betas[t]
                alpha = self.noiser.alphas[t]
                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x
                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

        centroids_norm = x  # [B, L, 3] normalized

        # Stage 2: Atom refinement
        if self.model.stage2 is not None:
            atoms_pred = self.model.forward_stage2(
                centroids_norm, aa_seq, chain_ids, res_idx, mask
            )
        else:
            # No stage 2, expand centroids to atoms (simple offset)
            # N, CA, C, O at fixed offsets from centroid
            offsets = torch.tensor(
                [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
                device=device,
            )
            atoms_pred = centroids_norm.unsqueeze(2) + offsets / sample.std

        inference_time_ms = (time.time() - start_time) * 1000

        # Denormalize: [B, L, 4, 3] -> [L, 4, 3]
        atoms_norm = atoms_pred[0].cpu().numpy()
        atoms = atoms_norm * sample.std + sample.centroid

        return PredictionResult(
            atoms_pred=atoms,
            inference_time_ms=inference_time_ms,
            metadata={"has_stage2": self.model.stage2 is not None},
        )


# Registry of adapters
ADAPTERS = {
    "resfold": ResFoldAdapter,
}


def create_adapter(model_type: str, checkpoint_path: str, **kwargs) -> BaseAdapter:
    """Create an adapter for the specified model type.

    Args:
        model_type: One of "resfold"
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional arguments for the adapter

    Returns:
        Configured adapter instance
    """
    if model_type not in ADAPTERS:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(ADAPTERS.keys())}"
        )

    return ADAPTERS[model_type](checkpoint_path, **kwargs)


def list_adapters() -> list[str]:
    """Return list of available adapter types."""
    return list(ADAPTERS.keys())
