"""Protocol definitions for TinyFold models.

Defines interfaces for model components to enable type checking and
ensure consistent APIs across different model implementations.
"""

from typing import Any, Protocol, TypedDict

from torch import Tensor


class Batch(TypedDict, total=False):
    """The residue/centroid training batch.

    Produced by ``tinyfold.training.data.collate_batch`` and consumed by the live
    model and samplers. This documents the shape that is otherwise threaded around
    as ``dict[str, Any]``. ``total=False`` because the template/ESM keys are present
    only when those features are enabled.
    """

    centroids: Tensor           # [B, L, 3] residue centroids
    coords_res: Tensor          # [B, L, 4, 3] backbone atoms per residue
    aa_seq: Tensor              # [B, L] amino-acid indices
    chain_ids: Tensor           # [B, L] chain id (0/1)
    res_idx: Tensor             # [B, L] residue index within chain
    mask_res: Tensor            # [B, L] bool, True for real residues
    coords: Tensor              # [B, N, 3] flattened atoms
    atom_types: Tensor          # [B, N]
    atom_to_res: Tensor         # [B, N]
    mask_atom: Tensor           # [B, N] bool
    esm_embed: Tensor           # [B, L, esm_dim] (ESM mode only)
    template_coords_res: Tensor  # [B, L, 4, 3] (template conditioning only)
    template_mask: Tensor
    template_frame_id: Tensor
    stds: list[float]           # per-sample coordinate std used for de-normalisation
    n_res: list[int]
    n_atoms: list[int]
    sample_ids: list[str]


class DiffusionDecoder(Protocol):
    """All diffusion decoders must implement this interface.

    A decoder predicts clean coordinates x0 from noisy input x_t at timestep t.
    """

    def forward(
        self,
        x_t: Tensor,                    # Noisy input [B, N, 3]
        t: Tensor,                      # Timesteps [B]
        conditioning: dict[str, Tensor], # Model-specific conditioning
        mask: Tensor | None = None,  # Valid positions [B, N]
    ) -> Tensor:
        """Predict clean x0 from noisy x_t at timestep t.

        Args:
            x_t: Noisy coordinates [B, N, 3]
            t: Timesteps [B]
            conditioning: Dict with model-specific inputs like:
                - aa_seq: Amino acid sequence [B, N]
                - chain_ids: Chain identifiers [B, N]
                - atom_types: Atom types [B, N]
            mask: Optional mask for valid positions [B, N]

        Returns:
            x0_pred: Predicted clean coordinates [B, N, 3]
        """
        ...


class ResidueEncoder(Protocol):
    """Encoder that produces per-residue representations."""

    def forward(
        self,
        aa_seq: Tensor,                 # [B, L]
        chain_ids: Tensor,              # [B, L]
        positions: Tensor,              # [B, L, 3]
        mask: Tensor | None = None,  # [B, L]
    ) -> Tensor:
        """Encode residue features to token embeddings.

        Args:
            aa_seq: Amino acid indices [B, L]
            chain_ids: Chain identifiers [B, L]
            positions: Centroid or CA positions [B, L, 3]
            mask: Optional mask for valid residues [B, L]

        Returns:
            embeddings: Per-residue embeddings [B, L, C]
        """
        ...


class AtomDecoder(Protocol):
    """Decoder that predicts atom positions from residue representations."""

    def forward(
        self,
        centroids: Tensor,              # [B, L, 3]
        residue_tokens: Tensor,         # [B, L, C]
        aa_seq: Tensor,                 # [B, L]
        mask: Tensor | None = None,  # [B, L]
    ) -> Tensor:
        """Predict atom positions from residue centroids and features.

        Args:
            centroids: Residue centroid positions [B, L, 3]
            residue_tokens: Per-residue embeddings [B, L, C]
            aa_seq: Amino acid indices [B, L]
            mask: Optional mask for valid residues [B, L]

        Returns:
            atoms: Atom positions [B, L, 4, 3] (N, CA, C, O)
        """
        ...


class StructurePredictor(Protocol):
    """Full structure prediction model (any architecture).

    Can predict either residue centroids (Stage 1) or full atom
    positions (Stage 2 / end-to-end).
    """

    def predict(
        self,
        batch: dict[str, Tensor],
        noiser: Any,
        n_steps: int = 50,
    ) -> dict[str, Tensor]:
        """Run full inference to predict structure.

        Args:
            batch: Input batch with keys like 'aa_seq', 'chain_ids', etc.
            noiser: Diffusion noiser for sampling
            n_steps: Number of diffusion steps

        Returns:
            Dict with predictions:
                - 'centroids': [B, L, 3] if predicting centroids
                - 'atoms': [B, N, 3] if predicting atoms
        """
        ...


class Noiser(Protocol):
    """Protocol for diffusion noise processes."""

    T: int  # Number of timesteps

    def add_noise(
        self,
        x0: Tensor,
        t: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Add noise to clean coordinates.

        Args:
            x0: Clean coordinates [B, N, 3]
            t: Timesteps [B]
            **kwargs: Noise-type specific args

        Returns:
            x_t: Noisy coordinates [B, N, 3]
            target: Training target (noise or x0 depending on type)
        """
        ...

    def to(self, device) -> "Noiser":
        """Move to device."""
        ...


class Schedule(Protocol):
    """Protocol for diffusion schedules."""

    T: int
    alpha_bar: Tensor
    sqrt_alpha_bar: Tensor
    sqrt_one_minus_alpha_bar: Tensor
    alphas: Tensor
    betas: Tensor

    def to(self, device) -> "Schedule":
        """Move to device."""
        ...
