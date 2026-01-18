"""Model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for TinyFold model."""

    # Token embedder
    n_aa: int = 21  # 20 AA + X
    max_seq_len: int = 1024

    # Dimensions
    c_s: int = 256  # Single embedding dim
    c_z: int = 128  # Pair embedding dim
    c_a: int = 128  # Atom hidden dim

    # Pairformer
    n_blocks: int = 12
    n_heads_single: int = 8
    n_heads_tri: int = 4
    c_tri_attn: int = 32
    c_tri_mul: int = 128
    transition_expansion: int = 4
    dropout_pair: float = 0.1
    chunk_size_tri: int = 16  # Chunk size for triangle attention

    # Denoiser
    n_egnn_layers: int = 6
    k_neighbors: int = 16
    n_atom_types: int = 4  # N, CA, C, O

    # Diffusion
    diffusion_steps: int = 16
    schedule_type: str = "cosine"

    # Training
    use_checkpoint: bool = True
