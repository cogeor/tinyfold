"""Top-level PPI model combining all components."""

import torch
import torch.nn as nn

from tinyfold.model.config import ModelConfig
from tinyfold.model.denoiser.conditioner import AtomConditioner, TimestepEmbedding
from tinyfold.model.denoiser.edges import build_edge_attr, build_knn_edges, merge_edges
from tinyfold.model.denoiser.egnn import EGNNDenoiser
from tinyfold.model.diffusion.sampler import DDIMSampler
from tinyfold.model.diffusion.schedule import DiffusionSchedule
from tinyfold.model.embeddings import TokenEmbedder
from tinyfold.model.pairformer import PairformerStack


class PPIModel(nn.Module):
    """AlphaFold3-style PPI prediction model.

    Components:
    1. TokenEmbedder: sequence -> single + pair representations
    2. PairformerStack: refine single + pair representations
    3. AtomConditioner: lift to atom-level conditioning
    4. EGNNDenoiser: predict noise on atom coordinates
    5. DiffusionSchedule + DDIMSampler: noising/denoising
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        # Token embedder
        self.embedder = TokenEmbedder(
            c_s=c.c_s,
            c_z=c.c_z,
            max_seq_len=c.max_seq_len,
            n_aa=c.n_aa,
        )

        # Pairformer trunk
        self.pairformer = PairformerStack(
            n_blocks=c.n_blocks,
            c_s=c.c_s,
            c_z=c.c_z,
            n_heads_single=c.n_heads_single,
            n_heads_tri=c.n_heads_tri,
            c_tri_attn=c.c_tri_attn,
            c_tri_mul=c.c_tri_mul,
            transition_expansion=c.transition_expansion,
            dropout=c.dropout_pair,
            chunk_size=c.chunk_size_tri,
            use_checkpoint=c.use_checkpoint,
        )

        # Atom conditioner
        self.atom_conditioner = AtomConditioner(
            c_s=c.c_s,
            c_z=c.c_z,
            c_a=c.c_a,
            n_atom_types=c.n_atom_types,
            c_pair_proj=32,
        )

        # Timestep embedding
        self.time_embed = TimestepEmbedding(dim=16)

        # Compute edge dimension: edge_type(2) + bond_type(4) + pair_proj(32) + time(16)
        edge_dim = 2 + 4 + 32 + 16

        # EGNN denoiser
        self.denoiser = EGNNDenoiser(
            n_layers=c.n_egnn_layers,
            h_dim=c.c_a,
            edge_dim=edge_dim,
        )

        # Diffusion schedule
        self.schedule = DiffusionSchedule(T=c.diffusion_steps)

    def forward(
        self,
        seq: torch.Tensor,
        chain_id_res: torch.Tensor,
        res_idx: torch.Tensor,
        atom_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_res: torch.Tensor,
        atom_type: torch.Tensor,
        bonds_src: torch.Tensor,
        bonds_dst: torch.Tensor,
        bond_type: torch.Tensor,
        t: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            seq: [L] amino acid indices
            chain_id_res: [L] chain IDs (0 or 1)
            res_idx: [L] within-chain residue indices
            atom_coords: [N_atom, 3] ground truth coordinates
            atom_mask: [N_atom] boolean mask for valid atoms
            atom_to_res: [N_atom] residue index per atom
            atom_type: [N_atom] atom type (0-3)
            bonds_src: [E_bond] bond source atoms
            bonds_dst: [E_bond] bond destination atoms
            bond_type: [E_bond] bond type indices
            t: optional fixed timestep (random if None)
        Returns:
            dict with:
                - eps_hat: [N_atom, 3] predicted noise
                - eps: [N_atom, 3] ground truth noise
                - x0_hat: [N_atom, 3] predicted clean coords
                - x_t: [N_atom, 3] noised coords
                - t: int timestep
        """
        device = seq.device
        L = seq.size(0)
        N_atom = atom_coords.size(0)

        # Build residue mask from atom mask
        res_mask = torch.ones(L, dtype=torch.bool, device=device)

        # Encode sequence
        s, z = self.embedder(seq, chain_id_res, res_idx)

        # Refine with Pairformer
        s, z = self.pairformer(s, z, res_mask)

        # Lift to atom conditioning
        h, pair_proj = self.atom_conditioner(s, z, atom_to_res, atom_type)

        # Sample timestep if not provided
        if t is None:
            t = torch.randint(0, self.config.diffusion_steps, (1,), device=device).item()

        # Sample noise and create noised coordinates
        eps = torch.randn_like(atom_coords)
        x_t = self.schedule.q_sample(atom_coords, t, eps)

        # Build edges (bonds + KNN from noised coords)
        knn_edges = build_knn_edges(x_t, k=self.config.k_neighbors)
        edge_index, edge_type = merge_edges(bonds_src, bonds_dst, knn_edges)

        # Timestep embedding
        t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
        t_embed = self.time_embed(t_tensor)

        # Build edge attributes
        edge_attr = build_edge_attr(
            edge_index=edge_index,
            edge_type=edge_type,
            bond_type=bond_type,
            pair_proj=pair_proj,
            atom_to_res=atom_to_res,
            t_embed=t_embed,
            n_bond_edges=bonds_src.size(0),
        )

        # Predict noise
        eps_hat = self.denoiser(x_t, h, edge_index, edge_attr)

        # Predict x0 for auxiliary losses
        x0_hat = self.schedule.predict_x0(x_t, t, eps_hat)

        # Mask outputs for missing atoms
        mask = atom_mask.unsqueeze(-1)  # [N_atom, 1]
        eps_hat = eps_hat * mask
        x0_hat = x0_hat * mask

        return {
            "eps_hat": eps_hat,
            "eps": eps,
            "x0_hat": x0_hat,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        seq: torch.Tensor,
        chain_id_res: torch.Tensor,
        res_idx: torch.Tensor,
        atom_to_res: torch.Tensor,
        atom_type: torch.Tensor,
        bonds_src: torch.Tensor,
        bonds_dst: torch.Tensor,
        bond_type: torch.Tensor,
        n_atom: int,
    ) -> torch.Tensor:
        """Inference: sample coordinates from noise.

        Args:
            seq: [L] amino acid indices
            chain_id_res: [L] chain IDs (0 or 1)
            res_idx: [L] within-chain residue indices
            atom_to_res: [N_atom] residue index per atom
            atom_type: [N_atom] atom type (0-3)
            bonds_src: [E_bond] bond source atoms
            bonds_dst: [E_bond] bond destination atoms
            bond_type: [E_bond] bond type indices
            n_atom: number of atoms
        Returns:
            x0: [N_atom, 3] sampled coordinates
        """
        device = seq.device
        L = seq.size(0)

        # Build residue mask
        res_mask = torch.ones(L, dtype=torch.bool, device=device)

        # Encode sequence
        s, z = self.embedder(seq, chain_id_res, res_idx)

        # Refine with Pairformer
        self.pairformer.eval()
        s, z = self.pairformer(s, z, res_mask)

        # Lift to atom conditioning
        h, pair_proj = self.atom_conditioner(s, z, atom_to_res, atom_type)

        # Create denoising function for sampler
        def denoise_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            # Build edges from current coordinates
            knn_edges = build_knn_edges(x_t, k=self.config.k_neighbors)
            edge_index, edge_type = merge_edges(bonds_src, bonds_dst, knn_edges)

            # Timestep embedding
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
            t_embed = self.time_embed(t_tensor)

            # Build edge attributes
            edge_attr = build_edge_attr(
                edge_index=edge_index,
                edge_type=edge_type,
                bond_type=bond_type,
                pair_proj=pair_proj,
                atom_to_res=atom_to_res,
                t_embed=t_embed,
                n_bond_edges=bonds_src.size(0),
            )

            # Predict noise
            return self.denoiser(x_t, h, edge_index, edge_attr)

        # Sample using DDIM
        sampler = DDIMSampler(self.schedule, eta=0.0)
        x0 = sampler.sample(denoise_fn, shape=(n_atom, 3), device=device)

        return x0

    def encode(
        self,
        seq: torch.Tensor,
        chain_id_res: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to single and pair representations.

        Useful for precomputing conditioning for multiple samples.

        Args:
            seq: [L] amino acid indices
            chain_id_res: [L] chain IDs
            res_idx: [L] residue indices
        Returns:
            s: [L, c_s] single representation
            z: [L, L, c_z] pair representation
        """
        device = seq.device
        L = seq.size(0)
        res_mask = torch.ones(L, dtype=torch.bool, device=device)

        s, z = self.embedder(seq, chain_id_res, res_idx)
        s, z = self.pairformer(s, z, res_mask)

        return s, z
