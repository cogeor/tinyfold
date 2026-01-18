"""Tests for model components.

Following the testing philosophy from CLAUDE.md:
- Focus on integration tests and shape invariants
- Test EGNN equivariance (critical for coordinate learning)
- Test diffusion schedule properties
"""

import pytest
import torch

from tinyfold.model.config import ModelConfig
from tinyfold.model.denoiser.edges import build_knn_edges, merge_edges
from tinyfold.model.denoiser.egnn import EGNNDenoiser, EGNNLayer
from tinyfold.model.diffusion.schedule import DiffusionSchedule
from tinyfold.model.embeddings import TokenEmbedder
from tinyfold.model.pairformer import PairformerStack
from tinyfold.model.pairformer.triangle_mul import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from tinyfold.model.ppi_model import PPIModel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return ModelConfig(
        c_s=64,
        c_z=32,
        c_a=32,
        n_blocks=2,
        n_heads_single=4,
        n_heads_tri=2,
        c_tri_attn=16,
        c_tri_mul=32,
        n_egnn_layers=2,
        k_neighbors=4,
        diffusion_steps=4,
        use_checkpoint=False,
    )


@pytest.fixture
def sample_batch():
    """Create a minimal sample batch for testing."""
    L = 20  # 10 residues per chain
    N_atom = L * 4  # 4 backbone atoms per residue

    # Sequence (random AA indices)
    seq = torch.randint(0, 20, (L,))

    # Chain IDs (first 10 residues chain A, last 10 chain B)
    chain_id_res = torch.cat([torch.zeros(10), torch.ones(10)]).long()

    # Residue indices within chain
    res_idx = torch.cat([torch.arange(10), torch.arange(10)]).long()

    # Atom coordinates (random, centered)
    atom_coords = torch.randn(N_atom, 3) * 10

    # Atom mask (all valid)
    atom_mask = torch.ones(N_atom, dtype=torch.bool)

    # Atom to residue mapping (4 atoms per residue)
    atom_to_res = torch.arange(L).repeat_interleave(4)

    # Atom types (N=0, CA=1, C=2, O=3 repeating)
    atom_type = torch.tensor([0, 1, 2, 3] * L)

    # Bonds: N-CA, CA-C, C-O for each residue, plus peptide bonds
    bonds_src = []
    bonds_dst = []
    bond_types = []

    for i in range(L):
        base = i * 4
        # N-CA (type 0)
        bonds_src.extend([base, base + 1])
        bonds_dst.extend([base + 1, base])
        bond_types.extend([0, 0])
        # CA-C (type 1)
        bonds_src.extend([base + 1, base + 2])
        bonds_dst.extend([base + 2, base + 1])
        bond_types.extend([1, 1])
        # C-O (type 2)
        bonds_src.extend([base + 2, base + 3])
        bonds_dst.extend([base + 3, base + 2])
        bond_types.extend([2, 2])

    # Peptide bonds (C-N between residues, within same chain)
    for i in range(L - 1):
        if chain_id_res[i] == chain_id_res[i + 1]:
            c_idx = i * 4 + 2  # C atom
            n_idx = (i + 1) * 4  # N atom of next residue
            bonds_src.extend([c_idx, n_idx])
            bonds_dst.extend([n_idx, c_idx])
            bond_types.extend([3, 3])

    bonds_src = torch.tensor(bonds_src)
    bonds_dst = torch.tensor(bonds_dst)
    bond_type = torch.tensor(bond_types)

    return {
        "seq": seq,
        "chain_id_res": chain_id_res,
        "res_idx": res_idx,
        "atom_coords": atom_coords,
        "atom_mask": atom_mask,
        "atom_to_res": atom_to_res,
        "atom_type": atom_type,
        "bonds_src": bonds_src,
        "bonds_dst": bonds_dst,
        "bond_type": bond_type,
    }


# ============================================================================
# TokenEmbedder Tests
# ============================================================================


class TestTokenEmbedder:
    """Tests for TokenEmbedder."""

    def test_output_shapes(self):
        """Test that embedder produces correct shapes."""
        c_s, c_z = 64, 32
        embedder = TokenEmbedder(c_s=c_s, c_z=c_z)

        L = 20
        seq = torch.randint(0, 20, (L,))
        chain_id_res = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        res_idx = torch.cat([torch.arange(10), torch.arange(10)]).long()

        s, z = embedder(seq, chain_id_res, res_idx)

        assert s.shape == (L, c_s), f"Expected ({L}, {c_s}), got {s.shape}"
        assert z.shape == (L, L, c_z), f"Expected ({L}, {L}, {c_z}), got {z.shape}"

    def test_no_nan(self):
        """Test that embedder doesn't produce NaN values."""
        embedder = TokenEmbedder()

        L = 20
        seq = torch.randint(0, 20, (L,))
        chain_id_res = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        res_idx = torch.cat([torch.arange(10), torch.arange(10)]).long()

        s, z = embedder(seq, chain_id_res, res_idx)

        assert not torch.isnan(s).any(), "Single representation contains NaN"
        assert not torch.isnan(z).any(), "Pair representation contains NaN"


# ============================================================================
# Triangle Multiplication Tests
# ============================================================================


class TestTriangleMul:
    """Tests for triangle multiplicative updates."""

    def test_output_shape(self):
        """Test output shape matches input."""
        c_z = 32
        L = 16

        tri_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=32)
        tri_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=32)

        z = torch.randn(L, L, c_z)
        pair_mask = torch.ones(L, L, dtype=torch.bool)

        dz_out = tri_out(z, pair_mask)
        dz_in = tri_in(z, pair_mask)

        assert dz_out.shape == z.shape
        assert dz_in.shape == z.shape

    def test_masking(self):
        """Test that masked positions are zero."""
        c_z = 32
        L = 16

        tri_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=32)

        z = torch.randn(L, L, c_z)
        pair_mask = torch.ones(L, L, dtype=torch.bool)
        # Mask out last 4 positions
        pair_mask[-4:, :] = False
        pair_mask[:, -4:] = False

        dz = tri_out(z, pair_mask)

        # Check masked positions are zero
        assert (dz[-4:, :, :] == 0).all(), "Masked rows should be zero"
        assert (dz[:, -4:, :] == 0).all(), "Masked columns should be zero"


# ============================================================================
# Pairformer Tests
# ============================================================================


class TestPairformer:
    """Tests for PairformerStack."""

    def test_output_shapes(self, small_config):
        """Test that Pairformer preserves shapes."""
        c = small_config
        pairformer = PairformerStack(
            n_blocks=c.n_blocks,
            c_s=c.c_s,
            c_z=c.c_z,
            n_heads_single=c.n_heads_single,
            n_heads_tri=c.n_heads_tri,
            c_tri_attn=c.c_tri_attn,
            c_tri_mul=c.c_tri_mul,
            use_checkpoint=False,
        )

        L = 20
        s = torch.randn(L, c.c_s)
        z = torch.randn(L, L, c.c_z)
        res_mask = torch.ones(L, dtype=torch.bool)

        s_out, z_out = pairformer(s, z, res_mask)

        assert s_out.shape == s.shape
        assert z_out.shape == z.shape

    def test_no_nan_or_inf(self, small_config):
        """Test that Pairformer doesn't produce NaN or Inf."""
        c = small_config
        pairformer = PairformerStack(
            n_blocks=2,
            c_s=c.c_s,
            c_z=c.c_z,
            n_heads_single=c.n_heads_single,
            n_heads_tri=c.n_heads_tri,
            use_checkpoint=False,
        )

        L = 16
        s = torch.randn(L, c.c_s)
        z = torch.randn(L, L, c.c_z)
        res_mask = torch.ones(L, dtype=torch.bool)

        s_out, z_out = pairformer(s, z, res_mask)

        assert not torch.isnan(s_out).any(), "Single contains NaN"
        assert not torch.isnan(z_out).any(), "Pair contains NaN"
        assert not torch.isinf(s_out).any(), "Single contains Inf"
        assert not torch.isinf(z_out).any(), "Pair contains Inf"


# ============================================================================
# EGNN Tests
# ============================================================================


class TestEGNN:
    """Tests for EGNN denoiser."""

    def test_output_shape(self):
        """Test EGNN output shape."""
        h_dim, edge_dim = 32, 16
        N = 40

        layer = EGNNLayer(h_dim=h_dim, edge_dim=edge_dim)
        denoiser = EGNNDenoiser(n_layers=2, h_dim=h_dim, edge_dim=edge_dim)

        x = torch.randn(N, 3)
        h = torch.randn(N, h_dim)
        edge_index = build_knn_edges(x, k=4)
        edge_attr = torch.randn(edge_index.size(1), edge_dim)

        # Test single layer
        x_new, h_new = layer(x, h, edge_index, edge_attr)
        assert x_new.shape == x.shape
        assert h_new.shape == h.shape

        # Test full denoiser
        eps_hat = denoiser(x, h, edge_index, edge_attr)
        assert eps_hat.shape == (N, 3)

    def test_translation_equivariance(self):
        """Test that EGNN is translation equivariant.

        If we translate input coordinates by t, output should also translate by t.
        """
        h_dim, edge_dim = 32, 16
        N = 20

        layer = EGNNLayer(h_dim=h_dim, edge_dim=edge_dim)
        layer.eval()

        x = torch.randn(N, 3)
        h = torch.randn(N, h_dim)
        edge_index = build_knn_edges(x, k=4)
        edge_attr = torch.randn(edge_index.size(1), edge_dim)

        # Forward pass
        x_out1, _ = layer(x, h, edge_index, edge_attr)

        # Translate input
        t = torch.randn(3) * 10
        x_translated = x + t

        # Note: KNN edges stay the same under translation (distance invariant)
        x_out2, _ = layer(x_translated, h, edge_index, edge_attr)

        # Output should be translated by t
        diff = (x_out2 - t) - x_out1
        assert diff.abs().max() < 1e-5, f"Translation equivariance violated: max diff {diff.abs().max()}"

    def test_rotation_equivariance(self):
        """Test that EGNN is approximately rotation equivariant.

        If we rotate input coordinates by R, output eps_hat should also rotate by R.
        Note: Due to numerical precision, this is approximate.
        """
        h_dim, edge_dim = 32, 16
        N = 20

        denoiser = EGNNDenoiser(n_layers=2, h_dim=h_dim, edge_dim=edge_dim)
        denoiser.eval()

        x = torch.randn(N, 3)
        h = torch.randn(N, h_dim)

        # Create a random rotation matrix
        def random_rotation():
            q = torch.randn(4)
            q = q / q.norm()
            w, x, y, z = q
            R = torch.tensor([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
            ])
            return R

        R = random_rotation()

        # Build edges from original coords
        edge_index = build_knn_edges(x, k=4)
        edge_attr = torch.randn(edge_index.size(1), edge_dim)

        # Forward pass with original coords
        eps1 = denoiser(x, h, edge_index, edge_attr)

        # Rotate coordinates
        x_rotated = (R @ x.T).T

        # Forward pass with rotated coords (same edges)
        eps2 = denoiser(x_rotated, h, edge_index, edge_attr)

        # Rotate eps1 and compare to eps2
        eps1_rotated = (R @ eps1.T).T

        diff = (eps2 - eps1_rotated).abs().max()
        # Allow some tolerance due to numerical precision
        assert diff < 0.1, f"Rotation equivariance violated: max diff {diff}"


# ============================================================================
# KNN Edge Tests
# ============================================================================


class TestKNNEdges:
    """Tests for KNN edge building."""

    def test_no_self_loops(self):
        """Test that KNN edges don't include self-loops."""
        x = torch.randn(20, 3)
        edge_index = build_knn_edges(x, k=4)

        src, dst = edge_index
        assert (src != dst).all(), "Found self-loops in KNN edges"

    def test_correct_count(self):
        """Test that we get k edges per node."""
        N, k = 20, 4
        x = torch.randn(N, 3)
        edge_index = build_knn_edges(x, k=k)

        assert edge_index.size(1) == N * k

    def test_merge_edges(self):
        """Test edge merging."""
        N = 20
        x = torch.randn(N, 3)

        # Some bond edges
        bonds_src = torch.tensor([0, 1, 2, 3])
        bonds_dst = torch.tensor([1, 0, 3, 2])

        knn_edges = build_knn_edges(x, k=4)

        edge_index, edge_type = merge_edges(bonds_src, bonds_dst, knn_edges)

        n_total = bonds_src.size(0) + knn_edges.size(1)
        assert edge_index.size(1) == n_total
        assert edge_type.size(0) == n_total
        assert (edge_type[:4] == 0).all()  # Bond edges
        assert (edge_type[4:] == 1).all()  # KNN edges


# ============================================================================
# Diffusion Schedule Tests
# ============================================================================


class TestDiffusionSchedule:
    """Tests for diffusion schedule."""

    def test_alpha_bar_monotonic(self):
        """Test that alpha_bar is monotonically decreasing."""
        schedule = DiffusionSchedule(T=16)

        # alpha_bar should decrease as t increases
        for t in range(1, schedule.T):
            assert schedule.alpha_bar[t] < schedule.alpha_bar[t - 1], \
                f"alpha_bar not decreasing at t={t}"

    def test_alpha_bar_bounds(self):
        """Test that alpha_bar is in (0, 1)."""
        schedule = DiffusionSchedule(T=16)

        assert (schedule.alpha_bar > 0).all(), "alpha_bar <= 0"
        assert (schedule.alpha_bar <= 1).all(), "alpha_bar > 1"

    def test_q_sample_variance(self):
        """Test that q_sample produces expected variance."""
        schedule = DiffusionSchedule(T=16)

        x0 = torch.randn(1000, 3)

        for t in [0, 7, 15]:  # Test at different timesteps
            x_t = schedule.q_sample(x0, t)

            # Expected: Var(x_t) = alpha_bar * Var(x0) + (1-alpha_bar) * Var(eps)
            # With Var(x0) ≈ 1 and Var(eps) = 1, Var(x_t) ≈ 1
            var = x_t.var()
            assert 0.5 < var < 2.0, f"Unexpected variance at t={t}: {var}"

    def test_predict_x0_roundtrip(self):
        """Test that predict_x0 inverts q_sample given true noise."""
        schedule = DiffusionSchedule(T=16)

        x0 = torch.randn(100, 3)
        eps = torch.randn_like(x0)
        t = 8

        x_t = schedule.q_sample(x0, t, eps)
        x0_pred = schedule.predict_x0(x_t, t, eps)

        diff = (x0 - x0_pred).abs().max()
        assert diff < 1e-5, f"predict_x0 roundtrip failed: max diff {diff}"


# ============================================================================
# Full Model Tests
# ============================================================================


class TestPPIModel:
    """Integration tests for full PPIModel."""

    def test_forward_shapes(self, small_config, sample_batch):
        """Test that forward pass produces correct shapes."""
        model = PPIModel(small_config)
        model.eval()

        output = model(**sample_batch)

        N_atom = sample_batch["atom_coords"].size(0)

        assert output["eps_hat"].shape == (N_atom, 3)
        assert output["eps"].shape == (N_atom, 3)
        assert output["x0_hat"].shape == (N_atom, 3)
        assert output["x_t"].shape == (N_atom, 3)
        assert isinstance(output["t"], int)

    def test_sample_shapes(self, small_config, sample_batch):
        """Test that sampling produces correct shapes."""
        model = PPIModel(small_config)
        model.eval()

        N_atom = sample_batch["atom_coords"].size(0)

        x0 = model.sample(
            seq=sample_batch["seq"],
            chain_id_res=sample_batch["chain_id_res"],
            res_idx=sample_batch["res_idx"],
            atom_to_res=sample_batch["atom_to_res"],
            atom_type=sample_batch["atom_type"],
            bonds_src=sample_batch["bonds_src"],
            bonds_dst=sample_batch["bonds_dst"],
            bond_type=sample_batch["bond_type"],
            n_atom=N_atom,
        )

        assert x0.shape == (N_atom, 3)

    def test_encode_shapes(self, small_config, sample_batch):
        """Test that encode produces correct shapes."""
        model = PPIModel(small_config)
        model.eval()

        L = sample_batch["seq"].size(0)

        s, z = model.encode(
            sample_batch["seq"],
            sample_batch["chain_id_res"],
            sample_batch["res_idx"],
        )

        assert s.shape == (L, small_config.c_s)
        assert z.shape == (L, L, small_config.c_z)

    def test_no_nan_in_forward(self, small_config, sample_batch):
        """Test that forward pass doesn't produce NaN."""
        model = PPIModel(small_config)
        model.eval()

        output = model(**sample_batch)

        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isnan(tensor).any(), f"NaN in {key}"
                assert not torch.isinf(tensor).any(), f"Inf in {key}"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
