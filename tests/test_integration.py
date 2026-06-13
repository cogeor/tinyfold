"""Integration tests for end-to-end pipeline and edge cases.

These tests verify that components work together correctly and handle
edge cases gracefully. They focus on meaningful behavior verification,
not just calling functions.
"""

import torch
import pytest
import numpy as np

from tinyfold.model.config import ModelConfig
from tinyfold.model.denoiser.edges import build_knn_edges, merge_edges, build_edge_attr
from tinyfold.model.pairformer.attn_pair_bias import AttentionPairBias
from tinyfold.data.collate import collate_ppi
from tinyfold.data.processing.atomization import atomize_chains, build_bonds
from tinyfold.constants import NUM_BOND_TYPES


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_config():
    """Small model config for fast testing."""
    return ModelConfig(
        c_s=64,
        c_z=32,
        c_a=32,
        n_blocks=2,
        n_egnn_layers=2,
        k_neighbors=4,
        diffusion_steps=4,  # Very few diffusion steps
    )


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def synthetic_sample():
    """Create a synthetic PPI sample mimicking real data."""
    LA, LB = 10, 8
    L = LA + LB
    N_atom = L * 4

    # Create realistic backbone coordinates
    # Each residue has N, CA, C, O atoms roughly 1.5A apart
    coords = np.zeros((L, 4, 3), dtype=np.float32)
    for i in range(L):
        # Backbone follows a rough helix
        theta = i * 100 * np.pi / 180
        x_base = i * 3.8 * np.cos(theta * 0.1)  # ~3.8A rise per residue
        y_base = i * 3.8 * np.sin(theta * 0.1)
        z_base = 0 if i < LA else 15  # Separate chains by 15A initially

        # N, CA, C, O positions within residue
        coords[i, 0] = [x_base, y_base, z_base]  # N
        coords[i, 1] = [x_base + 1.46, y_base + 0.3, z_base]  # CA
        coords[i, 2] = [x_base + 2.98, y_base + 0.5, z_base]  # C
        coords[i, 3] = [x_base + 3.5, y_base + 1.5, z_base]  # O

    mask = np.ones((L, 4), dtype=bool)

    coords_a = coords[:LA]
    coords_b = coords[LA:]
    mask_a = mask[:LA]
    mask_b = mask[LA:]

    atom_coords, atom_mask, atom_to_res, atom_type, chain_id_atom = atomize_chains(
        coords_a, mask_a, coords_b, mask_b
    )
    bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

    seq = np.zeros(L, dtype=np.int64)  # All alanine
    chain_id_res = np.concatenate([np.zeros(LA), np.ones(LB)]).astype(np.int64)
    res_idx = np.concatenate([np.arange(LA), np.arange(LB)]).astype(np.int64)
    iface_mask = np.zeros(L, dtype=bool)
    iface_mask[LA-3:LA] = True  # Last 3 of chain A
    iface_mask[LA:LA+3] = True  # First 3 of chain B

    return {
        "sample_id": "test_sample",
        "pdb_id": "TEST",
        "seq": torch.from_numpy(seq),
        "chain_id_res": torch.from_numpy(chain_id_res),
        "res_idx": torch.from_numpy(res_idx),
        "atom_coords": torch.from_numpy(atom_coords),
        "atom_mask": torch.from_numpy(atom_mask),
        "atom_to_res": torch.from_numpy(atom_to_res),
        "atom_type": torch.from_numpy(atom_type),
        "bonds_src": torch.from_numpy(bonds_src),
        "bonds_dst": torch.from_numpy(bonds_dst),
        "bond_type": torch.from_numpy(bond_type),
        "iface_mask": torch.from_numpy(iface_mask),
        "LA": LA,
        "LB": LB,
    }


# ============================================================================
# Collate Function Tests - Verify batching logic
# ============================================================================


class TestCollateFunction:
    """Tests for batch collation that verify correct padding and merging."""

    def test_padding_preserves_data(self, synthetic_sample):
        """Padded batch should preserve original sample data."""
        batch = [synthetic_sample]
        collated = collate_ppi(batch)

        # Extract and compare
        original_seq = synthetic_sample["seq"]
        collated_seq = collated["seq"][0, :len(original_seq)]

        assert torch.equal(original_seq, collated_seq), \
            "Collation should preserve sequence data"

        original_coords = synthetic_sample["atom_coords"]
        collated_coords = collated["atom_coords"][0, :len(original_coords)]

        assert torch.allclose(original_coords, collated_coords), \
            "Collation should preserve coordinates"

    def test_variable_length_batching(self, synthetic_sample):
        """Batch with different-sized samples should pad correctly."""
        # Create second sample with different size
        sample2 = synthetic_sample.copy()
        # Truncate to smaller size
        L2 = 12
        sample2["seq"] = sample2["seq"][:L2]
        sample2["chain_id_res"] = sample2["chain_id_res"][:L2]
        sample2["res_idx"] = sample2["res_idx"][:L2]
        sample2["iface_mask"] = sample2["iface_mask"][:L2]
        sample2["atom_coords"] = sample2["atom_coords"][:L2*4]
        sample2["atom_mask"] = sample2["atom_mask"][:L2*4]
        sample2["atom_to_res"] = sample2["atom_to_res"][:L2*4]
        sample2["atom_type"] = sample2["atom_type"][:L2*4]
        sample2["LA"] = 7
        sample2["LB"] = 5
        # Rebuild bonds for smaller sample
        sample2["bonds_src"] = torch.tensor([0, 1, 4, 5], dtype=torch.long)
        sample2["bonds_dst"] = torch.tensor([1, 0, 5, 4], dtype=torch.long)
        sample2["bond_type"] = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        batch = [synthetic_sample, sample2]
        collated = collate_ppi(batch)

        L1 = len(synthetic_sample["seq"])

        # Check shapes
        assert collated["seq"].shape[0] == 2, "Batch size should be 2"
        assert collated["seq"].shape[1] == L1, "Should pad to max length"

        # Check mask correctly identifies padding
        assert collated["res_mask"][0].sum() == L1
        assert collated["res_mask"][1].sum() == L2
        assert collated["res_mask"][1, L2:].sum() == 0, "Padding should be masked"

    def test_edge_offset_correctness(self, synthetic_sample):
        """Edge indices should be correctly offset when batching."""
        # Create two identical samples
        sample1 = synthetic_sample
        sample2 = synthetic_sample.copy()
        sample2["sample_id"] = "test_sample_2"

        batch = [sample1, sample2]
        collated = collate_ppi(batch)

        edge_index = collated["edge_index"]
        n_atoms_1 = len(sample1["atom_coords"])

        # Edges from sample 1 should be in [0, n_atoms_1)
        # Edges from sample 2 should be in [n_atoms_1, 2*n_atoms_1)
        n_edges_1 = len(sample1["bonds_src"])

        edges_sample1 = edge_index[:, :n_edges_1]
        edges_sample2 = edge_index[:, n_edges_1:2*n_edges_1]

        assert edges_sample1.max() < n_atoms_1, \
            "Sample 1 edges should reference atoms in [0, n_atoms_1)"
        assert edges_sample2.min() >= n_atoms_1, \
            "Sample 2 edges should be offset by n_atoms_1"

    def test_atom_batch_tracking(self, synthetic_sample):
        """atom_batch should correctly identify which sample each atom belongs to."""
        sample2 = synthetic_sample.copy()
        sample2["sample_id"] = "test_sample_2"

        batch = [synthetic_sample, sample2]
        collated = collate_ppi(batch)

        atom_batch = collated["atom_batch"]
        n_atoms = len(synthetic_sample["atom_coords"])

        assert (atom_batch[:n_atoms] == 0).all(), \
            "First sample atoms should have batch index 0"
        assert (atom_batch[n_atoms:] == 1).all(), \
            "Second sample atoms should have batch index 1"

    def test_single_sample_batch(self, synthetic_sample):
        """Single-sample batch should work correctly."""
        collated = collate_ppi([synthetic_sample])

        # Batch dimension should be 1
        assert collated["seq"].shape[0] == 1
        assert collated["atom_coords"].shape[0] == 1

        # No padding needed for single sample
        L = len(synthetic_sample["seq"])
        assert collated["res_mask"][0].sum() == L


# ============================================================================
# Edge Case Tests - Verify robustness
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_knn_with_few_nodes(self):
        """KNN should handle cases where k > number of nodes."""
        x = torch.randn(3, 3)  # Only 3 nodes
        edge_index = build_knn_edges(x, k=10)  # Request 10 neighbors

        # Should get at most 2 neighbors per node (3-1=2, excluding self)
        assert edge_index.shape[1] <= 3 * 2

        # No self-loops
        src, dst = edge_index
        assert (src != dst).all()

    def test_knn_single_node(self):
        """KNN with single node should return empty edges."""
        x = torch.randn(1, 3)
        edge_index = build_knn_edges(x, k=4)

        assert edge_index.shape == (2, 0), "Single node should have no edges"

    def test_attention_with_partial_mask(self):
        """Attention should handle partially masked sequences."""
        L, c_s, c_z = 10, 64, 32
        attn = AttentionPairBias(c_s=c_s, c_z=c_z, n_heads=4)

        s = torch.randn(L, c_s)
        z = torch.randn(L, L, c_z)

        # Mask out last 3 positions
        mask = torch.ones(L, dtype=torch.bool)
        mask[-3:] = False

        output = attn(s, z, mask)

        # Output should not have NaN
        assert not torch.isnan(output).any(), "Attention output should not contain NaN"

        # Masked positions should have zero output
        assert (output[-3:] == 0).all(), "Masked positions should be zeroed"

    def test_attention_fully_masked(self):
        """Attention with all positions masked should not crash or produce NaN."""
        L, c_s, c_z = 5, 64, 32
        attn = AttentionPairBias(c_s=c_s, c_z=c_z, n_heads=4)

        s = torch.randn(L, c_s)
        z = torch.randn(L, L, c_z)
        mask = torch.zeros(L, dtype=torch.bool)  # All masked

        output = attn(s, z, mask)

        # Should not crash or have NaN (due to our fix)
        assert not torch.isnan(output).any(), \
            "Fully masked attention should not produce NaN"

    def test_bond_types_all_present(self, synthetic_sample):
        """All 4 bond types should be generated for a complete structure."""
        bond_types = synthetic_sample["bond_type"].numpy()
        unique_types = set(bond_types)

        # Should have types 0, 1, 2 (within-residue) and 3 (peptide)
        assert 0 in unique_types, "Should have N-CA bonds (type 0)"
        assert 1 in unique_types, "Should have CA-C bonds (type 1)"
        assert 2 in unique_types, "Should have C-O bonds (type 2)"
        assert 3 in unique_types, "Should have peptide bonds (type 3)"
        assert max(unique_types) < NUM_BOND_TYPES, "Bond types should be < NUM_BOND_TYPES"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
