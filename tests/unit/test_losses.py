"""Unit tests for loss functions in tinyfold.model.losses.

Tests the consolidated loss functions including:
- Kabsch alignment
- MSE and RMSE computation
- Geometry losses
- Contact losses
- lDDT metrics
"""

import pytest
import torch

from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_distance_consistency_loss,
    GeometryLoss,
    ContactLoss,
    compute_lddt,
    compute_ilddt,
    compute_interface_mask,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def random_coords():
    """Random coordinates for testing."""
    torch.manual_seed(42)
    return torch.randn(2, 20, 3)  # [B=2, N=20, 3]


@pytest.fixture
def backbone_coords():
    """Backbone atom coordinates [B, L, 4, 3] for geometry tests."""
    torch.manual_seed(42)
    B, L = 2, 10
    # Create roughly realistic backbone geometry
    coords = torch.zeros(B, L, 4, 3)
    for i in range(L):
        # Place residues along x-axis
        base_x = i * 3.8 / 10  # Normalized CA-CA distance
        coords[:, i, 0, :] = torch.tensor([base_x - 0.04, 0.0, 0.0])  # N
        coords[:, i, 1, :] = torch.tensor([base_x, 0.0, 0.0])  # CA
        coords[:, i, 2, :] = torch.tensor([base_x + 0.04, 0.0, 0.0])  # C
        coords[:, i, 3, :] = torch.tensor([base_x + 0.06, 0.03, 0.0])  # O
    return coords


@pytest.fixture
def chain_ids():
    """Chain IDs for testing."""
    # First 5 residues chain A, next 5 chain B
    return torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])


# ============================================================================
# Test Kabsch Alignment
# ============================================================================


class TestKabschAlign:
    """Tests for Kabsch alignment."""

    def test_identity_alignment(self, random_coords):
        """Aligning identical coords should give zero RMSE."""
        aligned, target_c = kabsch_align(random_coords, random_coords.clone())
        rmse = torch.sqrt(((aligned - target_c) ** 2).sum(-1).mean())
        assert rmse < 1e-5

    def test_rotation_invariance(self, random_coords):
        """Alignment should recover rotated coordinates."""
        # Create a rotation matrix
        theta = torch.tensor(0.5)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])

        rotated = torch.einsum('...ij,jk->...ik', random_coords, R)
        aligned, target_c = kabsch_align(rotated, random_coords)
        rmse = torch.sqrt(((aligned - target_c) ** 2).sum(-1).mean())
        assert rmse < 1e-4

    def test_with_mask(self, random_coords):
        """Alignment with mask should only use valid positions."""
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 10:] = False  # Mask out second half

        aligned, target_c = kabsch_align(random_coords, random_coords.clone(), mask)
        # Alignment should still work
        assert aligned.shape == random_coords.shape


# ============================================================================
# Test MSE/RMSE Losses
# ============================================================================


class TestMSELoss:
    """Tests for MSE loss computation."""

    def test_zero_loss_identical(self, random_coords):
        """Identical coords should give zero MSE."""
        loss = compute_mse_loss(random_coords, random_coords.clone())
        assert loss < 1e-6

    def test_positive_loss_different(self, random_coords):
        """Different coords should give positive MSE."""
        perturbed = random_coords + 0.1
        loss = compute_mse_loss(random_coords, perturbed)
        assert loss > 0

    def test_with_mask(self, random_coords):
        """MSE with mask should only count valid positions."""
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 10:] = False

        loss = compute_mse_loss(random_coords, random_coords.clone(), mask)
        assert loss < 1e-6


class TestRMSE:
    """Tests for RMSE computation."""

    def test_rmse_with_random_noise(self, random_coords):
        """RMSE should be positive for random perturbation."""
        # Use non-uniform perturbation so Kabsch can't fully align
        perturbed = random_coords + torch.randn_like(random_coords) * 0.1
        rmse = compute_rmse(random_coords, perturbed)
        assert rmse > 0
        # RMSE should be roughly 0.1 * sqrt(3) for Gaussian noise
        assert 0.05 < rmse < 0.5


# ============================================================================
# Test Distance Consistency Loss
# ============================================================================


class TestDistanceConsistencyLoss:
    """Tests for distance consistency loss."""

    def test_zero_loss_identical(self, random_coords):
        """Identical coords should give zero distance loss."""
        loss = compute_distance_consistency_loss(random_coords, random_coords.clone())
        assert loss < 1e-6

    def test_positive_loss_different(self, random_coords):
        """Different coords should give positive distance loss."""
        perturbed = random_coords * 1.1  # Scale changes distances
        loss = compute_distance_consistency_loss(random_coords, perturbed)
        assert loss > 0


# ============================================================================
# Test Geometry Loss
# ============================================================================


class TestGeometryLoss:
    """Tests for geometry auxiliary losses."""

    def test_geometry_loss_runs(self, backbone_coords):
        """Geometry loss should compute without errors."""
        geom_loss = GeometryLoss(
            bond_length_weight=1.0,
            bond_angle_weight=1.0,
            omega_weight=1.0,
            o_chirality_weight=1.0,
        )

        mask = torch.ones(2, 10, dtype=torch.bool)
        losses = geom_loss(backbone_coords, mask)

        assert 'total' in losses
        assert 'bond_length' in losses
        assert losses['total'] >= 0

    def test_geometry_loss_bounded(self, backbone_coords):
        """With bound_losses=True, all losses should be < 1."""
        geom_loss = GeometryLoss(bound_losses=True)
        mask = torch.ones(2, 10, dtype=torch.bool)
        losses = geom_loss(backbone_coords, mask)

        for name, val in losses.items():
            if name != 'total':
                assert val <= 1.0, f"{name} loss not bounded: {val}"


# ============================================================================
# Test Contact Loss
# ============================================================================


class TestContactLoss:
    """Tests for contact-based losses."""

    def test_contact_loss_runs(self, backbone_coords, chain_ids):
        """Contact loss should compute without errors."""
        contact_loss = ContactLoss(
            threshold=1.0,
            min_seq_sep=2,
            stage="stage1",
        )

        # Get centroids from backbone
        centroids = backbone_coords.mean(dim=2)  # [B, L, 3]
        mask = torch.ones(2, 10, dtype=torch.bool)

        losses = contact_loss(
            pred_centroids=centroids,
            gt_centroids=centroids,
            chain_ids=chain_ids,
            mask=mask,
        )

        assert 'total' in losses
        assert losses['stage1'] >= 0


# ============================================================================
# Test lDDT Metrics
# ============================================================================


class TestLDDT:
    """Tests for lDDT metrics."""

    def test_lddt_perfect_prediction(self, random_coords):
        """Perfect prediction should give lDDT close to 1."""
        lddt = compute_lddt(random_coords, random_coords.clone(), coord_scale=1.0)
        assert lddt > 0.99

    def test_lddt_range(self, random_coords):
        """lDDT should be in [0, 1]."""
        perturbed = random_coords + torch.randn_like(random_coords) * 0.5
        lddt = compute_lddt(random_coords, perturbed, coord_scale=1.0)
        assert 0 <= lddt <= 1

    def test_interface_mask(self, backbone_coords, chain_ids):
        """Interface mask should identify cross-chain contacts."""
        # Bring chains closer together for interface
        coords = backbone_coords.clone()
        coords[:, 5:, :, 1] -= 0.5  # Move chain B closer in y

        ca_coords = coords[:, :, 1, :]  # CA atoms
        interface_mask = compute_interface_mask(
            ca_coords, chain_ids,
            interface_threshold=8.0,
            coord_scale=10.0,
        )

        # Should have some interface residues
        assert interface_mask.any()


# ============================================================================
# Test Import Compatibility
# ============================================================================


class TestImports:
    """Test that all expected functions are importable."""

    def test_all_mse_imports(self):
        """All MSE functions should be importable."""
        from tinyfold.model.losses import (
            kabsch_align,
            compute_mse_loss,
            compute_rmse,
            compute_distance_consistency_loss,
        )

    def test_all_geometry_imports(self):
        """All geometry functions should be importable."""
        from tinyfold.model.losses import (
            GeometryLoss,
            bond_length_loss,
            bond_angle_loss,
            omega_loss,
            o_chirality_loss,
            dihedral_angle,
            BOND_LENGTHS,
            BOND_ANGLES,
        )

    def test_all_contact_imports(self):
        """All contact functions should be importable."""
        from tinyfold.model.losses import (
            ContactLoss,
            compute_contact_mask,
            contact_loss_centroids,
            contact_loss_atoms,
        )

    def test_all_lddt_imports(self):
        """All lDDT functions should be importable."""
        from tinyfold.model.losses import (
            compute_lddt,
            compute_ilddt,
            compute_lddt_metrics,
            compute_interface_mask,
        )
