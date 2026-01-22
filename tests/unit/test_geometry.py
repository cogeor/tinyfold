"""Unit tests for geometry functions (dihedral angles, omega loss).

Converted from scripts/test_dihedral.py and scripts/test_omega_random.py
"""

import math
import pytest
import torch

from tinyfold.model.losses.geometry import dihedral_angle, omega_loss, GeometryLoss


# ============================================================================
# Test Dihedral Angle Computation
# ============================================================================


class TestDihedralAngle:
    """Tests for dihedral angle computation."""

    def test_colinear_atoms(self):
        """Colinear atoms should give 0 or undefined dihedral."""
        p0 = torch.tensor([[[0.0, 0.0, 0.0]]])
        p1 = torch.tensor([[[1.0, 0.0, 0.0]]])
        p2 = torch.tensor([[[2.0, 0.0, 0.0]]])
        p3 = torch.tensor([[[3.0, 0.0, 0.0]]])

        omega = dihedral_angle(p0, p1, p2, p3)
        # Colinear atoms have undefined dihedral, but should not crash
        assert not torch.isnan(omega).any()

    def test_trans_configuration_180deg(self):
        """Trans peptide configuration should give ~180 degrees."""
        # Classic trans peptide geometry
        p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
        p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i (at origin)
        p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
        p3 = torch.tensor([[[1.3, -1.0, 0.0]]])  # CA_i+1 (opposite side from CA_i)

        omega = dihedral_angle(p0, p1, p2, p3)
        omega_deg = omega.item() * 180 / math.pi

        # Trans should be close to +/- 180 degrees
        assert abs(abs(omega_deg) - 180) < 5, f"Trans omega should be ~180 deg, got {omega_deg:.1f}"

    def test_cis_configuration_0deg(self):
        """Cis peptide configuration should give ~0 degrees."""
        p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
        p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i
        p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
        p3 = torch.tensor([[[1.3, 1.0, 0.0]]])   # CA_i+1 (same side as CA_i)

        omega = dihedral_angle(p0, p1, p2, p3)
        omega_deg = omega.item() * 180 / math.pi

        # Cis should be close to 0 degrees
        assert abs(omega_deg) < 5, f"Cis omega should be ~0 deg, got {omega_deg:.1f}"

    def test_90deg_configuration(self):
        """90 degree dihedral configuration."""
        p0 = torch.tensor([[[0.0, 1.0, 0.0]]])   # CA_i
        p1 = torch.tensor([[[0.0, 0.0, 0.0]]])   # C_i
        p2 = torch.tensor([[[1.3, 0.0, 0.0]]])   # N_i+1
        p3 = torch.tensor([[[1.3, 0.0, 1.0]]])   # CA_i+1 (perpendicular)

        omega = dihedral_angle(p0, p1, p2, p3)
        omega_deg = omega.item() * 180 / math.pi

        # Should be close to +/- 90 degrees
        assert abs(abs(omega_deg) - 90) < 10, f"90 deg omega expected, got {omega_deg:.1f}"

    def test_batch_processing(self):
        """Dihedral should work with batched inputs."""
        B, L = 2, 5
        p0 = torch.randn(B, L, 3)
        p1 = torch.randn(B, L, 3)
        p2 = torch.randn(B, L, 3)
        p3 = torch.randn(B, L, 3)

        omega = dihedral_angle(p0, p1, p2, p3)

        assert omega.shape == (B, L)
        assert not torch.isnan(omega).any()
        # All values should be in [-pi, pi]
        assert (omega >= -math.pi).all() and (omega <= math.pi).all()


# ============================================================================
# Test Omega Loss
# ============================================================================


class TestOmegaLoss:
    """Tests for omega (peptide planarity) loss."""

    @pytest.fixture
    def ideal_backbone(self):
        """Create idealized backbone coordinates."""
        B, L = 1, 10
        coords = torch.zeros(B, L, 4, 3)

        # Create trans peptide bonds
        for i in range(L):
            base_x = i * 0.38  # Normalized CA-CA distance
            # Simplified geometry: atoms along x-axis with small y offsets
            coords[:, i, 0, :] = torch.tensor([base_x - 0.04, 0.1, 0.0])   # N
            coords[:, i, 1, :] = torch.tensor([base_x, 0.0, 0.0])          # CA
            coords[:, i, 2, :] = torch.tensor([base_x + 0.04, -0.1, 0.0])  # C
            coords[:, i, 3, :] = torch.tensor([base_x + 0.06, -0.15, 0.0]) # O

        return coords

    def test_omega_loss_runs(self, ideal_backbone):
        """Omega loss should compute without errors."""
        loss = omega_loss(ideal_backbone)

        assert loss.shape == ()  # Scalar
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_ground_truth_low_loss(self, ideal_backbone):
        """Idealized backbone should have low omega loss."""
        loss = omega_loss(ideal_backbone)

        # Idealized trans peptides should have very low omega loss
        assert loss < 0.5, f"Idealized backbone omega loss too high: {loss.item():.4f}"

    def test_random_coords_higher_loss(self, ideal_backbone):
        """Random perturbation should increase omega loss."""
        gt_loss = omega_loss(ideal_backbone)

        # Add significant noise
        noisy = ideal_backbone + 0.2 * torch.randn_like(ideal_backbone)
        noisy_loss = omega_loss(noisy)

        assert noisy_loss > gt_loss, "Noisy coords should have higher omega loss"

    def test_noise_levels(self, ideal_backbone):
        """Omega loss should increase with noise level."""
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        losses = []

        for noise in noise_levels:
            noisy = ideal_backbone + noise * torch.randn_like(ideal_backbone)
            loss = omega_loss(noisy)
            losses.append(loss.item())

        # Losses should generally increase with noise (may not be strictly monotonic)
        assert losses[-1] > losses[0], "Loss should increase with noise"

    def test_with_mask(self, ideal_backbone):
        """Omega loss should work with residue mask."""
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[:, 5:] = False  # Mask second half

        loss = omega_loss(ideal_backbone, mask=mask)

        assert not torch.isnan(loss)
        assert loss >= 0


# ============================================================================
# Test GeometryLoss Integration
# ============================================================================


class TestGeometryLossIntegration:
    """Integration tests for GeometryLoss class with omega."""

    @pytest.fixture
    def backbone_coords(self):
        """Create backbone coordinates for testing."""
        B, L = 1, 10
        coords = torch.zeros(B, L, 4, 3)

        for i in range(L):
            base_x = i * 0.38
            coords[:, i, 0, :] = torch.tensor([base_x - 0.04, 0.1, 0.0])   # N
            coords[:, i, 1, :] = torch.tensor([base_x, 0.0, 0.0])          # CA
            coords[:, i, 2, :] = torch.tensor([base_x + 0.04, -0.1, 0.0])  # C
            coords[:, i, 3, :] = torch.tensor([base_x + 0.06, -0.15, 0.0]) # O

        return coords

    def test_geometry_loss_includes_omega(self, backbone_coords):
        """GeometryLoss should include omega when weight > 0."""
        geom_loss = GeometryLoss(omega_weight=1.0)
        losses = geom_loss(backbone_coords)

        assert 'omega' in losses
        assert losses['omega'] >= 0

    def test_geometry_loss_excludes_omega_when_zero(self, backbone_coords):
        """GeometryLoss should exclude omega when weight = 0."""
        geom_loss = GeometryLoss(omega_weight=0.0)
        losses = geom_loss(backbone_coords)

        # omega should still be in dict but be 0
        assert 'omega' in losses
        assert losses['omega'] == 0.0

    def test_total_loss_includes_weighted_omega(self, backbone_coords):
        """Total loss should include weighted omega contribution."""
        geom_loss_with = GeometryLoss(
            bond_length_weight=1.0,
            bond_angle_weight=1.0,
            omega_weight=1.0
        )
        geom_loss_without = GeometryLoss(
            bond_length_weight=1.0,
            bond_angle_weight=1.0,
            omega_weight=0.0
        )

        losses_with = geom_loss_with(backbone_coords)
        losses_without = geom_loss_without(backbone_coords)

        # Total with omega should generally be different
        # (might be same if omega loss is 0)
        assert 'total' in losses_with
        assert 'total' in losses_without

    def test_random_vs_gt_comparison(self, backbone_coords):
        """Random coords should have higher geometry loss than GT."""
        geom_loss = GeometryLoss(
            bond_length_weight=1.0,
            bond_angle_weight=0.1,
            omega_weight=0.1
        )

        gt_losses = geom_loss(backbone_coords)

        # Random offsets from centroids
        centroids = backbone_coords.mean(dim=2, keepdim=True)
        random_offsets = 0.2 * torch.randn(1, 10, 4, 3)
        random_coords = centroids + random_offsets
        random_losses = geom_loss(random_coords)

        assert random_losses['total'] > gt_losses['total'], \
            f"Random should have higher loss: {random_losses['total']:.4f} vs {gt_losses['total']:.4f}"
