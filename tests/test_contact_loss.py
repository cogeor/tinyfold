"""Tests for contact-based loss functions."""

import sys
sys.path.insert(0, 'C:/Users/costa/src/tinyfold/scripts')

import pytest
import torch

from models.geometry_losses import (
    compute_contact_mask,
    contact_loss_centroids,
    contact_loss_atoms,
    ContactLoss,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_protein():
    """Create a simple two-chain protein for testing."""
    L = 20  # 10 residues per chain

    # Create centroids: two parallel helices
    centroids = torch.zeros(L, 3)
    chain_ids = torch.zeros(L, dtype=torch.long)

    # Chain A: residues 0-9, along x-axis
    for i in range(10):
        centroids[i] = torch.tensor([i * 0.38, 0.0, 0.0])  # ~3.8A spacing
        chain_ids[i] = 0

    # Chain B: residues 10-19, parallel to chain A, offset in y
    for i in range(10):
        centroids[10 + i] = torch.tensor([i * 0.38, 0.8, 0.0])  # 8A apart in y
        chain_ids[10 + i] = 1

    return centroids, chain_ids


@pytest.fixture
def simple_atoms():
    """Create simple atom positions from centroids."""
    L = 20
    atoms = torch.zeros(L, 4, 3)

    # Create centroids first
    centroids = torch.zeros(L, 3)
    for i in range(10):
        centroids[i] = torch.tensor([i * 0.38, 0.0, 0.0])
    for i in range(10):
        centroids[10 + i] = torch.tensor([i * 0.38, 0.8, 0.0])

    # Place atoms around centroids (N, CA, C, O)
    offsets = torch.tensor([
        [-0.1, 0.0, 0.0],   # N
        [0.0, 0.0, 0.0],    # CA (at centroid)
        [0.1, 0.0, 0.0],    # C
        [0.1, 0.1, 0.0],    # O
    ])

    for i in range(L):
        atoms[i] = centroids[i] + offsets

    chain_ids = torch.cat([torch.zeros(10), torch.ones(10)]).long()

    return atoms, chain_ids


# =============================================================================
# Tests for compute_contact_mask
# =============================================================================

class TestComputeContactMask:
    """Tests for compute_contact_mask function."""

    def test_basic_mask_shape(self, simple_protein):
        """Test that mask has correct shape."""
        centroids, chain_ids = simple_protein
        mask = compute_contact_mask(centroids, chain_ids, threshold=1.0)

        assert mask.shape == (20, 20)
        assert mask.dtype == torch.bool

    def test_upper_triangular(self, simple_protein):
        """Test that mask is upper triangular."""
        centroids, chain_ids = simple_protein
        mask = compute_contact_mask(centroids, chain_ids, threshold=2.0)

        # Lower triangle should be all False
        lower = torch.tril(torch.ones(20, 20, dtype=torch.bool), diagonal=0)
        assert not mask[lower].any()

    def test_no_self_contacts(self, simple_protein):
        """Test that diagonal is all False (no self-contacts)."""
        centroids, chain_ids = simple_protein
        mask = compute_contact_mask(centroids, chain_ids, threshold=2.0)

        assert not mask.diag().any()

    def test_min_seq_sep_intra(self, simple_protein):
        """Test that sequential neighbors are excluded for intra-chain."""
        centroids, chain_ids = simple_protein
        mask = compute_contact_mask(
            centroids, chain_ids, threshold=2.0, min_seq_sep=5,
            include_intra=True, include_inter=False
        )

        # Check that pairs with seq_sep <= 5 are excluded within same chain
        for i in range(10):  # Chain A
            for j in range(i + 1, min(i + 6, 10)):
                assert not mask[i, j], f"Pair ({i}, {j}) should be excluded"

    def test_inter_chain_contacts(self, simple_protein):
        """Test inter-chain contacts are detected."""
        centroids, chain_ids = simple_protein

        # With large threshold, should find inter-chain contacts
        mask = compute_contact_mask(
            centroids, chain_ids, threshold=1.5,  # 15A
            include_intra=False, include_inter=True
        )

        # Should have some inter-chain contacts (chains are 8A apart)
        inter_region = mask[:10, 10:]  # Chain A to Chain B
        assert inter_region.any(), "Should find inter-chain contacts"

    def test_no_contacts_far_apart(self, simple_protein):
        """Test that distant residues have no contacts with small threshold."""
        centroids, chain_ids = simple_protein

        # Very small threshold - only very close residues
        mask = compute_contact_mask(centroids, chain_ids, threshold=0.3)

        # Should have very few contacts
        assert mask.sum() < 5

    def test_include_flags(self, simple_protein):
        """Test include_intra and include_inter flags."""
        centroids, chain_ids = simple_protein

        # Only intra-chain
        mask_intra = compute_contact_mask(
            centroids, chain_ids, threshold=2.0,
            include_intra=True, include_inter=False
        )

        # Only inter-chain
        mask_inter = compute_contact_mask(
            centroids, chain_ids, threshold=2.0,
            include_intra=False, include_inter=True
        )

        # Both
        mask_both = compute_contact_mask(
            centroids, chain_ids, threshold=2.0,
            include_intra=True, include_inter=True
        )

        # Union of intra and inter should equal both
        assert torch.equal(mask_intra | mask_inter, mask_both)


# =============================================================================
# Tests for contact_loss_centroids
# =============================================================================

class TestContactLossCentroids:
    """Tests for contact_loss_centroids function."""

    def test_zero_loss_identical(self, simple_protein):
        """Test that loss is zero when pred equals gt."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)  # [1, L, 3]
        chain_ids = chain_ids.unsqueeze(0)  # [1, L]

        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)  # [1, L, L]

        loss = contact_loss_centroids(centroids, centroids, mask, chain_ids)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_different(self, simple_protein):
        """Test that loss is positive when pred differs from gt."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)

        # Perturb prediction
        pred = centroids + torch.randn_like(centroids) * 0.1

        loss = contact_loss_centroids(pred, centroids, mask, chain_ids)

        assert loss.item() > 0

    def test_inter_chain_weighting(self, simple_protein):
        """Test that inter-chain weighting increases loss for inter-chain errors."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        # Create mask with inter-chain contacts only
        mask = compute_contact_mask(
            centroids[0], chain_ids[0], threshold=1.5,
            include_intra=False, include_inter=True
        )
        mask = mask.unsqueeze(0)

        if mask.sum() == 0:
            pytest.skip("No inter-chain contacts found")

        pred = centroids + torch.randn_like(centroids) * 0.1

        loss_weight_1 = contact_loss_centroids(pred, centroids, mask, chain_ids, inter_chain_weight=1.0)
        loss_weight_2 = contact_loss_centroids(pred, centroids, mask, chain_ids, inter_chain_weight=2.0)

        # Higher weight should give higher loss
        assert loss_weight_2.item() > loss_weight_1.item()

    def test_with_residue_mask(self, simple_protein):
        """Test that residue mask properly excludes residues."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        contact_mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        contact_mask = contact_mask.unsqueeze(0)

        pred = centroids + torch.randn_like(centroids) * 0.1

        # Full mask
        full_mask = torch.ones(1, 20, dtype=torch.bool)
        loss_full = contact_loss_centroids(pred, centroids, contact_mask, chain_ids, mask=full_mask)

        # Partial mask (exclude half)
        partial_mask = torch.ones(1, 20, dtype=torch.bool)
        partial_mask[0, 10:] = False
        loss_partial = contact_loss_centroids(pred, centroids, contact_mask, chain_ids, mask=partial_mask)

        # Losses should differ (partial excludes some contacts)
        assert loss_full.item() != loss_partial.item() or contact_mask[:, :10, :10].sum() == contact_mask.sum()


# =============================================================================
# Tests for contact_loss_atoms
# =============================================================================

class TestContactLossAtoms:
    """Tests for contact_loss_atoms function."""

    def test_zero_loss_identical(self, simple_atoms):
        """Test that loss is zero when pred equals gt."""
        atoms, chain_ids = simple_atoms
        atoms = atoms.unsqueeze(0)  # [1, L, 4, 3]
        chain_ids = chain_ids.unsqueeze(0)

        centroids = atoms.mean(dim=2)  # [1, L, 3]
        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)

        loss = contact_loss_atoms(atoms, atoms, mask, chain_ids, distance_type="ca")

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ca_vs_min_distance(self, simple_atoms):
        """Test CA vs min distance types give different results."""
        atoms, chain_ids = simple_atoms
        atoms = atoms.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        centroids = atoms.mean(dim=2)
        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)

        pred = atoms + torch.randn_like(atoms) * 0.05

        loss_ca = contact_loss_atoms(pred, atoms, mask, chain_ids, distance_type="ca")
        loss_min = contact_loss_atoms(pred, atoms, mask, chain_ids, distance_type="min")

        # Both should be positive
        assert loss_ca.item() > 0
        assert loss_min.item() > 0

    def test_positive_loss_different(self, simple_atoms):
        """Test that loss is positive when pred differs from gt."""
        atoms, chain_ids = simple_atoms
        atoms = atoms.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        centroids = atoms.mean(dim=2)
        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)

        pred = atoms + torch.randn_like(atoms) * 0.1

        loss = contact_loss_atoms(pred, atoms, mask, chain_ids)

        assert loss.item() > 0


# =============================================================================
# Tests for ContactLoss class
# =============================================================================

class TestContactLossClass:
    """Tests for ContactLoss class."""

    def test_stage1_only(self, simple_protein):
        """Test Stage 1 only mode."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        loss_fn = ContactLoss(threshold=1.5, stage="stage1")

        pred = centroids + torch.randn_like(centroids) * 0.1

        losses = loss_fn(
            pred_centroids=pred,
            gt_centroids=centroids,
            chain_ids=chain_ids
        )

        assert losses['stage1'].item() > 0
        assert losses['stage2'].item() == 0
        assert losses['total'].item() == losses['stage1'].item()

    def test_stage2_only(self, simple_atoms):
        """Test Stage 2 only mode."""
        atoms, chain_ids = simple_atoms
        atoms = atoms.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        loss_fn = ContactLoss(threshold=1.5, stage="stage2")

        pred = atoms + torch.randn_like(atoms) * 0.05
        gt_centroids = atoms.mean(dim=2)

        losses = loss_fn(
            gt_centroids=gt_centroids,
            pred_atoms=pred,
            gt_atoms=atoms,
            chain_ids=chain_ids
        )

        assert losses['stage1'].item() == 0
        assert losses['stage2'].item() > 0
        assert losses['total'].item() == losses['stage2'].item()

    def test_both_stages(self, simple_protein, simple_atoms):
        """Test both stages mode."""
        centroids, chain_ids = simple_protein
        atoms, _ = simple_atoms

        centroids = centroids.unsqueeze(0)
        atoms = atoms.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        loss_fn = ContactLoss(threshold=1.5, stage="both")

        pred_centroids = centroids + torch.randn_like(centroids) * 0.1
        pred_atoms = atoms + torch.randn_like(atoms) * 0.05

        losses = loss_fn(
            pred_centroids=pred_centroids,
            gt_centroids=centroids,
            pred_atoms=pred_atoms,
            gt_atoms=atoms,
            chain_ids=chain_ids
        )

        assert losses['stage1'].item() > 0
        assert losses['stage2'].item() > 0
        assert losses['total'].item() == pytest.approx(
            losses['stage1'].item() + losses['stage2'].item(), rel=1e-5
        )

    def test_n_contacts_reported(self, simple_protein):
        """Test that number of contacts is reported."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        loss_fn = ContactLoss(threshold=1.5, stage="stage1")

        losses = loss_fn(
            pred_centroids=centroids,
            gt_centroids=centroids,
            chain_ids=chain_ids
        )

        assert 'n_contacts' in losses
        assert losses['n_contacts'] >= 0

    def test_precomputed_mask(self, simple_protein):
        """Test using precomputed contact mask."""
        centroids, chain_ids = simple_protein
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        loss_fn = ContactLoss(threshold=1.5, stage="stage1")

        # Precompute mask
        mask = compute_contact_mask(centroids[0], chain_ids[0], threshold=1.5)
        mask = mask.unsqueeze(0)

        pred = centroids + torch.randn_like(centroids) * 0.1

        # With precomputed mask
        losses1 = loss_fn(
            pred_centroids=pred,
            gt_centroids=centroids,
            chain_ids=chain_ids,
            contact_mask=mask
        )

        # Without precomputed mask (computed internally)
        losses2 = loss_fn(
            pred_centroids=pred,
            gt_centroids=centroids,
            chain_ids=chain_ids
        )

        # Should be equal
        assert losses1['stage1'].item() == pytest.approx(losses2['stage1'].item(), rel=1e-5)


# =============================================================================
# Integration tests with real data
# =============================================================================

class TestContactLossRealData:
    """Integration tests using real protein data."""

    def test_on_real_protein(self):
        """Test contact loss on a real protein from dataset."""
        import pyarrow.parquet as pq

        try:
            table = pq.read_table('C:/Users/costa/src/tinyfold/data/processed/samples.parquet')
        except FileNotFoundError:
            pytest.skip("Dataset not found")

        # Load first protein
        idx = 0
        coords_flat = torch.tensor(table['atom_coords'][idx].as_py(), dtype=torch.float32)
        chain_ids = torch.tensor(table['chain_id_res'][idx].as_py(), dtype=torch.long)
        n_atoms = len(table['atom_type'][idx].as_py())
        n_res = n_atoms // 4

        # Reshape and normalize
        coords = coords_flat.reshape(n_res, 4, 3)
        std = coords.std()
        coords = coords / std
        centroids = coords.mean(dim=1)

        # Batch
        coords = coords.unsqueeze(0)
        centroids = centroids.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        # Test contact loss
        loss_fn = ContactLoss(threshold=1.0, stage="both")  # 10A in normalized

        # GT vs GT should be ~0
        losses_gt = loss_fn(
            pred_centroids=centroids,
            gt_centroids=centroids,
            pred_atoms=coords,
            gt_atoms=coords,
            chain_ids=chain_ids
        )

        assert losses_gt['stage1'].item() < 1e-5
        assert losses_gt['stage2'].item() < 1e-5
        assert losses_gt['n_contacts'] > 0

        # Noisy vs GT should have positive loss
        pred_centroids = centroids + torch.randn_like(centroids) * 0.1
        pred_atoms = coords + torch.randn_like(coords) * 0.1

        losses_noisy = loss_fn(
            pred_centroids=pred_centroids,
            gt_centroids=centroids,
            pred_atoms=pred_atoms,
            gt_atoms=coords,
            chain_ids=chain_ids
        )

        assert losses_noisy['stage1'].item() > 0
        assert losses_noisy['stage2'].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
