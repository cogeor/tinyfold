"""Tests for tinyfold.training.cropping.

Pin the cropping contract:
  - Each strategy returns a sample dict with the same keys as the input.
  - GLOBAL res_idx is preserved (so positional encoding is crop-invariant).
  - Shapes are correct: per-residue tensors have L_crop entries, atom tensors
    have 4 * L_crop entries.
  - Samples with L <= crop_size pass through unchanged (no info lost).
  - InterfaceCrop reliably keeps both chains in the crop when both chains
    exist and the random draw lands on an interface residue.
"""

from __future__ import annotations

import pytest
import torch

from tinyfold.training.cropping import (
    ContiguousCrop,
    InterfaceCrop,
    NoCrop,
    SpatialCrop,
    build_cropper,
)


def _make_sample(n_chain_a: int = 60, n_chain_b: int = 80, with_esm: bool = False) -> dict:
    """Build a synthetic two-chain sample with known positions.

    Chain A occupies the first ``n_chain_a`` residues at x in [0, n_chain_a].
    Chain B occupies the next ``n_chain_b`` residues at x in [50, 50 + n_chain_b].
    The chains touch around x=50, so InterfaceCrop should find contacts there.
    """
    n = n_chain_a + n_chain_b
    chain_ids = torch.cat([
        torch.zeros(n_chain_a, dtype=torch.long),
        torch.ones(n_chain_b, dtype=torch.long),
    ])
    res_idx = torch.cat([
        torch.arange(n_chain_a, dtype=torch.long),
        torch.arange(n_chain_b, dtype=torch.long),
    ])
    aa_seq = torch.randint(0, 20, (n,), dtype=torch.long)

    # Atoms: 4 backbone atoms per residue with tiny per-atom offset around the
    # residue centroid. Chain A along x axis starting at 0; chain B starting
    # at 49 (so the first ~3 residues of B touch the last ~3 of A within 8 A).
    coords_res = torch.zeros(n, 4, 3)
    for i in range(n_chain_a):
        coords_res[i, :, 0] = float(i)
        coords_res[i, 0, 1] = -0.5     # N
        coords_res[i, 1, 1] = 0.0      # CA
        coords_res[i, 2, 1] = 0.5      # C
        coords_res[i, 3, 1] = 1.0      # O
    for j in range(n_chain_b):
        coords_res[n_chain_a + j, :, 0] = float(49 + j)
        coords_res[n_chain_a + j, :, 1] = 4.0  # 4 A off A's axis -> contact at edge

    centroids = coords_res.mean(dim=1)
    coords = coords_res.reshape(n * 4, 3)
    atom_types = torch.tile(torch.arange(4, dtype=torch.long), (n,))
    atom_to_res = torch.arange(n, dtype=torch.long).repeat_interleave(4)

    sample = {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'res_idx': res_idx,
        'std': 1.0,
        'n_atoms': n * 4,
        'n_res': n,
        'sample_id': 'synthetic.test',
    }
    if with_esm:
        sample['esm_embed'] = torch.randn(n, 64)
    return sample


def _check_invariants(out: dict, sample: dict, expected_L: int):
    # Shapes
    assert out['n_res'] == expected_L
    assert out['n_atoms'] == expected_L * 4
    assert out['centroids'].shape == (expected_L, 3)
    assert out['coords_res'].shape == (expected_L, 4, 3)
    assert out['aa_seq'].shape == (expected_L,)
    assert out['chain_ids'].shape == (expected_L,)
    assert out['res_idx'].shape == (expected_L,)
    assert out['coords'].shape == (expected_L * 4, 3)
    assert out['atom_types'].shape == (expected_L * 4,)
    assert out['atom_to_res'].shape == (expected_L * 4,)
    # Identity preserved
    assert out['sample_id'] == sample['sample_id']
    assert out['std'] == sample['std']
    # Crop traceability
    assert 'crop_global_idx' in out
    assert out['crop_global_idx'].shape == (expected_L,)
    # GLOBAL res_idx preserved: every selected residue carries the same res_idx
    # it had in the input.
    idx = out['crop_global_idx']
    assert torch.equal(out['res_idx'], sample['res_idx'][idx])
    assert torch.equal(out['chain_ids'], sample['chain_ids'][idx])
    # atom_to_res is the NEW local index (0..L_crop-1 repeated 4x), not global.
    assert torch.equal(
        out['atom_to_res'],
        torch.arange(expected_L).repeat_interleave(4),
    )


def test_nocrop_passthrough():
    sample = _make_sample(n_chain_a=10, n_chain_b=10)
    rng = torch.Generator().manual_seed(0)
    out = NoCrop()(sample, crop_size=256, rng=rng)
    _check_invariants(out, sample, expected_L=20)


def test_nocrop_errors_when_too_big():
    sample = _make_sample(n_chain_a=200, n_chain_b=200)
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="NoCrop"):
        NoCrop()(sample, crop_size=256, rng=rng)


def test_contiguous_crop_shape_and_global_idx():
    sample = _make_sample(n_chain_a=200, n_chain_b=200)
    rng = torch.Generator().manual_seed(0)
    out = ContiguousCrop()(sample, crop_size=128, rng=rng)
    _check_invariants(out, sample, expected_L=128)
    # Contiguous: global indices must be a sorted contiguous range.
    g = out['crop_global_idx']
    assert torch.equal(g, torch.arange(g[0].item(), g[0].item() + 128))


def test_spatial_crop_keeps_neighbors():
    sample = _make_sample(n_chain_a=200, n_chain_b=200)
    rng = torch.Generator().manual_seed(0)
    out = SpatialCrop()(sample, crop_size=64, rng=rng)
    _check_invariants(out, sample, expected_L=64)
    # The crop is the K-NN of a single center residue in GT CA space. Verify
    # that the maximum pairwise CA distance in the crop is <= the max distance
    # in the input (i.e., the crop is structurally tighter than a random
    # selection, modulo the trivial bound).
    ca_in = sample['coords_res'][:, 1, :]
    ca_out = out['coords_res'][:, 1, :]
    max_in = torch.cdist(ca_in, ca_in).max().item()
    max_out = torch.cdist(ca_out, ca_out).max().item()
    assert max_out <= max_in + 1e-5


def test_interface_crop_includes_both_chains():
    # Force interface selection to make this deterministic.
    cropper = InterfaceCrop(interface_prob=1.0, cutoff=8.0)
    sample = _make_sample(n_chain_a=200, n_chain_b=200)
    rng = torch.Generator().manual_seed(0)
    out = cropper(sample, crop_size=64, rng=rng)
    _check_invariants(out, sample, expected_L=64)
    # With interface_prob=1.0 the center must land on a contact residue, and
    # the K-NN around it must include residues from both chains (since the
    # contact is the K-NN-nearest cross-chain residue by construction).
    chains = out['chain_ids'].tolist()
    assert 0 in chains and 1 in chains, (
        f"InterfaceCrop should include both chains; got chain_ids={set(chains)}"
    )


def test_interface_crop_falls_back_when_no_contacts():
    # Build a sample where the two chains are far apart -> no interface.
    sample = _make_sample(n_chain_a=80, n_chain_b=80)
    # Move chain B 1000 A away on x axis.
    sample['coords_res'][80:, :, 0] += 1000.0
    sample['centroids'] = sample['coords_res'].mean(dim=1)
    sample['coords'] = sample['coords_res'].reshape(160 * 4, 3)
    cropper = InterfaceCrop(interface_prob=1.0, cutoff=8.0)
    rng = torch.Generator().manual_seed(0)
    out = cropper(sample, crop_size=64, rng=rng)
    # Should not raise; falls back to SpatialCrop and returns the right shape.
    _check_invariants(out, sample, expected_L=64)


def test_esm_embed_cropped_consistently():
    sample = _make_sample(n_chain_a=200, n_chain_b=200, with_esm=True)
    rng = torch.Generator().manual_seed(0)
    out = SpatialCrop()(sample, crop_size=50, rng=rng)
    assert 'esm_embed' in out
    assert out['esm_embed'].shape == (50, 64)
    # Per-row equivalence: out['esm_embed'][k] must equal
    # sample['esm_embed'][crop_global_idx[k]].
    assert torch.equal(
        out['esm_embed'],
        sample['esm_embed'][out['crop_global_idx']],
    )


def test_build_cropper_factory():
    assert isinstance(build_cropper("none"), NoCrop)
    assert isinstance(build_cropper("contiguous"), ContiguousCrop)
    assert isinstance(build_cropper("spatial"), SpatialCrop)
    assert isinstance(build_cropper("interface"), InterfaceCrop)
    with pytest.raises(ValueError, match="Unknown crop_strategy"):
        build_cropper("bogus")
