"""Tests for crop correctness hardening (C8).

Two guards: the un-cropped-key assertion in _apply_residue_indices fires on a
stray per-residue key, and InterfaceCrop keeps at least one residue from each
chain whenever the complex has an interface.
"""

import pytest
import torch

from tinyfold.training.cropping import (
    InterfaceCrop,
    SpatialCrop,
    _apply_residue_indices,
)


def _two_chain_sample(LA=8, LB=8, seed=0):
    """Minimal loaded-sample dict for the croppers: two chains, contacting."""
    torch.manual_seed(seed)
    L = LA + LB
    chain = torch.tensor([0] * LA + [1] * LB)
    # Place the two chains near each other so an interface exists.
    coords_a = torch.randn(LA, 4, 3)
    coords_b = torch.randn(LB, 4, 3) + torch.tensor([1.0, 0.0, 0.0])
    coords_res = torch.cat([coords_a, coords_b], dim=0)  # [L,4,3]
    coords = coords_res.reshape(L * 4, 3)
    return {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': coords_res.mean(dim=1),
        'atom_types': torch.tensor([0, 1, 2, 3] * L),
        'atom_to_res': torch.arange(L).repeat_interleave(4),
        'aa_seq': torch.randint(0, 20, (L,)),
        'chain_ids': chain,
        'res_idx': torch.arange(L),
        'std': 1.0,
        'n_atoms': L * 4,
        'n_res': L,
        'sample_id': 'test',
    }


# --- un-cropped-key guard ---------------------------------------------------

def test_guard_fires_on_stray_per_residue_key():
    sample = _two_chain_sample()
    # A per-residue key NOT in the crop list -> survives at parent length.
    sample['iface_mask'] = torch.ones(sample['n_res'], dtype=torch.bool)
    idx = torch.arange(6)  # crop to 6 < L
    with pytest.raises(AssertionError, match="iface_mask"):
        _apply_residue_indices(sample, idx)


def test_guard_passes_for_the_standard_schema():
    sample = _two_chain_sample()
    out = _apply_residue_indices(sample, torch.arange(6))
    assert out['n_res'] == 6
    # Every standard per-residue tensor is now length 6.
    for k in ('centroids', 'coords_res', 'aa_seq', 'chain_ids', 'res_idx'):
        assert out[k].shape[0] == 6
    # Atom tensors are 4x.
    assert out['coords'].shape[0] == 24
    assert out['atom_types'].shape[0] == 24


def test_crop_residue_and_atom_tensors_are_mutually_consistent():
    sample = _two_chain_sample()
    out = _apply_residue_indices(sample, torch.tensor([0, 5, 10, 12]))
    L_crop = out['n_res']
    assert out['coords'].shape[0] == L_crop * 4
    # coords_res is the reshaped flat coords.
    assert torch.equal(out['coords_res'], out['coords'].view(L_crop, 4, 3))
    # atom_to_res maps each atom to its new residue index.
    assert torch.equal(
        out['atom_to_res'],
        torch.arange(L_crop).repeat_interleave(4),
    )


# --- InterfaceCrop keeps both chains ---------------------------------------

def test_interface_crop_retains_both_chains_when_interface_exists():
    cropper = InterfaceCrop(interface_prob=1.0, cutoff=8.0)
    rng = torch.Generator().manual_seed(1)
    sample = _two_chain_sample(LA=20, LB=20)
    for _ in range(10):
        crop = cropper(sample, crop_size=16, rng=rng)
        has_a = bool((crop['chain_ids'] == 0).any())
        has_b = bool((crop['chain_ids'] == 1).any())
        assert has_a and has_b, "InterfaceCrop dropped a chain despite an interface"


def test_spatial_crop_size_is_honoured():
    sample = _two_chain_sample(LA=20, LB=20)
    rng = torch.Generator().manual_seed(2)
    crop = SpatialCrop()(sample, crop_size=12, rng=rng)
    assert crop['n_res'] == 12
