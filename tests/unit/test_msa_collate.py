"""msa_feats threads through collate + cropping as an O(L^2) pair feature.

Unlike every other feature (per-residue), coevolution features are [L, L, F], so
collate pads the [B, max_res, max_res, F] block and the cropper must slice BOTH
residue axes. These tests lock both behaviours.
"""

import torch

from tinyfold.msa.features import msa_feat_dim
from tinyfold.training.cropping import ContiguousCrop
from tinyfold.training.data import collate_batch


def _sample(sample_id: str, L: int, with_msa: bool = True) -> dict:
    """A minimal load_sample-shaped dict; msa_feats carries i*L+j so we can check
    the exact [:L,:L] block survives padding/cropping unpermuted."""
    F = msa_feat_dim()
    s = {
        "centroids": torch.randn(L, 3),
        "coords_res": torch.randn(L, 4, 3),
        "aa_seq": torch.randint(0, 20, (L,)),
        "chain_ids": torch.zeros(L, dtype=torch.long),
        "res_idx": torch.arange(L),
        "coords": torch.randn(L * 4, 3),
        "atom_types": torch.randint(0, 4, (L * 4,)),
        "atom_to_res": torch.arange(L).repeat_interleave(4),
        "std": 1.0,
        "n_res": L,
        "n_atoms": L * 4,
        "sample_id": sample_id,
    }
    if with_msa:
        base = torch.arange(L * L, dtype=torch.float32).view(L, L)
        s["msa_feats"] = base.unsqueeze(-1).repeat(1, 1, F)
    return s


def test_msa_collate_pads_l2_block_and_round_trips():
    samples = [_sample("a", 5), _sample("b", 8)]
    batch = collate_batch(samples, device=torch.device("cpu"))
    assert "msa_feats" in batch
    assert batch["msa_feats"].shape == (2, 8, 8, msa_feat_dim())
    # Each sample's [:L,:L] block is preserved exactly; the pad region is zero.
    for i, L in enumerate((5, 8)):
        got = batch["msa_feats"][i, :L, :L].float()
        assert torch.equal(got, samples[i]["msa_feats"].float())
        if L < 8:
            assert batch["msa_feats"][i, L:, :].abs().sum() == 0
            assert batch["msa_feats"][i, :, L:].abs().sum() == 0


def test_msa_absent_when_no_sample_has_it():
    batch = collate_batch([_sample("a", 5, with_msa=False)], device=torch.device("cpu"))
    assert "msa_feats" not in batch


def test_cropper_slices_both_axes():
    L, crop = 10, 4
    s = _sample("a", L)
    rng = torch.Generator().manual_seed(0)
    cropped = ContiguousCrop()(s, crop, rng)
    idx = cropped["crop_global_idx"]
    assert cropped["msa_feats"].shape == (crop, crop, msa_feat_dim())
    # The cropped block equals the gathered i,j sub-matrix of the original.
    expected = s["msa_feats"][idx][:, idx]
    assert torch.equal(cropped["msa_feats"], expected)
    # And it still collates cleanly after cropping.
    batch = collate_batch([cropped], device=torch.device("cpu"))
    assert batch["msa_feats"].shape == (1, crop, crop, msa_feat_dim())
