"""Symmetry-corrected sidechain loss (spec §9 -- MANDATORY).

The property under test: a prediction that places ASP's two carboxyl oxygens
perfectly but with the labels EXCHANGED is correct, and must cost ~nothing. A
naive MSE would charge it the full ~2.2 A separation and train the model to guess
an arbitrary labelling.
"""

import torch

from tinyfold.atom14 import atom14_mask, atom14_names
from tinyfold.constants import AA_TO_IDX
from tinyfold.model.losses.sidechain import (
    make_alt_gt,
    sidechain_rmsd,
    symmetric_sidechain_loss,
)


def _residue(aa1: str, seed: int = 0):
    """One-residue batch: (pred/gt [1,1,14,3], mask [1,1,14], aatype [1,1])."""
    g = torch.Generator().manual_seed(seed)
    gt = torch.randn(1, 1, 14, 3, generator=g)
    mask = torch.as_tensor(atom14_mask(aa1))[None, None]
    aatype = torch.tensor([[AA_TO_IDX[aa1]]])
    return gt, mask, aatype


def _swap(coords, aa1, a, b):
    """Exchange two named atoms' coordinates."""
    names = atom14_names(aa1)
    out = coords.clone()
    ia, ib = names.index(a), names.index(b)
    out[..., ia, :], out[..., ib, :] = coords[..., ib, :].clone(), coords[..., ia, :].clone()
    return out


class TestMakeAltGt:
    def test_identity_for_unambiguous_residue(self):
        gt, _, aatype = _residue("W")
        torch.testing.assert_close(make_alt_gt(gt, aatype), gt)

    def test_asp_alt_is_the_od1_od2_swap(self):
        gt, _, aatype = _residue("D")
        torch.testing.assert_close(make_alt_gt(gt, aatype), _swap(gt, "D", "OD1", "OD2"))

    def test_alt_of_alt_is_identity(self):
        gt, _, aatype = _residue("F")
        torch.testing.assert_close(make_alt_gt(make_alt_gt(gt, aatype), aatype), gt)

    def test_backbone_untouched(self):
        gt, _, aatype = _residue("E")
        torch.testing.assert_close(make_alt_gt(gt, aatype)[..., :4, :], gt[..., :4, :])


class TestSymmetryCorrection:
    def test_relabelled_asp_costs_nothing(self):
        # THE point of the whole module.
        gt, mask, aatype = _residue("D")
        pred = _swap(gt, "D", "OD1", "OD2")
        loss = symmetric_sidechain_loss(pred, gt, mask, aatype)
        assert loss.item() < 1e-6

    def test_naive_loss_would_have_punished_it(self):
        # Guard against the correction silently doing nothing: the uncorrected
        # error on the same prediction must be substantial.
        gt, mask, aatype = _residue("D")
        pred = _swap(gt, "D", "OD1", "OD2")
        naive = (((pred - gt) ** 2).sum(-1) * mask).sum()
        assert naive.item() > 1.0

    def test_relabelling_a_non_symmetric_residue_still_costs(self):
        # TRP's ring atoms are NOT interchangeable -- swapping them is a real
        # error and must be charged.
        gt, mask, aatype = _residue("W")
        pred = _swap(gt, "W", "CD1", "CD2")
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() > 0.1

    def test_phe_ring_flip_costs_nothing(self):
        gt, mask, aatype = _residue("F")
        pred = _swap(_swap(gt, "F", "CD1", "CD2"), "F", "CE1", "CE2")
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() < 1e-6

    def test_arg_nh1_nh2_flip_costs_nothing(self):
        # ARG is renaming-ambiguous even though it is not chi-pi-periodic.
        gt, mask, aatype = _residue("R")
        pred = _swap(gt, "R", "NH1", "NH2")
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() < 1e-6

    def test_partial_flip_is_not_free(self):
        # Flipping only ONE of PHE's two ring pairs is not a symmetry op.
        gt, mask, aatype = _residue("F")
        pred = _swap(gt, "F", "CD1", "CD2")
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() > 0.1


class TestPerResidueIndependence:
    def test_each_residue_picks_its_own_labelling(self):
        # Residue 0 flipped, residue 1 not. A global min would have to punish
        # one of them; a per-residue min frees both.
        gt = torch.randn(1, 2, 14, 3)
        aatype = torch.tensor([[AA_TO_IDX["D"], AA_TO_IDX["D"]]])
        mask = torch.as_tensor(atom14_mask("D"))[None, None].expand(1, 2, 14)
        pred = gt.clone()
        pred[:, 0] = _swap(gt[:, 0], "D", "OD1", "OD2")
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() < 1e-6

    def test_per_residue_reduction_shape(self):
        gt = torch.randn(2, 5, 14, 3)
        aatype = torch.full((2, 5), AA_TO_IDX["D"])
        mask = torch.as_tensor(atom14_mask("D"))[None, None].expand(2, 5, 14)
        out = symmetric_sidechain_loss(gt, gt, mask, aatype, reduction="per_residue")
        assert out.shape == (2, 5)


class TestMaskingAndDegenerateCases:
    def test_exact_prediction_is_zero(self):
        gt, mask, aatype = _residue("Y")
        assert symmetric_sidechain_loss(gt, gt, mask, aatype).item() < 1e-9

    def test_absent_atoms_are_ignored(self):
        # GLY has no sidechain: garbage in the unused slots must not register.
        gt, mask, aatype = _residue("G")
        pred = gt.clone()
        pred[..., 4:, :] += 100.0
        assert symmetric_sidechain_loss(pred, gt, mask, aatype).item() < 1e-6

    def test_all_masked_does_not_nan(self):
        gt, _, aatype = _residue("D")
        mask = torch.zeros(1, 1, 14, dtype=torch.bool)
        assert torch.isfinite(symmetric_sidechain_loss(gt, gt, mask, aatype))

    def test_gradients_flow(self):
        gt, mask, aatype = _residue("D")
        pred = gt.clone().requires_grad_(True)
        symmetric_sidechain_loss(pred + 0.1, gt, mask, aatype).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestSidechainRmsd:
    def test_zero_for_exact_prediction(self):
        gt, mask, aatype = _residue("Y")
        assert sidechain_rmsd(gt, gt, mask, aatype).item() < 1e-6

    def test_relabelled_asp_is_zero_rmsd(self):
        gt, mask, aatype = _residue("D")
        pred = _swap(gt, "D", "OD1", "OD2")
        assert sidechain_rmsd(pred, gt, mask, aatype).item() < 1e-6

    def test_ignores_the_frozen_backbone(self):
        # Stage 3 never predicts the backbone; a wrong backbone must not leak
        # into (or flatter) the gate metric.
        gt, mask, aatype = _residue("Y")
        pred = gt.clone()
        pred[..., :4, :] += 10.0
        assert sidechain_rmsd(pred, gt, mask, aatype).item() < 1e-6

    def test_reports_in_coordinate_units(self):
        # Displace every sidechain atom by exactly 0.5 -> RMSD 0.5.
        gt, mask, aatype = _residue("Y")
        pred = gt.clone()
        pred[..., 4:, 0] += 0.5
        assert abs(sidechain_rmsd(pred, gt, mask, aatype).item() - 0.5) < 1e-5
