"""B4: symmetry-corrected torsion loss (chi-pi-periodic).

The mandatory correction: a prediction off by exactly pi on a pi-periodic chi
(ASP chi2, GLU chi3, PHE/TYR chi2) is CORRECT and must score ~0; the same flip on
a non-periodic chi must score the full penalty.
"""

import math

import torch

from tinyfold.constants import AA_CODES
from tinyfold.model.losses.torsion import chi_mae_deg, torsion_symmetry_loss


def _vec(chi):
    return torch.stack([torch.cos(chi), torch.sin(chi)], dim=-1)


def test_perfect_prediction_is_zero():
    aa = torch.tensor([[AA_CODES.index("D")]])
    gt = torch.tensor([[[0.5, -1.2, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False, False]]])
    loss = torsion_symmetry_loss(_vec(gt), gt, mask, aa)
    assert loss.item() < 1e-6


def test_pi_flip_forgiven_on_periodic_chi():
    # ASP: chi2 (index 1) is pi-periodic; chi1 (index 0) is not.
    aa = torch.tensor([[AA_CODES.index("D")]])
    gt = torch.tensor([[[0.5, -1.2, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False, False]]])
    # Flip chi2 by pi -> should stay ~0 (periodic); flip chi1 by pi -> full penalty.
    pred_chi2_flip = gt.clone(); pred_chi2_flip[..., 1] += math.pi
    pred_chi1_flip = gt.clone(); pred_chi1_flip[..., 0] += math.pi
    loss_ok = torsion_symmetry_loss(_vec(pred_chi2_flip), gt, mask, aa)
    loss_bad = torsion_symmetry_loss(_vec(pred_chi1_flip), gt, mask, aa)
    assert loss_ok.item() < 1e-5
    assert loss_bad.item() > 1.0   # (cos,sin) L2 for a pi error is (2)^2 = 4 on that chi


def test_mask_excludes_absent_chis():
    aa = torch.tensor([[AA_CODES.index("S")]])   # SER: 1 chi
    gt = torch.zeros(1, 1, 4)
    mask = torch.tensor([[[True, False, False, False]]])
    # A huge error on a masked chi must not affect the loss.
    pred = gt.clone(); pred[..., 2] = 3.0
    loss = torsion_symmetry_loss(_vec(pred), gt, mask, aa)
    assert loss.item() < 1e-6


def test_chi_mae_degrees():
    aa = torch.tensor([[AA_CODES.index("D")]])
    gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False, False]]])
    # chi1 off by 90 deg (non-periodic) -> 90; chi2 off by pi (periodic) -> ~0.
    pred = torch.tensor([[[math.pi / 2, math.pi, 0.0, 0.0]]])
    mae = chi_mae_deg(pred, gt, mask, aa).item()
    assert math.isclose(mae, 45.0, abs_tol=1e-3)   # (90 + 0) / 2 valid chis
