"""B3: the torsion (wrapped-diffusion) sidechain head.

Locks the contract: predicts chi0 as unit (cos,sin) vectors -> angles in (-pi,pi];
sigma and position both change the output (no permutation-invariance floor); mask
zeroes padded residues.
"""

import math

import torch

from tinyfold.atom14 import NUM_CHI
from tinyfold.model.resfold.sidechain_torsion_head import SidechainTorsionHead, wrap_angle


def _head(**kw):
    torch.manual_seed(0)
    return SidechainTorsionHead(c_token=32, n_layers=2, n_heads=4, use_tokens=False, **kw)


def _inputs(B=2, L=6):
    torch.manual_seed(1)
    chi = (torch.rand(B, L, NUM_CHI) * 2 - 1) * math.pi
    aatype = torch.randint(0, 20, (B, L))
    sigma = torch.full((B,), 0.7)
    mask = torch.ones(B, L, dtype=torch.bool)
    bb = torch.randn(B, L, 4, 3)
    return chi, aatype, sigma, mask, bb


def test_wrap_angle_range():
    a = torch.tensor([0.0, math.pi, -math.pi, 3 * math.pi, -3 * math.pi])
    w = wrap_angle(a)
    assert (w <= math.pi + 1e-6).all() and (w > -math.pi - 1e-6).all()


def test_forward_shape_and_range():
    head = _head()
    chi, aa, sigma, mask, bb = _inputs()
    chi0, vec = head(chi, aa, sigma, mask=mask, backbone_feats=bb, return_vec=True)
    assert chi0.shape == (2, 6, NUM_CHI)
    assert vec.shape == (2, 6, NUM_CHI, 2)
    assert (chi0.abs() <= math.pi + 1e-5).all()
    # vec is unit-norm.
    assert torch.allclose(vec.norm(dim=-1), torch.ones_like(vec[..., 0]), atol=1e-4)


def test_sigma_changes_output():
    head = _head().eval()
    chi, aa, _, mask, bb = _inputs()
    with torch.no_grad():
        lo = head(chi, aa, torch.full((2,), 0.05), mask=mask, backbone_feats=bb)
        hi = head(chi, aa, torch.full((2,), 3.0), mask=mask, backbone_feats=bb)
    assert not torch.allclose(lo, hi)


def test_position_breaks_permutation_invariance():
    # Two residues of the SAME type must get DIFFERENT predictions (else the head
    # can only predict the per-restype mean rotamer -- the free-offset 2.5A floor).
    head = _head(backbone_cond=False).eval()
    L = 4
    chi = torch.zeros(1, L, NUM_CHI)                 # identical chi
    aa = torch.zeros(1, L, dtype=torch.long)         # identical type
    sigma = torch.full((1,), 1.0)
    with torch.no_grad():
        out = head(chi, aa, sigma)
    # Not all residues identical -> position is doing work.
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-4)


def test_backbone_cond_changes_output():
    head = _head(backbone_cond=True).eval()
    chi, aa, sigma, mask, bb = _inputs()
    with torch.no_grad():
        a = head(chi, aa, sigma, mask=mask, backbone_feats=bb)
        b = head(chi, aa, sigma, mask=mask, backbone_feats=bb * 0)
    assert not torch.allclose(a, b)


def test_mask_zeroes_padding():
    head = _head().eval()
    chi, aa, sigma, mask, bb = _inputs()
    mask[:, -2:] = False
    with torch.no_grad():
        out = head(chi, aa, sigma, mask=mask, backbone_feats=bb)
    assert torch.count_nonzero(out[:, -2:]) == 0
