"""Tests for EDM/Karras 2022 per-sample loss weighting.

Reference: Karras et al. 2022, "Elucidating the Design Space of Diffusion-
Based Generative Models", Eq. 7:

    lambda(sigma) = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2

Pins the formula against hand-computed reference values; catches the
historical Defect-1 regression where the denominator was (sigma + sigma_data)**2
(sum-then-square) instead of (sigma * sigma_data)**2 (product-then-square).
"""

import pytest
import torch

from tinyfold.training.utils import edm_loss_weight, af3_loss_weight


def test_edm_loss_weight_sigma_data_1():
    """sigma=1, sigma_data=1 -> (1+1)/(1*1)**2 = 2.0"""
    out = edm_loss_weight(torch.tensor([1.0]), sigma_data=1.0)
    torch.testing.assert_close(out, torch.tensor([2.0]), rtol=1e-6, atol=1e-6)


def test_edm_loss_weight_sigma_data_half():
    """sigma=0.5, sigma_data=0.5 -> (0.25+0.25)/(0.25)**2 = 0.5/0.0625 = 8.0"""
    out = edm_loss_weight(torch.tensor([0.5]), sigma_data=0.5)
    torch.testing.assert_close(out, torch.tensor([8.0]), rtol=1e-6, atol=1e-6)


def test_edm_loss_weight_three_point_reference():
    """Reference values at sigma=[0.1, 1.0, 10.0], sigma_data=0.5.

    - 0.1: (0.01 + 0.25) / (0.05)**2 = 0.26 / 0.0025 = 104.0
    - 1.0: (1.0 + 0.25) / (0.5)**2 = 1.25 / 0.25 = 5.0
    - 10.0: (100.0 + 0.25) / (5.0)**2 = 100.25 / 25.0 = 4.01
    """
    sigma = torch.tensor([0.1, 1.0, 10.0])
    expected = torch.tensor([104.0, 5.0, 4.01])
    out = edm_loss_weight(sigma, sigma_data=0.5)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_edm_loss_weight_zero_sigma_finite():
    """sigma=0 must not raise (clamp(min=1e-8) guard); result is finite + large."""
    out = edm_loss_weight(torch.tensor([0.0]), sigma_data=1.0)
    assert torch.isfinite(out).all(), f"Expected finite output, got {out}"
    # With clamp at 1e-8 and sigma_data=1: lambda ~ 1/sigma**2 = 1e16; just assert it's huge.
    assert (out > 1e10).all(), f"Expected large weight at sigma->0, got {out.item()}"


def test_edm_loss_weight_shape_preservation():
    """Input shape is preserved on output."""
    # 1D input
    sigma_1d = torch.tensor([0.1, 1.0, 10.0])
    out_1d = edm_loss_weight(sigma_1d, sigma_data=1.0)
    assert out_1d.shape == sigma_1d.shape, \
        f"1D shape mismatch: {out_1d.shape} vs {sigma_1d.shape}"

    # 2D input [B, 1]
    sigma_2d = torch.tensor([[0.1], [1.0], [10.0]])
    out_2d = edm_loss_weight(sigma_2d, sigma_data=1.0)
    assert out_2d.shape == sigma_2d.shape, \
        f"2D shape mismatch: {out_2d.shape} vs {sigma_2d.shape}"


def test_af3_loss_weight_deprecation_alias():
    """af3_loss_weight emits DeprecationWarning and returns same values as edm_loss_weight."""
    sigma = torch.tensor([0.1, 1.0, 10.0])
    with pytest.warns(DeprecationWarning, match="af3_loss_weight is deprecated"):
        out_legacy = af3_loss_weight(sigma, sigma_data=1.0)
    out_canonical = edm_loss_weight(sigma, sigma_data=1.0)
    torch.testing.assert_close(out_legacy, out_canonical, rtol=1e-6, atol=1e-6)
