"""Unit tests for the per-target ConfidenceHead.

Loop 06 (`.delegate/work/20260524-051951-prio01-retrain/06/PLAN.md`) Task 2:
locks the shape, mask-invariance, and gradient-flow contracts of the head.
"""

import pytest
import torch
import torch.nn.functional as F

from tinyfold.model.resfold import ConfidenceHead


class TestConfidenceHead:
    """Shape / mask / gradient contracts for ConfidenceHead."""

    def test_shape(self):
        """Output must be [B] and lie in [0, 1]."""
        torch.manual_seed(0)
        B, L, C = 3, 17, 128
        head = ConfidenceHead(c_token=C)
        tokens = torch.randn(B, L, C)
        # Mask: last 5 residues padded out on each sample.
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, -5:] = False

        pred = head(tokens, mask)
        assert pred.shape == (B,), f"expected [{B}], got {pred.shape}"
        assert torch.all((pred >= 0.0) & (pred <= 1.0)), (
            f"pred outside [0, 1]: {pred}"
        )

    def test_no_mask_equals_all_true_mask(self):
        """Passing mask=None must match an all-True mask."""
        torch.manual_seed(1)
        B, L, C = 2, 7, 64
        head = ConfidenceHead(c_token=C)
        tokens = torch.randn(B, L, C)
        mask_all = torch.ones(B, L, dtype=torch.bool)

        pred_no_mask = head(tokens, None)
        pred_full = head(tokens, mask_all)
        assert torch.allclose(pred_no_mask, pred_full, atol=1e-6)

    def test_mask_invariance(self):
        """Padding zero tokens with mask=False must not change the prediction."""
        torch.manual_seed(2)
        B, L_real, C = 2, 8, 64
        head = ConfidenceHead(c_token=C)
        head.eval()  # disable any dropout influence

        tokens_real = torch.randn(B, L_real, C)
        mask_real = torch.ones(B, L_real, dtype=torch.bool)
        pred_real = head(tokens_real, mask_real)

        # Now pad with 5 extra residues whose mask is False — those tokens carry
        # arbitrary garbage values and must NOT affect the masked-mean pool.
        L_pad = 5
        garbage = torch.randn(B, L_pad, C) * 100.0  # large-magnitude garbage
        tokens_padded = torch.cat([tokens_real, garbage], dim=1)
        mask_padded = torch.cat(
            [mask_real, torch.zeros(B, L_pad, dtype=torch.bool)],
            dim=1,
        )
        pred_padded = head(tokens_padded, mask_padded)
        assert torch.allclose(pred_real, pred_padded, atol=1e-6), (
            f"mask-invariance broken: real={pred_real} padded={pred_padded}"
        )

    def test_grad_flow(self):
        """Backprop a smooth-L1 loss; MLP weights must receive non-zero grads."""
        torch.manual_seed(3)
        B, L, C = 4, 11, 32
        head = ConfidenceHead(c_token=C)
        tokens = torch.randn(B, L, C, requires_grad=True)
        mask = torch.ones(B, L, dtype=torch.bool)
        target = torch.full((B,), 0.7)

        pred = head(tokens, mask)
        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        # MLP weights have grads
        for name, p in head.named_parameters():
            assert p.grad is not None, f"{name}: no grad attached"
            assert p.grad.abs().sum() > 0, f"{name}: grad is exactly zero"
        # Input tokens also have grads (the pool is differentiable).
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0

    def test_init_predicts_midpoint(self):
        """Output bias = 0 means the head predicts sigmoid(0) = 0.5 at init.

        With normal-distributed tokens and a Linear -> SiLU -> Linear MLP, the
        OUTPUT logits are not exactly zero (the activation is non-zero), so we
        only assert the predictions live in a sane band around 0.5. This is a
        sanity check that we did not accidentally pin the output to 0 or 1.
        """
        torch.manual_seed(0)
        head = ConfidenceHead(c_token=64)
        tokens = torch.randn(8, 12, 64)
        pred = head(tokens)
        assert pred.min() > 0.1 and pred.max() < 0.9, (
            f"init predictions look pathological: min={pred.min()} max={pred.max()}"
        )

    def test_param_count_is_tiny(self):
        """Sanity: ~2 * c_token^2 params (default hidden = c_token)."""
        C = 128
        head = ConfidenceHead(c_token=C)
        n = sum(p.numel() for p in head.parameters())
        # 2 linears: (C*C + C) + (C*1 + 1) = C^2 + 2C + 1 = 16641 for C=128.
        # Total roughly C^2; assert under 2x C^2 to allow for the bias rows.
        assert n < 2 * C * C, f"head too big: {n} params"
        assert n > C, "head suspiciously small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
