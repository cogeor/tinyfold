"""Integration tests for the ConfidenceHead wired into ResFoldOneStep.

Loop 06 Task 6: lock the 3-tuple return contract of ``forward_sigma`` /
``forward_sigma_with_trunk`` and the gradient flow path
(pred_lddt -> pool -> denoiser tokens).
"""

import pytest
import torch
import torch.nn.functional as F

from tinyfold.model.resfold.onestep import ModelOutput, ResFoldOneStep


def _tiny_model(confidence_head: bool, c_token: int = 32) -> ResFoldOneStep:
    """Small model that builds in <1s on CPU."""
    return ResFoldOneStep(
        c_token=c_token,
        trunk_layers=1,
        trunk_heads=2,
        denoiser_blocks=1,
        denoiser_heads=2,
        atom_head_layers=1,
        atom_head_heads=2,
        n_timesteps=10,
        confidence_head=confidence_head,
    )


def _toy_batch(B: int = 2, L: int = 8, device=None):
    """Minimal batch tensors compatible with forward_sigma."""
    device = device or torch.device("cpu")
    return {
        "x_t": torch.randn(B, L, 3, device=device),
        "aa_seq": torch.randint(0, 20, (B, L), device=device),
        "chain_ids": torch.zeros(B, L, dtype=torch.long, device=device),
        "res_idx": torch.arange(L, device=device).unsqueeze(0).expand(B, -1),
        "sigma": torch.full((B,), 0.5, device=device),
        "mask": torch.ones(B, L, dtype=torch.bool, device=device),
    }


class TestOneStepConfidence:
    """forward_sigma 3-tuple + optional head contract."""

    def test_onestep_returns_pred_lddt(self):
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=True)
        b = _toy_batch(B=2, L=8)
        out = model.forward_sigma(
            b["x_t"], b["aa_seq"], b["chain_ids"], b["res_idx"],
            b["sigma"], b["mask"],
        )
        assert isinstance(out, tuple) and len(out) == 3, (
            f"expected 3-tuple, got {type(out)} of length {len(out) if isinstance(out, tuple) else 'N/A'}"
        )
        centroid, atoms, pred_lddt = out
        assert centroid.shape == (2, 8, 3)
        assert atoms.shape == (2, 8, 4, 3)
        assert pred_lddt is not None
        assert pred_lddt.shape == (2,)
        assert torch.all((pred_lddt >= 0) & (pred_lddt <= 1))

    def test_onestep_no_head(self):
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=False)
        b = _toy_batch()
        out = model.forward_sigma(
            b["x_t"], b["aa_seq"], b["chain_ids"], b["res_idx"],
            b["sigma"], b["mask"],
        )
        assert isinstance(out, tuple) and len(out) == 3
        _, _, pred_lddt = out
        assert pred_lddt is None
        assert model.confidence_head is None

    def test_forward_sigma_with_trunk_returns_pred_lddt(self):
        """forward_sigma_with_trunk must share the 3-tuple contract."""
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=True)
        b = _toy_batch()
        trunk_tokens = model.get_trunk_tokens(
            b["aa_seq"], b["chain_ids"], b["res_idx"], b["mask"],
        )
        out = model.forward_sigma_with_trunk(
            b["x_t"], trunk_tokens, b["sigma"], b["mask"],
        )
        assert isinstance(out, tuple) and len(out) == 3
        _, _, pred_lddt = out
        assert pred_lddt is not None and pred_lddt.shape == (b["x_t"].shape[0],)

    def test_confidence_loss_gradient_reaches_denoiser(self):
        """Gradients from pred_lddt should reach both the head AND the denoiser.

        This is the "pool is differentiable" contract: the confidence head sees
        denoiser tokens, and we want the head's signal to shape those tokens.
        """
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=True)
        b = _toy_batch(B=4, L=10)
        out = model.forward_sigma(
            b["x_t"], b["aa_seq"], b["chain_ids"], b["res_idx"],
            b["sigma"], b["mask"],
        )
        _, _, pred_lddt = out
        target = torch.full_like(pred_lddt, 0.7)
        loss = F.smooth_l1_loss(pred_lddt, target)
        loss.backward()

        # Head params have grads.
        head_grads = [p.grad for p in model.confidence_head.parameters()]
        assert all(g is not None for g in head_grads)
        assert any(g.abs().sum() > 0 for g in head_grads)

        # Denoiser params have grads too (gradient flowed through the pool).
        denoiser_grads = [
            p.grad for p in model.diff_transformer.parameters()
            if p.grad is not None
        ]
        assert denoiser_grads, "no denoiser grads attached"
        assert any(g.abs().sum() > 0 for g in denoiser_grads), (
            "gradient did not flow back through the pool into the denoiser"
        )

    def test_forward_sigma_matches_with_trunk_path(self):
        """forward_sigma must equal (compute trunk) + forward_sigma_with_trunk.

        Guards the consolidation where forward_sigma delegates to the
        _with_trunk core after computing the trunk itself.
        """
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=True).eval()
        b = _toy_batch()
        full = model.forward_sigma(
            b["x_t"], b["aa_seq"], b["chain_ids"], b["res_idx"], b["sigma"], b["mask"],
        )
        trunk = model.get_trunk_tokens(b["aa_seq"], b["chain_ids"], b["res_idx"], b["mask"])
        split = model.forward_sigma_with_trunk(
            b["x_t"], trunk, b["sigma"], b["mask"],
            res_idx=b["res_idx"], chain_ids=b["chain_ids"],
        )
        for a, c in zip(full, split):
            assert torch.allclose(a, c, atol=1e-6)

    def test_forward_sigma_returns_named_modeloutput(self):
        """forward_sigma returns a ModelOutput NamedTuple: unpacks AND names.

        Back-compat: it is still a 3-tuple (unpacking + indexing keep working);
        forward-compat: it exposes .centroid_pred / .atoms_pred / .pred_lddt.
        """
        torch.manual_seed(0)
        model = _tiny_model(confidence_head=True)
        b = _toy_batch(B=2, L=8)
        out = model.forward_sigma(
            b["x_t"], b["aa_seq"], b["chain_ids"], b["res_idx"],
            b["sigma"], b["mask"],
        )
        assert isinstance(out, ModelOutput)
        assert isinstance(out, tuple) and len(out) == 3
        # Named accessors alias the positional slots.
        assert out.centroid_pred is out[0]
        assert out.atoms_pred is out[1]
        assert out.pred_lddt is out[2]
        # Still unpacks positionally like the old plain tuple.
        centroid, atoms, pred_lddt = out
        assert centroid is out.centroid_pred
        assert atoms is out.atoms_pred
        assert pred_lddt is out.pred_lddt

    def test_count_parameters_includes_confidence(self):
        """count_parameters must report the confidence head bucket."""
        # With head
        m_with = _tiny_model(confidence_head=True)
        pc = m_with.count_parameters()
        assert "confidence_head" in pc
        assert pc["confidence_head"] > 0
        assert pc["total"] == (
            pc["trunk"] + pc["denoiser"] + pc["atom_head"] + pc["confidence_head"]
        )
        # Without head
        m_without = _tiny_model(confidence_head=False)
        pc0 = m_without.count_parameters()
        assert pc0["confidence_head"] == 0
        assert pc0["total"] == pc0["trunk"] + pc0["denoiser"] + pc0["atom_head"]

    def test_count_parameters_includes_atom_diff(self):
        """With atom_diffusion=True the atom-diffusion head must be counted."""
        m = ResFoldOneStep(
            c_token=32,
            trunk_layers=1,
            trunk_heads=2,
            denoiser_blocks=1,
            denoiser_heads=2,
            atom_head_layers=1,
            atom_head_heads=2,
            n_timesteps=10,
            atom_diffusion=True,
        )
        pc = m.count_parameters()
        assert "atom_diff" in pc
        assert pc["atom_diff"] > 0
        # Total must include every bucket, atom_diff included.
        assert pc["total"] == (
            pc["trunk"] + pc["denoiser"] + pc["atom_head"]
            + pc["atom_diff"] + pc["confidence_head"]
        )
        # Disabled -> bucket present but zero, and total unchanged by it.
        m0 = _tiny_model(confidence_head=False)
        pc0 = m0.count_parameters()
        assert pc0["atom_diff"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
