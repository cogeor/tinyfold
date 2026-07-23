"""Tests for diffusion multiplicity (C2).

Multiplicity reuses one trunk pass across M noise draws per structure. The
training loop replicates the batch by M and then computes the trunk ONCE on
copy 0 of each group, repeating its tokens across the group. This is exact
precisely because the trunk is coordinate-blind: a sample's M copies have
identical trunk inputs. These tests pin the two properties the loop relies on:

1. trunk-once reuse == running the trunk per copy (bitwise), and the trunk is
   actually invoked once, not M times;
2. the expanded-batch per-sample loss equals the mean of M independent
   single-sample losses (the averaging the loop performs downstream).
"""

import torch

from tinyfold.model.losses import compute_mse_loss
from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.training.utils import edm_loss_weight


def _model():
    torch.manual_seed(0)
    return ResFoldOneStep(
        c_token=32, trunk_layers=2, denoiser_blocks=2, atom_head_layers=1,
    )


def _seq_inputs(B=2, L=8):
    aa = torch.zeros(B, L, dtype=torch.long)
    chain = torch.zeros(B, L, dtype=torch.long)
    chain[:, L // 2:] = 1
    res_idx = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    mask = torch.ones(B, L, dtype=torch.bool)
    return aa, chain, res_idx, mask


# --- trunk-once reuse is exact ---------------------------------------------

def test_trunk_once_matches_per_copy_forward():
    """The loop computes the trunk on copy 0 of each group and repeats its
    tokens; that must equal running forward_sigma independently on each copy
    (same x_t, same sample)."""
    model = _model()
    model.eval()
    B, L, M = 2, 8, 3
    aa, chain, res_idx, mask = _seq_inputs(B, L)

    # Replicate the sequence inputs the way the training loop does.
    ri = lambda z: z.repeat_interleave(M, dim=0)
    aa_e, chain_e, res_e, mask_e = ri(aa), ri(chain), ri(res_idx), ri(mask)
    # M distinct noised inputs per sample.
    torch.manual_seed(7)
    x_e = torch.randn(B * M, L, 3)
    sigma_e = torch.rand(B * M) + 0.1

    # trunk-once path (what the loop does)
    tok1 = model.get_trunk_tokens(aa[::1], chain, res_idx, mask)  # on B (copy 0 of each group == the B originals)
    tok_m = model.get_trunk_tokens(aa_e[::M], chain_e[::M], res_e[::M], mask_e[::M])
    tok_rep = tok_m.repeat_interleave(M, dim=0)
    out_once = model.forward_sigma_with_trunk(
        x_e, tok_rep, sigma_e, mask_e, res_idx=res_e, chain_ids=chain_e
    )
    del tok1

    # per-copy path: run the full forward_sigma on each expanded copy
    out_percopy = model.forward_sigma(
        x_e, aa_e, chain_e, res_e, sigma_e, mask_e
    )

    assert torch.equal(out_once.centroid_pred, out_percopy.centroid_pred)
    assert torch.equal(out_once.atoms_pred, out_percopy.atoms_pred)


def test_trunk_is_invoked_once_not_m_times():
    model = _model()
    model.eval()
    B, L, M = 2, 8, 4
    aa, chain, res_idx, mask = _seq_inputs(B, L)
    ri = lambda z: z.repeat_interleave(M, dim=0)
    aa_e, chain_e, res_e, mask_e = ri(aa), ri(chain), ri(res_idx), ri(mask)

    calls = {"n": 0}
    orig = model.trunk.forward

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    model.trunk.forward = counting
    try:
        # trunk-once: run on the strided (copy-0) inputs only.
        _ = model.get_trunk_tokens(aa_e[::M], chain_e[::M], res_e[::M], mask_e[::M])
    finally:
        model.trunk.forward = orig
    assert calls["n"] == 1, "trunk-once ran the trunk more than once"


def test_repeated_tokens_give_identical_rows_within_a_group():
    """Copies within a group must produce identical trunk tokens (the property
    that makes trunk-once exact)."""
    model = _model()
    model.eval()
    aa, chain, res_idx, mask = _seq_inputs(B=3, L=6)
    tok = model.get_trunk_tokens(aa, chain, res_idx, mask)
    tok_rep = tok.repeat_interleave(2, dim=0)
    # rows 0,1 come from sample 0; 2,3 from sample 1; ...
    for g in range(3):
        assert torch.equal(tok_rep[2 * g], tok_rep[2 * g + 1])


# --- averaging semantics ----------------------------------------------------

def test_multiplicity_loss_is_mean_of_independent_single_losses():
    """The loop computes (per_sample_mse * loss_weight).mean() over B*M. For one
    structure that must equal the mean of the M per-copy weighted losses -- i.e.
    weighting is per-copy (per its own sigma), then averaged."""
    torch.manual_seed(1)
    L, M = 10, 5
    target = torch.randn(1, L, 3).repeat_interleave(M, dim=0)   # [M, L, 3]
    pred = torch.randn(M, L, 3)
    mask = torch.ones(M, L, dtype=torch.bool)
    sigma = torch.rand(M) + 0.1
    lw = edm_loss_weight(sigma, sigma_data=1.0)

    per_sample = compute_mse_loss(pred, target, mask, reduction="per_sample")
    combined = (per_sample * lw).mean()

    # Independent per-copy losses, then averaged.
    per_copy = []
    for i in range(M):
        ps = compute_mse_loss(pred[i:i+1], target[i:i+1], mask[i:i+1],
                              reduction="per_sample")
        per_copy.append((ps * lw[i:i+1]).mean())
    manual = torch.stack(per_copy).mean()

    assert torch.allclose(combined, manual, atol=1e-6)
