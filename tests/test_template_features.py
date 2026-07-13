"""C5 tests: AF3-style relative template pair features.

Verifies the done-when criteria from the SPEC: correct shapes, coverage mask,
and rotation/translation invariance (the whole point of RELATIVE features).
"""

import torch

from tinyfold.retrieval import (
    TEMPLATE_FEAT_DIM,
    build_template_pair_features,
    template_feat_dim,
)


def _random_backbone(B, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Plausible backbone: CA on a wiggly chain, N/C offset from CA.
    ca = torch.cumsum(torch.randn(B, L, 3, generator=g) * 0.3, dim=1)
    n = ca + torch.randn(B, L, 3, generator=g) * 0.1 + torch.tensor([0.1, 0.0, 0.0])
    c = ca + torch.randn(B, L, 3, generator=g) * 0.1 + torch.tensor([0.0, 0.1, 0.0])
    o = ca + torch.randn(B, L, 3, generator=g) * 0.1
    return torch.stack([n, ca, c, o], dim=2)  # [B, L, 4, 3]


def test_shapes_and_dim():
    B, L, nrbf = 2, 7, 16
    coords = _random_backbone(B, L)
    mask = torch.ones(B, L, dtype=torch.bool)
    frame = torch.zeros(B, L, dtype=torch.long)
    feats, valid = build_template_pair_features(coords, mask, frame, n_rbf=nrbf)
    assert feats.shape == (B, L, L, template_feat_dim(nrbf))
    assert valid.shape == (B, L, L)
    assert TEMPLATE_FEAT_DIM == template_feat_dim(32)


def test_coverage_mask_zeros_invalid_pairs():
    B, L = 1, 6
    coords = _random_backbone(B, L)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 3:] = False  # last three residues uncovered
    frame = torch.zeros(B, L, dtype=torch.long)
    feats, valid = build_template_pair_features(coords, mask, frame)
    # Pairs touching an uncovered residue must be invalid and zeroed.
    assert not valid[0, 0, 4]
    assert torch.count_nonzero(feats[0, 0, 4]) == 0
    # A fully covered pair is valid and non-zero.
    assert valid[0, 0, 1]
    assert torch.count_nonzero(feats[0, 0, 1]) > 0


def test_frame_id_blocks_cross_frame_pairs():
    B, L = 1, 8
    coords = _random_backbone(B, L)
    mask = torch.ones(B, L, dtype=torch.bool)
    frame = torch.zeros(B, L, dtype=torch.long)
    frame[0, 4:] = 1  # two chains / frames
    feats, valid = build_template_pair_features(coords, mask, frame)
    assert valid[0, 0, 3]        # same frame -> valid
    assert not valid[0, 0, 5]    # cross frame -> invalid
    assert torch.count_nonzero(feats[0, 0, 5]) == 0


def test_rotation_translation_invariance():
    B, L = 2, 10
    coords = _random_backbone(B, L, seed=3)
    mask = torch.ones(B, L, dtype=torch.bool)
    frame = torch.zeros(B, L, dtype=torch.long)
    feats0, _ = build_template_pair_features(coords, mask, frame)

    # Random rotation + translation applied to the whole structure.
    g = torch.Generator().manual_seed(9)
    a = torch.randn(3, generator=g)
    # Rodrigues-free: build a rotation via QR of a random matrix.
    M = torch.randn(3, 3, generator=g)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diagonal(R))  # proper-ish; det may be -1
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, generator=g) * 5.0
    coords_rt = torch.einsum("ij,blaj->blai", Q, coords) + t
    feats1, _ = build_template_pair_features(coords_rt, mask, frame)

    assert torch.allclose(feats0, feats1, atol=1e-4), (
        f"relative features not invariant: max diff {(feats0 - feats1).abs().max().item():.2e}"
    )
