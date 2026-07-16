"""Coevolution pair features: mutual information + APC correction.

This is the signal the whole Phase-2 bet rests on, so the tests assert the
STATISTICS (does perfect covariation actually light up? do independent columns
stay dark?) rather than only shapes -- a features bug is silent downstream.

Design: notes/2026-07-14-msa-coevolution-pair-prior-SPEC.md §5, feature source (a)
"APC-corrected covariance / DCA couplings from the paired MSA".
"""

import math

import numpy as np
import pytest

from tinyfold.msa.features import (
    MSA_FEAT_DIM,
    build_msa_pair_features,
    coevolution_features,
    msa_feat_dim,
)

# 8 rows, L=4. Columns 0 and 3 covary PERFECTLY (A<->D, C<->E).
# Columns 1 and 2 cycle independently of column 0: each of K/L/M/N appears once
# with A and once with C, so MI(0, 1) is exactly zero.
#   chain A = columns 0,1   chain B = columns 2,3
COVARY = [
    "AKTD",
    "ALVD",
    "AMWD",
    "ANYD",
    "CKTE",
    "CLVE",
    "CMWE",
    "CNYE",
]


def mi_raw(feats):
    return feats[..., 1]


def mi_apc(feats):
    return feats[..., 0]


class TestShapeAndDims:
    def test_shape_is_L_L_F(self):
        f = coevolution_features(COVARY, reweight=False)
        assert f.shape == (4, 4, MSA_FEAT_DIM)

    def test_feat_dim_helper_agrees(self):
        assert msa_feat_dim() == MSA_FEAT_DIM
        assert coevolution_features(COVARY, reweight=False).shape[-1] == msa_feat_dim()

    def test_dtype_is_float32(self):
        assert coevolution_features(COVARY, reweight=False).dtype == np.float32

    def test_finite_everywhere(self):
        f = coevolution_features(COVARY, reweight=False)
        assert np.isfinite(f).all()


class TestMutualInformationSemantics:
    """The load-bearing assertions: does coevolution actually register?"""

    def test_perfectly_covarying_columns_have_high_mi(self):
        f = coevolution_features(COVARY, reweight=False)
        # Two equiprobable states, perfectly coupled => MI = ln(2) nats.
        assert mi_raw(f)[0, 3] == pytest.approx(math.log(2), abs=1e-5)

    def test_independent_columns_have_zero_mi(self):
        f = coevolution_features(COVARY, reweight=False)
        assert mi_raw(f)[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_covariation_beats_independence_after_apc(self):
        # APC is what makes the score usable as a contact prior.
        f = coevolution_features(COVARY, reweight=False)
        assert mi_apc(f)[0, 3] > mi_apc(f)[0, 1]

    def test_conserved_column_carries_no_information(self):
        # A column with no variation cannot coevolve with anything.
        seqs = ["AK", "AL", "AM", "AN"]
        f = coevolution_features(seqs, reweight=False)
        assert mi_raw(f)[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_single_sequence_msa_has_no_signal(self):
        # Depth 1 => nothing covaries. Must not NaN.
        f = coevolution_features(["AKTD"], reweight=False)
        assert np.isfinite(f).all()
        assert mi_raw(f).max() == pytest.approx(0.0, abs=1e-6)


class TestSymmetryAndDiagonal:
    def test_mi_is_symmetric(self):
        f = coevolution_features(COVARY, reweight=False)
        np.testing.assert_allclose(mi_raw(f), mi_raw(f).T, atol=1e-6)

    def test_apc_is_symmetric(self):
        f = coevolution_features(COVARY, reweight=False)
        np.testing.assert_allclose(mi_apc(f), mi_apc(f).T, atol=1e-6)

    def test_diagonal_is_zeroed(self):
        # A column is trivially perfectly informative about itself; that is not
        # a contact signal and would dominate the APC background.
        f = coevolution_features(COVARY, reweight=False)
        np.testing.assert_allclose(np.diag(mi_raw(f)), 0.0, atol=1e-6)


class TestChunking:
    def test_chunking_does_not_change_the_result(self):
        # The [L,L,21,21] joint is chunked over i to bound memory; the result
        # must not depend on the chunk size.
        a = coevolution_features(COVARY, reweight=False, chunk_size=1)
        b = coevolution_features(COVARY, reweight=False, chunk_size=1024)
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestCoverageAndDepthChannels:
    def test_coverage_is_one_when_no_gaps(self):
        f = coevolution_features(COVARY, reweight=False)
        assert f[0, 3, 2] == pytest.approx(1.0)

    def test_coverage_drops_with_gaps(self):
        # Half the rows are gapped at column 0.
        seqs = ["AKTD", "-KTD", "CKTE", "-KTE"]
        f = coevolution_features(seqs, reweight=False)
        assert f[0, 3, 2] == pytest.approx(0.5)

    def test_depth_channel_is_constant_across_pairs(self):
        # A per-complex scalar (log Neff) broadcast so a linear proj can learn
        # to discount shallow-MSA targets.
        f = coevolution_features(COVARY, reweight=False)
        assert np.allclose(f[..., 3], f[0, 0, 3])

    def test_depth_channel_grows_with_depth(self):
        shallow = coevolution_features(COVARY[:2], reweight=False)[0, 0, 3]
        deep = coevolution_features(COVARY, reweight=False)[0, 0, 3]
        assert deep > shallow


class TestReweighting:
    """Redundancy reweighting makes statistics reflect INDEPENDENT observations.

    Note it does not simply push MI down: a cluster of duplicates can just as
    easily be *drowning out* a distinct sequence, in which case reweighting
    raises MI. The invariant is about effective depth, not about MI's direction.
    """

    def test_duplicate_cluster_collapses_to_one_effective_sequence(self):
        from tinyfold.msa.features import encode_msa, sequence_weights

        w = sequence_weights(encode_msa(["AKTD"] * 100 + ["CKTE"]))
        # Each of the 100 duplicates is worth 1/100; the distinct row is worth 1.
        assert w[:100] == pytest.approx(0.01)
        assert w[100] == pytest.approx(1.0)
        assert w.sum() == pytest.approx(2.0)  # Meff = 2, not 101

    def test_reported_depth_discounts_redundancy(self):
        redundant = ["AKTD"] * 100 + ["CKTE"]
        deep_looking = coevolution_features(redundant, reweight=False)[0, 0, 3]
        honest = coevolution_features(redundant, reweight=True)[0, 0, 3]
        # log1p(101) vs log1p(2) -- the depth channel must not be fooled.
        assert honest < deep_looking

    def test_distinct_sequences_are_not_downweighted(self):
        from tinyfold.msa.features import encode_msa, sequence_weights

        w = sequence_weights(encode_msa(COVARY))
        assert w == pytest.approx(np.ones(len(COVARY)))


class TestBuildFromPairedMsas:
    def test_concatenates_chains_into_one_L(self):
        from tinyfold.msa.a3m import MsaRecord

        a = [MsaRecord("qa", "AK"), MsaRecord("h", "CK", 1)]
        b = [MsaRecord("qb", "TD"), MsaRecord("h", "TE", 1)]
        f = build_msa_pair_features(a, b)
        assert f.shape == (4, 4, MSA_FEAT_DIM)

    def test_rejects_row_count_mismatch(self):
        from tinyfold.msa.a3m import MsaRecord

        a = [MsaRecord("qa", "AK"), MsaRecord("h", "CK", 1)]
        b = [MsaRecord("qb", "TD")]
        with pytest.raises(ValueError, match="same number of rows"):
            build_msa_pair_features(a, b)
