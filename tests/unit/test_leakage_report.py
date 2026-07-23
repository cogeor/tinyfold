"""Tests for the leakage-stratified evaluation read-out.

An aggregate DockQ mean cannot distinguish generalization from memorization.
Splitting scale_6M_le200's per-target scores by train/test cluster overlap gave
0.251 (leaked, n=183) vs 0.044 (clean, n=17), with a monotone dose-response in
how many near-duplicate training copies each test complex had. These tests pin
the instrument that produced those tables.
"""

import pytest

from tinyfold.model.metrics import capri_band
from tinyfold.training.leakage_report import (
    annotate_leakage,
    format_report,
    leakage_strata,
    redundancy_bucket,
    redundancy_profile,
    train_cluster_counts,
)

# --- CAPRI bands -----------------------------------------------------------

@pytest.mark.parametrize("dockq,expected", [
    (0.0, "incorrect"),
    (0.229, "incorrect"),
    (0.23, "acceptable"),
    (0.489, "acceptable"),
    (0.49, "medium"),
    (0.799, "medium"),
    (0.80, "high"),
    (1.0, "high"),
])
def test_capri_band_boundaries(dockq, expected):
    assert capri_band(dockq) == expected


def test_capri_band_passes_through_none():
    assert capri_band(None) is None


# --- annotation ------------------------------------------------------------

def test_train_cluster_counts_ignores_unclustered():
    counts = train_cluster_counts(["a", "b", "c", "ghost"], {"a": 1, "b": 1, "c": 2})
    assert counts == {1: 2, 2: 1}


def test_annotate_marks_leaked_and_clean():
    clusters = {"tr0": 1, "tr1": 1, "te0": 1, "te1": 2}
    rows = [{"sample_id": "te0", "dockq": 0.5}, {"sample_id": "te1", "dockq": 0.1}]
    annotate_leakage(rows, ["tr0", "tr1"], clusters)

    assert rows[0]["cluster_leaked"] is True
    assert rows[0]["n_same_cluster_train"] == 2
    assert rows[1]["cluster_leaked"] is False
    assert rows[1]["n_same_cluster_train"] == 0


def test_annotate_handles_unclustered_test_sample():
    rows = [{"sample_id": "unknown", "dockq": 0.3}]
    annotate_leakage(rows, ["tr0"], {"tr0": 1})
    assert rows[0]["cluster"] is None
    assert rows[0]["cluster_leaked"] is False


# --- strata ----------------------------------------------------------------

def test_strata_separate_leaked_from_clean():
    clusters = {"tr": 1, "hot": 1, "cold": 2}
    rows = [{"sample_id": "hot", "dockq": 0.8}, {"sample_id": "cold", "dockq": 0.0}]
    annotate_leakage(rows, ["tr"], clusters)
    s = leakage_strata(rows)

    assert s["leaked"]["n"] == 1
    assert s["leaked"]["mean_dockq"] == pytest.approx(0.8)
    assert s["clean"]["n"] == 1
    assert s["clean"]["mean_dockq"] == pytest.approx(0.0)


def test_strata_skip_unscored_rows():
    rows = [
        {"sample_id": "a", "dockq": None, "cluster_leaked": True, "n_same_cluster_train": 3},
        {"sample_id": "b", "dockq": 0.4, "cluster_leaked": True, "n_same_cluster_train": 3},
    ]
    assert leakage_strata(rows)["leaked"]["n"] == 1


def test_empty_stratum_is_reported_not_crashed():
    rows = [{"sample_id": "a", "dockq": 0.5, "cluster_leaked": True,
             "n_same_cluster_train": 2}]
    s = leakage_strata(rows)
    assert s["clean"]["n"] == 0
    assert "leakage strata" in format_report(rows)


# --- redundancy dose-response ---------------------------------------------

@pytest.mark.parametrize("n,label", [
    (0, "0"), (1, "1-4"), (4, "1-4"), (5, "5-19"), (19, "5-19"), (20, "20+"), (500, "20+"),
])
def test_redundancy_bucket_edges(n, label):
    assert redundancy_bucket(n) == label


def test_redundancy_profile_reproduces_the_measured_shape():
    """Regression fixture for the dose-response that exposed the artifact.

    Synthetic rows reproducing the measured monotone relationship between
    training redundancy and DockQ (0 -> 0.044, 1-4 -> 0.171, 5-19 -> 0.182,
    20+ -> 0.353). A flat profile is what a genuinely clean split looks like.
    """
    rows = []
    for n_same, dockq, count in [(0, 0.044, 17), (2, 0.171, 51),
                                 (10, 0.182, 55), (40, 0.353, 77)]:
        for i in range(count):
            rows.append({
                "sample_id": f"s{n_same}_{i}", "dockq": dockq,
                "n_same_cluster_train": n_same, "cluster_leaked": n_same > 0,
            })
    prof = redundancy_profile(rows)

    assert prof["0"]["n"] == 17
    assert prof["20+"]["n"] == 77
    means = [prof[k]["mean_dockq"] for k in ("0", "1-4", "5-19", "20+")]
    assert means == sorted(means), "DockQ must rise monotonically with redundancy"
    assert prof["20+"]["mean_dockq"] > 5 * prof["0"]["mean_dockq"]


def test_fully_clean_split_is_called_out():
    rows = [{"sample_id": f"s{i}", "dockq": 0.05, "cluster_leaked": False,
             "n_same_cluster_train": 0} for i in range(5)]
    assert "fully clean" in format_report(rows)


def test_report_renders_both_tables():
    rows = [
        {"sample_id": "a", "dockq": 0.6, "cluster_leaked": True, "n_same_cluster_train": 25},
        {"sample_id": "b", "dockq": 0.03, "cluster_leaked": False, "n_same_cluster_train": 0},
    ]
    text = format_report(rows, title="ckpt.pt")
    assert "ckpt.pt" in text
    assert "leakage strata" in text
    assert "redundancy profile" in text
