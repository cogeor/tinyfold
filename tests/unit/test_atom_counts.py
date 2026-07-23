"""Tests for the vectorised atom-count path (B2).

Eligibility filtering used to call ``len(table['atom_type'][i].as_py())`` per
row, decoding a list of up to 6,000 elements. Measured on the real parquet:
1.345 ms/row, i.e. 56 s for one full-table scan -- performed on several code
paths per run. Samples are backbone-only (N, CA, C, O), so the count is exactly
``4 * (LA + LB)``, which reads two int64 columns instead.

These tests pin the invariant and prove the fast path selects the same rows the
materialising path did.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tinyfold.training.cluster_split import _eligible_rows
from tinyfold.training.data_split import (
    DataSplitConfig,
    atom_counts,
    get_eligible_samples,
    verify_atom_counts,
)

PARQUET = "data/processed/samples.parquet"


def make_table(sizes, *, atom_lists=None):
    """Synthetic table with ``sizes`` = [(LA, LB), ...].

    ``atom_type`` is generated consistently with LA/LB unless ``atom_lists`` is
    given, which lets a test deliberately violate the invariant.
    """
    la = [a for a, _ in sizes]
    lb = [b for _, b in sizes]
    if atom_lists is None:
        atom_lists = [list(range(4 * (a + b))) for a, b in sizes]
    return pa.table({
        "sample_id": [f"s{i:04d}" for i in range(len(sizes))],
        "LA": pa.array(la, type=pa.int64()),
        "LB": pa.array(lb, type=pa.int64()),
        "atom_type": pa.array(atom_lists, type=pa.list_(pa.int64())),
    })


def materialised_counts(table):
    """The pre-B2 implementation, kept as the reference oracle."""
    return [len(table["atom_type"][i].as_py()) for i in range(len(table))]


# --- the invariant ---------------------------------------------------------

def test_atom_counts_matches_materialised_on_synthetic_table():
    table = make_table([(10, 10), (1, 1), (300, 250), (40, 0)])
    assert list(atom_counts(table)) == materialised_counts(table)


def test_atom_counts_is_four_per_residue():
    table = make_table([(7, 13)])
    assert int(atom_counts(table)[0]) == 4 * 20


def test_atom_counts_returns_int64_not_float():
    # Downstream code indexes and compares these against int bounds; a float
    # dtype would silently work until a count exceeded 2**53 or a == became
    # inexact.
    assert atom_counts(make_table([(5, 5)])).dtype == np.int64


def test_missing_la_lb_raises_an_actionable_error():
    table = pa.table({"sample_id": ["s0"], "atom_type": [[0, 1, 2, 3]]})
    with pytest.raises(ValueError, match=r"canonical samples\.parquet schema"):
        atom_counts(table)


def test_verify_atom_counts_passes_on_consistent_table():
    verify_atom_counts(make_table([(10, 10), (20, 5)]))


def test_verify_atom_counts_raises_on_violation():
    # A row whose atom_type list disagrees with LA+LB -- what a parquet rebuild
    # that changed the atom set (e.g. adding CB) would look like.
    table = make_table([(10, 10), (10, 10)],
                       atom_lists=[list(range(80)), list(range(100))])
    with pytest.raises(ValueError, match="invariant"):
        verify_atom_counts(table)


def test_verify_atom_counts_honours_index_subset():
    table = make_table([(10, 10), (10, 10)],
                       atom_lists=[list(range(80)), list(range(100))])
    verify_atom_counts(table, indices=[0])          # clean row only
    with pytest.raises(ValueError):
        verify_atom_counts(table, indices=[1])


# --- eligibility selection is unchanged ------------------------------------

@pytest.mark.parametrize("lo,hi", [(0, 10**9), (80, 80), (81, 10**9), (0, 79)])
def test_cluster_eligible_rows_match_materialised_filter(lo, hi):
    sizes = [(10, 10), (1, 1), (300, 250), (40, 0), (20, 20)]
    table = make_table(sizes)
    counts = materialised_counts(table)
    expected = [(i, f"s{i:04d}") for i, n in enumerate(counts) if lo <= n <= hi]
    assert _eligible_rows(table, min_atoms=lo, max_atoms=hi) == expected


def test_cluster_eligible_rows_unbounded_max():
    table = make_table([(10, 10), (300, 300)])
    assert _eligible_rows(table, min_atoms=0, max_atoms=None) == [
        (0, "s0000"), (1, "s0001")
    ]


def test_get_eligible_samples_matches_materialised_filter():
    table = make_table([(10, 10), (1, 1), (300, 250), (40, 0), (20, 20)])
    cfg = DataSplitConfig(n_train=2, n_test=1, min_atoms=80, max_atoms=200)
    counts = materialised_counts(table)
    expected = sorted(
        [(i, f"s{i:04d}", n) for i, n in enumerate(counts) if 80 <= n <= 200],
        key=lambda x: x[1],
    )
    assert get_eligible_samples(table, cfg) == expected


def test_get_eligible_samples_select_smallest_orders_by_count():
    table = make_table([(300, 250), (1, 1), (20, 20)])
    cfg = DataSplitConfig(n_train=1, n_test=1, select_smallest=True)
    out = get_eligible_samples(table, cfg)
    # Two smallest are rows 1 (8 atoms) and 2 (160 atoms), returned sorted by id.
    assert out == [(1, "s0001", 8), (2, "s0002", 160)]


def test_eligible_row_indices_are_python_ints():
    # numpy int64 indices survive most use but break json.dump when a split is
    # written -- the failure would only show up at save_split time.
    rows = _eligible_rows(make_table([(10, 10)]), min_atoms=0, max_atoms=None)
    assert type(rows[0][0]) is int
    out = get_eligible_samples(make_table([(10, 10)]),
                               DataSplitConfig(n_train=1, n_test=1,
                                               min_atoms=0, max_atoms=10**9))
    assert type(out[0][0]) is int and type(out[0][2]) is int


# --- against the real table -------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
def test_invariant_holds_on_the_full_parquet():
    """4*(LA+LB) == len(atom_type) on all 41,883 rows. ~60 s."""
    table = pq.read_table(PARQUET)
    verify_atom_counts(table)
