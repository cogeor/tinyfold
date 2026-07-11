"""C2: leakage-clean train/test split by interface cluster.

The legacy split (``data_split.py`` test_strategy="random") shuffles individual
samples, so homologous complexes leak across train/test and inflate DockQ. Here
we split by whole CLUSTER (from ``cluster_interfaces.py``): every sample of a
cluster goes entirely to train OR entirely to test, so no test complex has a
homolog in training.

Optionally, a set of ``forced_test_clusters`` (e.g. the clusters a public PINDER
test complex falls into, C3) is placed test-side first, so our held-out set is
comparable to that external yardstick.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa


def load_clusters(path: str) -> Dict[str, int]:
    """Load {sample_id: cluster_id} from a clusters.json produced by C1."""
    with open(path) as f:
        data = json.load(f)
    return data["sample_to_cluster"]


def _eligible_rows(table: pa.Table, min_atoms: int, max_atoms: Optional[int]) -> List[Tuple[int, str]]:
    """(row_idx, sample_id) for samples whose atom count is in range."""
    out = []
    atom_types = table["atom_type"]
    sample_ids = table["sample_id"]
    for i in range(len(table)):
        n_atoms = len(atom_types[i].as_py())
        if n_atoms < min_atoms:
            continue
        if max_atoms is not None and n_atoms > max_atoms:
            continue
        out.append((i, sample_ids[i].as_py()))
    return out


def cluster_holdout_indices(
    table: pa.Table,
    clusters: Dict[str, int],
    n_train: int,
    n_test: int,
    min_atoms: int = 0,
    max_atoms: Optional[int] = None,
    seed: int = 42,
    forced_test_clusters: Optional[Set[int]] = None,
) -> Tuple[List[int], List[int], dict]:
    """Split eligible samples into train/test with ZERO cluster overlap.

    Returns (train_indices, test_indices, info). Clusters are assigned wholesale
    to test until ``n_test`` samples are collected (forced-test clusters first),
    then the remaining clusters supply up to ``n_train`` training samples.
    """
    forced_test_clusters = set(forced_test_clusters or set())
    eligible = _eligible_rows(table, min_atoms, max_atoms)
    # Group eligible samples by cluster id (unknown samples -> own singleton).
    by_cluster: Dict[int, List[int]] = {}
    next_synth = -1
    for idx, sid in eligible:
        cid = clusters.get(sid)
        if cid is None:
            cid = next_synth
            next_synth -= 1
        by_cluster.setdefault(cid, []).append(idx)

    all_cids = sorted(by_cluster.keys())
    rng = random.Random(seed)
    rng.shuffle(all_cids)
    # Forced-test clusters go to the front of the test-assignment order.
    forced = [c for c in all_cids if c in forced_test_clusters]
    rest = [c for c in all_cids if c not in forced_test_clusters]
    ordered = forced + rest

    test_idx: List[int] = []
    test_clusters: Set[int] = set()
    i = 0
    while i < len(ordered) and len(test_idx) < n_test:
        c = ordered[i]
        test_idx.extend(by_cluster[c])
        test_clusters.add(c)
        i += 1
    # Remaining clusters -> train pool.
    train_idx: List[int] = []
    for c in ordered[i:]:
        train_idx.extend(by_cluster[c])

    # Deterministic trim to requested sizes.
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    test_idx = test_idx[:n_test]
    train_idx = train_idx[:n_train]

    # Sanity: no cluster spans both sides (by construction whole clusters are
    # test OR train; the trim only drops samples, never moves a cluster).
    train_clusters = {clusters.get(table["sample_id"][ix].as_py(), None) for ix in train_idx}
    overlap = (train_clusters & test_clusters) - {None}

    info = {
        "n_eligible": len(eligible),
        "n_clusters_eligible": len(by_cluster),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_clusters": len(test_clusters),
        "n_forced_test": len(forced),
        "cluster_overlap": sorted(overlap),
        "seed": seed,
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
    }
    return train_idx, test_idx, info


def save_split(table, train_idx, test_idx, info, path: str) -> None:
    """Write a split JSON compatible with train_resfold --load_split.

    Includes the ``*_atom_range`` fields that ``data_split.load_split`` prints,
    so the file drops into the existing loader unchanged.
    """
    def atom_range(idxs):
        counts = [len(table["atom_type"][i].as_py()) for i in idxs]
        return [min(counts), max(counts)] if counts else [0, 0]

    data = {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_ids": [table["sample_id"][i].as_py() for i in train_idx],
        "test_ids": [table["sample_id"][i].as_py() for i in test_idx],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_atom_range": atom_range(train_idx),
        "test_atom_range": atom_range(test_idx),
        "strategy": "cluster",
        "info": info,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
