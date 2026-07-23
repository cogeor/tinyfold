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

import numpy as np
import pyarrow as pa

from tinyfold.training.data_split import atom_counts


def load_clusters(path: str) -> dict[str, int]:
    """Load {sample_id: cluster_id} from a clusters.json produced by C1."""
    with open(path) as f:
        data = json.load(f)
    return data["sample_to_cluster"]


def _eligible_rows(table: pa.Table, min_atoms: int, max_atoms: int | None) -> list[tuple[int, str]]:
    """(row_idx, sample_id) for samples whose atom count is in range."""
    n_atoms = atom_counts(table)
    keep = n_atoms >= min_atoms
    if max_atoms is not None:
        keep &= n_atoms <= max_atoms
    rows = np.nonzero(keep)[0]
    sample_ids = table["sample_id"].take(pa.array(rows)).to_pylist()
    return [(int(i), sid) for i, sid in zip(rows, sample_ids)]


def cluster_holdout_indices(
    table: pa.Table,
    clusters: dict[str, int],
    n_train: int,
    n_test: int,
    min_atoms: int = 0,
    max_atoms: int | None = None,
    seed: int = 42,
    forced_test_clusters: set[int] | None = None,
    per_cluster_cap: int | None = None,
    n_test_clusters: int | None = None,
    train_cluster_uniform: bool = False,
) -> tuple[list[int], list[int], dict]:
    """Split eligible samples into train/test with ZERO cluster overlap.

    Returns (train_indices, test_indices, info). Clusters are assigned wholesale
    to test until ``n_test`` samples are collected (forced-test clusters first),
    then the remaining clusters supply up to ``n_train`` training samples.

    Cluster disjointness alone is NOT enough to make a test set informative.
    Without ``per_cluster_cap`` a single large cluster can supply most of the
    test set, so ``n_test`` samples carry only a handful of clusters' worth of
    independent information. Measured on the splits this function produced
    before the cap existed:

        clean_le240: 200 samples /  41 clusters (top-5 = 52% of test)
        clean_le400: 200 samples /  44 clusters (top-5 = 55% of test)
        clean_le600: 200 samples /  21 clusters (ONE cluster = 137 of 200)

    Args:
        per_cluster_cap: Max test samples drawn from any one cluster. ``None``
            (default) preserves the historical uncapped behaviour bit-for-bit.
            ``1`` gives one test sample per cluster, so ``n_test`` samples carry
            ``n_test`` independent clusters -- the recommended setting.
        n_test_clusters: Stop collecting test clusters once this many have been
            taken, instead of once ``n_test`` samples have been collected. Use
            with ``per_cluster_cap > 1``. The final trim to ``n_test`` still
            applies (it only drops samples, so the cap invariant survives).
        train_cluster_uniform: Also return ``info["train_weights"]``, a
            per-train-sample weight of ``1/|cluster in train|`` so every training
            cluster carries equal total mass. PINDER reports cluster-uniform
            training sampling is itself a generalization win. The weights are
            recorded, not applied -- the trainer opts in separately.
    """
    if per_cluster_cap is not None and per_cluster_cap < 1:
        raise ValueError(f"per_cluster_cap must be >= 1 or None, got {per_cluster_cap}")
    if n_test_clusters is not None and n_test_clusters < 1:
        raise ValueError(f"n_test_clusters must be >= 1 or None, got {n_test_clusters}")

    forced_test_clusters = set(forced_test_clusters or set())
    eligible = _eligible_rows(table, min_atoms, max_atoms)
    # Group eligible samples by cluster id (unknown samples -> own singleton).
    by_cluster: dict[int, list[int]] = {}
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

    # Subsampling within a cluster draws from a SEPARATE stream so that the
    # default (cap=None) path leaves `rng` in exactly the state the historical
    # implementation left it in -- otherwise adding the cap would silently
    # re-roll every previously generated split.
    cap_rng = random.Random(seed) if per_cluster_cap is not None else None

    def _enough_test(n_samples: int, n_clusters: int) -> bool:
        if n_test_clusters is not None:
            return n_clusters >= n_test_clusters
        return n_samples >= n_test

    test_idx: list[int] = []
    test_clusters: set[int] = set()
    test_cluster_sizes: list[int] = []
    i = 0
    while i < len(ordered) and not _enough_test(len(test_idx), len(test_clusters)):
        c = ordered[i]
        members = by_cluster[c]
        if per_cluster_cap is not None and len(members) > per_cluster_cap:
            members = sorted(cap_rng.sample(members, per_cluster_cap))
        test_idx.extend(members)
        test_clusters.add(c)
        test_cluster_sizes.append(len(members))
        i += 1
    # Remaining clusters -> train pool. Whole clusters only: a cluster that was
    # visited for test never contributes to train, even if the cap dropped some
    # of its members (dropped samples are discarded, not recycled).
    train_idx: list[int] = []
    for c in ordered[i:]:
        train_idx.extend(by_cluster[c])

    # Deterministic trim to requested sizes.
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    test_idx = test_idx[:n_test]
    train_idx = train_idx[:n_train]

    # Sanity: no cluster spans both sides (by construction whole clusters are
    # test OR train; the trim only drops samples, never moves a cluster).
    sample_ids = table["sample_id"]
    train_cids = [clusters.get(sample_ids[ix].as_py()) for ix in train_idx]
    train_clusters = set(train_cids)
    overlap = (train_clusters & test_clusters) - {None}

    # Recompute the realised test-cluster profile AFTER the trim, so the
    # reported share is the one the consumer actually sees.
    realised: dict[int, int] = {}
    for ix in test_idx:
        cid = clusters.get(sample_ids[ix].as_py())
        realised[cid] = realised.get(cid, 0) + 1
    realised_sizes = sorted(realised.values(), reverse=True)
    max_share = (max(realised_sizes) / len(test_idx)) if test_idx else 0.0

    info = {
        "n_eligible": len(eligible),
        "n_clusters_eligible": len(by_cluster),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_clusters": len(realised),
        "n_forced_test": len(forced),
        "cluster_overlap": sorted(overlap),
        "seed": seed,
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "per_cluster_cap": per_cluster_cap,
        "n_test_clusters_requested": n_test_clusters,
        "test_cluster_sizes": realised_sizes,
        "max_test_cluster_share": max_share,
        "n_train_clusters": len(train_clusters - {None}),
    }

    if train_cluster_uniform:
        counts: dict[int, int] = {}
        for cid in train_cids:
            counts[cid] = counts.get(cid, 0) + 1
        info["train_weights"] = [1.0 / counts[cid] for cid in train_cids]

    return train_idx, test_idx, info


def audit_split_leakage(
    train_ids: list[str],
    test_ids: list[str],
    clusters: dict[str, int],
) -> dict:
    """Measure how much of ``test_ids`` is covered by ``train_ids``' clusters.

    This is the instrument that made the le200 result interpretable. Scoring
    `scale_6M_le200` per-target and splitting by this flag gave:

        cluster-LEAKED (n=183): mean DockQ 0.251, 44.3% success, 25.1% medium
        cluster-CLEAN  (n= 17): mean DockQ 0.044,  5.9% success,  0.0% medium

    and DockQ rose monotonically with the number of same-cluster training
    examples (0 -> 0.044; 1-4 -> 0.171; 5-19 -> 0.182; 20+ -> 0.353). Any run
    reporting a headline number without this audit is not interpretable.

    ``n_test_unclustered`` matters: samples absent from clusters.json cannot be
    checked, so a large count silently weakens the guarantee.
    """
    train_clusters = {clusters[s] for s in train_ids if s in clusters}
    test_cluster_of = {s: clusters.get(s) for s in test_ids}

    leaked = [s for s, c in test_cluster_of.items() if c is not None and c in train_clusters]
    unclustered = [s for s, c in test_cluster_of.items() if c is None]

    sizes: dict[int, int] = {}
    for c in test_cluster_of.values():
        if c is not None:
            sizes[c] = sizes.get(c, 0) + 1
    n_test = len(test_ids)

    return {
        "n_train_clusters": len(train_clusters),
        "n_test_clusters": len(sizes),
        "n_test_leaked": len(leaked),
        "frac_test_leaked": (len(leaked) / n_test) if n_test else 0.0,
        "n_test_unclustered": len(unclustered),
        "max_test_cluster_share": (max(sizes.values()) / n_test) if (sizes and n_test) else 0.0,
        "leaked_test_ids": sorted(leaked)[:20],
    }


def save_cluster_split(table, train_idx, test_idx, info, path: str) -> None:
    """Write a cluster-holdout split JSON in the canonical data_split format.

    Includes the ``*_atom_range`` fields that ``data_split.load_split`` reads,
    so the file drops into the existing loader unchanged. Named distinctly from
    ``data_split.save_split`` (which has a different signature) to avoid the
    historical name collision.
    """
    n_atoms_all = atom_counts(table)

    def atom_range(idxs):
        counts = [int(n_atoms_all[i]) for i in idxs]
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
    # Surface the per-sample sampling weights at the top level, positionally
    # aligned with train_indices, so a trainer can consume them without
    # reaching into `info`.
    if "train_weights" in info:
        data["train_weights"] = info["train_weights"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
