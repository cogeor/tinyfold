"""Leakage-stratified read-out for per-target evaluation results.

An aggregate DockQ mean hides whether a model generalizes or memorizes. Scoring
``scale_6M_le200`` per-target and splitting by train/test cluster overlap gave:

    cluster-LEAKED (n=183): mean DockQ 0.251, 44.3% success, 25.1% medium
    cluster-CLEAN  (n= 17): mean DockQ 0.044,  5.9% success,  0.0% medium

and a clean monotone dose-response in how many near-duplicate training copies a
test complex had:

    0 same-cluster train examples -> DockQ 0.044
    1-4                          -> 0.171
    5-19                         -> 0.182
    20+                          -> 0.353

The aggregate (0.233) was reported for months and is a memorization artifact.
Every evaluation should print these two tables, even when the split is believed
clean -- a flat redundancy profile is the evidence that it is.
"""

from __future__ import annotations

from collections.abc import Sequence

# Buckets over "how many training complexes share this test complex's cluster".
REDUNDANCY_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 0, "0"),
    (1, 4, "1-4"),
    (5, 19, "5-19"),
    (20, 10**9, "20+"),
)


def redundancy_bucket(n_same_cluster_train: int) -> str:
    """Label the redundancy bucket for a test complex."""
    for lo, hi, label in REDUNDANCY_BUCKETS:
        if lo <= n_same_cluster_train <= hi:
            return label
    return REDUNDANCY_BUCKETS[-1][2]


def train_cluster_counts(
    train_ids: Sequence[str],
    clusters: dict[str, int],
) -> dict[int, int]:
    """How many training samples each cluster contributes."""
    counts: dict[int, int] = {}
    for sid in train_ids:
        cid = clusters.get(sid)
        if cid is not None:
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def annotate_leakage(
    rows: list[dict],
    train_ids: Sequence[str],
    clusters: dict[str, int],
) -> list[dict]:
    """Add ``cluster``, ``n_same_cluster_train``, ``cluster_leaked`` to each row.

    ``rows`` must each carry a ``sample_id``. Mutates and returns ``rows``.
    """
    counts = train_cluster_counts(train_ids, clusters)
    for r in rows:
        cid = clusters.get(r["sample_id"])
        n_same = counts.get(cid, 0) if cid is not None else 0
        r["cluster"] = cid
        r["n_same_cluster_train"] = n_same
        r["cluster_leaked"] = n_same > 0
    return rows


def _summarise(dockqs: Sequence[float]) -> dict:
    n = len(dockqs)
    if n == 0:
        return {"n": 0, "mean_dockq": float("nan"), "succ": float("nan"),
                "medium_plus": float("nan")}
    return {
        "n": n,
        "mean_dockq": sum(dockqs) / n,
        "succ": 100.0 * sum(1 for d in dockqs if d >= 0.23) / n,
        "medium_plus": 100.0 * sum(1 for d in dockqs if d >= 0.49) / n,
    }


def leakage_strata(rows: Sequence[dict]) -> dict:
    """Summarise scored rows split by ``cluster_leaked``."""
    scored = [r for r in rows if r.get("dockq") is not None]
    return {
        "leaked": _summarise([r["dockq"] for r in scored if r["cluster_leaked"]]),
        "clean": _summarise([r["dockq"] for r in scored if not r["cluster_leaked"]]),
    }


def redundancy_profile(rows: Sequence[dict]) -> dict[str, dict]:
    """Summarise scored rows bucketed by training redundancy."""
    scored = [r for r in rows if r.get("dockq") is not None]
    out: dict[str, dict] = {}
    for _lo, _hi, label in REDUNDANCY_BUCKETS:
        vals = [r["dockq"] for r in scored
                if redundancy_bucket(r["n_same_cluster_train"]) == label]
        out[label] = _summarise(vals)
    return out


def format_report(rows: Sequence[dict], title: str = "") -> str:
    """Render both tables as text. Safe to print after any evaluation."""
    lines = []
    if title:
        lines.append(f"=== {title} ===")

    strata = leakage_strata(rows)
    lines.append("  leakage strata (test complexes sharing a cluster with train):")
    lines.append(f"    {'stratum':8s} {'n':>5s} {'meanDockQ':>10s} {'succ':>7s} {'medium+':>8s}")
    for name in ("leaked", "clean"):
        s = strata[name]
        if s["n"] == 0:
            lines.append(f"    {name:8s} {0:5d} {'-':>10s} {'-':>7s} {'-':>8s}")
        else:
            lines.append(
                f"    {name:8s} {s['n']:5d} {s['mean_dockq']:10.4f} "
                f"{s['succ']:6.1f}% {s['medium_plus']:7.1f}%"
            )

    prof = redundancy_profile(rows)
    lines.append("  redundancy profile (# same-cluster training complexes):")
    lines.append(f"    {'bucket':8s} {'n':>5s} {'meanDockQ':>10s} {'succ':>7s} {'medium+':>8s}")
    for _lo, _hi, label in REDUNDANCY_BUCKETS:
        s = prof[label]
        if s["n"] == 0:
            continue
        lines.append(
            f"    {label:8s} {s['n']:5d} {s['mean_dockq']:10.4f} "
            f"{s['succ']:6.1f}% {s['medium_plus']:7.1f}%"
        )
    if strata["clean"]["n"] == len(
        [r for r in rows if r.get("dockq") is not None]
    ) and strata["clean"]["n"] > 0:
        lines.append("  -> split is fully clean; the profile above is the evidence.")
    return "\n".join(lines)
