"""Join all benchmarks/results/{model}.csv files into one master table and
emit a size-vs-error cliff plot.

Output:
  benchmarks/results/cliff_comparison.csv  -- one row per (model, bin)
                                              with mean/median C-RMSD, DockQ, n
  benchmarks/results/cliff_comparison.png  -- C-RMSD by size bin, one line per model
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = [
    "le200", "200_400", "400_600", "600_1000", "ge1000",
    "clean_le200", "clean_200_400", "clean_400_600", "clean_600_1000", "clean_ge1000",
]


def _agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    median = (
        vals_sorted[mid] if len(vals_sorted) % 2
        else 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
    )
    return sum(vals) / len(vals), median


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    if s is None or s == "" or s == "None":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="benchmarks/results")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_plot", default=None)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    res_root = REPO_ROOT / args.results_root
    csvs = sorted(p_ for p_ in res_root.glob("*.csv") if p_.name != "cliff_comparison.csv")
    if not csvs:
        sys.exit(f"ERROR: no per-model CSVs in {res_root}")
    print(f"Joining {len(csvs)} per-model CSVs from {res_root}")

    master_rows: list[dict] = []
    for csv_path in csvs:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        model = rows[0].get("model", csv_path.stem)
        by_bin: dict[str, list[dict]] = {}
        for r in rows:
            by_bin.setdefault(r["bin"], []).append(r)
        for b in SPLITS:
            rs = by_bin.get(b, [])
            if not rs:
                continue
            c_vals = [_to_float(r["c_rmsd_aligned_A"]) for r in rs]
            d_vals = [_to_float(r["dockq"]) for r in rs]
            c_mean, c_med = _agg(c_vals)
            d_mean, d_med = _agg(d_vals)
            master_rows.append({
                "model": model,
                "bin": b,
                "n": len(rs),
                "c_rmsd_mean_A": round(c_mean, 3),
                "c_rmsd_median_A": round(c_med, 3),
                "dockq_mean": round(d_mean, 4),
                "dockq_median": round(d_med, 4),
            })

    out_csv = Path(args.out_csv) if args.out_csv else res_root / "cliff_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        w.writeheader()
        w.writerows(master_rows)
    print(f"Wrote {len(master_rows)} (model, bin) rows -> {out_csv}")

    print("\nMaster table:")
    print(f"  {'model':>30} {'bin':>10} {'n':>4} {'c_rmsd':>10} {'dockq':>8}")
    for r in master_rows:
        print(f"  {r['model']:>30} {r['bin']:>10} {r['n']:>4} "
              f"{r['c_rmsd_mean_A']:>10.2f} {r['dockq_mean']:>8.3f}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot.")
        return

    out_plot = Path(args.out_plot) if args.out_plot else res_root / "cliff_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted({r["model"] for r in master_rows})
    x_labels = SPLITS
    xs = list(range(len(x_labels)))
    for m in models:
        ys = []
        for b in x_labels:
            hit = [r for r in master_rows if r["model"] == m and r["bin"] == b]
            ys.append(hit[0]["c_rmsd_mean_A"] if hit else float("nan"))
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Complex size (LA + LB, residues)")
    ax.set_ylabel("Mean C-RMSD (Angstroms, Kabsch-aligned)")
    ax.set_title("Size-stratified PPI cliff: TinyFold vs baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=120)
    print(f"Wrote plot -> {out_plot}")


if __name__ == "__main__":
    main()
