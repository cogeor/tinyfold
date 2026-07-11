"""Sampler sweep: what does multi-step diffusion buy, vs #samples K?

For a trained 6M onestep checkpoint, evaluate a grid of
  steps T  in {1 (=one-shot), 2, 4, 8, 16, 32, 50}   (VE Euler for T>1)
  samples K in {1, 5, 20, 40}
on the test split, reporting per (T,K):
  * DockQ_ranked   — confidence-head-ranked pick, mean over targets
  * DockQ_best     — best-of-K (oracle), mean over targets
  * succ%          — fraction of ranked picks with DockQ >= 0.23
  * RMSE_ranked / RMSE_oracle  — centroid RMSE (A)
  * consistency    — mean RMSD (A) of the K samples to sample-0 (lower = the
                     K draws agree more; the thing multi-step should tighten)

Usage:
    uv run python scripts/eval_sampler_sweep.py \
        --checkpoint outputs/resfold/scale_6M_le200/<run>/best_model.pt \
        --split data/processed/splits/clean_le240.json \
        --T-list 1,2,4,8,16,32,50 --K-list 1,5,20,40 \
        --n-targets 100 --out benchmarks/results/sampler_sweep.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from tinyfold.inference.build import load_onestep_run  # noqa: E402
from tinyfold.inference.samplers import sample_k_centroids  # noqa: E402
from tinyfold.model.diffusion import KarrasSchedule, VENoiser  # noqa: E402
from tinyfold.model.metrics import compute_dockq, compute_rmse  # noqa: E402
from tinyfold.training import load_sample_raw, collate_batch  # noqa: E402


def _noiser_for_T(T, cfg, device):
    """VE noiser whose schedule has T Euler steps (T+1 sigma knots)."""
    sched = KarrasSchedule(
        n_steps=T,
        sigma_min=cfg.get("sigma_min", 0.002),
        sigma_max=cfg.get("sigma_max", 10.0),
        rho=7.0,
    )
    return VENoiser(sched, sigma_data=cfg.get("sigma_data", 1.0)).to(device)


def _consistency(samples_c, K, std):
    """Mean RMSD (A) of samples 1..K-1 to sample 0 (Kabsch-aligned)."""
    if K < 2:
        return 0.0
    ref = samples_c[0]
    ds = [compute_rmse(samples_c[i], ref).item() * std for i in range(1, K)]
    return sum(ds) / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--T-list", default="1,2,4,8,16,32,50")
    ap.add_argument("--K-list", default="1,5,20,40")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--self-cond", action="store_true",
                    help="Use self-conditioning in the VE sampler (set only for "
                         "models TRAINED with self_cond_prob>0).")
    ap.add_argument("--out", default="benchmarks/results/sampler_sweep.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_onestep_run(args.checkpoint, device)
    T_list = [int(x) for x in args.T_list.split(",")]
    K_list = sorted(int(x) for x in args.K_list.split(","))
    Kmax = max(K_list)

    # Test samples (reuse the run's data recipe).
    table = pq.read_table(args.parquet)
    sp = json.load(open(args.split))
    test_ids = sp["test_ids"][:args.n_targets]
    id_to_row = {table["sample_id"][i].as_py(): i for i in range(len(table))}
    esm_dir = cfg.get("esm_cache_dir") if cfg.get("aa_embed", "learned") != "learned" else None
    gscale = cfg.get("global_scale")
    per_chain = cfg.get("per_chain_res_idx", False)
    samples = []
    for sid in test_ids:
        row = id_to_row[sid]
        samples.append(load_sample_raw(table, row, normalize=True, esm_cache_dir=esm_dir,
                                       per_chain_res_idx=per_chain, global_scale=gscale))
    print(f"Loaded {len(samples)} test targets; grid T={T_list} x K={K_list}")

    rows = []
    for T in T_list:
        noiser = _noiser_for_T(T, cfg, device)
        one_shot = (T == 1)
        # accumulators keyed by K
        acc = {K: {"dq_rank": [], "dq_best": [], "succ": [], "rmse_rank": [],
                   "rmse_oracle": [], "consist": []} for K in K_list}
        for ti, s in enumerate(samples):
            batch = collate_batch([s], device)
            with torch.no_grad():
                sc, sa, slddt = sample_k_centroids(
                    model, batch, noiser, device, K=Kmax, base_seed=args.seed,
                    target_idx=ti, is_onestep=True, one_shot=one_shot,
                    self_cond=args.self_cond,
                )
            n_res = s["n_res"]
            gt_c = batch["centroids"]
            std = s["std"]
            # per-sample centroid RMSE + confidence
            rmses = [compute_rmse(sc[k], gt_c, batch["mask_res"]).item() * std for k in range(Kmax)]
            confs = [float(slddt[k]) if slddt is not None else 0.0 for k in range(Kmax)]
            for K in K_list:
                sub_r = rmses[:K]
                rank = max(range(K), key=lambda i: confs[i]) if slddt is not None else 0
                best = min(range(K), key=lambda i: sub_r[i])
                dq_rank = compute_dockq(sa[rank][0, :n_res], batch["coords_res"][0, :n_res],
                                        batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res],
                                        std=std)["dockq"]
                dq_best = compute_dockq(sa[best][0, :n_res], batch["coords_res"][0, :n_res],
                                        batch["aa_seq"][0, :n_res], batch["chain_ids"][0, :n_res],
                                        std=std)["dockq"]
                a = acc[K]
                if dq_rank is not None:
                    a["dq_rank"].append(dq_rank)
                    a["succ"].append(1.0 if dq_rank >= 0.23 else 0.0)
                if dq_best is not None:
                    a["dq_best"].append(dq_best)
                a["rmse_rank"].append(sub_r[rank])
                a["rmse_oracle"].append(sub_r[best])
                a["consist"].append(_consistency(sc, K, std))
        for K in K_list:
            a = acc[K]
            mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
            row = {
                "T": T, "K": K, "n": len(samples),
                "dockq_ranked": round(mean(a["dq_rank"]), 4),
                "dockq_best": round(mean(a["dq_best"]), 4),
                "succ_pct": round(100 * mean(a["succ"]), 1),
                "rmse_ranked": round(mean(a["rmse_rank"]), 3),
                "rmse_oracle": round(mean(a["rmse_oracle"]), 3),
                "consistency_A": round(mean(a["consist"]), 3),
            }
            rows.append(row)
            print(f"  T={T:>2} K={K:>2} | DockQ rank {row['dockq_ranked']:.3f} "
                  f"best {row['dockq_best']:.3f} succ {row['succ_pct']:.0f}% | "
                  f"RMSE rank {row['rmse_ranked']:.2f} oracle {row['rmse_oracle']:.2f} | "
                  f"consist {row['consistency_A']:.2f} A")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
