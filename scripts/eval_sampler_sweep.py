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

import pyarrow.parquet as pq
import torch

from tinyfold.inference.build import load_onestep_run
from tinyfold.inference.samplers import sample_k_centroids
from tinyfold.model.diffusion import KarrasSchedule, VENoiser
from tinyfold.model.losses import compute_rmse
from tinyfold.model.metrics import compute_dockq
from tinyfold.model.metrics.cluster import (
    score_geometric_energy,
    score_self_consistency,
)
from tinyfold.retrieval import make_template_inputs
from tinyfold.training import collate_batch, load_sample_raw


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
    ap.add_argument("--template-source", default=None,
                    help="Override template source (else read from cfg). "
                         "oracle/oracle_monomer/retrieved/none.")
    ap.add_argument("--template-cache-dir", default=None,
                    help="Template npz dir for --template-source retrieved.")
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
    tsource = args.template_source or cfg.get("template_source", "none")
    tcache = args.template_cache_dir or cfg.get("template_cache_dir")
    tcache = tcache if tsource == "retrieved" else None
    samples = []
    for sid in test_ids:
        row = id_to_row[sid]
        s = load_sample_raw(table, row, normalize=True, esm_cache_dir=esm_dir,
                            per_chain_res_idx=per_chain, global_scale=gscale,
                            template_cache_dir=tcache)
        s["iface_mask"] = torch.tensor(table["iface_mask"][row].as_py(), dtype=torch.bool)
        samples.append(s)
    print(f"Loaded {len(samples)} test targets; grid T={T_list} x K={K_list}; "
          f"template_source={tsource}")

    # Rankers compared per cell (all training-free except confidence).
    RANKERS = ["conf", "consist", "energy", "oracle"]
    rows = []
    for T in T_list:
        noiser = _noiser_for_T(T, cfg, device)
        one_shot = (T == 1)
        acc = {K: {f"dq_{r}": [] for r in RANKERS} for K in K_list}
        for K in K_list:
            acc[K]["rmse_oracle"] = []
            acc[K]["consist"] = []
        for ti, s in enumerate(samples):
            batch = collate_batch([s], device)
            tc, tm, tf = make_template_inputs(batch, source=tsource)
            if tc is not None:
                batch["template_coords_res"] = tc
                batch["template_mask"] = tm
                batch["template_frame_id"] = tf
            with torch.no_grad():
                sc, sa, slddt = sample_k_centroids(
                    model, batch, noiser, device, K=Kmax, base_seed=args.seed,
                    target_idx=ti, is_onestep=True, one_shot=one_shot,
                    self_cond=args.self_cond,
                )
            n_res = s["n_res"]
            gt_c = batch["centroids"]
            std = s["std"]
            iface = s["iface_mask"].to(device)
            chain = batch["chain_ids"][0, :n_res]
            valid = batch["mask_res"][0, :n_res]
            rmses = [compute_rmse(sc[k], gt_c, batch["mask_res"]).item() * std for k in range(Kmax)]
            confs = [float(slddt[k]) if slddt is not None else 0.0 for k in range(Kmax)]

            def dockq_of(idx, _cache={}):  # noqa: B006  (intentional memoization cache)
                if idx not in _cache:
                    _cache[idx] = compute_dockq(
                        sa[idx][0, :n_res], batch["coords_res"][0, :n_res],
                        batch["aa_seq"][0, :n_res], chain, std=std)["dockq"]
                return _cache[idx]

            for K in K_list:
                sub_r = rmses[:K]
                # per-ranker pick
                pick = {}
                pick["conf"] = max(range(K), key=lambda i: confs[i]) if slddt is not None else 0
                cons_scores = score_self_consistency(sc[:K, 0, :n_res], iface)
                pick["consist"] = int(torch.argmin(cons_scores).item()) if K > 1 else 0
                en_scores = score_geometric_energy(sa[:K, 0, :n_res], chain, valid)
                pick["energy"] = int(torch.argmin(en_scores).item()) if K > 1 else 0
                pick["oracle"] = min(range(K), key=lambda i: sub_r[i])
                a = acc[K]
                for r in RANKERS:
                    dq = dockq_of(pick[r])
                    if dq is not None:
                        a[f"dq_{r}"].append(dq)
                a["rmse_oracle"].append(sub_r[pick["oracle"]])
                a["consist"].append(_consistency(sc, K, std))
        for K in K_list:
            a = acc[K]
            mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
            row = {"T": T, "K": K, "n": len(samples)}
            for r in RANKERS:
                row[f"dockq_{r}"] = round(mean(a[f"dq_{r}"]), 4)
            row["rmse_oracle"] = round(mean(a["rmse_oracle"]), 3)
            row["consistency_A"] = round(mean(a["consist"]), 3)
            rows.append(row)
            print(f"  T={T:>2} K={K:>2} | DockQ conf {row['dockq_conf']:.3f} "
                  f"consist {row['dockq_consist']:.3f} energy {row['dockq_energy']:.3f} "
                  f"| oracle {row['dockq_oracle']:.3f} | consistency {row['consistency_A']:.2f} A")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
