"""Run pretrained DiffDock-PP on a TinyFold split and write NPZ predictions.

Pipeline:
  1. adapt.py our parquet split -> per-sample PDB pairs in a DPP dataset dir.
  2. Patch the dips_esm_inference YAML to point at our dataset.
  3. Spawn `python src/main_inf.py` inside the DPP repo with --logger tensorboard.
     This runs 40-sample diffusion + confidence reranking and dumps a pickle.
  4. Load the pickle. Each complex i has results[i][0]=(gt_data, inf) and
     results[i][1]=(top1_pred, conf_score). HeteroData stores CA-only positions
     in DPP's internal (centered) frame.
  5. Reconstruct full backbone in the parquet's Angstrom frame:
        - Kabsch (output GT receptor CA) -> (parquet receptor CA): global T
        - Apply T to predicted ligand CA: gives pred ligand CA in parquet frame
        - Kabsch (pred ligand CA, parquet frame) -> (parquet ligand CA): T_lig
          gives the ligand's rigid motion.
        - receptor backbone = parquet receptor backbone (DPP keeps it fixed)
        - ligand backbone = apply T_lig (input->predicted) to parquet ligand
          backbone N, C, O.
  6. Write `benchmarks/predictions/diffdock_pp/{split_tag}/{sample_id}.npz`
     with `pred_atoms: [L, 4, 3]` and `sample_id`.

NOTE on the Kabsch math: DPP does rigid docking, so the "ligand motion" we
recover IS the model's only degree of freedom for the ligand. The N/C/O
positions inferred this way are exact (not an approximation), modulo the
fact that DPP internally only ever sees CA.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
DPP_REPO = THIS_DIR / "repo"
DPP_VENV_PY = REPO_ROOT / ".venv_diffdock_pp" / "Scripts" / "python.exe"


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find R, t such that R @ P + t ~= Q (minimizes RMSD)."""
    p_c = P.mean(0)
    q_c = Q.mean(0)
    A = P - p_c
    B = Q - q_c
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_c - R @ p_c
    return R, t


def reconstruct_backbone(
    parquet_atoms: np.ndarray,    # [L, 4, 3] real Angstroms
    parquet_chains: np.ndarray,   # [L] {0, 1}
    dpp_gt_rec_ca: np.ndarray,    # [LR, 3] DPP frame
    dpp_gt_lig_ca: np.ndarray,    # [LL, 3] DPP frame
    dpp_pred_rec_ca: np.ndarray,  # [LR, 3] should equal gt_rec (rigid docking)
    dpp_pred_lig_ca: np.ndarray,  # [LL, 3] DPP frame, predicted ligand
) -> np.ndarray:
    """Lift DPP's CA-only prediction back to our [L,4,3] backbone frame."""
    chain_a = parquet_chains == 0
    chain_b = parquet_chains == 1
    assert chain_a.sum() == dpp_gt_rec_ca.shape[0], \
        f"receptor size mismatch parquet={chain_a.sum()} dpp={dpp_gt_rec_ca.shape[0]}"
    assert chain_b.sum() == dpp_gt_lig_ca.shape[0], \
        f"ligand size mismatch parquet={chain_b.sum()} dpp={dpp_gt_lig_ca.shape[0]}"

    # 1. Global transform (DPP frame -> parquet frame) via receptor CA Kabsch.
    parquet_rec_ca = parquet_atoms[chain_a, 1, :]  # CA index = 1
    parquet_lig_ca = parquet_atoms[chain_b, 1, :]
    R_g, t_g = _kabsch(dpp_gt_rec_ca, parquet_rec_ca)

    # 2. Bring predicted ligand CA into parquet frame.
    pred_lig_ca_pframe = dpp_pred_lig_ca @ R_g.T + t_g

    # 3. Ligand rigid motion: input ligand CA -> predicted ligand CA (parquet frame).
    R_l, t_l = _kabsch(parquet_lig_ca, pred_lig_ca_pframe)

    # 4. Transform ligand's full backbone (N, CA, C, O) by (R_l, t_l).
    lig_bb_in = parquet_atoms[chain_b]            # [LL, 4, 3]
    lig_bb_out = lig_bb_in @ R_l.T + t_l           # [LL, 4, 3]

    # 5. Receptor backbone unchanged.
    out = parquet_atoms.copy()
    out[chain_b] = lig_bb_out
    return out


def _patch_dips_esm_config(template_path: Path, split_root: Path) -> Path:
    """Clone the dips_esm_inference config and rewrite data paths to our split."""
    cfg = yaml.safe_load(template_path.read_text())
    cfg["data"]["dataset"] = "db5"   # use the DB5Loader for {name}_r_b.pdb / _l_b.pdb format
    cfg["data"]["data_path"] = str(split_root.resolve()).replace("\\", "/")
    cfg["data"]["data_file"] = str((split_root / "splits_test.csv").resolve()).replace("\\", "/")
    cfg["data"]["resolution"] = "residue"
    cfg["data"]["no_graph_cache"] = True
    cfg["data"]["multiplicity"] = 1
    cfg["data"]["use_unbound"] = False
    cfg["data"]["use_orientation_features"] = False
    out_path = split_root / "config.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, type=Path,
                   help="benchmarks/splits/clean_*.json")
    p.add_argument("--num_samples", type=int, default=40,
                   help="DPP sampler K (default 40 = paper protocol).")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only adapt and predict the first N samples.")
    p.add_argument("--out_predictions_dir", default=None,
                   help="Default benchmarks/predictions/diffdock_pp/{split_tag}.")
    p.add_argument("--run_tag", default=None,
                   help="DPP run name (defaults to clean_{bin}).")
    p.add_argument("--reuse_existing", action="store_true",
                   help="Skip the DPP inference call if storage pickle already exists.")
    p.add_argument("--config_template", type=Path,
                   default=DPP_REPO / "config" / "dips_esm_inference.yaml")
    args = p.parse_args()

    split_tag = args.split.stem  # e.g. 'clean_400_600'
    run_tag = args.run_tag or split_tag
    print(f"=== DPP pretrained eval: {split_tag} ===")

    # 1. Adapt parquet -> DPP input PDBs.
    split_root = DPP_REPO / "datasets" / f"benchmarks_{split_tag}"
    if split_root.exists():
        # Always refresh — adapt is cheap and avoids stale data on rerun.
        shutil.rmtree(split_root)
    adapt_cmd = [
        sys.executable,
        str(THIS_DIR / "adapt.py"),
        "--split", str(args.split),
        "--out", str(split_root),
    ]
    if args.limit > 0:
        adapt_cmd += ["--limit", str(args.limit)]
    print(f"[adapt] {' '.join(adapt_cmd)}")
    subprocess.run(adapt_cmd, check=True)

    # 2. Patch config for our dataset.
    cfg_path = _patch_dips_esm_config(args.config_template, split_root)
    print(f"[config] wrote {cfg_path}")

    # 3. Run DPP inference.
    storage_path = DPP_REPO / "storage" / f"{run_tag}.pkl"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    (DPP_REPO / "ckpts").mkdir(exist_ok=True)
    (DPP_REPO / "tb").mkdir(exist_ok=True)
    (DPP_REPO / "visualization").mkdir(exist_ok=True)

    if args.reuse_existing and storage_path.exists():
        print(f"[infer] reusing existing {storage_path}")
    else:
        cmd = [
            str(DPP_VENV_PY),
            "src/main_inf.py",
            "--mode", "test",
            "--config_file", str(cfg_path.relative_to(DPP_REPO)).replace("\\", "/"),
            "--run_name", run_tag,
            "--save_path", f"ckpts/{run_tag}",
            "--batch_size", "1",
            "--num_folds", "1",
            "--num_gpu", "1",
            "--gpu", "0",
            "--seed", "0",
            "--logger", "tensorboard",
            "--tensorboard_path", f"tb/{run_tag}",
            "--filtering_model_path", "checkpoints/confidence_model_dips/fold_0/",
            "--score_model_path", "checkpoints/large_model_dips/fold_0/",
            "--num_samples", str(args.num_samples),
            "--prediction_storage", f"storage/{run_tag}.pkl",
        ]
        print(f"[infer] cwd={DPP_REPO}")
        print(f"        {' '.join(cmd)}")
        env = {**os.environ, "VIRTUAL_ENV": str(DPP_VENV_PY.parent.parent)}
        subprocess.run(cmd, cwd=DPP_REPO, env=env, check=True)

    # 4. Extract NPZs (must run inside DPP venv: pickle requires torch_geometric).
    out_pred_dir = (
        Path(args.out_predictions_dir) if args.out_predictions_dir
        else REPO_ROOT / "benchmarks" / "predictions" / "diffdock_pp" / split_tag
    )
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    extract_cmd = [
        str(DPP_VENV_PY),
        str(THIS_DIR / "extract_npz.py"),
        "--pickle", str(storage_path),
        "--split", str(args.split.resolve()),
        "--out_dir", str(out_pred_dir.resolve()),
        "--parquet", str((REPO_ROOT / "data/processed/samples.parquet").resolve()),
    ]
    if args.limit > 0:
        extract_cmd += ["--limit", str(args.limit)]
    print(f"[extract] {' '.join(extract_cmd)}")
    subprocess.run(extract_cmd, check=True, cwd=REPO_ROOT)

    print(f"Score with: python benchmarks/scripts/compute_metrics.py "
          f"--model diffdock_pp --splits {split_tag}")


if __name__ == "__main__":
    main()
