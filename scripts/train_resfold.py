#!/usr/bin/env python
"""
ResFold training script.

Two-stage architecture:
- Stage 1: Residue-level diffusion on centroids
- Stage 2: Atom refinement from centroids

Training modes:
- stage1_only: Train residue diffusion only
- stage2_only: Train atom refinement only (using GT centroids)
- end_to_end: Train both stages together

Usage:
    python train_resfold.py --mode stage1_only --n_train 80 --n_steps 10000
    python train_resfold.py --mode stage2_only --n_train 80 --n_steps 5000
    python train_resfold.py --mode end_to_end --n_train 80 --n_steps 15000
"""

import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq
import argparse
import os
import time
from datetime import datetime
import yaml

# Shared utilities
from script_utils import (
    Logger,
    set_seed,
    save_config,
    get_data_path,
    plot_prediction,
)

from tinyfold.training.run_naming import generate_run_name
from tinyfold.training.registry_append import append_registry_row

# Training utilities from tinyfold.training
from tinyfold.training import (
    load_sample_raw,
    collate_batch,
    random_rotation_matrix,
    apply_rigid_augment,
    get_or_create_split,
    create_diffusion_components,
    load_model_checkpoint,
    create_train_sampler,
)

# Model imports
from tinyfold.model.diffusion import (
    create_schedule,
    create_noiser,
    kabsch_align_to_target,
    create_sampler,
)
from tinyfold.model.geometry import kabsch_rigid
from tinyfold.model.resfold import ResFoldPipeline
from tinyfold.model.metrics import (
    compute_dockq,
    cluster_poses,
    interface_mask_from_gt,
    score_geometric_energy,
    score_self_consistency,
)
from tinyfold.training.utils import (
    MultiCopyTrainer,
    VectorizedMultiCopyTrainer,
    edm_loss_weight,
)

# Loss imports
from tinyfold.model.losses import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_c_rmsd,
    compute_distance_consistency_loss,
    GeometryLoss,
    ContactLoss,
    compute_lddt,
    compute_lddt_metrics,
)


from tinyfold.inference import (
    sample_centroids,
    sample_centroids_one_shot,
    sample_centroids_ve,
    sample_k_centroids,
    sample_centroids_with_sampler,
)



@torch.no_grad()
def generate_stage1_predictions(
    model, samples, indices, noiser, device, batch_size=1, logger=None,
    align_per_step=False, recenter=False, sampler=None
):
    """Generate Stage 1 centroid predictions for all samples.

    Args:
        model: ResFoldPipeline model with trained Stage 1
        samples: dict of sample_idx -> sample dict
        indices: list of sample indices to process
        noiser: DiffusionNoiser
        device: torch device
        batch_size: batch size for inference (1 for variable lengths)
        logger: optional logger
        align_per_step: Kabsch-align x0_pred each step (fixes drift)
        recenter: Re-center each step (avoids translation drift)

    Returns:
        dict of sample_idx -> predicted centroids tensor [L, 3] (normalized)
    """
    model.eval()
    predictions = {}

    total = len(indices)
    for i, idx in enumerate(indices):
        s = samples[idx]
        batch = collate_batch([s], device)

        # Run Stage 1 diffusion sampling
        if sampler is not None:
            centroids_pred = sample_centroids_with_sampler(model, batch, noiser, device, sampler)
        else:
            centroids_pred = sample_centroids(
                model, batch, noiser, device,
                align_per_step=align_per_step, recenter=recenter
            )

        # Store prediction (trim to actual length)
        n_res = s['n_res']
        predictions[idx] = centroids_pred[0, :n_res].cpu()

        if logger and (i + 1) % 500 == 0:
            logger.log(f"    Generated {i + 1}/{total} predictions...")

    return predictions


def save_stage1_predictions(predictions, path):
    """Save Stage 1 predictions to file."""
    torch.save(predictions, path)


def load_stage1_predictions(path):
    """Load Stage 1 predictions from file."""
    return torch.load(path)




# =============================================================================
# Main
# =============================================================================

def parse_args():
    # Optional profile config loaded before full arg parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="YAML config profile for default arguments")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="ResFold training")

    # Training mode
    parser.add_argument("--mode", type=str, default="end_to_end",
                        choices=["stage1_only", "stage2_only", "end_to_end"],
                        help="Training mode")

    # Data
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=14)
    parser.add_argument("--n_eval_train", type=int, default=200)
    parser.add_argument("--eval_train_dockq", action="store_true",
                        help="Also compute DockQ + C-RMSD on the train-eval "
                             "subset (onestep + one_shot only). Default off so "
                             "legacy runs stay byte-identical. Essential for "
                             "overfit / capacity-ladder experiments where the "
                             "question is whether the model can memorise the "
                             "INTERFACE (train DockQ), not just the backbone "
                             "(train centroid RMSE).")
    parser.add_argument("--min_atoms", type=int, default=200)
    parser.add_argument("--max_atoms", type=int, default=400)
    parser.add_argument("--select_smallest", action="store_true",
                        help="Select N smallest proteins instead of filtering by atom range")
    parser.add_argument("--test_strategy", type=str, default="random",
                        choices=["random", "stratified"],
                        help="Test-set sampling: 'random' (default) shuffles "
                             "eligible pool and takes n_test; 'stratified' bins by "
                             "LA+LB total residues and takes an equal share per bin "
                             "(so the headline number isn't dominated by small "
                             "complexes in datasets with a long size tail).")
    parser.add_argument("--test_size_bins", type=str, default=None,
                        help="Comma-separated bin lower-edges (LA+LB residues) for "
                             "--test_strategy stratified. Default: '0,400,600,1000,1500' "
                             "(5 bins; last is [1500, inf)).")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Don't normalize coordinates to unit variance - work in Angstroms directly")
    parser.add_argument("--global_scale", type=float, default=None,
                        help="Fixed coordinate divisor (Angstroms) used INSTEAD of "
                             "per-sample std when normalizing. Decouples coordinate "
                             "scale from complex size (an 8A contact maps to the same "
                             "normalized value at every size), addressing the OOD "
                             "size-collapse. ~15 matches the dataset-global coord std / "
                             "AF3's sigma_data convention. None = legacy per-sample std.")
    parser.add_argument("--per_chain_res_idx", action="store_true",
                        help="Use per-chain-reset res_idx ([0..LA-1] then [0..LB-1]) "
                             "instead of the legacy single arange(L_total). The legacy "
                             "encoding makes the model size-dependent — chain B's "
                             "positional features change with chain A's length. "
                             "See scripts/test_positional_invariance.py.")
    parser.add_argument("--crop_strategy", type=str, default="none",
                        choices=["none", "contiguous", "spatial", "interface"],
                        help="Training-time crop strategy. 'none' = no crop "
                             "(legacy behavior). 'interface' = bias center "
                             "toward GT contact residues (recommended for PPI; "
                             "see src/tinyfold/training/cropping.py).")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Token budget per cropped sample. With 12M params, "
                             "256 gives ~46k params/token (vs 12k uncropped, vs "
                             "AF-Multimer's 242k cropped).")
    parser.add_argument("--crop_interface_prob", type=float, default=0.8,
                        help="P(center on interface) for InterfaceCrop. The "
                             "remaining 1-p crops are uniform random so non-"
                             "interface regions still receive proportional signal.")
    parser.add_argument("--relpos_bias", action="store_true",
                        help="Add AF-M-style relative-position bias (clipped to "
                             "±32 plus same-chain bit) to trunk + denoiser "
                             "attention. Combined with --crop_strategy, this is "
                             "the v2 fix for the cliff (see notes/phase_d_"
                             "stratified_finding.md and the test_positional_"
                             "invariance.py diagnostic).")
    parser.add_argument("--relpos_clip", type=int, default=32,
                        help="Clip distance for the relpos bucket (default 32, "
                             "AF-M convention).")
    parser.add_argument("--pair_repr", action="store_true",
                        help="Add a minimal Pairmixer-style pair representation "
                             "(triangle multiplication + transition, no triangle "
                             "attention) to the trunk; emits a per-head attention "
                             "bias. Phase H: the decisive test of whether an "
                             "explicit (i,j) channel moves the >=200-res cliff. "
                             "Pair track runs once per sample (trunk); pair with "
                             "--crop_strategy interface to keep L^2 affordable.")
    parser.add_argument("--c_pair", type=int, default=64,
                        help="Pair-channel width for --pair_repr (default 64).")
    parser.add_argument("--pair_layers", type=int, default=3,
                        help="Number of triangle-mul+transition blocks for "
                             "--pair_repr (default 3).")
    parser.add_argument("--pair_hidden", type=int, default=64,
                        help="Triangle hidden width for --pair_repr (default 64).")
    parser.add_argument("--load_split", type=str, default=None,
                        help="Load train/test split from JSON (for Stage 2 to reuse Stage 1 split)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_bucketing", action="store_true",
                        help="Use length bucketing for efficient batching")
    parser.add_argument("--n_buckets", type=int, default=8,
                        help="Number of length buckets (default: 8)")
    parser.add_argument("--dynamic_batch", action="store_true",
                        help="Use dynamic batch sizing based on sequence length")
    parser.add_argument("--max_tokens", type=int, default=30000,
                        help="Max tokens per batch for dynamic batching (batch_size * max_seq_len)")

    # Training
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for cosine schedule")
    parser.add_argument("--grad_accum", type=int, default=1)

    # Model selection (resfold = original two-stage pipeline; onestep = single
    # network with parallel centroid + atom heads, trained end-to-end)
    parser.add_argument("--model_kind", type=str, default="resfold",
                        choices=["resfold", "onestep"],
                        help="Model architecture: 'resfold' (ResFoldPipeline) or 'onestep' (ResFoldOneStep)")

    # Model - Stage 1
    parser.add_argument("--c_token_s1", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=9)
    parser.add_argument("--denoiser_blocks", type=int, default=7)

    # Model - AA representation (Loop 05 / Task F)
    # learned   = historical nn.Embedding(n_aa_types, c_token), default
    # esm2_35M  = frozen ESM-2-35M cached embeddings (480d) + projection
    # esm2_150M = frozen ESM-2-150M cached embeddings (640d) + projection
    parser.add_argument("--aa_embed", type=str, default="learned",
                        choices=["learned", "esm2_35M", "esm2_150M"],
                        help="AA representation: learned nn.Embedding (default) or frozen "
                             "ESM-2 cached embeddings (35M=480d, 150M=640d).")
    parser.add_argument("--esm_cache_dir", type=str, default=None,
                        help="Directory containing per-sample ESM NPZ files. Required when "
                             "--aa_embed != learned. Default: data/processed/esm2_{variant}.")

    # Model - OneStep atom head (only used when model_kind=onestep)
    parser.add_argument("--atom_head_layers", type=int, default=2,
                        help="Number of transformer layers in the atom head (onestep only)")
    parser.add_argument("--atom_head_heads", type=int, default=4,
                        help="Number of attention heads in the atom head (onestep only)")
    parser.add_argument("--atom_weight", type=float, default=0.5,
                        help="Weight on atom-MSE loss (onestep only)")
    parser.add_argument("--atom_warmup_steps", type=int, default=500,
                        help="Linear warmup steps for the atom-MSE weight (onestep only)")

    # Model - OneStep confidence head (Loop 06 — optional per-target lDDT regressor
    # for multi-sample ranking at eval; small weight by default so a noisy
    # regression target can't tank centroid quality).
    parser.add_argument("--confidence_head", action="store_true",
                        help="Enable the per-target confidence head on ResFoldOneStep "
                             "(predicts lDDT in [0,1] from pooled denoiser tokens). "
                             "Requires --model_kind onestep.")
    parser.add_argument("--confidence_head_weight", type=float, default=0.0,
                        help="Aux loss coefficient for smooth-L1(pred_lddt, gt_lddt). "
                             ">0 requires --confidence_head. Default 0 keeps the head "
                             "frozen out of the loss even when instantiated.")

    # Model - Stage 2 (AtomRefinerV2: ~15M params with defaults)
    parser.add_argument("--c_token_s2", type=int, default=256)
    parser.add_argument("--s2_layers", type=int, default=18)
    parser.add_argument("--s2_heads", type=int, default=8)
    parser.add_argument("--centroid_noise", type=float, default=0.0,
                        help="Noise std for centroid augmentation in Stage 2 (normalized units)")

    # Diffusion
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--continuous_sigma", action="store_true",
                        help="Use AF3-style continuous sigma instead of discrete timesteps")
    parser.add_argument("--sigma_data", type=float, default=1.0,
                        help="Sigma_data for normalized coordinates (1.0 for std=1)")
    parser.add_argument("--sigma_min", type=float, default=0.002,
                        help="Minimum sigma for VE noise")
    parser.add_argument("--sigma_max", type=float, default=10.0,
                        help="Maximum sigma for VE noise")

    # Self-conditioning
    parser.add_argument("--self_cond_prob", type=float, default=0.5,
                        help="Probability of using self-conditioning during training (0 to disable)")

    # Training augmentation
    parser.add_argument("--translate_aug", type=float, default=0.0,
                        help="Random translation augmentation scale (in normalized units, try 0.1-0.3)")

    # Loss weights
    parser.add_argument("--dist_weight", type=float, default=0.1,
                        help="Weight for distance consistency loss")

    # Geometry loss weights (set to 0 to disable)
    parser.add_argument("--geom_weight", type=float, default=0.1,
                        help="Overall weight for geometry loss (0 to disable all)")
    parser.add_argument("--bond_length_weight", type=float, default=1.0,
                        help="Weight for bond length loss within geometry loss")
    parser.add_argument("--bond_angle_weight", type=float, default=0.1,
                        help="Weight for bond angle loss within geometry loss")
    parser.add_argument("--omega_weight", type=float, default=0.1,
                        help="Weight for omega dihedral loss within geometry loss")
    parser.add_argument("--o_chirality_weight", type=float, default=0.1,
                        help="Weight for O chirality loss within geometry loss")
    parser.add_argument("--cb_chirality_weight", type=float, default=0.0,
                        help="Weight for virtual CB chirality loss (default 0, experimental)")

    # Contact loss
    parser.add_argument("--contact_weight", type=float, default=0.0,
                        help="Weight for contact-based loss (0 to disable)")
    parser.add_argument("--contact_threshold", type=float, default=1.0,
                        help="Contact distance threshold in normalized units (1.0 = 10A)")
    parser.add_argument("--contact_min_seq_sep", type=int, default=5,
                        help="Minimum sequence separation for intra-chain contacts")
    parser.add_argument("--contact_inter_weight", type=float, default=2.0,
                        help="Weight multiplier for inter-chain contacts")
    parser.add_argument("--contact_stage", type=str, default="stage1",
                        choices=["stage1", "stage2", "both"],
                        help="Which stage(s) to apply contact loss")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load (e.g., Stage 1 checkpoint for Stage 2 training)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load --checkpoint and run one eval pass on "
                             "the test split, then write a REGISTRY row with the metrics.")

    # Sampling (evaluation)
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["ddpm", "ddpm_kabsch", "ddpm_kabsch_recenter", "heun", "ddim", "edm"],
                        help="Sampler for evaluation (default: use --align_per_step/--recenter flags)")
    parser.add_argument("--one_shot_sample", action="store_true",
                        help="Eval with single-forward EDM inference instead of multi-step VE sampling. Useful for overfit checks where the multi-step trajectory accumulates drift but a high-sigma one-shot recovers the memorized structure.")
    parser.add_argument("--align_per_step", action="store_true",
                        help="Kabsch-align x0_pred to x_t each step (fixes drift, Boltz-1 style)")
    parser.add_argument("--recenter", action="store_true",
                        help="Re-center coordinates each step (avoids translation drift)")
    parser.add_argument("--kabsch_interp", action="store_true",
                        help="Boltz Kabsch-interpolation sampler: after each Euler step, "
                             "rigid-align x_new onto the previous step's x. Targets the "
                             "+2 A multi-step drift documented in notes/phase_a_findings.md. "
                             "Orthogonal to --align_per_step (different anchor).")

    # Multi-sample inference (HDOCK-style cluster-then-rank). K=1 (default) is
    # the legacy single-sample path and produces byte-identical eval output.
    parser.add_argument("--n_samples", type=int, default=1,
                        help="K samples per target for multi-sample eval (default 1 = old behaviour)")
    parser.add_argument("--cluster_radius", type=float, default=5.0,
                        help="Interface-CA RMSD radius (Angstroms) for HDOCK-style clustering")
    parser.add_argument("--eval_K_list", type=str, default="1,5,40",
                        help="Comma-separated K values to report; only used when --n_samples > 1. "
                             "Max(eval_K_list) must be <= --n_samples.")
    # Loop 06 — third ranker uses the confidence head's predicted lDDT.
    parser.add_argument("--rank_by", type=str, default="cluster",
                        choices=["oracle", "cluster", "confidence",
                                 "self_consistency", "geometric"],
                        help="Multi-sample ranking strategy for downstream metrics "
                             "(DockQ, atom RMSE, C-RMSD). cluster=HDOCK-style "
                             "cluster-rep (default), oracle=argmin per-sample RMSE "
                             "vs GT (cheats; sanity-check upper bound), "
                             "confidence=argmax predicted lDDT (requires "
                             "--confidence_head), self_consistency=min mean "
                             "interface-RMSD to other K-1 samples, "
                             "geometric=min cross-chain clash count minus contact "
                             "count. ranked_{cluster,conf,consistency,energy}@K "
                             "are always reported when applicable.")

    # AF3-style training (multi-copy with trunk reuse)
    parser.add_argument("--multi_copy", type=int, default=0,
                        help="Number of augmented copies per sample (0=disabled, 48=AF3-style)")
    parser.add_argument("--augment_rotation", action="store_true", default=True,
                        help="Apply random rotation augmentation during training (default: True)")
    parser.add_argument("--no_augment_rotation", dest="augment_rotation", action="store_false",
                        help="Disable rotation augmentation")
    parser.add_argument("--loss_weighting", action="store_true",
                        help="Apply AF3-style loss weighting by noise level")
    parser.add_argument("--no_stratified_sigma", dest="stratified_sigma", action="store_false",
                        help="Disable stratified sigma sampling (use AF3 default sampling instead)")
    parser.set_defaults(stratified_sigma=True)  # Stratified sampling ON by default

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/resfold")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config profile for default arguments")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Apply defaults from config profile, then parse final args.
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            config_defaults = yaml.safe_load(f) or {}
        valid_dests = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in config_defaults.items() if k in valid_dests}
        parser.set_defaults(**filtered)

    return parser.parse_args()


def _run_test_eval(
    model,
    test_samples,
    test_indices,
    noiser,
    eval_sampler,
    device,
    args,
    is_onestep,
    logger,
    train_avg=None,
    n_eval=None,
):
    """Run one full eval pass over ``test_indices`` and log the summary line.

    This is the body originally inlined inside ``if step % args.eval_every == 0:``
    in the training loop. Factored out so the ``--eval_only`` re-eval path can
    invoke the exact same code without re-running training.

    Pure function w.r.t. the ``progress`` dict — caller is responsible for
    storing returned metrics if they want them surfaced to REGISTRY.md.

    Returns:
        tuple ``(test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg, extra_tokens)``.
        Each of ``dockq_avg``, ``dockq_success_pct``, ``c_rmsd_avg`` may be
        ``None`` if the corresponding metric was not collected (e.g., DockQ is
        None for non-onestep stage1_only modes). ``extra_tokens`` is a possibly
        empty list of pre-formatted ``oracle@K`` / ``mean@K`` / ``ranked@K``
        strings (one trio per K > 1 in ``args.eval_K_list`` when
        ``--n_samples > 1``).
    """
    model.eval()
    # K-list parsing: when --n_samples 1 (default), the K=1 fast path keeps
    # current behaviour byte-identical; the multi-sample machinery is silent.
    if getattr(args, "n_samples", 1) > 1:
        k_list = sorted({int(x) for x in args.eval_K_list.split(",")})
        assert max(k_list) <= args.n_samples, \
            f"eval_K_list max ({max(k_list)}) exceeds --n_samples ({args.n_samples})"
    else:
        k_list = [1]
    per_k_oracle = {k: [] for k in k_list}
    per_k_mean = {k: [] for k in k_list}
    per_k_ranked = {k: [] for k in k_list}
    # Loop 06: per-K ranked-by-confidence RMSE (only populated when the OneStep
    # model has a confidence head and emits pred_lddt during sampling).
    per_k_ranked_conf = {k: [] for k in k_list}
    # Energy-style rankers (post-Loop-06 add-on):
    # - consistency: argmin mean interface-RMSD to other K-1 samples
    # - energy:      argmin clash_count - 0.1 * contact_count
    per_k_ranked_consistency = {k: [] for k in k_list}
    per_k_ranked_energy = {k: [] for k in k_list}
    # Loop 06: flat lists across all (target, sample) pairs for Spearman.
    all_pred_lddts: list = []
    all_neg_rmses: list = []
    # Loop 06 smoke instrumentation: stddev of pred_lddt across K samples per
    # target. A near-zero stddev for every target = head collapsed.
    pred_lddt_std_per_target: list = []
    rank_by = getattr(args, "rank_by", "cluster")
    with torch.no_grad():
        # Evaluate on test set
        test_rmses = []
        test_dockq_scores = []
        test_lddt_scores = []
        test_ilddt_scores = []
        test_atom_rmses = []  # OneStep only
        test_c_rmsds = []  # Complex-RMSD (chain-A-aligned, all CA)
        for target_pos, idx in enumerate(test_indices):
            s = test_samples[idx]
            batch = collate_batch([s], device)

            if args.mode == "stage1_only":
                atoms_pred_onestep = None
                if args.continuous_sigma and args.n_samples > 1:
                    # Multi-sample path: K reproducible samples per target.
                    samples_c, samples_a, samples_lddt = sample_k_centroids(
                        model, batch, noiser, device,
                        K=args.n_samples, base_seed=args.seed, target_idx=target_pos,
                        is_onestep=is_onestep, one_shot=args.one_shot_sample,
                        align_per_step=args.align_per_step, recenter=args.recenter,
                        kabsch_interp=args.kabsch_interp,
                    )
                    n_res = s['n_res']
                    # Per-sample RMSE vs GT (Kabsch-aligned by compute_rmse).
                    gt_centroids = batch['centroids']
                    mask_res = batch['mask_res']
                    per_sample_rmses = []
                    for k_i in range(args.n_samples):
                        r = compute_rmse(
                            samples_c[k_i], gt_centroids, mask_res
                        ).item() * s['std']
                        per_sample_rmses.append(r)
                    # Loop 06: flatten pred_lddt vs -RMSE for Spearman across all
                    # (target, sample) pairs.
                    if samples_lddt is not None:
                        # samples_lddt: [K, B=1]. Per-sample scores list.
                        per_sample_conf = samples_lddt[:, 0].cpu().tolist()
                        all_pred_lddts.extend(per_sample_conf)
                        all_neg_rmses.extend([-r for r in per_sample_rmses])
                        pred_lddt_std_per_target.append(
                            float(torch.tensor(per_sample_conf).std().item())
                        )
                    else:
                        per_sample_conf = None
                    # Interface mask is GT-only, computed once.
                    iface_mask = interface_mask_from_gt(
                        batch['centroids'][0, :n_res].cpu(),
                        batch['chain_ids'][0, :n_res].cpu(),
                        batch['mask_res'][0, :n_res].cpu(),
                    )
                    if not iface_mask.any():
                        # Sanity guard: chains too far apart in GT — cluster on all residues.
                        iface_mask = torch.ones(n_res, dtype=torch.bool)
                    # Pose tensor for clustering: [K, n_res, 3] on CPU, in Angstroms.
                    poses_cpu = (samples_c[:, 0, :n_res, :] * s['std']).cpu()
                    # Track per-K best indices for each ranker so the downstream
                    # selection (DockQ/atom/C-RMSD) can honour --rank_by.
                    cluster_rep_per_k = {}
                    conf_rep_per_k = {}
                    oracle_rep_per_k = {}
                    consistency_rep_per_k = {}
                    energy_rep_per_k = {}
                    # Atom coords for the geometric scorer: [K, n_res, 4, 3] in
                    # Angstroms, CPU (cluster.py helpers are CPU-only).
                    if is_onestep and samples_a is not None:
                        atoms_cpu = (samples_a[:, 0, :n_res, :, :] * s['std']).cpu()
                    else:
                        atoms_cpu = None
                    chain_ids_cpu = batch['chain_ids'][0, :n_res].cpu()
                    valid_cpu = batch['mask_res'][0, :n_res].cpu()
                    for k in k_list:
                        sub_rmses = per_sample_rmses[:k]
                        per_k_oracle[k].append(min(sub_rmses))
                        per_k_mean[k].append(sum(sub_rmses) / k)
                        clusters = cluster_poses(poses_cpu[:k], iface_mask, args.cluster_radius)
                        cluster_idx = clusters[0]["representative"]
                        cluster_rep_per_k[k] = cluster_idx
                        per_k_ranked[k].append(sub_rmses[cluster_idx])
                        oracle_rep_per_k[k] = int(
                            min(range(k), key=lambda i: sub_rmses[i])
                        )
                        if per_sample_conf is not None:
                            sub_conf = per_sample_conf[:k]
                            conf_idx = int(max(range(k), key=lambda i: sub_conf[i]))
                            conf_rep_per_k[k] = conf_idx
                            per_k_ranked_conf[k].append(sub_rmses[conf_idx])
                        # Self-consistency: needs K>=2 to be meaningful; for K=1
                        # the scorer returns zeros and consistency_idx degenerates
                        # to sample 0, matching the existing K=1 fast-path.
                        consistency_scores = score_self_consistency(
                            poses_cpu[:k], iface_mask
                        )
                        consistency_idx = int(torch.argmin(consistency_scores).item())
                        consistency_rep_per_k[k] = consistency_idx
                        per_k_ranked_consistency[k].append(sub_rmses[consistency_idx])
                        # Geometric energy: requires per-sample atom coords; skip
                        # silently when not available (non-onestep path).
                        if atoms_cpu is not None:
                            energy_scores = score_geometric_energy(
                                atoms_cpu[:k], chain_ids_cpu, valid_cpu,
                            )
                            energy_idx = int(torch.argmin(energy_scores).item())
                            energy_rep_per_k[k] = energy_idx
                            per_k_ranked_energy[k].append(sub_rmses[energy_idx])
                    # Downstream metrics (DockQ, atom RMSE, C-RMSD) use the sample
                    # chosen by the active ranker at the FULL K to stay consistent
                    # with the headline ranked@K column. Default cluster keeps
                    # legacy sample-0 behaviour only if the cluster-rep happens
                    # to be sample 0 (no behavioural change vs Loop 02 in the
                    # typical small-K case).
                    K = args.n_samples
                    if rank_by == "oracle":
                        pick = oracle_rep_per_k.get(K, 0)
                    elif rank_by == "confidence" and per_sample_conf is not None:
                        pick = conf_rep_per_k.get(K, 0)
                    elif rank_by == "self_consistency":
                        pick = consistency_rep_per_k.get(K, 0)
                    elif rank_by == "geometric" and atoms_cpu is not None:
                        pick = energy_rep_per_k.get(K, 0)
                    else:  # cluster (default) or unavailable head/atoms
                        pick = cluster_rep_per_k.get(K, 0)
                    centroids_pred = samples_c[pick]
                    if is_onestep:
                        atoms_pred_onestep = samples_a[pick]
                    rmse = per_sample_rmses[pick]
                elif args.continuous_sigma:
                    if args.one_shot_sample:
                        sample_out = sample_centroids_one_shot(
                            model, batch, noiser, device, is_onestep=is_onestep,
                        )
                    else:
                        sample_out = sample_centroids_ve(
                            model, batch, noiser, device,
                            align_per_step=args.align_per_step,
                            recenter=args.recenter,
                            kabsch_interp=args.kabsch_interp,
                            is_onestep=is_onestep,
                        )
                    if is_onestep:
                        centroids_pred, atoms_pred_onestep = sample_out
                    else:
                        centroids_pred = sample_out
                    rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']
                elif eval_sampler is not None:
                    centroids_pred = sample_centroids_with_sampler(model, batch, noiser, device, eval_sampler)
                    rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']
                else:
                    centroids_pred = sample_centroids(
                        model, batch, noiser, device,
                        align_per_step=args.align_per_step,
                        recenter=args.recenter
                    )
                    rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']
                if is_onestep and atoms_pred_onestep is not None:
                    # Atom RMSE in Angstroms (Kabsch-aligned).
                    B_, L_ = centroids_pred.shape[:2]
                    atom_rmse = compute_rmse(
                        atoms_pred_onestep.reshape(B_, L_ * 4, 3),
                        batch['coords_res'].reshape(B_, L_ * 4, 3),
                        batch['mask_atom'],
                    ).item() * s['std']
                    test_atom_rmses.append(atom_rmse)

                # --- DockQ + C-RMSD for stage1_only onestep ---
                # C-RMSD only needs centroids + chain IDs (works regardless of is_onestep).
                n_res = s['n_res']
                c_rmsd = compute_c_rmsd(
                    pred_ca=centroids_pred[:, :n_res],
                    gt_ca=batch['centroids'][:, :n_res],
                    chain_ids=batch['chain_ids'][:, :n_res],
                    mask=batch['mask_res'][:, :n_res],
                ).item() * s['std']
                test_c_rmsds.append(c_rmsd)

                # DockQ only when we have atom-level predictions (i.e. onestep).
                if is_onestep and atoms_pred_onestep is not None:
                    pred_coords_res = atoms_pred_onestep[0, :n_res]   # [L, 4, 3]
                    gt_coords_res   = batch['coords_res'][0, :n_res]
                    dockq_result = compute_dockq(
                        pred_coords_res, gt_coords_res,
                        batch['aa_seq'][0, :n_res], batch['chain_ids'][0, :n_res],
                        std=s['std'],
                    )
                    if dockq_result['dockq'] is not None:
                        test_dockq_scores.append(dockq_result['dockq'])
            elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                # For Stage 2 with cached predictions: use Stage 1 predictions directly
                atoms_pred = model.forward_stage2(
                    batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                    batch['res_idx'], batch['mask_res'],
                    esm_embed=batch.get('esm_embed'),
                )
                atoms_pred_flat = atoms_pred.view(1, -1, 3)
                rmse = compute_rmse(atoms_pred_flat, batch['coords'], batch['mask_atom']).item() * s['std']

                # Compute DockQ, lDDT/ilDDT for Stage 2
                n_res = s['n_res']
                pred_coords_res = atoms_pred[:, :n_res]  # [1, L, 4, 3]
                gt_coords_res = batch['coords_res'][:, :n_res]

                # DockQ
                dockq_result = compute_dockq(
                    pred_coords_res[0], gt_coords_res[0],
                    batch['aa_seq'][0, :n_res], batch['chain_ids'][0, :n_res],
                    std=s['std']
                )
                if dockq_result['dockq'] is not None:
                    test_dockq_scores.append(dockq_result['dockq'])

                # lDDT/ilDDT
                lddt_result = compute_lddt_metrics(
                    pred_coords_res, gt_coords_res,
                    batch['chain_ids'][:, :n_res],
                    batch['mask_res'][:, :n_res],
                    coord_scale=s['std']
                )
                test_lddt_scores.append(lddt_result['lddt'])
                if lddt_result['n_interface'] > 0:
                    test_ilddt_scores.append(lddt_result['ilddt'])
            else:
                # For end_to_end or stage2 without cached: full sampling
                atoms_pred = model.sample(
                    batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                    noiser, batch['mask_res'],
                    esm_embed=batch.get('esm_embed'),
                )
                rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']

                # Compute DockQ and lDDT/ilDDT for Stage 2 / end-to-end
                n_res = s['n_res']
                pred_coords_res = atoms_pred[0].view(n_res, 4, 3)
                gt_coords_res = batch['coords_res'][0, :n_res]
                aa_seq = batch['aa_seq'][0, :n_res]
                chain_ids = batch['chain_ids'][0, :n_res]
                dockq_result = compute_dockq(pred_coords_res, gt_coords_res, aa_seq, chain_ids, std=s['std'])
                if dockq_result['dockq'] is not None:
                    test_dockq_scores.append(dockq_result['dockq'])

                # lDDT/ilDDT
                lddt_result = compute_lddt_metrics(
                    pred_coords_res.unsqueeze(0), gt_coords_res.unsqueeze(0),
                    chain_ids.unsqueeze(0),
                    batch['mask_res'][:, :n_res],
                    coord_scale=s['std']
                )
                test_lddt_scores.append(lddt_result['lddt'])
                if lddt_result['n_interface'] > 0:
                    test_ilddt_scores.append(lddt_result['ilddt'])

            test_rmses.append(rmse)
        test_avg = sum(test_rmses) / len(test_rmses)

        metric_name = "Centroid RMSE" if args.mode == "stage1_only" else "Atom RMSE"
        if train_avg is not None and n_eval is not None:
            log_msg = (
                f"         >>> Train {metric_name} ({n_eval}): {train_avg:.4f} A "
                f"| Test {metric_name} ({len(test_indices)}): {test_avg:.4f} A"
            )
        else:
            # --eval_only: no train-side numbers to print.
            log_msg = f"         >>> Test {metric_name} ({len(test_indices)}): {test_avg:.4f} A"
        dockq_avg = None
        dockq_success_pct = None
        c_rmsd_avg = None
        if test_dockq_scores:
            dockq_avg = sum(test_dockq_scores) / len(test_dockq_scores)
            dockq_success_pct = 100.0 * sum(
                1 for d in test_dockq_scores if d >= 0.23
            ) / len(test_dockq_scores)
            log_msg += f" | DockQ: {dockq_avg:.4f} (succ {dockq_success_pct:.1f}%)"
        if test_lddt_scores:
            lddt_avg = sum(test_lddt_scores) / len(test_lddt_scores)
            log_msg += f" | lDDT: {lddt_avg:.4f}"
        if test_ilddt_scores:
            ilddt_avg = sum(test_ilddt_scores) / len(test_ilddt_scores)
            log_msg += f" | ilDDT: {ilddt_avg:.4f}"
        if test_atom_rmses:
            atom_avg = sum(test_atom_rmses) / len(test_atom_rmses)
            log_msg += f" | Atom RMSE: {atom_avg:.4f}"
        if test_c_rmsds:
            c_rmsd_avg = sum(test_c_rmsds) / len(test_c_rmsds)
            log_msg += f" | C-RMSD: {c_rmsd_avg:.4f} A"
        # Multi-sample (Loop 02) tokens: K=1 oracle/mean/ranked all equal the
        # printed test RMSE, so skip K=1 to keep the cell tidy.
        extra_tokens: list = []
        for k in k_list:
            if k == 1:
                continue
            o = sum(per_k_oracle[k]) / len(per_k_oracle[k])
            m = sum(per_k_mean[k]) / len(per_k_mean[k])
            r = sum(per_k_ranked[k]) / len(per_k_ranked[k])
            log_msg += f" | oracle@{k}: {o:.4f} A | mean@{k}: {m:.4f} A | ranked@{k}: {r:.4f} A"
            extra_tokens.append(f"oracle@{k} {o:.3f} A")
            extra_tokens.append(f"mean@{k} {m:.3f} A")
            extra_tokens.append(f"ranked@{k} {r:.3f} A")
            # Loop 06: confidence-ranked sample (only when the head fired).
            if per_k_ranked_conf[k]:
                rc = sum(per_k_ranked_conf[k]) / len(per_k_ranked_conf[k])
                log_msg += f" | ranked_conf@{k}: {rc:.4f} A"
                extra_tokens.append(f"ranked_conf@{k} {rc:.3f} A")
            # Energy-style rankers: self-consistency and geometric clash/contact.
            if per_k_ranked_consistency[k]:
                rcs = sum(per_k_ranked_consistency[k]) / len(per_k_ranked_consistency[k])
                log_msg += f" | ranked_consistency@{k}: {rcs:.4f} A"
                extra_tokens.append(f"ranked_consistency@{k} {rcs:.3f} A")
            if per_k_ranked_energy[k]:
                re = sum(per_k_ranked_energy[k]) / len(per_k_ranked_energy[k])
                log_msg += f" | ranked_energy@{k}: {re:.4f} A"
                extra_tokens.append(f"ranked_energy@{k} {re:.3f} A")
        # Loop 06: Spearman(pred_lddt, -RMSE) across all (target, sample) pairs.
        # Falls back to torch.corrcoef Pearson if scipy isn't installed.
        if all_pred_lddts:
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(all_pred_lddts, all_neg_rmses)
                log_msg += f" | Spearman(pred_lddt,-RMSE): {rho:.3f}"
                extra_tokens.append(f"spearman {rho:.3f}")
            except Exception:
                a = torch.tensor(all_pred_lddts, dtype=torch.float32)
                b = torch.tensor(all_neg_rmses, dtype=torch.float32)
                stacked = torch.stack([a, b], dim=0)
                if a.numel() > 1 and a.std() > 0 and b.std() > 0:
                    rho = torch.corrcoef(stacked)[0, 1].item()
                    log_msg += f" | Pearson(pred_lddt,-RMSE): {rho:.3f}"
                    extra_tokens.append(f"pearson {rho:.3f}")
            # Mean of per-target stddev: a value near 0 means the head's score
            # is invariant to which sample it sees (collapse-to-mean indicator).
            if pred_lddt_std_per_target:
                std_mean = sum(pred_lddt_std_per_target) / len(pred_lddt_std_per_target)
                log_msg += f" | pred_lddt_std(per-target): {std_mean:.4f}"
                extra_tokens.append(f"pred_lddt_std {std_mean:.4f}")
        logger.log(log_msg)

        return test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg, extra_tokens


def _run_training(args, progress):
    """Body of the training run.

    ``progress`` is a mutable dict whose ``"best_rmse"`` key is updated every
    time a new best test RMSE is recorded. ``main()`` reads ``progress`` from
    the surrounding ``finally:`` block so that a crash mid-training still
    surfaces the last metric we reached.
    """
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # === Loop 05: resolve ESM cache + esm_dim once, up-front ===
    # We mirror the small dim table from prepare_esm2_embeddings.py and the
    # model/ResidueEncoder ESM_DIMS so the three sites stay in sync. Fail fast
    # with a friendly message if the cache directory is missing — re-running
    # the prep script is the only fix and the error should make that obvious.
    ESM_DIMS = {"esm2_35M": 480, "esm2_150M": 640}
    if args.aa_embed != "learned":
        if args.esm_cache_dir is None:
            variant = args.aa_embed.split("_", 1)[1]  # "esm2_35M" -> "35M"
            args.esm_cache_dir = os.path.join("data", "processed", f"esm2_{variant}")
        if not os.path.isdir(args.esm_cache_dir):
            raise FileNotFoundError(
                f"ESM cache dir not found: {args.esm_cache_dir}\n"
                f"Build it first:\n"
                f"  python scripts/prepare_esm2_embeddings.py "
                f"--parquet data/processed/samples.parquet "
                f"--output-dir {args.esm_cache_dir} "
                f"--variant {args.aa_embed.split('_', 1)[1]} --device cuda"
            )
        args._esm_dim = ESM_DIMS[args.aa_embed]
    else:
        args._esm_dim = None
        # Leave esm_cache_dir alone (likely None) so load_sample passes None and
        # the dataloader is byte-identical to pre-Loop-05.

    # Save config at startup (before training, in case of crash)
    save_config(args, args.output_dir)

    # Setup logging
    log_path = os.path.join(args.output_dir, 'train.log')
    logger = Logger(log_path)

    # Log header
    logger.log("=" * 70)
    logger.log("ResFold Training")
    logger.log("=" * 70)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Script: {os.path.abspath(__file__)}")
    logger.log(f"Command: python {' '.join(sys.argv)}")
    logger.log("")

    # Log config
    logger.log("Configuration:")
    logger.log(f"  mode:          {args.mode}")
    logger.log(f"  output_dir:    {args.output_dir}")
    logger.log(f"  n_train:       {args.n_train}")
    logger.log(f"  n_test:        {args.n_test}")
    logger.log(f"  batch_size:    {args.batch_size}")
    logger.log(f"  grad_accum:    {args.grad_accum}")
    logger.log(f"  eff_batch:     {args.batch_size * args.grad_accum}")
    logger.log(f"  n_steps:       {args.n_steps}")
    logger.log(f"  eval_every:    {args.eval_every}")
    logger.log(f"  lr:            {args.lr}")
    logger.log(f"  T:             {args.T}")
    logger.log(f"  aa_embed:      {args.aa_embed}")
    if args.aa_embed != "learned":
        logger.log(f"  esm_cache_dir: {args.esm_cache_dir}")
        logger.log(f"  esm_dim:       {args._esm_dim}")
    logger.log(f"  centroid_noise:{args.centroid_noise}")
    logger.log(f"  dist_weight:   {args.dist_weight}")
    logger.log(f"  geom_weight:   {args.geom_weight}")
    if args.geom_weight > 0:
        logger.log(f"    bond_length: {args.bond_length_weight}")
        logger.log(f"    bond_angle:  {args.bond_angle_weight}")
        logger.log(f"    omega:       {args.omega_weight}")
        logger.log(f"    o_chirality: {args.o_chirality_weight}")
        logger.log(f"    cb_chirality:{args.cb_chirality_weight}")
    logger.log("")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.log("")

    # Load data
    data_path = get_data_path()
    table = pq.read_table(data_path)

    # Deterministic split (either load from file or create new)
    train_indices, test_indices, split_info = get_or_create_split(args, table, logger, args.output_dir)
    logger.log("")

    # Preload samples
    normalize = not args.no_normalize
    logger.log(f"Preloading samples... (normalize={normalize})")
    _esm_dir = args.esm_cache_dir if args.aa_embed != "learned" else None
    per_chain = getattr(args, "per_chain_res_idx", False)
    gscale = getattr(args, "global_scale", None)
    if gscale is not None:
        logger.log(f"  Fixed-scale normalization: coords / {gscale:.2f} A (size-invariant)")
    train_samples = {idx: load_sample_raw(table, idx, normalize=normalize, esm_cache_dir=_esm_dir, per_chain_res_idx=per_chain, global_scale=gscale) for idx in train_indices}
    test_samples = {idx: load_sample_raw(table, idx, normalize=normalize, esm_cache_dir=_esm_dir, per_chain_res_idx=per_chain, global_scale=gscale) for idx in test_indices}
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples")

    # If not normalizing, warn about sigma values
    if args.no_normalize:
        # Compute what the typical STD would have been
        # We need to compute it from centered coords before the std=1 assignment
        sample_stds = []
        for idx in list(train_indices)[:100]:
            coords_raw = torch.tensor(table['atom_coords'][idx].as_py(), dtype=torch.float32).reshape(-1, 3)
            coords_centered = coords_raw - coords_raw.mean(dim=0, keepdim=True)
            sample_stds.append(coords_centered.std().item())
        avg_std = np.mean(sample_stds)
        logger.log(f"  Working in ANGSTROM space (typical coord std ~{avg_std:.1f}A)")
        logger.log(f"  Recommended sigma_max ~{avg_std * 10:.0f} (currently {args.sigma_max})")
        if args.sigma_max < avg_std * 5:
            logger.log(f"  WARNING: sigma_max={args.sigma_max} may be too small for Angstrom space!")

    # For Stage 2: load or generate Stage 1 predictions
    if args.mode == "stage2_only" and args.load_split:
        stage1_dir = os.path.dirname(args.load_split)
        predictions_path = os.path.join(stage1_dir, "stage1_predictions.pt")

        if os.path.exists(predictions_path):
            logger.log(f"  Loading Stage 1 predictions from: {predictions_path}")
            s1_predictions = load_stage1_predictions(predictions_path)
            logger.log(f"    Loaded {len(s1_predictions)} predictions")
        else:
            # Generate predictions using Stage 1 checkpoint
            s1_checkpoint = os.path.join(stage1_dir, "best_model.pt")
            if not os.path.exists(s1_checkpoint):
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_checkpoint}")

            logger.log(f"  Generating Stage 1 predictions (this may take a while)...")
            logger.log(f"    Loading Stage 1 checkpoint: {s1_checkpoint}")

            # Create temporary model for Stage 1
            s1_model = ResFoldPipeline(
                c_token_s1=args.c_token_s1,
                trunk_layers=args.trunk_layers,
                denoiser_blocks=args.denoiser_blocks,
                c_token_s2=args.c_token_s2,
                s2_layers=args.s2_layers,
                s2_heads=args.s2_heads,
                n_timesteps=args.T,
                dropout=0.0,
                aa_embed=args.aa_embed,
                esm_dim=args._esm_dim,
            ).to(device)

            ckpt = torch.load(s1_checkpoint, map_location=device)
            s1_model.load_state_dict(ckpt['model_state_dict'], strict=False)

            # Create noiser for sampling
            schedule = create_schedule("linear", T=args.T)
            s1_noiser = create_noiser("gaussian", schedule)

            # Generate predictions for train and test
            all_samples = {**train_samples, **test_samples}
            all_indices = train_indices + test_indices
            s1_predictions = generate_stage1_predictions(
                s1_model, all_samples, all_indices, s1_noiser, device, logger=logger
            )

            # Save predictions
            save_stage1_predictions(s1_predictions, predictions_path)
            logger.log(f"    Saved predictions to: {predictions_path}")

            # Clean up
            del s1_model
            torch.cuda.empty_cache()

        # Inject predictions into samples
        for idx, pred in s1_predictions.items():
            if idx in train_samples:
                train_samples[idx]['centroids_pred'] = pred
            if idx in test_samples:
                test_samples[idx]['centroids_pred'] = pred
        logger.log(f"  Injected Stage 1 predictions into samples")

    # Create sampler for efficient batching
    train_sampler = create_train_sampler(args, train_samples, logger)

    # Create model. Two architectures share this script:
    #   resfold  -> ResFoldPipeline (Stage 1 + optional Stage 2)
    #   onestep  -> ResFoldOneStep  (centroid diffusion + parallel atom head)
    is_onestep = (args.model_kind == "onestep")
    if is_onestep:
        from tinyfold.model.resfold.onestep import ResFoldOneStep
        if args.mode != "stage1_only":
            raise ValueError("model_kind=onestep requires --mode stage1_only")
        if args.multi_copy and args.multi_copy > 0:
            raise ValueError("model_kind=onestep does not yet support --multi_copy training")
        # Loop 06: confidence-head weight requires the head be instantiated.
        if args.confidence_head_weight > 0 and not args.confidence_head:
            raise ValueError(
                "--confidence_head_weight > 0 requires --confidence_head to be set "
                f"(got weight={args.confidence_head_weight}, head=False)."
            )
        if getattr(args, "rank_by", "cluster") == "confidence" and not args.confidence_head:
            raise ValueError("--rank_by confidence requires --confidence_head.")
        model = ResFoldOneStep(
            c_token=args.c_token_s1,
            trunk_layers=args.trunk_layers,
            denoiser_blocks=args.denoiser_blocks,
            relpos_bias=getattr(args, "relpos_bias", False),
            relpos_clip=getattr(args, "relpos_clip", 32),
            pair_repr=getattr(args, "pair_repr", False),
            c_pair=getattr(args, "c_pair", 64),
            pair_layers=getattr(args, "pair_layers", 3),
            pair_hidden=getattr(args, "pair_hidden", 64),
            atom_head_layers=args.atom_head_layers,
            atom_head_heads=args.atom_head_heads,
            n_timesteps=args.T,
            dropout=0.0,
            aa_embed=args.aa_embed,
            esm_dim=args._esm_dim,
            confidence_head=args.confidence_head,
            sigma_data=args.sigma_data,
        ).to(device)
        # In OneStep the model itself is the "stage 1" denoiser; alias for forward calls.
        stage1_module = model
        if args.checkpoint:
            load_model_checkpoint(model, args.checkpoint, args.mode, device, logger)
        pc = model.count_parameters()
        logger.log(f"Model: ResFoldOneStep ({args.mode})")
        logger.log(f"  Trunk params:     {pc['trunk']:,} ({pc['trunk_pct']:.1f}%)")
        logger.log(f"  Denoiser params:  {pc['denoiser']:,} ({pc['denoiser_pct']:.1f}%)")
        logger.log(f"  Atom-head params: {pc['atom_head']:,} ({pc['atom_head_pct']:.1f}%)")
        if pc.get('confidence_head', 0) > 0:
            logger.log(
                f"  Confidence-head params: {pc['confidence_head']:,} "
                f"({pc['confidence_head_pct']:.1f}%)"
            )
        logger.log(f"  Total params:     {pc['total']:,}")
        logger.log("")
    else:
        # Original two-stage path.
        model = ResFoldPipeline(
            c_token_s1=args.c_token_s1,
            trunk_layers=args.trunk_layers,
            denoiser_blocks=args.denoiser_blocks,
            c_token_s2=args.c_token_s2,
            s2_layers=args.s2_layers,
            s2_heads=args.s2_heads,
            n_timesteps=args.T,
            dropout=0.0,
            stage1_only=(args.mode == "stage1_only"),
            aa_embed=args.aa_embed,
            esm_dim=args._esm_dim,
        ).to(device)
        stage1_module = model.stage1

        # Load checkpoint if provided
        if args.checkpoint:
            load_model_checkpoint(model, args.checkpoint, args.mode, device, logger)

        # Set training mode (freeze/unfreeze stages)
        model.set_training_mode(args.mode)

        param_counts = model.count_parameters()
        logger.log(f"Model: ResFold ({args.mode})")
        logger.log(f"  Stage 1 params: {param_counts['stage1']:,} ({param_counts['stage1_pct']:.1f}%)")
        logger.log(f"  Stage 2 params: {param_counts['stage2']:,} ({param_counts['stage2_pct']:.1f}%)")
        logger.log(f"  Total params:   {param_counts['total']:,}")
        logger.log("")

    # Create diffusion components (for Stage 1)
    schedule, noiser = create_diffusion_components(args, device, logger)

    # Create sampler for evaluation (if specified)
    eval_sampler = None
    if args.sampler:
        eval_sampler = create_sampler(args.sampler)
        logger.log(f"  Sampler: {args.sampler}")
    elif args.align_per_step or args.recenter:
        logger.log(f"  Sampling: align_per_step={args.align_per_step}, recenter={args.recenter}")
    if args.kabsch_interp:
        logger.log(f"  kabsch_interp: True (Boltz trajectory-frame alignment)")
    logger.log("")

    # Geometry loss (Stage 2 atoms, or onestep atom head)
    geom_loss_fn = None
    if args.geom_weight > 0 and (args.mode != "stage1_only" or is_onestep):
        geom_loss_fn = GeometryLoss(
            bond_length_weight=args.bond_length_weight,
            bond_angle_weight=args.bond_angle_weight,
            omega_weight=args.omega_weight,
            o_chirality_weight=args.o_chirality_weight,
            cb_chirality_weight=args.cb_chirality_weight,
        )
        logger.log(f"Geometry loss: {geom_loss_fn}")
        logger.log("")

    # Contact loss (can be used in Stage 1 and/or Stage 2)
    contact_loss_fn = None
    if args.contact_weight > 0:
        contact_loss_fn = ContactLoss(
            threshold=args.contact_threshold,
            min_seq_sep=args.contact_min_seq_sep,
            inter_chain_weight=args.contact_inter_weight,
            stage=args.contact_stage,
        )
        logger.log(f"Contact loss: {contact_loss_fn}")
        logger.log("")

    # --- Re-eval mode: skip training entirely and run one test pass ---
    # Loaded checkpoint is already in ``model`` above (the load_model_checkpoint
    # call at the top of _run_training). We just need to drive the test split
    # through the same eval body the in-loop check uses.
    if args.eval_only:
        assert args.checkpoint is not None, "--eval_only requires --checkpoint"
        logger.log("=" * 70)
        logger.log(f"Eval-only: scoring {args.checkpoint} on test split (N={len(test_indices)})")
        logger.log("=" * 70)
        test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg, extra_tokens = _run_test_eval(
            model, test_samples, test_indices, noiser, eval_sampler,
            device, args, is_onestep, logger,
        )
        # Surface the metrics for the outer finally: -> REGISTRY append.
        progress["best_rmse"] = test_avg
        if dockq_avg is not None:
            progress["dockq_avg"] = dockq_avg
        if dockq_success_pct is not None:
            progress["dockq_success_pct"] = dockq_success_pct
        if c_rmsd_avg is not None:
            progress["c_rmsd"] = c_rmsd_avg
        if extra_tokens:
            progress["extra_tokens"] = extra_tokens
        logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()
        return test_avg

    # Optimizer (only on trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps, eta_min=args.min_lr)

    # Build the training-time cropper. Eval paths never crop.
    if getattr(args, "crop_strategy", "none") != "none":
        from tinyfold.training.cropping import build_cropper
        cropper = build_cropper(
            args.crop_strategy,
            interface_prob=args.crop_interface_prob,
            interface_cutoff=8.0,
        )
        # Own RNG for crop sampling — seeded off args.seed so a rerun with the
        # same seed produces the same crop sequence. CPU generator: croppers
        # operate on CPU tensors (samples are preloaded in main memory).
        crop_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        logger.log(f"Cropper: {args.crop_strategy} @ crop_size={args.crop_size}"
                   + (f" interface_prob={args.crop_interface_prob}"
                      if args.crop_strategy == "interface" else ""))
    else:
        cropper = None
        crop_rng = None

    # Training loop
    logger.log(f"Training for {args.n_steps} steps...")
    logger.log("=" * 70)

    best_rmse = float('inf')
    progress["best_rmse"] = best_rmse
    start_time = time.time()

    model.train()
    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(args.grad_accum):
            # Sample batch (using bucketing if enabled)
            if train_sampler is not None:
                if args.dynamic_batch:
                    batch_indices, current_batch_size = train_sampler.sample_batch()
                else:
                    batch_indices = train_sampler.sample_batch(args.batch_size)
                    current_batch_size = args.batch_size
            else:
                batch_indices = random.choices(train_indices, k=args.batch_size)
                current_batch_size = args.batch_size

            batch_samples = [train_samples[idx] for idx in batch_indices]
            batch = collate_batch(
                batch_samples, device,
                cropper=cropper, crop_size=args.crop_size if cropper is not None else None,
                rng=crop_rng,
            )

            # Sample noise levels and add noise to centroids (for Stage 1)
            noise = torch.randn_like(batch['centroids'])

            # Target for loss (may be rotated by augmentation)
            centroids_target = batch['centroids']

            if args.continuous_sigma:
                # AF3-style: continuous sigma with VE noise (x_t = x0 + sigma * noise)
                if args.stratified_sigma:
                    sigma = noiser.sample_sigma_stratified(current_batch_size, device)
                else:
                    sigma = noiser.sample_sigma_af3(current_batch_size, device)
                x_t = batch['centroids'] + sigma.view(-1, 1, 1) * noise

                # Rotation augmentation: rotate BOTH x_t AND target by same rotation
                # This preserves the coordinate frame relationship
                if args.augment_rotation:
                    R = random_rotation_matrix(current_batch_size, device)
                    x_t = torch.bmm(x_t, R.transpose(1, 2))
                    centroids_target = torch.bmm(centroids_target, R.transpose(1, 2))

                # Translation augmentation: add small random shift to x_t
                # This makes the model robust to drift during inference
                if args.translate_aug > 0:
                    translation = args.translate_aug * torch.randn(current_batch_size, 1, 3, device=device)
                    x_t = x_t + translation

                # EDM/Karras 2022 per-sample loss weighting (Eq. 7).
                # See tinyfold/training/utils.py::edm_loss_weight for derivation.
                # MUST be applied at per-sample MSE level, not after batch reduction.
                loss_weight = (
                    edm_loss_weight(sigma, sigma_data=noiser.sigma_data)
                    if args.loss_weighting else None
                )
            else:
                # Standard: discrete timesteps with VP noise
                t = torch.randint(0, noiser.T, (current_batch_size,), device=device)
                sqrt_ab = noiser.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
                sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
                x_t = sqrt_ab * batch['centroids'] + sqrt_one_minus_ab * noise
                loss_weight = None

            # Forward pass
            if args.mode == "stage1_only":
                # === AF3-STYLE MULTI-COPY TRAINING ===
                # Trunk processes sequence-only (no coords), so we can run it ONCE
                # and reuse tokens for multiple augmented/noisy copies.
                if args.multi_copy > 0:
                    # Run trunk ONCE on sequence features (no coordinates!)
                    trunk_tokens = model.stage1.get_trunk_tokens(
                        batch['aa_seq'], batch['chain_ids'],
                        batch['res_idx'], batch['mask_res'],
                        esm_embed=batch.get('esm_embed'),
                    )

                    # Expand for n_copies
                    n_copies = args.multi_copy
                    B_orig, L, _ = batch['centroids'].shape

                    # Expand tensors: [B, ...] -> [B * n_copies, ...]
                    trunk_tokens_exp = trunk_tokens.unsqueeze(1).expand(-1, n_copies, -1, -1)
                    trunk_tokens_exp = trunk_tokens_exp.reshape(B_orig * n_copies, L, -1)

                    centroids_exp = batch['centroids'].unsqueeze(1).expand(-1, n_copies, -1, -1)
                    centroids_exp = centroids_exp.reshape(B_orig * n_copies, L, 3)

                    mask_exp = batch['mask_res'].unsqueeze(1).expand(-1, n_copies, -1)
                    mask_exp = mask_exp.reshape(B_orig * n_copies, L)

                    # Apply different augmentations to each copy
                    if args.augment_rotation:
                        centroids_aug = random_rigid_augment(centroids_exp, mask_exp, rotation=True)
                    else:
                        centroids_aug = centroids_exp

                    # Sample different timesteps for each copy
                    t_exp = torch.randint(0, noiser.T, (B_orig * n_copies,), device=device)

                    # Add noise
                    noise = torch.randn_like(centroids_aug)
                    sqrt_ab = noiser.schedule.sqrt_alpha_bar[t_exp].view(-1, 1, 1)
                    sqrt_one_minus_ab = noiser.schedule.sqrt_one_minus_alpha_bar[t_exp].view(-1, 1, 1)
                    x_t_exp = sqrt_ab * centroids_aug + sqrt_one_minus_ab * noise

                    # Run denoiser with pre-computed trunk tokens
                    centroids_pred = model.stage1.forward_with_trunk(
                        x_t_exp, trunk_tokens_exp, t_exp, mask_exp
                    )

                    # Loss: MSE on centroids (target is augmented centroids)
                    loss_mse = compute_mse_loss(centroids_pred, centroids_aug, mask_exp)
                    loss_dist = compute_distance_consistency_loss(
                        centroids_pred, centroids_aug, mask_exp
                    )
                    loss = loss_mse + args.dist_weight * loss_dist

                    # Track loss components
                    if accum_step == args.grad_accum - 1:
                        loss_components = {
                            'mse': loss_mse.item(),
                            'dist': loss_dist.item(),
                            'contact': 0.0,
                            'n_copies': n_copies,
                            'eff_batch': B_orig * n_copies,
                        }

                else:
                    # === STANDARD TRAINING ===
                    pred_lddt = None  # Loop 06: confidence-head output (onestep only)
                    if args.continuous_sigma:
                        # AF3-style with continuous sigma
                        # Self-conditioning: with probability p, first run model to get x0_prev
                        x0_prev = None
                        if args.self_cond_prob > 0 and torch.rand(1).item() < args.self_cond_prob:
                            with torch.no_grad():
                                sc_out = stage1_module.forward_sigma(
                                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                    sigma, batch['mask_res'], x0_prev=None,
                                    esm_embed=batch.get('esm_embed'),
                                )
                                # OneStep returns (centroid, atoms, pred_lddt); self-conditioning only uses centroids.
                                x0_prev = (sc_out[0] if is_onestep else sc_out).detach()

                        # Main forward pass (with or without self-conditioning)
                        fwd_out = stage1_module.forward_sigma(
                            x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                            sigma, batch['mask_res'], x0_prev=x0_prev,
                            esm_embed=batch.get('esm_embed'),
                        )
                        if is_onestep:
                            # Loop 06: 3-tuple (centroid, atoms, pred_lddt_or_None).
                            centroids_pred, atoms_pred, pred_lddt = fwd_out
                        else:
                            centroids_pred = fwd_out
                            atoms_pred = None
                    else:
                        # Original behavior with discrete timesteps
                        if is_onestep:
                            centroids_pred, atoms_pred = stage1_module(
                                x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                t, batch['mask_res'],
                                esm_embed=batch.get('esm_embed'),
                            )
                        else:
                            centroids_pred = model.forward_stage1(
                                x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                t, batch['mask_res'],
                                esm_embed=batch.get('esm_embed'),
                            )
                            atoms_pred = None
                    # Loss: MSE on centroids + distance consistency
                    # Use centroids_target which may be rotated by augmentation
                    if loss_weight is not None:
                        # EDM/Karras 2022 per-sample weighting: lambda(sigma) * MSE,
                        # then average across batch. See edm_loss_weight derivation.
                        per_sample_mse = compute_mse_loss(
                            centroids_pred, centroids_target, batch['mask_res'],
                            reduction='per_sample',
                        )
                        loss_mse = (per_sample_mse * loss_weight).mean()
                    else:
                        loss_mse = compute_mse_loss(
                            centroids_pred, centroids_target, batch['mask_res']
                        )
                    # dist loss is a geometric regularizer (not the EDM-preconditioned
                    # objective), so we leave it unweighted by lambda(sigma).
                    loss_dist = compute_distance_consistency_loss(
                        centroids_pred, centroids_target, batch['mask_res']
                    )
                    loss = loss_mse + args.dist_weight * loss_dist

                    # OneStep: add atom-MSE (with warmup) and geometry losses on atoms_pred.
                    loss_atom = 0.0
                    loss_geom = 0.0
                    loss_bond = 0.0
                    loss_angle = 0.0
                    loss_omega = 0.0
                    alpha_atom = 0.0
                    if is_onestep and atoms_pred is not None:
                        B, L = centroids_pred.shape[:2]
                        atoms_target_BL43 = batch['coords_res']  # [B, L, 4, 3]
                        # Flatten to [B, L*4, 3] for compute_mse_loss with mask_atom.
                        atoms_pred_flat = atoms_pred.reshape(B, L * 4, 3)
                        atoms_target_flat = atoms_target_BL43.reshape(B, L * 4, 3)
                        atom_mse_tensor = compute_mse_loss(
                            atoms_pred_flat, atoms_target_flat, batch['mask_atom']
                        )
                        # Linear warmup so the atom head doesn't poison the trunk early.
                        warmup = args.atom_warmup_steps
                        ramp = min(1.0, float(step) / float(warmup)) if warmup > 0 else 1.0
                        alpha_atom = ramp * args.atom_weight
                        loss = loss + alpha_atom * atom_mse_tensor
                        loss_atom = atom_mse_tensor.item()

                        if geom_loss_fn is not None and args.geom_weight > 0:
                            geom_losses = geom_loss_fn(
                                atoms_pred,
                                batch['mask_res'],
                                gt_coords=batch['coords_res'],
                            )
                            loss = loss + args.geom_weight * geom_losses['total']
                            loss_geom = geom_losses['total'].item()
                            loss_bond = geom_losses['bond_length'].item()
                            loss_angle = geom_losses['bond_angle'].item()
                            loss_omega = geom_losses['omega'].item()

                    # === Loop 06: confidence head auxiliary loss ===
                    # Regress predicted lDDT against GT lDDT of the EDM-blended
                    # one-step centroid prediction. GT is computed under no_grad
                    # so only the head/denoiser learn from this term — never the
                    # path that produced `centroids_pred`. coord_scale=1.0
                    # because trainer coords are normalized (std~1.0 per sample,
                    # absorbed at eval time via s['std']); lDDT thresholds in
                    # the normalized space then scale-compare correctly.
                    loss_conf = 0.0
                    gt_lddt_mean = 0.0
                    pred_lddt_mean = 0.0
                    if (
                        is_onestep
                        and pred_lddt is not None
                        and args.confidence_head_weight > 0
                    ):
                        with torch.no_grad():
                            gt_lddt = compute_lddt(
                                centroids_pred.detach(),
                                centroids_target,
                                mask=batch['mask_res'],
                                coord_scale=1.0,
                                reduction="per_sample",
                            )
                        conf_loss_tensor = torch.nn.functional.smooth_l1_loss(
                            pred_lddt, gt_lddt
                        )
                        loss = loss + args.confidence_head_weight * conf_loss_tensor
                        loss_conf = conf_loss_tensor.item()
                        gt_lddt_mean = gt_lddt.mean().item()
                        pred_lddt_mean = pred_lddt.mean().item()

                    # Contact loss for Stage 1
                    loss_contact = 0.0
                    if contact_loss_fn is not None and args.contact_stage in ["stage1", "both"]:
                        contact_losses = contact_loss_fn(
                            pred_centroids=centroids_pred,
                            gt_centroids=centroids_target,
                            chain_ids=batch['chain_ids'],
                            mask=batch['mask_res']
                        )
                        loss_contact = contact_losses['stage1'].item()
                        loss = loss + args.contact_weight * contact_losses['stage1']

                    # Track loss components for logging
                    if accum_step == args.grad_accum - 1:
                        loss_components = {
                            'mse': loss_mse.item(),
                            'dist': loss_dist.item(),
                            'contact': loss_contact,
                            'loss_weight': loss_weight.mean().item() if loss_weight is not None else 1.0,
                        }
                        if is_onestep:
                            loss_components.update({
                                'atom_mse': loss_atom,
                                'alpha_atom': alpha_atom,
                                'geom': loss_geom,
                                'bond': loss_bond,
                                'angle': loss_angle,
                                'omega': loss_omega,
                                # Loop 06: confidence-head aux loss (0 when disabled).
                                'conf': loss_conf,
                                'gt_lddt_mean': gt_lddt_mean,
                                'pred_lddt_mean': pred_lddt_mean,
                            })

            elif args.mode == "stage2_only":
                # Only Stage 2
                # Use Stage 1 predicted centroids if available, else GT
                if 'centroids_pred' in batch:
                    centroids_for_s2 = batch['centroids_pred']
                else:
                    centroids_for_s2 = batch['centroids']

                # Compute trunk tokens from sequence features (no coordinates)
                trunk_tokens = model.get_trunk_tokens(
                    batch['aa_seq'], batch['chain_ids'],
                    batch['res_idx'], batch['mask_res'],
                    esm_embed=batch.get('esm_embed'),
                )
                # Add noise augmentation to centroids input (optional)
                centroids_input = centroids_for_s2
                if args.centroid_noise > 0:
                    centroids_input = centroids_for_s2 + args.centroid_noise * torch.randn_like(centroids_for_s2)
                # Forward stage 2 with trunk tokens
                atoms_pred = model.forward_stage2(
                    centroids_input, batch['aa_seq'], batch['chain_ids'],
                    batch['res_idx'], batch['mask_res'], trunk_tokens=trunk_tokens,
                    esm_embed=batch.get('esm_embed'),
                )
                # Loss: MSE on atom positions
                # Reshape coords_res to [B, L*4, 3] for comparison
                B, L = batch['centroids'].shape[:2]
                atoms_target = batch['coords_res'].view(B, L * 4, 3)
                atoms_pred_flat = atoms_pred.view(B, L * 4, 3)
                loss_mse = compute_mse_loss(atoms_pred_flat, atoms_target, batch['mask_atom'])
                loss = loss_mse

                # Add geometry loss if enabled
                loss_geom = 0.0
                loss_bond = 0.0
                loss_angle = 0.0
                loss_omega = 0.0
                if geom_loss_fn is not None:
                    # Pass GT coords to detect chain breaks (don't penalize valid structural gaps)
                    geom_losses = geom_loss_fn(atoms_pred, batch['mask_res'], gt_coords=batch['coords_res'])
                    loss_geom = geom_losses['total'].item()
                    loss_bond = geom_losses['bond_length'].item()
                    loss_angle = geom_losses['bond_angle'].item()
                    loss_omega = geom_losses['omega'].item()
                    loss = loss + args.geom_weight * geom_losses['total']

                # Add contact loss for Stage 2
                loss_contact = 0.0
                if contact_loss_fn is not None and args.contact_stage in ["stage2", "both"]:
                    contact_losses = contact_loss_fn(
                        gt_centroids=batch['centroids'],
                        pred_atoms=atoms_pred,
                        gt_atoms=batch['coords_res'],
                        chain_ids=batch['chain_ids'],
                        mask=batch['mask_res']
                    )
                    loss_contact = contact_losses['stage2'].item()
                    loss = loss + args.contact_weight * contact_losses['stage2']

                # Track loss components for logging
                if accum_step == args.grad_accum - 1:
                    loss_components = {
                        'mse': loss_mse.item(),
                        'geom': loss_geom,
                        'bond': loss_bond,
                        'angle': loss_angle,
                        'omega': loss_omega,
                        'contact': loss_contact,
                    }

            else:  # end_to_end
                # Full pipeline
                result = model(
                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                    t, batch['mask_res'], mode="end_to_end",
                    esm_embed=batch.get('esm_embed'),
                )
                centroids_pred = result['centroids_pred']
                atoms_pred = result['atoms_pred']

                # Loss: centroid loss + distance consistency + atom loss
                loss_centroid = compute_mse_loss(centroids_pred, batch['centroids'], batch['mask_res'])
                loss_dist = compute_distance_consistency_loss(
                    centroids_pred, batch['centroids'], batch['mask_res']
                )

                B, L = batch['centroids'].shape[:2]
                atoms_target = batch['coords_res'].view(B, L * 4, 3)
                atoms_pred_flat = atoms_pred.view(B, L * 4, 3)
                loss_atoms = compute_mse_loss(atoms_pred_flat, atoms_target, batch['mask_atom'])

                loss = loss_centroid + args.dist_weight * loss_dist + loss_atoms

                # Add geometry loss if enabled
                loss_geom = 0.0
                if geom_loss_fn is not None:
                    # Pass GT coords to detect chain breaks (don't penalize valid structural gaps)
                    geom_losses = geom_loss_fn(atoms_pred, batch['mask_res'], gt_coords=batch['coords_res'])
                    loss_geom = geom_losses['total'].item()
                    loss = loss + args.geom_weight * geom_losses['total']

                # Add contact loss for end-to-end (both stages)
                loss_contact = 0.0
                if contact_loss_fn is not None:
                    contact_losses = contact_loss_fn(
                        pred_centroids=centroids_pred,
                        gt_centroids=batch['centroids'],
                        pred_atoms=atoms_pred,
                        gt_atoms=batch['coords_res'],
                        chain_ids=batch['chain_ids'],
                        mask=batch['mask_res']
                    )
                    loss_contact = contact_losses['total'].item()
                    loss = loss + args.contact_weight * contact_losses['total']

                # Track individual losses (for last accumulation step)
                if accum_step == args.grad_accum - 1:
                    loss_components = {
                        'centroid': loss_centroid.item(),
                        'dist': loss_dist.item(),
                        'atoms': loss_atoms.item(),
                        'geom': loss_geom,
                        'contact': loss_contact,
                    }

            # Backward
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        loss = accum_loss

        # Log every 100 steps OR on the eval boundary (so short runs still
        # surface a representative per-step line). Loop 06 adds `conf` token.
        if step % 100 == 0 or step % args.eval_every == 0:
            elapsed = time.time() - start_time
            if args.mode == "end_to_end" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" | cnt: {lc.get('contact', 0):.4f}" if args.contact_weight > 0 else ""
                logger.log(f"Step {step:5d} | loss: {loss:.4f} | ctr: {lc['centroid']:.4f} | atm: {lc['atoms']:.4f} | dst: {lc['dist']:.4f}{contact_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            elif args.mode == "stage2_only" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" cnt:{lc.get('contact', 0):.3f}" if args.contact_weight > 0 else ""
                if args.geom_weight > 0:
                    logger.log(f"Step {step:5d} | loss: {loss:.4f} | mse: {lc['mse']:.4f} | geom: {lc['geom']:.4f} (bnd:{lc['bond']:.3f} ang:{lc['angle']:.3f} omg:{lc['omega']:.3f}{contact_str}) | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
                else:
                    logger.log(f"Step {step:5d} | loss: {loss:.6f} | mse: {lc['mse']:.6f}{contact_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            elif args.mode == "stage1_only" and 'loss_components' in dir():
                lc = loss_components
                contact_str = f" | cnt: {lc.get('contact', 0):.4f}" if args.contact_weight > 0 else ""
                weight_str = f" | w: {lc.get('loss_weight', 1.0):.2f}" if args.continuous_sigma and args.loss_weighting else ""
                # Loop 06: when the confidence head is active, surface its aux
                # loss + raw lddt means so collapse-to-mean is visible at a glance.
                conf_str = (
                    f" | conf: {lc.get('conf', 0):.4f} (pred={lc.get('pred_lddt_mean', 0):.3f} gt={lc.get('gt_lddt_mean', 0):.3f})"
                    if getattr(args, "confidence_head", False)
                    else ""
                )
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | mse: {lc['mse']:.4f} | dst: {lc['dist']:.4f}{contact_str}{weight_str}{conf_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            else:
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on train set
                n_eval = min(args.n_eval_train, len(train_indices))
                eval_train_indices = random.sample(train_indices, n_eval)
                train_rmses = []
                train_dockq_scores = []  # only filled when --eval_train_dockq
                for idx in eval_train_indices:
                    s = train_samples[idx]
                    batch = collate_batch([s], device)

                    if args.mode == "stage1_only":
                        # For Stage 1: evaluate centroid RMSE via diffusion sampling
                        if args.continuous_sigma:
                            if args.one_shot_sample:
                                sample_out = sample_centroids_one_shot(
                                    model, batch, noiser, device, is_onestep=is_onestep,
                                )
                            else:
                                sample_out = sample_centroids_ve(
                                    model, batch, noiser, device,
                                    align_per_step=args.align_per_step,
                                    recenter=args.recenter,
                                    kabsch_interp=args.kabsch_interp,
                                    is_onestep=is_onestep,
                                )
                            centroids_pred = sample_out[0] if is_onestep else sample_out
                        elif eval_sampler is not None:
                            centroids_pred = sample_centroids_with_sampler(model, batch, noiser, device, eval_sampler)
                        else:
                            centroids_pred = sample_centroids(
                                model, batch, noiser, device,
                                align_per_step=args.align_per_step,
                                recenter=args.recenter
                            )
                        rmse = compute_rmse(centroids_pred, batch['centroids'], batch['mask_res']).item() * s['std']

                        # Optional train-set DockQ (overfit/capacity experiments):
                        # mirrors the test-eval DockQ using the same one-shot atom
                        # head output. Gated so default runs are unchanged.
                        if (
                            args.eval_train_dockq
                            and is_onestep
                            and args.one_shot_sample
                            and isinstance(sample_out, tuple)
                            and len(sample_out) > 1
                            and sample_out[1] is not None
                        ):
                            n_res_t = s['n_res']
                            dq = compute_dockq(
                                sample_out[1][0, :n_res_t],
                                batch['coords_res'][0, :n_res_t],
                                batch['aa_seq'][0, :n_res_t],
                                batch['chain_ids'][0, :n_res_t],
                                std=s['std'],
                            )
                            if dq['dockq'] is not None:
                                train_dockq_scores.append(dq['dockq'])
                    elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                        # For Stage 2 with cached predictions: use Stage 1 predictions directly
                        atoms_pred = model.forward_stage2(
                            batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                            batch['res_idx'], batch['mask_res'],
                            esm_embed=batch.get('esm_embed'),
                        )
                        atoms_pred = atoms_pred.view(1, -1, 3)
                        rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']
                    else:
                        # For end_to_end or stage2 without cached: full sampling
                        atoms_pred = model.sample(
                            batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                            noiser, batch['mask_res'],
                            esm_embed=batch.get('esm_embed'),
                        )
                        rmse = compute_rmse(atoms_pred, batch['coords'], batch['mask_atom']).item() * s['std']
                    train_rmses.append(rmse)
                train_avg = sum(train_rmses) / len(train_rmses)

                # Train-set DockQ summary (overfit/capacity experiments only).
                if train_dockq_scores:
                    tdq = sum(train_dockq_scores) / len(train_dockq_scores)
                    tdq_succ = 100.0 * sum(1 for d in train_dockq_scores if d >= 0.23) / len(train_dockq_scores)
                    logger.log(
                        f"         >>> Train DockQ ({len(train_dockq_scores)}): "
                        f"{tdq:.4f} (succ {tdq_succ:.1f}%)"
                    )

                # Evaluate on test set (delegated; see _run_test_eval).
                test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg, extra_tokens = _run_test_eval(
                    model, test_samples, test_indices, noiser, eval_sampler,
                    device, args, is_onestep, logger,
                    train_avg=train_avg, n_eval=n_eval,
                )

                # Plot first sample
                s = train_samples[train_indices[0]]
                batch = collate_batch([s], device)

                if args.mode == "stage1_only":
                    # Plot centroids for Stage 1
                    if args.continuous_sigma:
                        if args.one_shot_sample:
                            sample_out = sample_centroids_one_shot(
                                model, batch, noiser, device, is_onestep=is_onestep,
                            )
                        else:
                            sample_out = sample_centroids_ve(
                                model, batch, noiser, device,
                                align_per_step=args.align_per_step,
                                recenter=args.recenter,
                                kabsch_interp=args.kabsch_interp,
                                is_onestep=is_onestep,
                            )
                        centroids_pred = sample_out[0] if is_onestep else sample_out
                    elif eval_sampler is not None:
                        centroids_pred = sample_centroids_with_sampler(model, batch, noiser, device, eval_sampler)
                    else:
                        centroids_pred = sample_centroids(
                            model, batch, noiser, device,
                            align_per_step=args.align_per_step,
                            recenter=args.recenter
                        )
                    n_res = s['n_res']
                    pred = centroids_pred[0, :n_res] * s['std']
                    target = batch['centroids'][0, :n_res] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred_aligned, target_c).item()
                    chain_ids_plot = batch['chain_ids'][0, :n_res]
                elif args.mode == "stage2_only" and 'centroids_pred' in batch:
                    # Plot atoms for Stage 2 with cached predictions
                    atoms_pred = model.forward_stage2(
                        batch['centroids_pred'], batch['aa_seq'], batch['chain_ids'],
                        batch['res_idx'], batch['mask_res'],
                        esm_embed=batch.get('esm_embed'),
                    )
                    n = s['n_atoms']
                    pred = atoms_pred[0].view(-1, 3)[:n] * s['std']
                    target = batch['coords'][0, :n] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred_aligned, target_c).item()
                    chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]
                else:
                    # Plot atoms for end_to_end (or stage2 without cached)
                    atoms_pred = model.sample(
                        batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                        noiser, batch['mask_res'],
                        esm_embed=batch.get('esm_embed'),
                    )
                    n = s['n_atoms']
                    pred = atoms_pred[0, :n] * s['std']
                    target = batch['coords'][0, :n] * s['std']
                    pred_aligned, target_c = kabsch_align(pred.unsqueeze(0), target.unsqueeze(0))
                    rmse_viz = compute_rmse(pred_aligned, target_c).item()
                    chain_ids_plot = batch['chain_ids'][0].unsqueeze(-1).expand(-1, 4).reshape(-1)[:n]

                plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                plot_prediction(pred_aligned[0], target_c[0], chain_ids_plot,
                               s['sample_id'], rmse_viz, plot_path)
                logger.log(f"         >>> Saved plot: {plot_path}")

                # Save best model
                if test_avg < best_rmse:
                    best_rmse = test_avg
                    progress["best_rmse"] = best_rmse
                    # Stash the latest auxiliary metrics from the best-eval step
                    # so the outer finally: block can write them into REGISTRY.md.
                    if dockq_avg is not None:
                        progress["dockq_avg"] = dockq_avg
                    if dockq_success_pct is not None:
                        progress["dockq_success_pct"] = dockq_success_pct
                    if c_rmsd_avg is not None:
                        progress["c_rmsd"] = c_rmsd_avg
                    if extra_tokens:
                        progress["extra_tokens"] = extra_tokens
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    logger.log(f"         >>> New best test RMSE! Saved.")

                # Always save best-on-train (useful for N=1 overfit runs where test
                # is noise and best_model.pt freezes early).
                if not hasattr(_run_training, "_best_train") or train_avg < _run_training._best_train:
                    _run_training._best_train = train_avg
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_train_model.pt'))

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log(f"Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.4f} A")
    logger.log("")
    # Save final-step checkpoint regardless of eval outcomes. This is the source of
    # truth for "what the model looks like at the end of training" — useful when
    # the eval-tracked best is set early (e.g., N=1 overfit) and then never moves.
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'train_rmse': progress.get("best_rmse", float('inf')),
        'args': vars(args),
    }, os.path.join(args.output_dir, 'final_model.pt'))
    logger.log(f"Saved final-step checkpoint: {os.path.join(args.output_dir, 'final_model.pt')}")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()

    return best_rmse


def main():
    args = parse_args()

    # Reproducibility: set seed before anything else
    set_seed(args.seed)

    # Loop 08: never overwrite a prior run — timestamped subdir.
    run_name = generate_run_name("resfold", vars(args))
    args.output_dir = os.path.join(args.output_dir, run_name)

    # Track best metric across the try/except so finally: can write to REGISTRY.md.
    progress = {"best_rmse": float('inf')}
    outcome = "crashed: NoExitReached"
    try:
        _run_training(args, progress)
        outcome = "converged"
    except KeyboardInterrupt:
        outcome = "killed by user"
        raise
    except Exception as e:
        outcome = f"crashed: {type(e).__name__}"
        raise
    finally:
        best = progress.get("best_rmse", float('inf'))
        final_metric = best if best != float('inf') else None
        try:
            registry_path = append_registry_row(
                run_name=run_name,
                model="resfold",
                config_path=getattr(args, "config", None),
                final_metric=final_metric,
                outcome=outcome,
                output_dir=getattr(args, "output_dir", None),
                dockq_avg=progress.get("dockq_avg"),
                dockq_success_pct=progress.get("dockq_success_pct"),
                c_rmsd=progress.get("c_rmsd"),
                extra_tokens=progress.get("extra_tokens"),
            )
            print(f"[registry] appended row to {registry_path}")
        except Exception as reg_err:
            # Registry write must never mask the real failure.
            print(f"[registry] WARNING: failed to append row: {reg_err}")


if __name__ == "__main__":
    main()
