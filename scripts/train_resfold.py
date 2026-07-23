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

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

# Shared utilities
from script_utils import (
    Logger,
    get_data_path,
    plot_prediction,
    save_config,
    set_seed,
)

from tinyfold.inference import (
    sample_centroids,
    sample_centroids_with_sampler,
    sample_k_centroids,
    self_cond_rollout,
)

# Model imports
from tinyfold.model.diffusion import (
    create_noiser,
    create_sampler,
    create_schedule,
)

# Loss imports
from tinyfold.model.losses import (
    ContactLoss,
    GeometryLoss,
    apply_chain_swap,
    choose_chain_permutation,
    compute_c_rmsd,
    compute_distance_consistency_loss,
    compute_lddt,
    compute_lddt_metrics,
    compute_mse_loss,
    compute_rmse,
    kabsch_align,
)
from tinyfold.model.losses.torsion import torsion_symmetry_loss
from tinyfold.model.metrics import (
    cluster_poses,
    compute_dockq,
    interface_mask_from_gt,
    score_geometric_energy,
    score_self_consistency,
)
from tinyfold.model.resfold import ResFoldPipeline
from tinyfold.model.resfold.sidechain_torsion_head import wrap_angle
from tinyfold.retrieval import make_template_inputs
from tinyfold.sidechain_geometry import extract_chi

# Training utilities from tinyfold.training
from tinyfold.training import (
    EMA,
    MergedSampleStore,
    SampleStore,
    collate_batch,
    create_diffusion_components,
    create_train_sampler,
    get_or_create_split,
    load_model_checkpoint,
    random_rotation_matrix,
    read_train_table,
)
from tinyfold.training.data_split import verify_atom_counts
from tinyfold.training.eval import sample_centroids_continuous, summarize_eval_metrics
from tinyfold.training.objective import atom_loss_ramp
from tinyfold.training.registry_append import append_registry_row
from tinyfold.training.run_naming import generate_run_name
from tinyfold.training.utils import edm_loss_weight


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
                        choices=["random", "stratified", "cluster"],
                        help="Test-set sampling: 'random' (default) shuffles "
                             "eligible pool and takes n_test; 'stratified' bins by "
                             "LA+LB total residues and takes an equal share per bin "
                             "(so the headline number isn't dominated by small "
                             "complexes in datasets with a long size tail); "
                             "'cluster' holds out whole sequence clusters so the "
                             "test set is leakage-free. NEITHER 'random' NOR "
                             "'stratified' is leakage-safe -- on a random le200 "
                             "split 183/200 test complexes shared a cluster with "
                             "training (DockQ 0.251 leaked vs 0.044 clean).")
    parser.add_argument("--clusters", type=str, default="data/processed/clusters.json",
                        help="clusters.json used to audit train/test leakage on "
                             "every run, and to build the split when "
                             "--test_strategy cluster.")
    parser.add_argument("--require_clean_split", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Refuse to train when any test sample shares a "
                             "sequence cluster with training (default: on). Pass "
                             "--no-require-clean-split to train on a leaky split "
                             "anyway; the run is then labelled LEAKY in its log "
                             "and split.json.")
    parser.add_argument("--per_cluster_cap", type=int, default=1,
                        help="With --test_strategy cluster: max test samples from "
                             "any one cluster. 1 (default) means n_test samples "
                             "carry n_test independent clusters.")
    parser.add_argument("--n_test_clusters", type=int, default=None,
                        help="With --test_strategy cluster: stop after this many "
                             "test clusters instead of after --n_test samples.")
    parser.add_argument("--sample_cache", type=str, default="eager",
                        choices=["eager", "lru"],
                        help="How training samples are held in RAM. 'eager' "
                             "(default) decodes every sample before the first "
                             "step -- fine up to a few thousand complexes, but "
                             "the full 41,883-complex dataset costs ~46 GB that "
                             "way (~42 GB of it fp32 ESM embeddings). 'lru' "
                             "decodes on access and evicts by bytes, bounding "
                             "residency at --sample_cache_mb. Results are "
                             "identical either way: decoding is deterministic "
                             "and crops are still drawn per step at collate.")
    parser.add_argument("--sample_cache_mb", type=float, default=8192.0,
                        help="Byte budget for --sample_cache lru (default 8 GB). "
                             "Budgeting by bytes rather than entry count is "
                             "deliberate: complexes span 8-3,106 residues, so a "
                             "count-based bound would not bound memory.")
    parser.add_argument("--no_registry", action="store_true",
                        help="Do not append a row to experiments/REGISTRY.md. "
                             "For smoke tests and throwaway runs that should "
                             "not enter the experiment record.")
    parser.add_argument("--verify_atom_counts", action="store_true",
                        help="Debug: re-check the 4*(LA+LB) atom-count invariant "
                             "against the materialised atom_type lists before "
                             "splitting. Costs a ~56 s full-table scan; use after "
                             "rebuilding the parquet.")
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
    # --- Efficiency: "everyone does this" ---
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Gradient-checkpoint the pair-track triangle-mul "
                             "blocks (recompute in backward). Frees the O(L^2) "
                             "activation memory -> larger batch / L.")
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast for the training forward/backward "
                             "(halves activation memory, faster on Ampere+). "
                             "WARNING: diverges EDM training (large loss weights "
                             "-> NaN in bf16); leave off unless guarded.")
    parser.add_argument("--pair_to_single", action="store_true",
                        help="Higher-bandwidth template conditioning: inject the "
                             "pair representation back into the token content "
                             "(not just the per-head attention logit bias). "
                             "Requires --pair_repr.")
    parser.add_argument("--frame_atom_head", action="store_true",
                        help="Frame-based atom head: predict a per-residue rigid "
                             "frame (rotation+translation) and place an idealized "
                             "backbone template, instead of free global offsets. "
                             "Guarantees near-rigid backbone geometry.")
    parser.add_argument("--atom_diffusion", action="store_true",
                        help="Replace the one-shot atom head with an atom-DIFFUSION "
                             "stage: EDM diffusion over per-residue backbone offsets, "
                             "conditioned on the centroid stage's tokens. Iterative "
                             "refinement (sees current atom state). Use with "
                             "--no augment (frame-consistent offsets).")
    parser.add_argument("--atom_sigma_data", type=float, default=0.15,
                        help="EDM sigma_data for the atom-offset diffusion "
                             "(offset std in normalized units; ~1.5A/scale).")
    parser.add_argument("--atom_sigma_min", type=float, default=0.002)
    parser.add_argument("--atom_sigma_max", type=float, default=1.0)
    parser.add_argument("--atom_steps", type=int, default=8,
                        help="Atom-diffusion sampling steps at eval.")
    # --- Sidechain (third) diffusion stage: torsion (chi) packing ---
    parser.add_argument("--sidechain_diffusion", action="store_true",
                        help="Add a THIRD diffusion stage: torsion (chi) sidechain "
                             "packing conditioned on denoiser tokens + the predicted "
                             "backbone. Requires --atom14_cache_dir and (in practice) "
                             "--atom_diffusion. Default off = byte-identical.")
    parser.add_argument("--atom14_cache_dir", type=str, default=None,
                        help="Directory of per-sample atom14 npz "
                             "(coords_atom14/mask_atom14/seq_indices), keyed by "
                             "sample_id. Required for --sidechain_diffusion.")
    parser.add_argument("--sc_head_layers", type=int, default=2)
    parser.add_argument("--sc_head_heads", type=int, default=4)
    parser.add_argument("--sc_neighbor_graph", action="store_true",
                        help="C5 fallback: restrict sidechain-head attention to CA "
                             "neighbors within --sc_neighbor_radius (clash resolution). "
                             "Default off = dense global attention.")
    parser.add_argument("--sc_neighbor_radius", type=float, default=10.0)
    parser.add_argument("--sc_weight", type=float, default=0.5,
                        help="Weight of the sidechain chi loss (C2).")
    parser.add_argument("--sc_warmup_steps", type=int, default=500,
                        help="Warmup before the chi loss engages (C2).")
    parser.add_argument("--sc_sigma_min", type=float, default=0.02,
                        help="Min angular noise (radians) for chi diffusion.")
    parser.add_argument("--sc_sigma_max", type=float, default=3.0,
                        help="Max angular noise (radians) for chi diffusion.")
    parser.add_argument("--fixed_sigma", type=float, default=None,
                        help="Isolation test: train at this single fixed sigma "
                             "(near-clean input) to decouple the atom head from "
                             "diffusion noise. Pair with --sigma_max = same value "
                             "so eval one-shot uses the same near-clean sigma.")
    # --- Template conditioning (retrieval library) ---
    parser.add_argument("--template_cond", action="store_true",
                        help="Enable AF3-style template conditioning: relative "
                             "pair features (distogram + local-frame unit "
                             "vectors) built from per-residue template coords are "
                             "injected into the pair track. Requires --pair_repr.")
    parser.add_argument("--template_source", type=str, default="none",
                        choices=["none", "oracle", "oracle_monomer", "retrieved"],
                        help="Where template coords come from: 'oracle' "
                             "(self-template from GT, whole-complex frame; the "
                             "E2a positive control), 'oracle_monomer' (GT folds "
                             "but per-chain frames; docking hidden), 'retrieved' "
                             "(real retrieval, Milestone B), or 'none'.")
    parser.add_argument("--template_rbf", type=int, default=32,
                        help="Number of RBF distogram bins for template features "
                             "(default 32).")
    parser.add_argument("--template_d_max", type=float, default=4.0,
                        help="Max CA-CA distance (in the model's normalized "
                             "coord units) spanned by the distogram RBF centers. "
                             "Default 4.0 ~= 40 A at global_scale ~= 11.")
    parser.add_argument("--template_dropout", type=float, default=0.0,
                        help="Bernoulli per-residue template coverage dropout at "
                             "train time (0 = always full template; keep 0 for "
                             "the pure oracle upper-bound probe).")
    parser.add_argument("--template_cache_dir", type=str, default=None,
                        help="Directory of per-sample retrieved-template npz "
                             "(prepare_templates.py). Required for "
                             "--template_source retrieved.")
    # --- Coevolution conditioning (MSA pair prior; mirror of templates) ---
    parser.add_argument("--msa_cond", action="store_true",
                        help="Enable coevolution conditioning: cached APC-corrected "
                             "MSA pair features [L,L,F] are injected into the pair "
                             "track (additive with templates). Requires --pair_repr.")
    parser.add_argument("--msa_cache_dir", type=str, default=None,
                        help="Directory of per-sample coevolution npz "
                             "(prepare_msa_features.py, key 'msa_feats'). Required "
                             "for --msa_cond.")
    parser.add_argument("--msa_dropout", type=float, default=0.0,
                        help="Per-complex Bernoulli dropout of the whole coev "
                             "signal at train time (graceful degradation on "
                             "shallow-/no-MSA targets; 0 = always full MSA).")
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
    parser.add_argument("--atom_start_step", type=int, default=0,
                        help="Step at which the atom loss first enters (onestep only). "
                             "0 (default) = legacy schedule (ramp from step 0). A "
                             "positive value lets the centroid stage converge first, "
                             "THEN anneals atoms in over --atom_warmup_steps -- the "
                             "loss-balance lever for the measured 'atom loss slows "
                             "centroid convergence' effect. Do NOT instead detach the "
                             "atom conditioning: that was measured much worse for "
                             "atoms (0.32 A -> 1.14 A).")

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
    parser.add_argument("--self_cond_steps", type=int, default=1,
                        help="AF3 detached mini-rollout length for self-conditioning "
                             "(C4). 1 = single forward at the current sigma; 2-3 runs "
                             "a short reverse-diffusion rollout from the noised state.")

    # Recycling (C1)
    parser.add_argument("--n_recycle", type=int, default=0,
                        help="Max trunk recycling passes during training. Per step "
                             "the count is drawn uniformly from {0..n_recycle} "
                             "(0 = recycling off, bitwise-identical to pre-C1).")
    parser.add_argument("--n_recycle_eval", type=int, default=0,
                        help="Fixed number of trunk recycling passes at eval time.")

    # Diffusion multiplicity (C2)
    parser.add_argument("--diffusion_multiplicity", type=int, default=1,
                        help="Reuse one trunk pass across M noise draws per structure "
                             "(default 1 = current behaviour). Multiplies the denoiser "
                             "gradient signal without growing the effective batch; "
                             "onestep + continuous-sigma only.")

    # EMA of weights (C3)
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="Exponential moving average decay for weights "
                             "(0.0 = off, 0.999 typical). When on, eval and the "
                             "saved ema_state_dict use the smoothed weights; "
                             "model_state_dict stays the raw weights.")

    # Chain-permutation-aware loss (C5) -- correctness fix, on by default.
    parser.add_argument("--chain_perm_loss", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="For homodimers, score against whichever chain "
                             "assignment (identity or A<->B swap) the prediction is "
                             "closer to, applied consistently across all coordinate "
                             "losses. On by default (correctness fix, not a tunable).")

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
        with open(pre_args.config, encoding="utf-8") as f:
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

            # Template conditioning at eval: same source as training. For the
            # oracle positive control this feeds the GT self-template; the
            # samplers read batch['template_*'] (like esm_embed). No-op when
            # --template_source none.
            tmpl_c, tmpl_m, tmpl_f = make_template_inputs(
                batch, source=getattr(args, "template_source", "none"),
            )
            if tmpl_c is not None:
                batch['template_coords_res'] = tmpl_c
                batch['template_mask'] = tmpl_m
                batch['template_frame_id'] = tmpl_f

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
                        n_recycle=args.n_recycle_eval,
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
                    sample_out = sample_centroids_continuous(
                        model, batch, noiser, device,
                        one_shot=args.one_shot_sample,
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

        (
            test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg,
            extra_tokens, log_msg,
        ) = summarize_eval_metrics(
            mode=args.mode,
            k_list=k_list,
            test_rmses=test_rmses,
            test_dockq_scores=test_dockq_scores,
            test_lddt_scores=test_lddt_scores,
            test_ilddt_scores=test_ilddt_scores,
            test_atom_rmses=test_atom_rmses,
            test_c_rmsds=test_c_rmsds,
            per_k_oracle=per_k_oracle,
            per_k_mean=per_k_mean,
            per_k_ranked=per_k_ranked,
            per_k_ranked_conf=per_k_ranked_conf,
            per_k_ranked_consistency=per_k_ranked_consistency,
            per_k_ranked_energy=per_k_ranked_energy,
            all_pred_lddts=all_pred_lddts,
            all_neg_rmses=all_neg_rmses,
            pred_lddt_std_per_target=pred_lddt_std_per_target,
            n_test=len(test_indices),
            train_avg=train_avg,
            n_eval=n_eval,
        )
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
    # Column-projected read: the bond columns are 4.29 GB of the 7.37 GB table
    # and belong to the retired atom-graph model, which this script never uses.
    table = read_train_table(data_path)

    if getattr(args, "verify_atom_counts", False):
        logger.log("Verifying atom-count invariant 4*(LA+LB) (slow full scan)...")
        verify_atom_counts(table)
        logger.log(f"  OK on all {len(table)} rows")

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
    _tmpl_dir = getattr(args, "template_cache_dir", None)
    if getattr(args, "template_source", "none") == "retrieved" and _tmpl_dir is None:
        raise ValueError("--template_source retrieved requires --template_cache_dir")
    _msa_dir = getattr(args, "msa_cache_dir", None)
    if getattr(args, "msa_cond", False) and _msa_dir is None:
        raise ValueError("--msa_cond requires --msa_cache_dir")
    _atom14_dir = getattr(args, "atom14_cache_dir", None)
    if getattr(args, "sidechain_diffusion", False) and _atom14_dir is None:
        raise ValueError("--sidechain_diffusion requires --atom14_cache_dir")
    _loader_kwargs = dict(normalize=normalize, esm_cache_dir=_esm_dir,
                          per_chain_res_idx=per_chain, global_scale=gscale,
                          template_cache_dir=_tmpl_dir, msa_feats_cache_dir=_msa_dir,
                          atom14_cache_dir=_atom14_dir)
    _cache_mode = getattr(args, "sample_cache", "eager")
    _cache_mb = getattr(args, "sample_cache_mb", 8192.0)
    # The test split stays eager regardless: it is small (hundreds), it is
    # re-read every eval, and keeping it resident makes eval timing independent
    # of the training cache policy.
    train_samples = SampleStore(table, train_indices, mode=_cache_mode,
                                cache_mb=_cache_mb, loader_kwargs=_loader_kwargs)
    test_samples = SampleStore(table, test_indices, mode="eager",
                               loader_kwargs=_loader_kwargs)
    logger.log(f"  Loaded {len(train_samples)} train, {len(test_samples)} test samples "
               f"(train cache={_cache_mode}"
               + (f", budget {_cache_mb:.0f} MB" if _cache_mode == "lru" else "")
               + f"; resident {train_samples.resident_mb + test_samples.resident_mb:.0f} MB)")
    # Coverage diagnostics read a bounded sample: touching every entry for a
    # mean would materialise the whole split and defeat the cache.
    _diag = train_samples.subset(256, seed=args.seed) \
        if (_tmpl_dir is not None or _msa_dir is not None) else []
    if _tmpl_dir is not None:
        _cov = [float(s['template_mask'].float().mean()) for s in _diag if 'template_mask' in s]
        if _cov:
            logger.log(f"  Template coverage (train, n={len(_diag)} sampled): "
                       f"{100*sum(_cov)/len(_cov):.1f}% residues mean")
    if _msa_dir is not None:
        _n_msa = sum(1 for s in _diag if 'msa_feats' in s)
        logger.log(f"  Coevolution cache (train, n={len(_diag)} sampled): "
                   f"{_n_msa}/{len(_diag)} samples have msa_feats")

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

            logger.log("  Generating Stage 1 predictions (this may take a while)...")
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
            all_samples = MergedSampleStore(train_samples, test_samples)
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
        # set_overlay, not direct mutation: an lru store may evict the dict we
        # would have written into, silently dropping the Stage 1 prediction.
        for idx, pred in s1_predictions.items():
            if idx in train_samples:
                train_samples.set_overlay(idx, 'centroids_pred', pred)
            if idx in test_samples:
                test_samples.set_overlay(idx, 'centroids_pred', pred)
        logger.log("  Injected Stage 1 predictions into samples")

    # Create sampler for efficient batching
    # length_index(): the samplers bucket by 'n_res' and read nothing else, so
    # they get lengths from LA/LB rather than forcing every sample resident.
    train_sampler = create_train_sampler(args, train_samples.length_index(), logger)

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
            template_cond=getattr(args, "template_cond", False),
            template_rbf=getattr(args, "template_rbf", 32),
            template_d_max=getattr(args, "template_d_max", 4.0),
            msa_cond=getattr(args, "msa_cond", False),
            grad_checkpoint=getattr(args, "grad_checkpoint", False),
            pair_to_single=getattr(args, "pair_to_single", False),
            frame_atom_head=getattr(args, "frame_atom_head", False),
            global_scale=(getattr(args, "global_scale", None) or 11.0),
            atom_diffusion=getattr(args, "atom_diffusion", False),
            atom_sigma_data=getattr(args, "atom_sigma_data", 0.15),
            atom_sigma_min=getattr(args, "atom_sigma_min", 0.002),
            atom_sigma_max=getattr(args, "atom_sigma_max", 1.0),
            sidechain_diffusion=getattr(args, "sidechain_diffusion", False),
            sc_head_layers=getattr(args, "sc_head_layers", 2),
            sc_head_heads=getattr(args, "sc_head_heads", 4),
            sc_neighbor_graph=getattr(args, "sc_neighbor_graph", False),
            sc_neighbor_radius=getattr(args, "sc_neighbor_radius", 10.0),
            atom_head_layers=args.atom_head_layers,
            atom_head_heads=args.atom_head_heads,
            n_timesteps=args.T,
            dropout=0.0,
            aa_embed=args.aa_embed,
            esm_dim=args._esm_dim,
            confidence_head=args.confidence_head,
            sigma_data=args.sigma_data,
        ).to(device)
        model._atom_eval_steps = getattr(args, "atom_steps", 8)
        # In OneStep the model itself is the "stage 1" denoiser; alias for forward calls.
        stage1_module = model
        if args.checkpoint:
            load_model_checkpoint(model, args.checkpoint, args.mode, device, logger)
        pc = model.count_parameters()
        logger.log(f"Model: ResFoldOneStep ({args.mode})")
        logger.log(f"  Trunk params:     {pc['trunk']:,} ({pc['trunk_pct']:.1f}%)")
        logger.log(f"  Denoiser params:  {pc['denoiser']:,} ({pc['denoiser_pct']:.1f}%)")
        logger.log(f"  Atom-head params: {pc['atom_head']:,} ({pc['atom_head_pct']:.1f}%)")
        if pc.get('atom_diff', 0) > 0:
            logger.log(
                f"  Atom-diff params: {pc['atom_diff']:,} "
                f"({pc['atom_diff_pct']:.1f}%)"
            )
        if pc.get('confidence_head', 0) > 0:
            logger.log(
                f"  Confidence-head params: {pc['confidence_head']:,} "
                f"({pc['confidence_head_pct']:.1f}%)"
            )
        if pc.get('sc_head', 0) > 0:
            logger.log(
                f"  Sidechain-head params: {pc['sc_head']:,} "
                f"({pc['sc_head_pct']:.1f}%)"
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
        logger.log("  kabsch_interp: True (Boltz trajectory-frame alignment)")
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
    # Everything before this point is setup and validation. Only past here has
    # a run earned a REGISTRY.md row -- see the finally: block in main().
    progress["training_started"] = True
    start_time = time.time()

    # C3: EMA of weights. Off when --ema_decay <= 0 (the shadow is never built,
    # so the training step is bitwise-unchanged). When on, the shadow updates
    # after each optimizer step; eval and the saved ema_state_dict read it.
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

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

            # C2: diffusion multiplicity. Replicate each sample M times so the
            # whole downstream pipeline (noise, sigma, augmentation, every loss)
            # operates on B*M with no further changes. The trunk is coordinate-
            # blind, so a sample's M copies share identical trunk inputs; the
            # forward below computes the trunk ONCE and repeats its tokens, which
            # is the point -- M noise draws share one (expensive) trunk pass. M=1
            # replicates nothing and is bitwise-identical to the pre-C2 step.
            _mult = getattr(args, "diffusion_multiplicity", 1)
            if _mult > 1:
                if not (args.continuous_sigma and is_onestep):
                    raise ValueError(
                        "--diffusion_multiplicity>1 requires --continuous_sigma and "
                        "the onestep model (the forward_sigma_with_trunk seam)."
                    )

                def _rep(v, m=_mult):
                    if torch.is_tensor(v):
                        return v.repeat_interleave(m, dim=0)
                    if isinstance(v, list):
                        return [x for x in v for _ in range(m)]
                    return v
                batch = {k: _rep(v) for k, v in batch.items()}
                current_batch_size = batch['centroids'].shape[0]

            # Sample noise levels and add noise to centroids (for Stage 1)
            noise = torch.randn_like(batch['centroids'])

            # Target for loss (may be rotated by augmentation)
            centroids_target = batch['centroids']
            # Rotation applied to the centroid frame this step (None if aug off).
            # The atom-diffusion offsets must live in the SAME frame as the
            # conditioning tokens (derived from the rotated x_t), so we reuse it.
            aug_R = None

            if args.continuous_sigma:
                # AF3-style: continuous sigma with VE noise (x_t = x0 + sigma * noise)
                if getattr(args, "fixed_sigma", None) is not None:
                    # Isolation test: train at a single small sigma so the
                    # denoiser input is near-clean (x_t ~= GT centroids) and the
                    # atom head trains as a regression, decoupled from diffusion
                    # noise. Used to compare atom-head encodings cleanly.
                    sigma = torch.full((current_batch_size,), float(args.fixed_sigma), device=device)
                elif args.stratified_sigma:
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
                    aug_R = R

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

                    # Apply a different random rotation to each expanded copy.
                    if args.augment_rotation:
                        R = random_rotation_matrix(centroids_exp.shape[0], device)
                        centroids_aug = torch.bmm(centroids_exp, R.transpose(1, 2))
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
                    # Template conditioning: build per-residue template inputs
                    # once per batch (oracle self-template for E2a, etc.). Returns
                    # (None, None, None) when --template_source none.
                    tmpl_coords, tmpl_mask, tmpl_frame = make_template_inputs(
                        batch,
                        source=getattr(args, "template_source", "none"),
                        dropout=getattr(args, "template_dropout", 0.0),
                    )
                    # Coevolution: cached [B,L,L,F] already in the batch (via
                    # collate). Per-complex dropout zeros the whole signal for a
                    # sampled subset so the model degrades gracefully on targets
                    # with no MSA. Mirrors template_dropout but per-complex (the
                    # feature is O(L^2), not per-residue).
                    msa_feats = batch.get('msa_feats')
                    _msa_p = getattr(args, "msa_dropout", 0.0)
                    if msa_feats is not None and _msa_p > 0.0:
                        keep = (torch.rand(msa_feats.shape[0], device=msa_feats.device) >= _msa_p)
                        msa_feats = msa_feats * keep.view(-1, 1, 1, 1).to(msa_feats.dtype)
                    if args.continuous_sigma:
                        # AF3-style with continuous sigma
                        # Recycling (C1): draw the pass count uniformly from
                        # {0..n_recycle} per step (PairMixer/Boltz recipe). 0 when
                        # --n_recycle is 0, so the trunk runs exactly once as before.
                        _n_rc = (
                            int(torch.randint(0, args.n_recycle + 1, (1,)).item())
                            if args.n_recycle > 0 else 0
                        )
                        # Self-conditioning (C4): with probability p, run an AF3
                        # detached mini-rollout from the current noised state and
                        # feed its clean-coord estimate back as x0_prev for the one
                        # trained step. Shared with inference via self_cond_rollout
                        # (no_grad + detached inside), so train and eval cannot
                        # drift. self_cond_prob=0 skips this entirely -> x0_prev
                        # stays None, bitwise-identical to the no-self-cond step.
                        x0_prev = None
                        _amp = getattr(args, "amp", False) and torch.cuda.is_available()
                        if args.self_cond_prob > 0 and torch.rand(1).item() < args.self_cond_prob:
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
                                x0_prev = self_cond_rollout(
                                    stage1_module, batch, x_t, sigma,
                                    n_steps=args.self_cond_steps, is_onestep=is_onestep,
                                    sigma_min=args.sigma_min, n_recycle=_n_rc,
                                    template_kwargs={
                                        'template_coords_res': tmpl_coords,
                                        'template_mask': tmpl_mask,
                                        'template_frame_id': tmpl_frame,
                                        'msa_feats': msa_feats,
                                    },
                                )

                        # Main forward pass (with or without self-conditioning)
                        atom_cond_tokens = None
                        _atom_diff = is_onestep and getattr(stage1_module, "atom_diffusion", False)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
                            if _atom_diff:
                                # Atom-diffusion path: centroid stage returns tokens;
                                # atoms come from the atom-diffusion loss below.
                                centroids_pred, atom_cond_tokens, pred_lddt = stage1_module.centroid_tokens(
                                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                    sigma, batch['mask_res'], x0_prev=x0_prev,
                                    esm_embed=batch.get('esm_embed'),
                                    template_coords_res=tmpl_coords,
                                    template_mask=tmpl_mask,
                                    template_frame_id=tmpl_frame,
                                    msa_feats=msa_feats,
                                    n_recycle=_n_rc,
                                )
                                atoms_pred = None
                            elif (
                                _mult > 1
                                and tmpl_coords is None
                                and msa_feats is None
                            ):
                                # C2 trunk-once: a sample's M copies share identical
                                # trunk inputs (the trunk is coordinate-blind), so
                                # run the trunk on copy 0 of each group and repeat
                                # its tokens across the group. Exact -- the trunk is
                                # a pure function of these (copy-invariant) inputs.
                                # Only valid without templates/msa, whose per-copy
                                # dropout would differ across copies; with those on
                                # we fall through to forward_sigma (trunk runs M
                                # times -- correct, just not memory-optimal).
                                _esm = batch.get('esm_embed')
                                trunk_tokens_1 = stage1_module.get_trunk_tokens(
                                    batch['aa_seq'][::_mult], batch['chain_ids'][::_mult],
                                    batch['res_idx'][::_mult], batch['mask_res'][::_mult],
                                    esm_embed=(_esm[::_mult] if _esm is not None else None),
                                    n_recycle=_n_rc,
                                )
                                trunk_tokens_m = trunk_tokens_1.repeat_interleave(_mult, dim=0)
                                fwd_out = stage1_module.forward_sigma_with_trunk(
                                    x_t, trunk_tokens_m, sigma, batch['mask_res'],
                                    x0_prev=x0_prev,
                                    res_idx=batch['res_idx'], chain_ids=batch['chain_ids'],
                                )
                            else:
                                fwd_out = stage1_module.forward_sigma(
                                    x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                    sigma, batch['mask_res'], x0_prev=x0_prev,
                                    esm_embed=batch.get('esm_embed'),
                                    template_coords_res=tmpl_coords,
                                    template_mask=tmpl_mask,
                                    template_frame_id=tmpl_frame,
                                    msa_feats=msa_feats,
                                    n_recycle=_n_rc,
                                )
                        # Cast predictions back to fp32 for the loss (stable).
                        if _atom_diff:
                            centroids_pred = centroids_pred.float()
                            atom_cond_tokens = atom_cond_tokens.float()
                        elif is_onestep:
                            fwd_out = tuple(o.float() if torch.is_tensor(o) else o for o in fwd_out)
                            centroids_pred, atoms_pred, pred_lddt = fwd_out
                        else:
                            centroids_pred = fwd_out.float()
                            atoms_pred = None
                    else:
                        # Original behavior with discrete timesteps
                        if is_onestep:
                            centroids_pred, atoms_pred = stage1_module(
                                x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                t, batch['mask_res'],
                                esm_embed=batch.get('esm_embed'),
                                template_coords_res=tmpl_coords,
                                template_mask=tmpl_mask,
                                template_frame_id=tmpl_frame,
                                msa_feats=msa_feats,
                            )
                        else:
                            centroids_pred = model.forward_stage1(
                                x_t, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                                t, batch['mask_res'],
                                esm_embed=batch.get('esm_embed'),
                            )
                            atoms_pred = None

                    # C5: chain-permutation-aware target. For sequence-identical
                    # (homodimer) chains -- 34.5% of DIPS -- the GT "A"/"B"
                    # labelling is arbitrary, so pick per sample the assignment
                    # (identity vs A<->B swap) that minimises the centroid MSE and
                    # apply that SAME swap to every coordinate target (centroids,
                    # coords_res, atoms), so all loss terms stay consistent. Chosen
                    # once, on the DETACHED prediction. Off -> targets unchanged
                    # (bitwise). Heterodimers are never swapped.
                    if getattr(args, "chain_perm_loss", True) and is_onestep:
                        _swap = choose_chain_permutation(
                            centroids_pred.detach(), centroids_target,
                            batch['chain_ids'], batch['aa_seq'], batch['mask_res'],
                        )
                        if bool(_swap.any()):
                            centroids_target = apply_chain_swap(
                                centroids_target, batch['chain_ids'],
                                batch['mask_res'], _swap)
                            for _k in ('coords_res', 'centroids', 'atom14_gt'):
                                if _k in batch and torch.is_tensor(batch[_k]):
                                    batch[_k] = apply_chain_swap(
                                        batch[_k], batch['chain_ids'],
                                        batch['mask_res'], _swap)

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
                    loss_chi = 0.0
                    loss_geom = 0.0
                    loss_bond = 0.0
                    loss_angle = 0.0
                    loss_omega = 0.0
                    alpha_atom = 0.0
                    if _atom_diff and atom_cond_tokens is not None:
                        # === ATOM-DIFFUSION loss ===
                        B, L = centroids_pred.shape[:2]
                        adh = stage1_module.atom_diff_head
                        sd_a = adh.sigma_data
                        # Offsets from GT centroids, in the SAME frame as the
                        # conditioning tokens. With rotation augmentation the
                        # centroid frame is rotated by aug_R, so rotate the GT
                        # centroids AND atoms by the same R (offsets are
                        # frame-equivariant: R*(atoms - centroid)). Translation
                        # aug is irrelevant -- offsets are translation-invariant.
                        cen_anchor = batch['centroids']                     # [B,L,3]
                        coords_gt = batch['coords_res']                     # [B,L,4,3]
                        if aug_R is not None:
                            cen_anchor = torch.bmm(cen_anchor, aug_R.transpose(1, 2))
                            coords_gt = torch.bmm(
                                coords_gt.reshape(B, L * 4, 3), aug_R.transpose(1, 2)
                            ).reshape(B, L, 4, 3)
                        delta0 = coords_gt - cen_anchor.unsqueeze(2)        # [B,L,4,3]
                        # log-uniform atom sigma in [atom_sigma_min, atom_sigma_max]
                        lo = math.log(stage1_module.atom_sigma_min); hi = math.log(stage1_module.atom_sigma_max)
                        sig_a = torch.exp(torch.rand(B, device=delta0.device) * (hi - lo) + lo)
                        eps_a = torch.randn_like(delta0)
                        delta_t = delta0 + sig_a.view(B, 1, 1, 1) * eps_a
                        # Full e2e conditioning (NOT detached): measured decisively
                        # better atom placement -- atoms given GT centroids reach
                        # 0.32 A / DockQ 0.96 e2e vs 1.14 A / 0.77 when detached and
                        # 1.37 A / 0.63 for the regression head. The e2e gradient
                        # helps the atom head; the only cost is slightly slower
                        # centroid convergence (loss-balance problem, handled
                        # separately, not by starving the atom gradient).
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
                            delta_pred = stage1_module.denoise_atoms(delta_t, atom_cond_tokens, sig_a, batch['mask_res'])
                        delta_pred = delta_pred.float()
                        # EDM loss on the OFFSET (well-conditioned: lambda*c_out^2=1).
                        # NOT on absolute atoms -- that would multiply the (un-scaled)
                        # centroid error by the huge low-sigma weight. Offsets are in
                        # a fixed frame (aug off) so no Kabsch.
                        lam_a = (sig_a ** 2 + sd_a ** 2) / (sig_a * sd_a) ** 2        # EDM weight [B]
                        per_sample_atom = compute_mse_loss(
                            delta_pred.reshape(B, L * 4, 3),
                            delta0.reshape(B, L * 4, 3),
                            batch['mask_atom'], use_kabsch=False, reduction='per_sample',
                        )
                        atom_loss_t = (per_sample_atom * lam_a).mean()
                        alpha_atom = atom_loss_ramp(
                            step,
                            weight=args.atom_weight,
                            warmup_steps=args.atom_warmup_steps,
                            start_step=getattr(args, "atom_start_step", 0),
                        )
                        loss = loss + alpha_atom * atom_loss_t
                        loss_atom = atom_loss_t.item()
                        atoms_pred_ad = cen_anchor.unsqueeze(2) + delta_pred   # GT centroid + pred offset (same frame)
                        if geom_loss_fn is not None and args.geom_weight > 0:
                            geom_losses = geom_loss_fn(atoms_pred_ad, batch['mask_res'], gt_coords=coords_gt)
                            loss = loss + args.geom_weight * geom_losses['total']
                            loss_geom = geom_losses['total'].item()
                            loss_bond = geom_losses['bond_length'].item()
                            loss_angle = geom_losses['bond_angle'].item()
                            loss_omega = geom_losses['omega'].item()

                        # === SIDECHAIN (third-stage) chi loss (C2) ===
                        # Co-trained on top of the atom-diff backbone. Denoise
                        # noised chi conditioned on the denoiser tokens + the
                        # model's OWN predicted backbone (detached so chi grads
                        # cannot perturb the backbone/centroid stages). chi is a
                        # dihedral -> rigid+scale-invariant, so the atom14 frame
                        # vs the training frame is irrelevant to the target.
                        _sc_on = getattr(stage1_module, "sidechain_diffusion", False)
                        if _sc_on and stage1_module.sc_head is not None and 'atom14_gt' in batch:
                            aatype_sc = batch['aa_seq'].long()
                            gt_chi, chi_mask = extract_chi(
                                batch['atom14_gt'], aatype_sc, batch['atom14_mask'])
                            # Per-residue CA-centered local frame; global context
                            # arrives via atom_cond_tokens + the head's attention.
                            pred_bb = atoms_pred_ad.detach()                 # [B,L,4,3]
                            bb_feats = pred_bb - pred_bb[:, :, 1:2, :]
                            ca_pos_sc = pred_bb[:, :, 1, :]                  # absolute CA (neighbor graph)
                            lo_c = math.log(args.sc_sigma_min)
                            hi_c = math.log(args.sc_sigma_max)
                            sig_c = torch.exp(
                                torch.rand(B, device=gt_chi.device) * (hi_c - lo_c) + lo_c)
                            chi_t = wrap_angle(
                                gt_chi + sig_c.view(B, 1, 1) * torch.randn_like(gt_chi))
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
                                _, vec = stage1_module.denoise_chi(
                                    chi_t, atom_cond_tokens, sig_c, bb_feats,
                                    aatype_sc, batch['mask_res'], ca_pos=ca_pos_sc)
                            vec = vec.float()
                            chi_loss_t = torsion_symmetry_loss(
                                vec, gt_chi, chi_mask, aatype_sc)
                            warmup_sc = args.sc_warmup_steps
                            ramp_sc = (min(1.0, float(step) / float(warmup_sc))
                                       if warmup_sc > 0 else 1.0)
                            loss = loss + ramp_sc * args.sc_weight * chi_loss_t
                            loss_chi = chi_loss_t.item()
                    elif is_onestep and atoms_pred is not None:
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
                                'chi': loss_chi,
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
        if ema is not None:
            ema.update(model)
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
                chi_str = f" | chi: {lc['chi']:.4f}" if lc.get('chi', 0) else ""
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | mse: {lc['mse']:.4f} | dst: {lc['dist']:.4f}{contact_str}{weight_str}{conf_str}{chi_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            else:
                logger.log(f"Step {step:5d} | loss: {loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0:
            model.eval()
            # C3: evaluate from the EMA weights when enabled (store the live
            # weights first, swap the shadow in; restored to raw before saving so
            # model_state_dict stays the raw weights).
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            # Release cached-but-unallocated blocks before eval. The pair track
            # is O(L^2); thousands of training steps fragment the caching
            # allocator, so eval's large contiguous allocation can fail
            # ("reserved but unallocated") even though total free memory suffices.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                            sample_out = sample_centroids_continuous(
                                model, batch, noiser, device,
                                one_shot=args.one_shot_sample,
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
                        sample_out = sample_centroids_continuous(
                            model, batch, noiser, device,
                            one_shot=args.one_shot_sample,
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

                # Plotting is cosmetic; never let a viz/IO error abort a long
                # training run. (Also recreate plots_dir defensively in case it
                # was removed mid-run.)
                try:
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_path = os.path.join(plots_dir, f'step_{step:06d}.png')
                    plot_prediction(pred_aligned[0], target_c[0], chain_ids_plot,
                                   s['sample_id'], rmse_viz, plot_path)
                    logger.log(f"         >>> Saved plot: {plot_path}")
                except Exception as e:
                    logger.log(f"         >>> Plot skipped ({type(e).__name__}: {e})")

                # C3: restore raw (live) weights before saving, so
                # model_state_dict is always the raw weights. The EMA weights
                # ride along under 'ema_state_dict'.
                if ema is not None:
                    ema.restore(model)
                _ema_sd = ema.state_dict(model) if ema is not None else None

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
                        'ema_state_dict': _ema_sd,
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    logger.log("         >>> New best test RMSE! Saved.")

                # Always save best-on-train (useful for N=1 overfit runs where test
                # is noise and best_model.pt freezes early).
                if not hasattr(_run_training, "_best_train") or train_avg < _run_training._best_train:
                    _run_training._best_train = train_avg
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': _ema_sd,
                        'train_rmse': train_avg,
                        'test_rmse': test_avg,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_train_model.pt'))

            model.train()

    # Final summary
    total_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log("Training complete")
    logger.log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best test RMSE: {best_rmse:.4f} A")
    logger.log("")
    # Save final-step checkpoint regardless of eval outcomes. This is the source of
    # truth for "what the model looks like at the end of training" — useful when
    # the eval-tracked best is set early (e.g., N=1 overfit) and then never moves.
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': (ema.state_dict(model) if ema is not None else None),
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
        # A run that never reached the training loop is a rejected
        # configuration, not an experiment. Appending it anyway filled the
        # registry with rows like "crashed: SplitLeakageError" for runs that
        # never trained a step -- noise in the one file that is supposed to be
        # the experiment record.
        if not progress.get("training_started"):
            print(f"[registry] no row: run ended before training started "
                  f"({outcome})")
        elif getattr(args, "no_registry", False):
            print("[registry] no row: --no_registry")
        else:
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
