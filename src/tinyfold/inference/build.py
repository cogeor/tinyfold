"""Build a ResFoldOneStep model from a run's saved config.json.

Eval/showcase tooling used to hardcode the architecture
(``ResFoldOneStep(c_token=256, trunk_layers=6, ...)``), which silently mismatches
any checkpoint trained with different hyperparameters. Every training run writes a
``config.json`` next to its checkpoint; this reads it so the model is reconstructed
exactly as trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep


def build_onestep_from_config(cfg: dict) -> ResFoldOneStep:
    """Instantiate ResFoldOneStep with the architecture recorded in ``cfg``."""
    return ResFoldOneStep(
        c_token=cfg["c_token_s1"],
        trunk_layers=cfg["trunk_layers"],
        denoiser_blocks=cfg["denoiser_blocks"],
        relpos_bias=cfg.get("relpos_bias", False),
        relpos_clip=cfg.get("relpos_clip", 32),
        pair_repr=cfg.get("pair_repr", False),
        c_pair=cfg.get("c_pair", 64),
        pair_layers=cfg.get("pair_layers", 3),
        pair_hidden=cfg.get("pair_hidden", 64),
        template_cond=cfg.get("template_cond", False),
        template_rbf=cfg.get("template_rbf", 32),
        template_d_max=cfg.get("template_d_max", 4.0),
        pair_to_single=cfg.get("pair_to_single", False),
        atom_head_layers=cfg["atom_head_layers"],
        atom_head_heads=cfg["atom_head_heads"],
        n_timesteps=cfg.get("T", 50),
        dropout=0.0,
        aa_embed=cfg.get("aa_embed", "learned"),
        esm_dim=cfg.get("_esm_dim", cfg.get("esm_dim", 480)),
        confidence_head=cfg.get("confidence_head", False),
        sigma_data=cfg.get("sigma_data", 1.0),
    )


def load_onestep_run(checkpoint_path, device) -> Tuple[ResFoldOneStep, dict]:
    """Load a trained ResFoldOneStep from a checkpoint + its sibling config.json.

    Returns ``(model_in_eval_mode, config_dict)``. Raises if the config.json is
    missing (older runs) or the run is not a onestep model.
    """
    ckpt_path = Path(checkpoint_path)
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.json next to {ckpt_path}. The architecture cannot be "
            f"inferred for this checkpoint (older runs predate config dumping)."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model_kind") != "onestep":
        raise ValueError(
            f"load_onestep_run expects model_kind=onestep, got {cfg.get('model_kind')!r}"
        )
    model = build_onestep_from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/architecture mismatch for {ckpt_path}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
            f"config.json does not describe this checkpoint."
        )
    model.eval()
    return model, cfg
