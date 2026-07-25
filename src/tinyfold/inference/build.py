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

import torch

from tinyfold.model.resfold.config import ResFoldConfig
from tinyfold.model.resfold.onestep import ResFoldOneStep

# Params introduced by off-by-default capabilities (zero-init, unused at eval)
# that a pre-capability checkpoint may legitimately lack. Matched as substrings of
# the state-dict key. Keep this list tight: only additive, eval-inert modules.
_OFFDEFAULT_PARAM_TOKENS = (
    "recycle_norm", "recycle_proj",            # C1 recycling (single track)
    "recycle_pair_norm", "recycle_pair_proj",  # C1 recycling (pair track)
)


def _is_offdefault_param(key: str) -> bool:
    return any(tok in key for tok in _OFFDEFAULT_PARAM_TOKENS)


def build_onestep_from_config(cfg: dict) -> ResFoldOneStep:
    """Instantiate ResFoldOneStep with the architecture recorded in ``cfg``.

    The run-config -> constructor key translation lives in
    :meth:`ResFoldConfig.from_config`.
    """
    return ResFoldOneStep(**ResFoldConfig.from_config(cfg).to_kwargs())


def load_onestep_run(checkpoint_path, device, ema: bool = False) -> tuple[ResFoldOneStep, dict]:
    """Load a trained ResFoldOneStep from a checkpoint + its sibling config.json.

    Returns ``(model_in_eval_mode, config_dict)``. Raises if the config.json is
    missing (older runs) or the run is not a onestep model.

    ``ema=True`` selects the EMA weights (``ema_state_dict``, C3) when the
    checkpoint carries them; raises if requested but absent.
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
    # weights_only=True: our checkpoints hold only tensors + plain dicts/numbers,
    # so this is safe and blocks arbitrary code execution from an untrusted .pt.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if ema:
        if ckpt.get("ema_state_dict") is None:
            raise ValueError(
                f"ema=True but {ckpt_path} has no ema_state_dict (run trained "
                f"without --ema_decay, or an older checkpoint)."
            )
        state = ckpt["ema_state_dict"]
    else:
        state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Off-by-default capabilities added after a checkpoint was trained introduce
    # zero-init params that are unused at eval, so a pre-capability checkpoint
    # legitimately lacks them and they stay at their (no-op) constructed values.
    # C1 recycling is the case: recycle_proj is zero-init and the recycle path is
    # only taken when n_recycle>0, which eval never does by default. Tolerate
    # exactly those missing keys; any other missing key, or ANY unexpected key,
    # is a real architecture mismatch and still raises.
    hard_missing = [k for k in missing if not _is_offdefault_param(k)]
    if hard_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/architecture mismatch for {ckpt_path}: "
            f"{len(hard_missing)} missing, {len(unexpected)} unexpected keys "
            f"(ignoring {len(missing) - len(hard_missing)} benign off-by-default "
            f"params). config.json does not describe this checkpoint."
        )
    model.eval()
    return model, cfg
