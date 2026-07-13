"""Typed configuration for :class:`ResFoldOneStep`.

``ResFoldConfig`` mirrors the ~30 constructor arguments of ``ResFoldOneStep``
(same names, same defaults) so callers can pass a validated object instead of a
loose kwargs dict, and so the run-config key translation (``c_token_s1`` ->
``c_token``, ``T`` -> ``n_timesteps``, ...) lives in one place.

Contract: ``ResFoldOneStep(**ResFoldConfig(...).to_kwargs())`` is identical to
calling ``ResFoldOneStep(...)`` directly. A signature-drift test guards that the
field set stays in lock-step with the constructor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ResFoldConfig:
    """Architecture hyperparameters for ResFoldOneStep (1:1 with its __init__)."""

    c_token: int = 128
    trunk_layers: int = 4
    trunk_heads: int = 8
    denoiser_blocks: int = 4
    denoiser_heads: int = 8
    atom_head_layers: int = 2
    atom_head_heads: int = 4
    n_timesteps: int = 50
    n_aa_types: int = 21
    n_chains: int = 2
    dropout: float = 0.0
    aa_embed: str = "learned"
    esm_dim: int | None = None
    confidence_head: bool = False
    sigma_data: float = 1.0
    relpos_bias: bool = False
    relpos_clip: int = 32
    pair_repr: bool = False
    c_pair: int = 64
    pair_layers: int = 3
    pair_hidden: int = 64
    template_cond: bool = False
    template_rbf: int = 32
    template_d_max: float = 4.0
    grad_checkpoint: bool = False
    pair_to_single: bool = False
    frame_atom_head: bool = False
    global_scale: float = 11.0
    atom_diffusion: bool = False
    atom_sigma_data: float = 0.15
    atom_sigma_min: float = 0.002
    atom_sigma_max: float = 1.0

    def to_kwargs(self) -> dict:
        """Return the fields as a kwargs dict for ``ResFoldOneStep(**...)``."""
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict) -> ResFoldConfig:
        """Translate a training run's ``config.json`` dict into a config.

        Reproduces the exact key mapping that ``build_onestep_from_config`` used:
        run-config names (``c_token_s1``, ``T``, ``_esm_dim``) differ from the
        constructor's, and a few values carry hardcoded conventions (dropout=0.0
        at inference; esm_dim defaults to 480; global_scale falsy -> 11.0). Keys
        not present in the run config fall back to the dataclass defaults.
        """
        return cls(
            c_token=cfg["c_token_s1"],
            trunk_layers=cfg["trunk_layers"],
            denoiser_blocks=cfg["denoiser_blocks"],
            atom_head_layers=cfg["atom_head_layers"],
            atom_head_heads=cfg["atom_head_heads"],
            n_timesteps=cfg.get("T", 50),
            dropout=0.0,
            aa_embed=cfg.get("aa_embed", "learned"),
            esm_dim=cfg.get("_esm_dim", cfg.get("esm_dim", 480)),
            confidence_head=cfg.get("confidence_head", False),
            sigma_data=cfg.get("sigma_data", 1.0),
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
            frame_atom_head=cfg.get("frame_atom_head", False),
            global_scale=(cfg.get("global_scale") or 11.0),
            atom_diffusion=cfg.get("atom_diffusion", False),
            atom_sigma_data=cfg.get("atom_sigma_data", 0.15),
            atom_sigma_min=cfg.get("atom_sigma_min", 0.002),
            atom_sigma_max=cfg.get("atom_sigma_max", 1.0),
        )
