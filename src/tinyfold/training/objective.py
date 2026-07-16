"""Composable training objectives with pluggable loss terms.

This keeps loss wiring declarative and allows scripts to swap terms
without editing core training loops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

LossFn = Callable[..., torch.Tensor]


def atom_loss_ramp(
    step: int,
    *,
    weight: float,
    warmup_steps: int,
    start_step: int = 0,
) -> float:
    """Weight on the atom-diffusion loss at ``step`` (linear warmup).

    The atom stage shares the trunk with the centroid stage, so its gradient
    competes with centroid convergence. This is a loss-BALANCE knob: it shapes
    how much atom gradient enters over time WITHOUT detaching the conditioning
    (detaching was measured decisively worse for atoms: 0.32 A -> 1.14 A).

    Args:
        step:         current global training step.
        weight:       target weight once warmed up (``--atom_weight``).
        warmup_steps: linear ramp length. ``<= 0`` applies ``weight`` at once.
        start_step:   step at which the atom loss first enters. ``0`` (default)
                      reproduces the original schedule; a positive value lets
                      centroids converge first, then anneals atoms in.

    Returns:
        The scalar weight in ``[0, weight]``, non-decreasing in ``step``.
    """
    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return weight
    ramp = min(1.0, (step - start_step) / warmup_steps)
    return ramp * weight


@dataclass(frozen=True)
class LossTerm:
    """A named objective term with an optional scalar weight."""

    name: str
    fn: LossFn
    weight: float = 1.0
    enabled: bool = True


class LossRegistry:
    """Registry for named loss functions."""

    def __init__(self):
        self._terms: dict[str, LossFn] = {}

    def register(self, name: str, fn: LossFn) -> None:
        if not name:
            raise ValueError("Loss name must be non-empty")
        self._terms[name] = fn

    def get(self, name: str) -> LossFn:
        if name not in self._terms:
            available = ", ".join(sorted(self._terms))
            raise KeyError(f"Unknown loss '{name}'. Available: {available}")
        return self._terms[name]

    def has(self, name: str) -> bool:
        return name in self._terms

    def names(self) -> list[str]:
        return sorted(self._terms)


class LossComposer:
    """Combine multiple loss terms into a single scalar objective."""

    def __init__(self, terms: list[LossTerm]):
        self.terms = terms

    def __call__(self, **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        metrics: dict[str, float] = {}

        for term in self.terms:
            if not term.enabled or term.weight == 0.0:
                continue

            value = term.fn(**kwargs)
            weighted = value * term.weight
            total = weighted if total is None else total + weighted
            metrics[term.name] = value.detach().item()

        if total is None:
            raise ValueError("No enabled loss terms found")

        metrics["total"] = total.detach().item()
        return total, metrics

