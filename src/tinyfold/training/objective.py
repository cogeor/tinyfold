"""Composable training objectives with pluggable loss terms.

This keeps loss wiring declarative and allows scripts to swap terms
without editing core training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch


LossFn = Callable[..., torch.Tensor]


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
        self._terms: Dict[str, LossFn] = {}

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

