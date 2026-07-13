"""Per-target confidence head for ResFold one-step.

A tiny 2-layer MLP that takes mask-aware mean-pooled denoiser tokens as input
and produces a scalar predicted lDDT in ``[0, 1]`` (sigmoid output). Trained
jointly with the main centroid + atom losses at a small auxiliary weight so a
noisy regression target on small datasets cannot tank centroid quality. At
multi-sample eval the predicted scores rank the K samples without needing
ground truth at inference (see Loop 06 plan).

Shape contract (forward):
    tokens: [B, L, c_token]
    mask:   [B, L] boolean (True = real residue) or None
    return: [B] predicted lDDT, each in ``[0, 1]``.
"""


import torch
import torch.nn as nn
from torch import Tensor


class ConfidenceHead(nn.Module):
    """Per-target confidence regressor.

    Architecture: ``Linear(c_token -> hidden) -> SiLU -> Dropout -> Linear(hidden -> 1)``
    followed by ``sigmoid`` so the prediction is bounded in ``[0, 1]`` (the lDDT
    domain). Pooling over residues is a mask-aware mean of the denoiser tokens.

    Init detail: the output bias is set to 0 so that at initialization the head
    predicts ``sigmoid(0) = 0.5`` — the midpoint of the lDDT range. This keeps
    gradients live in both directions during the early steps.
    """

    def __init__(
        self,
        c_token: int = 128,
        hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.hidden = c_token if hidden is None else hidden

        self.fc1 = nn.Linear(c_token, self.hidden)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden, 1)

        # Predict 0.5 (midpoint of lDDT range) at init: bias = inverse_sigmoid(0.5) = 0.
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        """Predict per-target lDDT in ``[0, 1]``.

        Args:
            tokens: ``[B, L, c_token]`` denoiser tokens.
            mask: ``[B, L]`` boolean mask (True = real residue). If ``None``,
                every residue is treated as valid.

        Returns:
            ``[B]`` predicted lDDT, each in ``[0, 1]``.
        """
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            mask_f = mask.to(tokens.dtype).unsqueeze(-1)  # [B, L, 1]
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            pooled = (tokens * mask_f).sum(dim=1) / denom  # [B, c_token]

        h = self.fc1(pooled)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h).squeeze(-1)  # [B]
        return torch.sigmoid(logits)
