"""Produce per-residue template inputs for a training/eval batch.

Returns the three per-residue tensors the trunk's pair track consumes:
``template_coords_res [B, L, 4, 3]``, ``template_mask [B, L]`` (coverage), and
``template_frame_id [B, L]`` (rigid-group id; pairs across frames are invalid).

Sources
-------
* ``"none"``      — no template (returns all None); model runs template-free.
* ``"oracle"``    — self-template from the batch's GT ``coords_res``, single
                    frame (frame_id = 0) so ALL pairs are valid, including the
                    cross-chain docking. This is the E2a positive control: the
                    perfect answer is handed to the model; DockQ MUST approach
                    ceiling or the conditioning is unwired (SPEC §4.1).
* ``"oracle_monomer"`` — self-template but frame_id = chain_id, so only
                    intra-chain pairs are valid. Isolates "can the model DOCK
                    given perfect monomer folds?" — the realistic ceiling for
                    monomer-fold retrieval.
* ``"retrieved"`` — real retrieved monomer templates from the cache (Milestone
                    B; wired in a later commit). Not yet implemented here.

``dropout`` randomly zeroes per-residue coverage (Bernoulli) so a template-
conditioned model does not become fully dependent on always having a template.
Keep it 0 for the pure oracle upper-bound probe.
"""

from __future__ import annotations

from typing import Optional

import torch


ORACLE_SOURCES = {"oracle", "oracle_monomer"}
VALID_SOURCES = {"none", "retrieved"} | ORACLE_SOURCES


def make_template_inputs(
    batch: dict,
    source: str = "none",
    dropout: float = 0.0,
    generator: Optional[torch.Generator] = None,
):
    """Return (template_coords_res, template_mask, template_frame_id) or Nones.

    ``batch`` must carry ``coords_res [B, L, 4, 3]``, ``mask_res [B, L]`` and
    ``chain_ids [B, L]`` (the standard collate output).
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"unknown template source {source!r}; expected {sorted(VALID_SOURCES)}")
    if source == "none":
        return None, None, None
    if source == "retrieved":
        raise NotImplementedError(
            "template source 'retrieved' is wired in Milestone B (needs the "
            "retriever + template cache)."
        )

    coords_res = batch["coords_res"]
    mask = batch["mask_res"].clone()
    chain_ids = batch["chain_ids"]

    if dropout and dropout > 0.0:
        keep = torch.rand(mask.shape, device=mask.device, generator=generator) >= dropout
        mask = mask & keep

    if source == "oracle_monomer":
        frame_id = chain_ids.clone()
    else:  # "oracle" — whole complex in one frame (docking included)
        frame_id = torch.zeros_like(chain_ids)

    return coords_res, mask, frame_id
