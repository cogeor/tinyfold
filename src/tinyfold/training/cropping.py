"""Training-time cropping strategies for TinyFold.

The active model has ~12M trainable params spread over up to 1000 residues per
training token — about 12k params/token vs AlphaFold-Multimer's 242k or
Protenix-Mini's 780k. Cropping training to a fixed-token-budget brings the
per-token capacity back into a workable regime without throwing away the long-
tail samples (the way an explicit chain-length cap does).

The strategies here are *training-time* croppers: they take a fully-loaded
sample dict (from ``load_sample``) and return a new dict where the per-residue
tensors are sliced to <= ``crop_size``. The same strategies are reused at
inference time by the tiled / interface multi-pass reconstruction code in
``tinyfold/inference/`` (Strategies 1+2 from the v2 multi-pass plan).

GLOBAL ``res_idx`` is preserved: a crop of residues 50-150 carries
``res_idx = [50..150]`` not ``[0..100]``. Combined with ``per_chain_res_idx``
this makes the positional encoding crop-invariant — the same residue gets the
same positional feature whether it appears in a crop or in the full complex.

Four strategies, increasing PPI-awareness:
- ``NoCrop``:        passthrough; errors if a sample is larger than crop_size
- ``ContiguousCrop``: random window across the (concatenated) chain axis
- ``SpatialCrop``:    pick a random center, take K nearest GT neighbors
- ``InterfaceCrop``:  bias the center toward interface residues; default for PPI
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor


class Cropper(Protocol):
    """Crop a single sample to <= ``crop_size`` residues.

    The returned dict must:
    - have the same keys as ``sample``
    - have per-residue tensors of length L_crop <= crop_size
    - preserve GLOBAL ``res_idx`` so positional encoding stays consistent
    - preserve ``chain_ids`` (some chain may be absent if cropped out)
    - update ``n_res`` and ``n_atoms`` to reflect the crop
    - leave ``std``, ``sample_id`` untouched
    """

    def __call__(
        self,
        sample: dict[str, Any],
        crop_size: int,
        rng: torch.Generator,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Implementation utilities
# ---------------------------------------------------------------------------


def _apply_residue_indices(sample: dict[str, Any], idx: Tensor) -> dict[str, Any]:
    """Slice the per-residue tensors in ``sample`` by ``idx``.

    ``idx`` is a LongTensor of selected residue positions in [0, L). Order is
    preserved (we sort idx so chain ordering and centroid topology survive).
    """
    idx, _ = torch.sort(idx)
    L_crop = int(idx.numel())

    # Per-residue tensors that always exist.
    per_res_keys = ['centroids', 'coords_res', 'aa_seq', 'chain_ids', 'res_idx']
    out = dict(sample)
    for key in per_res_keys:
        out[key] = sample[key][idx]

    # ESM cache is optional.
    if 'esm_embed' in sample:
        out['esm_embed'] = sample['esm_embed'][idx]

    # Coevolution features are O(L^2): crop BOTH residue axes so [L,L,F] stays
    # aligned with the cropped residue set (a one-axis slice would silently
    # desync i/j and crash collate on the [:L,:L] assignment).
    if 'msa_feats' in sample:
        out['msa_feats'] = sample['msa_feats'][idx][:, idx]

    # Re-derive the flat atom-level tensors from the residue slice. The atom
    # ordering is (res_0_N, res_0_CA, res_0_C, res_0_O, res_1_N, ...) so we
    # expand each residue index to its 4 atom positions.
    atom_idx = (idx.unsqueeze(1) * 4 + torch.arange(4, device=idx.device).unsqueeze(0)).reshape(-1)
    out['coords'] = sample['coords'][atom_idx]
    out['atom_types'] = sample['atom_types'][atom_idx]

    # Re-center the crop on its OWN centroid. The parent complex was centered at
    # load time, but a subset is not: a spatially-compact crop (Spatial/
    # Interface) sits off-origin in the parent frame. The EDM preconditioning
    # (sigma_data) and the sampler init (x = sigma_max * randn, origin-centered)
    # both assume centered data, so feeding an off-center crop to the denoiser
    # is a train/inference mismatch. Centering here restores the contract.
    # Scale is left to the (fixed) global-scale convention: with a size-
    # independent divisor a crop already carries the correct absolute scale, so
    # no per-crop rescale is needed (and rescaling would re-introduce the
    # size-coupling we are trying to remove).
    crop_centroid = out['coords'].mean(dim=0, keepdim=True)  # [1, 3]
    out['coords'] = out['coords'] - crop_centroid
    out['centroids'] = sample['centroids'][idx] - crop_centroid
    out['coords_res'] = out['coords'].view(L_crop, 4, 3)
    # atom_to_res maps to the NEW residue index (0..L_crop-1), not the global one.
    out['atom_to_res'] = torch.arange(L_crop, device=idx.device).repeat_interleave(4)

    out['n_res'] = L_crop
    out['n_atoms'] = L_crop * 4

    # Trace where the crop came from, for debugging.
    out['crop_global_idx'] = idx
    return out


def _is_two_chain(sample: dict[str, Any]) -> bool:
    chain_ids = sample['chain_ids']
    return bool((chain_ids == 0).any().item() and (chain_ids == 1).any().item())


def _interface_residue_indices(sample: dict[str, Any], cutoff: float = 8.0) -> Tensor:
    """Residues whose CA is within ``cutoff`` Å of any CA on the other chain.

    Returns indices into the residue axis. Empty tensor if no contacts (degenerate
    single-chain samples or rare disjoint complexes).
    """
    chain_ids = sample['chain_ids']
    # CA is atom index 1 within each residue: (N, CA, C, O).
    ca = sample['coords_res'][:, 1, :]  # [L, 3]
    a = chain_ids == 0
    b = chain_ids == 1
    if not (a.any() and b.any()):
        return torch.empty(0, dtype=torch.long, device=chain_ids.device)
    ca_a = ca[a]
    ca_b = ca[b]
    # Pairwise CA-CA distances between chains.
    d = torch.cdist(ca_a, ca_b)  # [LA, LB]
    a_iface = (d.min(dim=1).values < cutoff)
    b_iface = (d.min(dim=0).values < cutoff)
    a_pos = torch.nonzero(a, as_tuple=False).squeeze(1)
    b_pos = torch.nonzero(b, as_tuple=False).squeeze(1)
    iface_idx = torch.cat([a_pos[a_iface], b_pos[b_iface]])
    return iface_idx


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class NoCrop:
    """Passthrough cropper. Raises if a sample is larger than crop_size.

    Used as the control arm in ablations and as the inference-time strategy
    when the whole complex fits within crop_size.
    """

    def __call__(self, sample, crop_size, rng):
        L = int(sample['n_res'])
        if crop_size < L:
            raise ValueError(
                f"NoCrop: sample has L={L} > crop_size={crop_size}. "
                f"Use a real cropper or raise crop_size."
            )
        # Still mark crop_global_idx for downstream consistency.
        out = dict(sample)
        out['crop_global_idx'] = torch.arange(L, device=sample['centroids'].device)
        return out


class ContiguousCrop:
    """Random contiguous window of ``crop_size`` residues across the chain axis.

    Cheapest crop. The chain boundary may fall inside the crop, outside it, or
    coincide with a crop edge — the strategy makes no attempt to preserve both
    chains. For pure folding objectives this is fine; for PPI it is a poor
    default because many crops will contain only one chain. Kept as a
    control / baseline for ablations.
    """

    def __call__(self, sample, crop_size, rng):
        L = int(sample['n_res'])
        if crop_size >= L:
            return NoCrop()(sample, crop_size, rng)
        start = int(torch.randint(0, L - crop_size + 1, (1,), generator=rng).item())
        idx = torch.arange(start, start + crop_size, device=sample['centroids'].device)
        return _apply_residue_indices(sample, idx)


class SpatialCrop:
    """Random center residue, take ``crop_size - 1`` nearest GT-CA neighbors.

    Preserves local 3D structure (a small protein domain) better than
    ContiguousCrop but is chain-blind: the K neighbors may all come from one
    chain. Same caveat for PPI as ContiguousCrop, but the crop is at least
    structurally compact.
    """

    def __call__(self, sample, crop_size, rng):
        L = int(sample['n_res'])
        if crop_size >= L:
            return NoCrop()(sample, crop_size, rng)
        ca = sample['coords_res'][:, 1, :]  # [L, 3]
        center = int(torch.randint(0, L, (1,), generator=rng).item())
        d = torch.norm(ca - ca[center], dim=-1)
        # K nearest (including self).
        _, idx = torch.topk(d, k=crop_size, largest=False)
        return _apply_residue_indices(sample, idx)


class InterfaceCrop:
    """Bias the crop center toward interface residues; PPI default.

    With probability ``interface_prob`` (default 0.8), the crop center is
    sampled uniformly from interface residues (CA-CA contact < 8 Å between
    chains). With probability 1-interface_prob the center is uniform random,
    so non-interface regions still receive training signal proportional to
    their abundance — pure interface-only training would teach the model that
    every residue is in contact.

    Once a center is chosen, the crop is the ``crop_size - 1`` nearest GT-CA
    neighbors of the center. For a center on an interface residue, this
    naturally pulls in residues from BOTH chains because the contact is what
    makes the residue "interface" in the first place. So InterfaceCrop is the
    only strategy that reliably keeps both sides of the contact in every crop.

    Degenerate cases:
    - No interface residues found (rare): fall back to SpatialCrop.
    - Single-chain sample (no chain B): fall back to SpatialCrop.
    """

    def __init__(self, interface_prob: float = 0.8, cutoff: float = 8.0):
        if not 0.0 <= interface_prob <= 1.0:
            raise ValueError(f"interface_prob must be in [0,1], got {interface_prob}")
        self.interface_prob = float(interface_prob)
        self.cutoff = float(cutoff)
        self._spatial = SpatialCrop()

    def __call__(self, sample, crop_size, rng):
        L = int(sample['n_res'])
        if crop_size >= L:
            return NoCrop()(sample, crop_size, rng)
        if not _is_two_chain(sample):
            return self._spatial(sample, crop_size, rng)

        use_iface = torch.rand(1, generator=rng).item() < self.interface_prob
        if use_iface:
            iface = _interface_residue_indices(sample, cutoff=self.cutoff)
            if iface.numel() == 0:
                # No contacts found; fall back to spatial.
                return self._spatial(sample, crop_size, rng)
            center = int(iface[torch.randint(0, iface.numel(), (1,), generator=rng).item()].item())
        else:
            center = int(torch.randint(0, L, (1,), generator=rng).item())

        ca = sample['coords_res'][:, 1, :]
        d = torch.norm(ca - ca[center], dim=-1)
        _, idx = torch.topk(d, k=crop_size, largest=False)
        return _apply_residue_indices(sample, idx)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_REGISTRY = {
    'none': NoCrop,
    'contiguous': ContiguousCrop,
    'spatial': SpatialCrop,
    'interface': InterfaceCrop,
}


def build_cropper(
    strategy: str,
    interface_prob: float = 0.8,
    interface_cutoff: float = 8.0,
) -> Cropper:
    """Build a cropper by name. ``strategy`` matches the YAML config key."""
    s = strategy.lower()
    if s not in _REGISTRY:
        raise ValueError(
            f"Unknown crop_strategy={strategy!r}; expected one of {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[s]
    if cls is InterfaceCrop:
        return InterfaceCrop(interface_prob=interface_prob, cutoff=interface_cutoff)
    return cls()
