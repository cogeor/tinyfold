"""Data loading and batching utilities for TinyFold training.

Provides:
- load_sample: Load a single sample from parquet table
- collate_batch: Collate samples into padded batches
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch import Tensor


def load_sample(
    table,
    i: int,
    normalize: bool = True,
    esm_cache_dir: Optional[str | Path] = None,
    per_chain_res_idx: bool = False,
) -> Dict[str, Any]:
    """Load sample at residue level (4 atoms per residue).

    Args:
        table: PyArrow table from samples.parquet
        i: Sample index
        normalize: If True, normalize coords to unit variance
        esm_cache_dir: Optional directory containing per-sample ESM-2 embeddings
            (NPZ files keyed by ``sample_id``). When given, the returned dict
            includes ``'esm_embed'``: float32 [L, esm_dim]. The cache key is
            ``sample_id`` (NOT ``pdb_id``) so different bioassemblies do not
            collide. Raises ``ValueError`` if the cache file is missing or its
            residue count disagrees with the parquet row.

    Returns:
        Dict with:
            - coords: [N_atoms, 3] all atom coordinates
            - coords_res: [L, 4, 3] atoms per residue (N, CA, C, O)
            - centroids: [L, 3] residue centroids
            - aa_seq: [L] amino acid sequence
            - chain_ids: [L] chain IDs
            - res_idx: [L] residue indices
            - std: normalization factor
            - n_atoms, n_res: counts
            - sample_id: identifier
            - esm_embed (optional): [L, esm_dim] float32, present only when
              ``esm_cache_dir`` is set.
    """
    coords = torch.tensor(table['atom_coords'][i].as_py(), dtype=torch.float32)
    atom_types = torch.tensor(table['atom_type'][i].as_py(), dtype=torch.long)
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py(), dtype=torch.long)
    seq_res = torch.tensor(table['seq'][i].as_py(), dtype=torch.long)
    chain_res = torch.tensor(table['chain_id_res'][i].as_py(), dtype=torch.long)

    n_atoms = len(atom_types)
    n_res = n_atoms // 4
    coords = coords.reshape(n_atoms, 3)

    # Center coordinates
    centroid = coords.mean(dim=0, keepdim=True)
    coords = coords - centroid

    # Compute std
    original_std = coords.std()

    if normalize:
        coords = coords / original_std
        std = original_std
    else:
        std = torch.tensor(1.0)

    # Compute residue centroids
    coords_res = coords.view(n_res, 4, 3)
    centroids = coords_res.mean(dim=1)

    sample_id = table['sample_id'][i].as_py()

    # res_idx encoding choice — see scripts/test_positional_invariance.py
    # for why this matters:
    #   - per_chain_res_idx=False (legacy / Phase D): a single continuous
    #     index 0..L_total-1. Chain B residues end up offset by LA, so the
    #     same chain B residue gets a different positional feature depending
    #     on chain A's length. The model learns absolute position as a
    #     structural cue and catastrophically loses size invariance.
    #   - per_chain_res_idx=True (the fix): use the per-chain-reset indices
    #     already stored in the parquet ([0..LA-1] for chain A, [0..LB-1]
    #     for chain B). The chain_id embedding disambiguates which chain;
    #     positional features now generalize across complex sizes.
    if per_chain_res_idx:
        res_idx_tensor = torch.tensor(
            table['res_idx'][i].as_py(), dtype=torch.long
        )
    else:
        res_idx_tensor = torch.arange(n_res)

    out = {
        'coords': coords,
        'coords_res': coords_res,
        'centroids': centroids,
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': seq_res,
        'chain_ids': chain_res,
        'res_idx': res_idx_tensor,
        'std': std.item(),
        'n_atoms': n_atoms,
        'n_res': n_res,
        'sample_id': sample_id,
    }

    if esm_cache_dir is not None:
        cache_path = Path(esm_cache_dir) / f"{sample_id}.npz"
        if not cache_path.exists():
            raise ValueError(
                f"ESM cache missing for {sample_id}: {cache_path}"
            )
        # mmap so DataLoader workers don't multiply RAM usage.
        # `np.load` keeps the file open; copying into a torch tensor below
        # materialises only the slice we need, then the npz handle goes out
        # of scope and closes.
        with np.load(cache_path, mmap_mode='r') as npz:
            emb_np = np.asarray(npz['embeddings'])
        if emb_np.shape[0] != n_res:
            raise ValueError(
                f"ESM cache shape mismatch for {sample_id}: "
                f"got {emb_np.shape[0]} residues, expected {n_res} (LA+LB)"
            )
        # Cast fp16 -> fp32 here, off the GPU hot path.
        out['esm_embed'] = torch.from_numpy(emb_np).float()

    return out


def collate_batch(
    samples: List[Dict],
    device: torch.device,
    cropper: Optional[Any] = None,
    crop_size: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Collate residue-level samples into a padded batch.

    Args:
        samples: List of sample dicts from load_sample
        device: Target device
        cropper: Optional ``tinyfold.training.cropping.Cropper``. Each sample
            with ``n_res > crop_size`` is cropped before padding so the model
            never sees more than ``crop_size`` tokens per gradient step. The
            global ``res_idx`` is preserved by the cropper so positional
            encoding stays consistent. Samples that already fit pass through.
        crop_size: Token budget per sample when ``cropper`` is set. Ignored
            when ``cropper`` is None.
        rng: torch.Generator used by stochastic croppers. Stateful — pass a
            generator owned by the training loop so crops are reproducible
            from a seed. Ignored when ``cropper`` is None.

    Returns:
        Batched dict with padded tensors
    """
    if cropper is not None:
        if crop_size is None:
            raise ValueError("crop_size must be provided when cropper is set")
        if rng is None:
            # Stochastic croppers REQUIRE an RNG; a fresh one each call would
            # silently break reproducibility, so fail loudly.
            raise ValueError("rng must be provided when cropper is set")
        samples = [cropper(s, crop_size, rng) for s in samples]

    B = len(samples)
    max_res = max(s['n_res'] for s in samples)
    max_atoms = max_res * 4

    # Residue-level tensors
    centroids = torch.zeros(B, max_res, 3)
    coords_res = torch.zeros(B, max_res, 4, 3)
    aa_seq = torch.zeros(B, max_res, dtype=torch.long)
    chain_ids = torch.zeros(B, max_res, dtype=torch.long)
    res_idx = torch.zeros(B, max_res, dtype=torch.long)
    mask_res = torch.zeros(B, max_res, dtype=torch.bool)

    # Atom-level tensors
    coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_to_res = torch.zeros(B, max_atoms, dtype=torch.long)
    mask_atom = torch.zeros(B, max_atoms, dtype=torch.bool)

    stds = []

    # Optional ESM-2 embedding pathway. We allocate the padded tensor lazily
    # so the default ``aa_embed="learned"`` path (no per-sample ``esm_embed``)
    # never touches this branch and the returned dict is byte-identical to
    # the pre-Loop-05 behaviour.
    have_esm = any('esm_embed' in s for s in samples)
    esm_embed_padded = None
    if have_esm:
        esm_dim = samples[0]['esm_embed'].shape[1]
        esm_embed_padded = torch.zeros(B, max_res, esm_dim)

    for i, s in enumerate(samples):
        L = s['n_res']
        N = s['n_atoms']

        centroids[i, :L] = s['centroids']
        coords_res[i, :L] = s['coords_res']
        aa_seq[i, :L] = s['aa_seq']
        chain_ids[i, :L] = s['chain_ids']
        res_idx[i, :L] = s['res_idx']
        mask_res[i, :L] = True

        coords[i, :N] = s['coords']
        atom_types[i, :N] = s['atom_types']
        atom_to_res[i, :N] = s['atom_to_res']
        mask_atom[i, :N] = True

        if have_esm:
            esm_embed_padded[i, :L] = s['esm_embed']

        stds.append(s['std'])

    out = {
        'centroids': centroids.to(device),
        'coords_res': coords_res.to(device),
        'aa_seq': aa_seq.to(device),
        'chain_ids': chain_ids.to(device),
        'res_idx': res_idx.to(device),
        'mask_res': mask_res.to(device),
        'coords': coords.to(device),
        'atom_types': atom_types.to(device),
        'atom_to_res': atom_to_res.to(device),
        'mask_atom': mask_atom.to(device),
        'stds': stds,
        'n_res': [s['n_res'] for s in samples],
        'n_atoms': [s['n_atoms'] for s in samples],
        'sample_ids': [s['sample_id'] for s in samples],
    }
    if esm_embed_padded is not None:
        out['esm_embed'] = esm_embed_padded.to(device)
    return out


# Aliases for backward compatibility
load_sample_raw = load_sample
collate_batch_residue = collate_batch
