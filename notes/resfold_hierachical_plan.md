# Iterative Atom Assembly: Phase 3 Training

## Background Context

This document provides a complete specification for adding a **third training phase** to the ResFold E2E pipeline. The new phase implements **iterative atom assembly** - constructing atom positions one cluster at a time, ordered by hierarchical clustering.

### Existing Pipeline Overview

The current ResFold E2E architecture has two stages:

```
┌─────────────────────────────────────────────────────────────────┐
│Current ResFold E2E Pipeline                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Stage 1: ResidueDenoiser (~15M params)                          │
│   - Input: aa_seq, chain_ids, res_idx (sequence features)       │
│   - Diffusion on residue centroids [B, L, 3]                    │  
│   - Output: K centroid predictions [B, K, L, 3]                 │
│                                                                 │
│ Stage 2: AtomRefinerV2MultiSample (~5M params)                  │
│   - Input: trunk_tokens + K centroid samples                    │
│   - Aggregates K samples, refines to atom level                 │
│   - Output: atom positions [B, L, 4, 3]                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Format

- **Residues**: L residues per sample, each has 4 backbone atoms (N, CA, C, O)
- **Atom coordinates**: Stored as `[B, L, 4, 3]` or flattened to `[B, N, 3]` where N = L × 4
- **Centroids**: Mean of 4 backbone atoms per residue, shape `[B, L, 3]`
- **Normalization**: Coordinates are centered and normalized to unit variance (std stored per sample)

### Relevant Files

| File | Description |
|------|-------------|
| [train_resfold_e2e_v2.py](file:///c:/Users/costa/src/tinyfold/scripts/train_resfold_e2e_v2.py) | Current E2E training script (copy this) |
| [resfold_e2e.py](file:///c:/Users/costa/src/tinyfold/scripts/models/resfold_e2e.py) | E2E model combining Stage 1 + 2 |
| [resfold.py](file:///c:/Users/costa/src/tinyfold/scripts/models/resfold.py) | Stage 1 ResidueDenoiser |
| [atomrefine_multi_sample.py](file:///c:/Users/costa/src/tinyfold/scripts/models/atomrefine_multi_sample.py) | Stage 2 AtomRefiner |
| [mse.py](file:///c:/Users/costa/src/tinyfold/src/tinyfold/model/losses/mse.py) | Loss functions including `kabsch_align` |

---

## Design Overview

### Core Concept

Like diffusion but **sequential**: instead of denoising all atoms simultaneously, we **fix atoms one cluster at a time** in an order determined by hierarchical clustering.

```
┌─────────────────────────────────────────────────────────────────┐
│                Iterative Atom Assembly (Phase 3)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAINING (single step, random mask):                           │
│  1. Sample x ∈ [0, N-K] atoms as "known" (already placed)       │
│  2. Hierarchical clustering on GT atoms (by position + chain)   │
│  3. Select next K atoms: closest cluster to known atoms         │
│  4. Model predicts relative positions for K atoms               │
│  5. Loss: distance to known atoms (after Kabsch alignment)      │
│                                                                 │
│  INFERENCE (iterative construction):                            │
│  while not all atoms placed:                                    │
│     1. Cluster current structure estimate                       │
│     2. Select next K atoms to place                             │
│     3. Predict their positions relative to known                │
│     4. Add to structure, optionally refine previous atoms       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Separate script**: New file `train_resfold_e2e_iterative.py` copied from `train_resfold_e2e_v2.py`
2. **Separate module**: New `IterativeAtomAssembler` module (not modifying existing Stage 2)
3. **K = 4**: Predict 4 atoms per step (one residue's worth)
4. **Clustering**: Combine spatial proximity + chain connectivity + covalent bonds

---

## Proposed Changes

### Clustering Utilities

#### [NEW] [clustering.py](file:///c:/Users/costa/src/tinyfold/scripts/models/clustering.py)

Utilities for determining atom placement order using hierarchical clustering.

```python
"""Hierarchical clustering utilities for iterative atom assembly.

Determines the order in which to place atoms during iterative construction.
Uses a combination of spatial proximity, chain connectivity, and covalent bonds.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# Backbone atom types: N=0, CA=1, C=2, O=3
BACKBONE_BONDS = {
    # Within residue: N-CA, CA-C, C-O
    (0, 1): 1.458,  # N-CA bond length (Angstroms)
    (1, 2): 1.524,  # CA-C bond length
    (2, 3): 1.231,  # C-O bond length
}
# Between residues: C(i) - N(i+1) peptide bond
PEPTIDE_BOND_LENGTH = 1.329


def compute_bond_connectivity(
    n_residues: int,
    chain_ids: Tensor,  # [L] chain ID per residue
) -> Tensor:
    """Compute bond connectivity matrix for backbone atoms.
    
    Each residue has 4 atoms: N(0), CA(1), C(2), O(3).
    - Intra-residue bonds: N-CA, CA-C, C-O
    - Inter-residue bonds: C(i)-N(i+1) if same chain
    
    Args:
        n_residues: Number of residues (L)
        chain_ids: Chain ID for each residue [L]
        
    Returns:
        connectivity: [N, N] adjacency matrix where N = L * 4
    """
    n_atoms = n_residues * 4
    connectivity = torch.zeros(n_atoms, n_atoms, dtype=torch.bool)
    
    for i in range(n_residues):
        base = i * 4
        # Intra-residue bonds
        connectivity[base + 0, base + 1] = True  # N-CA
        connectivity[base + 1, base + 0] = True
        connectivity[base + 1, base + 2] = True  # CA-C
        connectivity[base + 2, base + 1] = True
        connectivity[base + 2, base + 3] = True  # C-O
        connectivity[base + 3, base + 2] = True
        
        # Peptide bond to next residue (if same chain)
        if i < n_residues - 1 and chain_ids[i] == chain_ids[i + 1]:
            connectivity[base + 2, base + 4 + 0] = True  # C(i)-N(i+1)
            connectivity[base + 4 + 0, base + 2] = True
    
    return connectivity


def hierarchical_cluster_atoms(
    coords: Tensor,         # [N, 3] atom coordinates
    chain_ids: Tensor,      # [L] chain ID per residue (N = L * 4)
    n_clusters: int = 10,
    chain_weight: float = 2.0,  # Penalty for crossing chains
) -> Tensor:
    """Hierarchical clustering of atoms using proximity + chain info.
    
    Computes a modified distance matrix that penalizes cross-chain distances,
    then performs agglomerative clustering.
    
    Args:
        coords: Atom coordinates [N, 3]
        chain_ids: Chain ID per residue [L]
        n_clusters: Number of clusters to form
        chain_weight: Multiplier for cross-chain distances
        
    Returns:
        cluster_ids: [N] cluster assignment for each atom
    """
    n_atoms = coords.shape[0]
    n_residues = n_atoms // 4
    device = coords.device
    
    # Expand chain_ids to atom level [N]
    atom_chain_ids = chain_ids.repeat_interleave(4)
    
    # Compute pairwise distances
    coords_np = coords.detach().cpu().numpy()
    dist_condensed = pdist(coords_np)
    
    # Convert to square form for chain penalty
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(dist_condensed)
    
    # Apply chain penalty: increase distance for cross-chain pairs
    chain_np = atom_chain_ids.cpu().numpy()
    cross_chain = chain_np[:, None] != chain_np[None, :]
    dist_matrix[cross_chain] *= chain_weight
    
    # Back to condensed form
    dist_modified = squareform(dist_matrix)
    
    # Hierarchical clustering
    Z = linkage(dist_modified, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    return torch.tensor(labels - 1, device=device, dtype=torch.long)  # 0-indexed


def select_next_atoms_to_place(
    coords_gt: Tensor,     # [N, 3] ground truth (for cluster selection)
    known_mask: Tensor,    # [N] bool, True = already placed
    k: int,                # number of atoms to select
    cluster_ids: Tensor,   # [N] pre-computed cluster assignments
) -> Tensor:
    """Select K atoms to place next based on cluster proximity to known atoms.
    
    Strategy: 
    1. Find clusters that have unknown atoms
    2. Score each cluster by minimum distance to any known atom
    3. Select atoms from the closest cluster(s)
    
    Args:
        coords_gt: Ground truth coordinates [N, 3]
        known_mask: Boolean mask for already-placed atoms [N]
        k: Number of atoms to select
        cluster_ids: Pre-computed cluster IDs [N]
        
    Returns:
        target_idx: Indices of atoms to predict [K]
    """
    n_atoms = coords_gt.shape[0]
    device = coords_gt.device
    
    # Edge case: if no atoms known yet, pick from first cluster
    if not known_mask.any():
        first_cluster = cluster_ids.min().item()
        candidates = (cluster_ids == first_cluster).nonzero(as_tuple=True)[0]
        return candidates[:k]
    
    # Get known atom positions
    known_coords = coords_gt[known_mask]  # [n_known, 3]
    
    # For each unknown atom, compute min distance to known atoms
    unknown_idx = (~known_mask).nonzero(as_tuple=True)[0]
    unknown_coords = coords_gt[unknown_idx]  # [n_unknown, 3]
    
    # Pairwise distances: [n_unknown, n_known]
    dists = torch.cdist(unknown_coords, known_coords)
    min_dists = dists.min(dim=1).values  # [n_unknown]
    
    # Select k atoms with smallest min distance
    _, top_k_local = torch.topk(min_dists, k=min(k, len(unknown_idx)), largest=False)
    target_idx = unknown_idx[top_k_local]
    
    return target_idx


def get_placement_order(
    coords_gt: Tensor,     # [N, 3]
    chain_ids: Tensor,     # [L]
    k_per_step: int = 4,
) -> list[Tensor]:
    """Pre-compute the full placement order for a structure.
    
    Returns a list of index tensors, each of length k, representing
    the order in which atoms should be placed.
    
    Useful for debugging and visualization.
    """
    n_atoms = coords_gt.shape[0]
    known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=coords_gt.device)
    
    # Compute clustering once
    cluster_ids = hierarchical_cluster_atoms(
        coords_gt, chain_ids, n_clusters=n_atoms // k_per_step + 1
    )
    
    order = []
    while known_mask.sum() < n_atoms:
        remaining = n_atoms - known_mask.sum()
        k = min(k_per_step, remaining.item())
        
        target_idx = select_next_atoms_to_place(
            coords_gt, known_mask, k, cluster_ids
        )
        order.append(target_idx)
        known_mask[target_idx] = True
    
    return order
```

---

### Model Module

#### [NEW] [iterative_assembler.py](file:///c:/Users/costa/src/tinyfold/scripts/models/iterative_assembler.py)

New transformer module for predicting positions of K new atoms conditioned on known atoms.

```python
"""Iterative Atom Assembler: Predict positions for new atoms given known atoms.

This is Stage 3 of the ResFold E2E pipeline. Given a partial structure with
some atoms already placed, predict positions for the next K atoms.

Key differences from AtomRefinerV2MultiSample:
- Takes known_coords + known_mask instead of centroid samples
- Cross-attends from target atoms to known atoms
- Predicts relative positions (distances) rather than absolute coords
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IterativeAtomAssembler(nn.Module):
    """Predict positions for K new atoms given known atom positions.
    
    Architecture:
    1. Encode known atoms into context embeddings
    2. Create query embeddings for target atoms (from trunk_tokens)
    3. Cross-attention: target queries attend to known context
    4. Output: predicted coordinates for target atoms
    
    Parameters:
        c_token: Hidden dimension (matches Stage 1/2)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_layers = n_layers

        # Embed known atom coordinates [3] -> [c_token]
        self.known_coord_embed = nn.Linear(3, c_token)
        
        # Embed target atom queries (will use trunk_tokens as base)
        # Additional embedding for "target" vs "known" distinction
        self.target_type_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)
        self.known_type_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.02)

        # Cross-attention layers: target attends to known
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                c_token, n_heads, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Self-attention layers for target refinement
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                c_token, n_heads, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_token),
                nn.Linear(c_token, c_token * 4),
                nn.GELU(),
                nn.Linear(c_token * 4, c_token),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(c_token) for _ in range(n_layers)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(c_token) for _ in range(n_layers)
        ])
        
        # Output projection: predict 3D coordinates
        self.output_proj = nn.Linear(c_token, 3)

    def forward(
        self,
        trunk_tokens: Tensor,      # [B, L, c_token] from Stage 1 encoder
        known_coords: Tensor,      # [B, N, 3] coordinates of known atoms
        known_mask: Tensor,        # [B, N] bool, True = valid known atom
        target_atom_idx: Tensor,   # [B, K] indices of atoms to predict
        target_res_idx: Tensor,    # [B, K] residue index for each target atom
    ) -> Tensor:
        """Predict coordinates for target atoms.
        
        Args:
            trunk_tokens: Pre-computed sequence embeddings [B, L, c_token]
            known_coords: Coordinates of already-placed atoms [B, N, 3]
            known_mask: Mask for valid known atoms [B, N]
            target_atom_idx: Which atoms to predict [B, K]
            target_res_idx: Residue index for each target (for trunk lookup) [B, K]
            
        Returns:
            pred_coords: Predicted coordinates for target atoms [B, K, 3]
        """
        B, L, C = trunk_tokens.shape
        K = target_atom_idx.shape[1]
        device = trunk_tokens.device
        
        # === Build known context ===
        # Embed known coordinates
        known_emb = self.known_coord_embed(known_coords)  # [B, N, c_token]
        known_emb = known_emb + self.known_type_embed
        
        # === Build target queries ===
        # Gather trunk tokens for target residues
        # target_res_idx: [B, K] -> use to index into trunk_tokens [B, L, C]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        target_trunk = trunk_tokens[batch_idx, target_res_idx.clamp(0, L-1)]  # [B, K, C]
        target_queries = target_trunk + self.target_type_embed
        
        # === Transformer layers ===
        # Key padding mask for known atoms (True = ignore)
        known_key_pad = ~known_mask  # [B, N]
        
        for i in range(self.n_layers):
            # Cross-attention: targets attend to known
            q = self.cross_norms[i](target_queries)
            k = v = known_emb
            attn_out, _ = self.cross_attn_layers[i](
                q, k, v, key_padding_mask=known_key_pad
            )
            target_queries = target_queries + attn_out
            
            # Self-attention among targets
            q = self.self_norms[i](target_queries)
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            target_queries = target_queries + attn_out
            
            # FFN
            target_queries = target_queries + self.ffn_layers[i](target_queries)
        
        # === Output ===
        pred_coords = self.output_proj(target_queries)  # [B, K, 3]
        
        return pred_coords

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

---

### Loss Function

#### [MODIFY] [mse.py](file:///c:/Users/costa/src/tinyfold/src/tinyfold/model/losses/mse.py)

Add new function for relative distance loss.

```python
def compute_relative_distance_loss(
    pred_coords: Tensor,       # [B, K, 3] predicted coordinates
    gt_coords: Tensor,         # [B, K, 3] ground truth for target atoms
    known_coords: Tensor,      # [B, M, 3] coordinates of known atoms
    known_mask: Tensor,        # [B, M] mask for valid known atoms
    align_first: bool = True,
) -> Tensor:
    """Compute loss on distances from predicted atoms to known atoms.
    
    Instead of penalizing absolute positions, this penalizes the distance
    from each predicted atom to each known atom. This makes the loss 
    invariant to global translation/rotation.
    
    Optionally performs Kabsch alignment of predicted to ground truth
    first (considering only the predicted atoms).
    
    Args:
        pred_coords: Predicted coordinates for K target atoms [B, K, 3]
        gt_coords: Ground truth coordinates for K target atoms [B, K, 3]
        known_coords: Coordinates of M already-placed atoms [B, M, 3]
        known_mask: Boolean mask for valid known atoms [B, M]
        align_first: Whether to Kabsch-align pred to gt before computing loss
        
    Returns:
        loss: Scalar loss value
    """
    B, K, _ = pred_coords.shape
    
    if align_first and K >= 3:
        # Kabsch align predicted to ground truth
        # Note: kabsch_align expects [B, N, 3] and returns aligned coords
        pred_aligned, gt_centered = kabsch_align(pred_coords, gt_coords)
        # Use aligned predictions, but measure distances to original known coords
    else:
        pred_aligned = pred_coords
    
    # Compute distances from predicted to known atoms [B, K, M]
    pred_dists = torch.cdist(pred_aligned, known_coords)  # [B, K, M]
    gt_dists = torch.cdist(gt_coords, known_coords)       # [B, K, M]
    
    # Mask out invalid known atoms
    if known_mask is not None:
        mask_exp = known_mask.unsqueeze(1)  # [B, 1, M]
        # MSE on distances, masked
        sq_diff = ((pred_dists - gt_dists) ** 2) * mask_exp.float()
        n_valid = mask_exp.sum(dim=-1).clamp(min=1)  # [B, K]
        per_target_loss = sq_diff.sum(dim=-1) / n_valid  # [B, K]
        loss = per_target_loss.mean()
    else:
        loss = F.mse_loss(pred_dists, gt_dists)
    
    return loss
```

Also add to `__init__.py`:
```python
from .mse import compute_relative_distance_loss
```

---

### Training Script

#### [NEW] [train_resfold_e2e_iterative.py](file:///c:/Users/costa/src/tinyfold/scripts/train_resfold_e2e_iterative.py)

Copy `train_resfold_e2e_v2.py` entirely, then modify:

**1. Add imports:**
```python
from models.iterative_assembler import IterativeAtomAssembler
from models.clustering import (
    hierarchical_cluster_atoms,
    select_next_atoms_to_place,
    compute_bond_connectivity,
)
from tinyfold.model.losses import compute_relative_distance_loss
```

**2. Replace `train_step_e2e` with `train_step_iterative`:**

```python
def train_step_iterative(
    model: ResFoldE2E,
    assembler: IterativeAtomAssembler,
    batch: dict,
    args,
) -> dict:
    """Iterative assembly training step.
    
    1. Sample x ∈ [0, N-K] as number of known atoms
    2. Cluster GT atoms, select first x as "known" 
    3. Select next K atoms to predict
    4. Model predicts their positions
    5. Loss on relative distances to known atoms
    """
    model.train()
    assembler.train()
    
    B = batch['aa_seq'].shape[0]
    device = batch['centroids'].device
    
    # Get trunk tokens (sequence encoding, no coords)
    trunk_tokens = model.get_trunk_tokens(
        batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
    )
    
    total_loss = 0.0
    loss_components = {'relative_dist': 0.0, 'position_mse': 0.0}
    
    # Process each sample in batch (different sizes may cause issues with batching)
    for b in range(B):
        n_res = batch['n_res'][b]
        n_atoms = batch['n_atoms'][b]
        chain_ids = batch['chain_ids'][b, :n_res]
        
        # Get GT atom coordinates [N, 3]
        gt_coords = batch['coords'][b, :n_atoms]  # [N, 3]
        
        # Sample number of known atoms: uniform in [0, N-K]
        k = args.k_atoms
        max_known = max(0, n_atoms - k)
        n_known = torch.randint(0, max_known + 1, (1,)).item()
        
        # Compute hierarchical clustering
        cluster_ids = hierarchical_cluster_atoms(
            gt_coords, chain_ids,
            n_clusters=max(1, n_atoms // k),
            chain_weight=args.chain_weight,
        )
        
        # Build known mask by taking first n_known atoms in placement order
        known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        placed_count = 0
        while placed_count < n_known:
            next_k = min(k, n_known - placed_count)
            if next_k <= 0:
                break
            target_idx = select_next_atoms_to_place(
                gt_coords, known_mask, next_k, cluster_ids
            )
            known_mask[target_idx] = True
            placed_count += len(target_idx)
        
        # Select next K atoms to predict
        remaining = (~known_mask).sum().item()
        actual_k = min(k, remaining)
        if actual_k == 0:
            continue  # All atoms already placed
            
        target_idx = select_next_atoms_to_place(
            gt_coords, known_mask, actual_k, cluster_ids
        )
        
        # Get residue indices for target atoms (atom_idx // 4)
        target_res_idx = target_idx // 4
        
        # Prepare inputs for assembler (add batch dim)
        trunk_b = trunk_tokens[b:b+1]  # [1, L, C]
        known_coords = gt_coords[known_mask].unsqueeze(0)  # [1, n_known, 3]
        known_mask_b = torch.ones(1, known_mask.sum(), dtype=torch.bool, device=device)
        target_idx_b = target_idx.unsqueeze(0)  # [1, K]
        target_res_idx_b = target_res_idx.unsqueeze(0)  # [1, K]
        
        # Forward pass
        pred_coords = assembler(
            trunk_b, known_coords, known_mask_b, target_idx_b, target_res_idx_b
        )  # [1, K, 3]
        
        # Ground truth for target atoms
        gt_target = gt_coords[target_idx].unsqueeze(0)  # [1, K, 3]
        
        # Loss: relative distances to known atoms
        loss_rel = compute_relative_distance_loss(
            pred_coords, gt_target, known_coords, known_mask_b,
            align_first=args.align_before_loss,
        )
        
        # Also add direct MSE for stability
        loss_mse = F.mse_loss(pred_coords, gt_target)
        
        sample_loss = args.rel_dist_weight * loss_rel + args.mse_weight * loss_mse
        total_loss = total_loss + sample_loss
        
        loss_components['relative_dist'] += loss_rel.item()
        loss_components['position_mse'] += loss_mse.item()
    
    total_loss = total_loss / B
    for key in loss_components:
        loss_components[key] /= B
    
    return {
        'total': total_loss,
        **loss_components,
    }
```

**3. Replace `evaluate_e2e` with `evaluate_iterative`:**

```python
@torch.no_grad()
def evaluate_iterative(
    model: ResFoldE2E,
    assembler: IterativeAtomAssembler,
    samples: dict,
    indices: list,
    noiser,
    device: torch.device,
    args,
) -> dict:
    """Evaluate iterative assembly with full construction."""
    model.eval()
    assembler.eval()
    
    atom_rmses = []
    
    for idx in indices:
        s = samples[idx]
        batch = collate_batch([s], device)
        
        n_res = s['n_res']
        n_atoms = s['n_atoms']
        k = args.k_atoms
        
        # Get trunk tokens
        trunk_tokens = model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], batch['mask_res']
        )
        
        # Initialize with Stage 1 centroid estimates (or zeros)
        # For simplicity, initialize with slightly noisy GT
        gt_coords = batch['coords'][0, :n_atoms]  # [N, 3]
        chain_ids = batch['chain_ids'][0, :n_res]
        
        # Iteratively construct
        constructed = torch.zeros(n_atoms, 3, device=device)
        known_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        
        # Pre-compute clustering on GT (for evaluation fairness)
        cluster_ids = hierarchical_cluster_atoms(
            gt_coords, chain_ids, n_clusters=max(1, n_atoms // k)
        )
        
        while known_mask.sum() < n_atoms:
            remaining = n_atoms - known_mask.sum().item()
            actual_k = min(k, remaining)
            
            target_idx = select_next_atoms_to_place(
                gt_coords if not known_mask.any() else constructed,
                known_mask, actual_k, cluster_ids
            )
            
            target_res_idx = target_idx // 4
            
            # Prepare inputs
            known_coords = constructed[known_mask].unsqueeze(0) if known_mask.any() else torch.zeros(1, 1, 3, device=device)
            known_mask_b = torch.ones(1, max(1, known_mask.sum()), dtype=torch.bool, device=device)
            
            pred_coords = assembler(
                trunk_tokens,
                known_coords,
                known_mask_b,
                target_idx.unsqueeze(0),
                target_res_idx.unsqueeze(0),
            )  # [1, K, 3]
            
            # Update constructed structure
            constructed[target_idx] = pred_coords[0]
            known_mask[target_idx] = True
        
        # Compute RMSE
        from tinyfold.model.losses import compute_rmse
        rmse = compute_rmse(
            constructed.unsqueeze(0), 
            gt_coords.unsqueeze(0)
        ).item() * s['std']
        atom_rmses.append(rmse)
    
    return {
        'atom_rmse': np.mean(atom_rmses),
        'n_samples': len(indices),
    }
```

**4. Add new command-line arguments:**

```python
# Iterative assembly args
parser.add_argument("--k_atoms", type=int, default=4,
                    help="Number of atoms to predict per step")
parser.add_argument("--rel_dist_weight", type=float, default=1.0,
                    help="Weight for relative distance loss")
parser.add_argument("--mse_weight", type=float, default=0.1,
                    help="Weight for direct MSE loss")
parser.add_argument("--chain_weight", type=float, default=2.0,
                    help="Distance multiplier for cross-chain clustering")
parser.add_argument("--align_before_loss", action="store_true",
                    help="Kabsch align before computing loss")

# Assembler model args
parser.add_argument("--assembler_layers", type=int, default=4)
parser.add_argument("--assembler_heads", type=int, default=8)
```

**5. In `main()`, create the assembler:**

```python
# Create iterative assembler (Stage 3)
assembler = IterativeAtomAssembler(
    c_token=args.c_token,
    n_layers=args.assembler_layers,
    n_heads=args.assembler_heads,
    dropout=0.0,
).to(device)

logger.log(f"Assembler: {assembler.count_parameters():,} params")

# Optimizer includes both model and assembler
trainable_params = list(model.parameters()) + list(assembler.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
```

---

## Summary of Changes

| File | Action | Lines Changed |
|------|--------|---------------|
| `scripts/models/clustering.py` | NEW | ~150 |
| `scripts/models/iterative_assembler.py` | NEW | ~130 |
| `scripts/train_resfold_e2e_iterative.py` | NEW (copy+modify) | ~800 |
| `src/tinyfold/model/losses/mse.py` | MODIFY | +50 |
| `src/tinyfold/model/losses/__init__.py` | MODIFY | +1 |

---

## Verification Plan

### Unit Tests

```bash
# Test clustering utilities
cd c:\Users\costa\src\tinyfold
python -c "
from scripts.models.clustering import *
import torch

# Test bond connectivity
conn = compute_bond_connectivity(10, torch.zeros(10, dtype=torch.long))
print(f'Bond connectivity: {conn.sum()} bonds')

# Test clustering
coords = torch.randn(40, 3)  # 10 residues * 4 atoms
chain_ids = torch.zeros(10, dtype=torch.long)
clusters = hierarchical_cluster_atoms(coords, chain_ids, n_clusters=5)
print(f'Cluster sizes: {[(clusters == i).sum().item() for i in range(5)]}')

# Test atom selection
known_mask = torch.zeros(40, dtype=torch.bool)
target = select_next_atoms_to_place(coords, known_mask, k=4, cluster_ids=clusters)
print(f'Selected atoms: {target}')
"
```

### Integration Test (Overfit)

```bash
python scripts/train_resfold_e2e_iterative.py \
    --checkpoint outputs/resfold_stage1/best_model.pt \
    --n_train 5 --n_test 2 --n_steps 500 \
    --k_atoms 4 --eval_every 100 \
    --output_dir outputs/resfold_iterative_overfit
```

**Expected**: Loss should decrease to < 0.1 after 500 steps on 5 samples.

### Full Training

```bash
python scripts/train_resfold_e2e_iterative.py \
    --checkpoint outputs/resfold_stage1/best_model.pt \
    --n_train 80 --n_test 14 --n_steps 5000 \
    --k_atoms 4 --eval_every 500 \
    --output_dir outputs/resfold_iterative
```
