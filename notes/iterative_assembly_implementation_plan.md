# Iterative Atom Assembly Implementation Plan

## Overview

Phase 3 of ResFold E2E: **Iterative Atom Assembly** - constructing atom positions one cluster at a time, ordered by hierarchical clustering. Like diffusion but sequential.

## Core Concept

```
Training (single step, random mask):
1. Sample x ∈ [0, N-K] atoms as "known" (already placed)
2. Hierarchical clustering on GT atoms
3. Select next K atoms to predict (closest cluster to known)
4. Model predicts positions for K atoms
5. Loss: relative distance to known atoms

Inference (iterative):
while not all atoms placed:
   1. Select next K atoms
   2. Predict positions via cross-attention to known
   3. Add to structure
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/models/clustering.py` | CREATE | Hierarchical clustering utilities |
| `scripts/models/iterative_assembler.py` | CREATE | IterativeAtomAssembler transformer |
| `scripts/train_resfold_e2e_iterative.py` | CREATE | Training script (copy from v2) |
| `src/tinyfold/model/losses/mse.py` | MODIFY | Add `compute_relative_distance_loss` |
| `src/tinyfold/model/losses/__init__.py` | MODIFY | Export new function |
| `scripts/models/__init__.py` | MODIFY | Export new classes |

---

## Implementation Steps

### Step 1: Create `clustering.py`

**Functions to implement:**
- `compute_bond_connectivity(n_residues, chain_ids)` → bond adjacency matrix
- `hierarchical_cluster_atoms(coords, chain_ids, n_clusters)` → cluster IDs
- `select_next_atoms_to_place(coords_gt, known_mask, k, cluster_ids)` → target indices
- `get_placement_order(coords_gt, chain_ids, k_per_step)` → full placement order

**Key details:**
- Uses scipy `linkage` and `fcluster` for hierarchical clustering
- Cross-chain distances penalized by `chain_weight` multiplier
- Selection based on minimum distance to known atoms

### Step 2: Create `iterative_assembler.py`

**IterativeAtomAssembler architecture:**
```
Input:
- trunk_tokens [B, L, c_token] from Stage 1 encoder
- known_coords [B, M, 3] coordinates of placed atoms
- known_mask [B, M] validity mask
- target_atom_idx [B, K] indices to predict
- target_res_idx [B, K] residue indices for trunk lookup

Processing:
- Embed known_coords → known_emb
- Get trunk tokens for target residues → target_queries
- Cross-attention: targets attend to known (n_layers)
- Self-attention among targets
- FFN

Output:
- pred_coords [B, K, 3]
```

**~2M parameters** with default config (c_token=256, n_layers=4)

### Step 3: Add `compute_relative_distance_loss` to mse.py

```python
def compute_relative_distance_loss(
    pred_coords: Tensor,       # [B, K, 3]
    gt_coords: Tensor,         # [B, K, 3]
    known_coords: Tensor,      # [B, M, 3]
    known_mask: Tensor,        # [B, M]
    align_first: bool = True,
) -> Tensor:
    """Loss on distances from predicted to known atoms."""
```

**Key idea:** Instead of absolute position MSE, penalize difference in distances to known atoms. Invariant to global translation/rotation.

### Step 4: Create training script

Copy `train_resfold_e2e_v2.py` and modify:

1. **Imports**: Add clustering, assembler, new loss
2. **New args**: `--k_atoms`, `--rel_dist_weight`, `--chain_weight`, `--assembler_layers`
3. **Replace `train_step_e2e`** with `train_step_iterative`:
   - Sample n_known ∈ [0, N-K]
   - Cluster GT atoms
   - Build known_mask by simulating placement order
   - Select next K atoms
   - Forward through assembler
   - Compute relative distance loss + MSE

4. **Replace `evaluate_e2e`** with `evaluate_iterative`:
   - Full iterative construction from scratch
   - Compute final RMSE

5. **main()**: Create assembler, add to optimizer

### Step 5: Update exports

- Add to `scripts/models/__init__.py`
- Add to `src/tinyfold/model/losses/__init__.py`

---

## Verification Plan

### Unit Tests

```bash
# Test clustering
python -c "
from scripts.models.clustering import *
import torch

coords = torch.randn(40, 3)  # 10 residues
chain_ids = torch.zeros(10, dtype=torch.long)
clusters = hierarchical_cluster_atoms(coords, chain_ids, n_clusters=5)
print(f'Clusters: {clusters}')

known_mask = torch.zeros(40, dtype=torch.bool)
target = select_next_atoms_to_place(coords, known_mask, k=4, cluster_ids=clusters)
print(f'First targets: {target}')
"
```

### Overfit Test

```bash
python scripts/train_resfold_e2e_iterative.py \
    --checkpoint outputs/train_10k_continuous/best_model.pt \
    --n_train 5 --n_test 2 --n_steps 500 \
    --k_atoms 4 --eval_every 100 \
    --output_dir outputs/iterative_overfit
```

**Expected:** Loss < 0.1 after 500 steps on 5 samples

---

## Model Parameters

| Component | Params | Notes |
|-----------|--------|-------|
| Stage 1 (frozen) | ~15M | ResidueDenoiser |
| Stage 2 (optional) | ~5M | AtomRefinerV2MultiSample |
| Stage 3 (Assembler) | ~2M | IterativeAtomAssembler |

---

## Key Design Decisions

1. **K = 4**: Predict one residue's atoms per step (N, CA, C, O)
2. **Clustering uses scipy**: Fast enough for training, ~10ms per sample
3. **Cross-attention architecture**: Targets attend to known atoms
4. **Relative distance loss**: Invariant to global frame
5. **Teacher forcing**: Use GT coords for known atoms during training
