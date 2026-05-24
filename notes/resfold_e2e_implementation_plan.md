# ResFold E2E Implementation Plan

## Summary

Two-phase end-to-end training where Stage 2 receives K=5 diffusion samples from Stage 1 and learns to aggregate them for atom prediction.

---

## Phase 1: Create AtomRefinerV2MultiSample

**File:** `scripts/models/atomrefine_multi_sample.py`

### Implementation

```python
class AtomRefinerV2MultiSample(nn.Module):
    """Stage 2 with multi-sample centroid conditioning."""

    def __init__(
        self,
        c_token: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_samples: int = 5,
        dropout: float = 0.0,
        aggregation: str = "learned",  # "learned" or "mean"
    ):
        # Components:
        # 1. centroid_embed: Linear(3, c_token) - embed each centroid sample
        # 2. sample_agg: learned aggregation across K samples
        # 3. transformer: TransformerEncoder for sequence processing
        # 4. output_proj: Linear(c_token, 4*3) for atom offsets
```

### Key Design Decisions

1. **Aggregation method**: Start with learned weighted mean (`Linear(n_samples, 1)`)
2. **Conditioning**: Add aggregated embeddings to trunk_tokens (additive)
3. **Output**: Predict offsets from mean centroid (use `mean(samples)` as anchor)

---

## Phase 2: Create ResFoldE2E Wrapper

**File:** `scripts/models/resfold_e2e.py`

### Implementation

```python
class ResFoldE2E(nn.Module):
    """End-to-end ResFold with multi-sample diffusion."""

    def __init__(
        self,
        # Stage 1 (existing ResidueDenoiser)
        c_token: int = 256,
        trunk_layers: int = 9,
        denoiser_blocks: int = 7,
        # Stage 2 (new multi-sample refiner)
        s2_layers: int = 6,
        n_samples: int = 5,
    ):
        self.stage1 = ResidueDenoiser(...)
        self.stage2 = AtomRefinerV2MultiSample(...)
        self.n_samples = n_samples

    def forward_e2e(self, gt_centroids, aa_seq, chain_ids, res_idx, mask, noiser):
        # 1. Run trunk once
        trunk_tokens = self.stage1.get_trunk_tokens(...)

        # 2. Generate K diffusion samples
        centroids_samples = []
        for k in range(self.n_samples):
            sigma = noiser.sample_sigma(B, device)
            noise = torch.randn_like(gt_centroids)
            x_t = gt_centroids + sigma.view(-1, 1, 1) * noise
            x0_pred = self.stage1.forward_sigma_with_trunk(x_t, trunk_tokens, sigma, mask)
            centroids_samples.append(x0_pred)

        centroids_stack = torch.stack(centroids_samples, dim=1)  # [B, K, L, 3]

        # 3. Stage 2: atoms from multi-sample centroids
        atoms_pred = self.stage2(trunk_tokens, centroids_stack, mask)

        return {
            'centroids_samples': centroids_stack,
            'atoms_pred': atoms_pred,
        }
```

---

## Phase 3: Create Training Script

**File:** `scripts/train_resfold_e2e.py` (update existing or new)

### Training Modes

1. **`--mode stage1_only`**: Train Stage 1 diffusion (existing, unchanged)
2. **`--mode stage2_e2e`**: Load Stage 1 checkpoint, train Stage 2 with multi-sample conditioning

### Key Training Loop (Stage 2 E2E)

```python
def train_step(model, batch, noiser, geom_loss_fn, args):
    result = model.forward_e2e(
        batch['centroids'], batch['aa_seq'], batch['chain_ids'],
        batch['res_idx'], batch['mask_res'], noiser
    )

    # Stage 1 loss: average over K samples
    loss_s1 = 0.0
    for k in range(model.n_samples):
        loss_s1 += compute_mse_loss(
            result['centroids_samples'][:, k],
            batch['centroids'],
            batch['mask_res']
        )
    loss_s1 /= model.n_samples

    # Stage 2 loss: atoms + geometry
    atoms_flat = result['atoms_pred'].view(B, -1, 3)
    gt_atoms_flat = batch['coords_res'].view(B, -1, 3)
    loss_mse = compute_mse_loss(atoms_flat, gt_atoms_flat, batch['mask_atom'])
    loss_geom = geom_loss_fn(result['atoms_pred'], batch['mask_res'])
    loss_s2 = loss_mse + args.geom_weight * loss_geom

    # Combined loss (gradients flow through both stages!)
    total_loss = args.s1_weight * loss_s1 + args.s2_weight * loss_s2
    return total_loss
```

### Command-Line Args

```bash
# Phase 1: Train Stage 1
python train_resfold_e2e.py --mode stage1_only \
    --n_train 80 --n_steps 10000 --continuous_sigma \
    --output_dir outputs/resfold_stage1

# Phase 2: Train Stage 2 E2E
python train_resfold_e2e.py --mode stage2_e2e \
    --checkpoint outputs/resfold_stage1/best_model.pt \
    --n_samples 5 --s2_layers 6 \
    --n_train 80 --n_steps 5000 \
    --output_dir outputs/resfold_e2e
```

---

## Phase 4: Inference/Evaluation Updates

**File:** `scripts/eval_two_stage.py` or integrate into training

### Multi-Sample Inference

```python
@torch.no_grad()
def sample_e2e(model, batch, noiser, device, n_samples=5):
    """Full E2E sampling with multi-sample aggregation."""
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Run trunk once
    trunk_tokens = model.stage1.get_trunk_tokens(...)

    # Generate K centroid samples via full diffusion
    all_centroids = []
    for k in range(n_samples):
        centroids_k = sample_centroids_ve(model.stage1, batch, noiser, device)
        all_centroids.append(centroids_k)

    centroids_stack = torch.stack(all_centroids, dim=1)  # [B, K, L, 3]

    # Stage 2: atoms from multi-sample
    atoms_pred = model.stage2(trunk_tokens, centroids_stack, mask)

    return atoms_pred
```

---

## Implementation Order (Tasks)

### Step 1: AtomRefinerV2MultiSample
- [ ] Create `scripts/models/atomrefine_multi_sample.py`
- [ ] Implement `centroid_embed`, `sample_agg`, `transformer`, `output_proj`
- [ ] Add unit test to verify shapes

### Step 2: ResFoldE2E Wrapper
- [ ] Create `scripts/models/resfold_e2e.py`
- [ ] Implement `forward_e2e()` with K sample generation
- [ ] Add `load_stage1_checkpoint()` method
- [ ] Verify gradient flow: both stages receive gradients

### Step 3: Update Training Script
- [ ] Modify `scripts/train_resfold.py` OR create new `train_resfold_e2e.py`
- [ ] Add `--mode stage2_e2e` option
- [ ] Implement dual loss (Stage 1 + Stage 2)
- [ ] Add `--n_samples`, `--s2_layers` args

### Step 4: Evaluation
- [ ] Update evaluation to use multi-sample inference
- [ ] Compute DockQ, lDDT, ilDDT for E2E predictions

### Step 5: Verification
- [ ] Overfit test: train on 5 samples, verify both losses decrease
- [ ] Gradient flow test: check `grad` on Stage 1 and Stage 2 params
- [ ] Memory test: verify K=5 fits in GPU memory

---

## Memory Considerations

With K=5 samples per batch:
- Stage 1 generates 5 x0_pred tensors: 5 * [B, L, c_token] gradients
- Stage 2 processes [B, K, L, c_token] tensor

**Mitigations if memory is tight:**
1. Reduce batch_size, increase grad_accum
2. Use gradient checkpointing on Stage 1 transformer
3. Reduce K to 3 during training (can use K=5 at inference)

**Recommended settings for ~24GB GPU:**
```bash
--batch_size 8 --grad_accum 4 --n_samples 5
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/models/atomrefine_multi_sample.py` | CREATE | Multi-sample Stage 2 |
| `scripts/models/resfold_e2e.py` | CREATE | E2E wrapper |
| `scripts/models/__init__.py` | MODIFY | Export new classes |
| `scripts/train_resfold.py` | MODIFY | Add stage2_e2e mode |
| `scripts/eval_two_stage.py` | MODIFY | Multi-sample inference |

---

## Success Criteria

1. **Stage 1 alone**: Centroid RMSE < 3.0 A on test set
2. **Stage 2 E2E**: Atom RMSE < 3.5 A, lDDT > 0.4 on test set
3. **Gradient flow**: Both stages show non-zero gradients during E2E training
4. **Memory**: Training runs without OOM on 24GB GPU with K=5, batch=8
