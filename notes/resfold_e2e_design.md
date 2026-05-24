# ResFold E2E: Two-Phase End-to-End Training

## Overview

Two-phase training approach that trains Stage 1 first, then trains Stage 2 end-to-end with multi-sample diffusion conditioning.

**Model sizes:**
- Stage 1: 15M params (kept exactly as is)
- Stage 2: ~5M params  
- Total: ~20M params

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ResFold E2E Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔═══════════════════════════════════════════════════════╗     │
│  ║ ResidueEncoder (Trunk) - shared between stages        ║     │
│  ║   Input: aa_seq, chain_ids, res_idx                   ║     │
│  ║   Output: trunk_tokens [B, L, c_token]                ║     │
│  ╚═══════════════════════════════════════════════════════╝     │
│                          ↓                                      │
│  ╔═══════════════════════════════════════════════════════╗     │
│  ║ Stage 1: Residue Diffusion Denoiser                   ║     │
│  ║   Generate K=5 diffusion samples:                     ║     │
│  ║   - Sample σ_k for each k ∈ [1..5]                    ║     │
│  ║   - x_t = gt_centroids + σ * noise                    ║     │
│  ║   - x0_pred_k = denoiser(x_t, trunk_tokens, σ)        ║     │
│  ║   Output: centroids_samples [B, K, L, 3]              ║     │
│  ╚═══════════════════════════════════════════════════════╝     │
│                          ↓                                      │
│  ╔═══════════════════════════════════════════════════════╗     │
│  ║ Stage 2: AtomRefinerV2 (Modified)                     ║     │
│  ║   Input:                                              ║     │
│  ║     - trunk_tokens [B, L, c_token]                    ║     │
│  ║     - centroids_samples [B, K, L, 3]                  ║     │
│  ║   Process:                                            ║     │
│  ║     - Embed each sample → [B, K, L, c_token]          ║     │
│  ║     - Aggregate (mean/attention) → [B, L, c_token]    ║     │
│  ║     - Add trunk_tokens conditioning                   ║     │
│  ║     - Transformer → atom offsets                      ║     │
│  ║   Output: atom_coords [B, L, 4, 3]                    ║     │
│  ╚═══════════════════════════════════════════════════════╝     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Strategy

### Phase 1: Train Stage 1 Only

```bash
python scripts/train_resfold.py --mode stage1_only \
    --n_train 80 --n_steps 10000 --continuous_sigma \
    --output_dir outputs/resfold_stage1
```

**Loss:** MSE on centroid predictions (existing implementation)

### Phase 2: Train Stage 2 E2E (with frozen or trainable Stage 1)

1. Load Stage 1 checkpoint
2. For each training batch:
   - Run trunk once → `trunk_tokens`
   - Generate K=5 diffusion samples from Stage 1 → `centroids_samples`
   - Pass both to Stage 2 → `atoms_pred`
3. Compute dual losses:
   - **Residue loss:** MSE(centroids_samples, gt_centroids)
   - **Atom loss:** MSE(atoms_pred, gt_atoms) + geometry losses
4. Backprop through both stages (full E2E)

---

## Model Changes

### [NEW] `models/resfold_e2e.py`

```python
class ResFoldE2E(nn.Module):
    def __init__(
        self,
        # Stage 1 (15M)
        c_token: int = 256,
        trunk_layers: int = 9,
        denoiser_blocks: int = 7,
        # Stage 2 (~5M)  
        s2_layers: int = 6,
        n_samples: int = 5,
    ):
        self.stage1 = ResidueDenoiser(...)
        self.stage2 = AtomRefinerV2MultiSample(
            c_token=c_token,
            n_layers=s2_layers,
            n_samples=n_samples,
        )
    
    def forward_e2e(self, gt_centroids, aa_seq, chain_ids, res_idx, mask, noiser):
        B, L, _ = gt_centroids.shape
        
        # 1. Trunk (shared, runs once)
        trunk_tokens = self.stage1.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)
        
        # 2. Stage 1: Generate K diffusion samples
        centroids_samples = []
        for k in range(self.n_samples):
            sigma = noiser.sample_sigma(B, device)
            noise = torch.randn_like(gt_centroids)
            x_t = gt_centroids + sigma.view(-1, 1, 1) * noise
            x0_pred = self.stage1.forward_sigma_with_trunk(x_t, trunk_tokens, sigma, mask)
            centroids_samples.append(x0_pred)
        
        centroids_stack = torch.stack(centroids_samples, dim=1)  # [B, K, L, 3]
        
        # 3. Stage 2: Atoms from trunk + multi-sample centroids
        atoms_pred = self.stage2(trunk_tokens, centroids_stack, mask)
        
        return {
            'centroids_samples': centroids_stack,  # [B, K, L, 3]
            'atoms_pred': atoms_pred,              # [B, L, 4, 3]
        }
```

### [NEW] `AtomRefinerV2MultiSample`

Modified Stage 2 that accepts K centroid samples:

```python
class AtomRefinerV2MultiSample(nn.Module):
    def __init__(self, c_token=256, n_layers=6, n_heads=8, n_samples=5):
        super().__init__()
        self.n_samples = n_samples
        
        # Embed each centroid sample
        self.centroid_embed = nn.Linear(3, c_token)
        
        # Aggregate K samples (learned attention or mean)
        self.sample_agg = nn.Sequential(
            nn.Linear(n_samples, 1),  # Simple weighted mean
        )
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(...)
        
        # Output: atom offsets
        self.output_proj = nn.Linear(c_token, 4 * 3)
    
    def forward(self, trunk_tokens, centroids_samples, mask):
        # centroids_samples: [B, K, L, 3]
        B, K, L, _ = centroids_samples.shape
        
        # Embed each sample
        sample_emb = self.centroid_embed(centroids_samples)  # [B, K, L, c_token]
        
        # Aggregate across samples
        sample_emb = sample_emb.permute(0, 2, 3, 1)  # [B, L, c_token, K]
        agg_emb = self.sample_agg(sample_emb).squeeze(-1)  # [B, L, c_token]
        
        # Combine with trunk tokens
        tokens = trunk_tokens + agg_emb
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Output offsets
        offsets = self.output_proj(tokens).view(B, L, 4, 3)
        
        # Get mean centroid for final position
        mean_centroid = centroids_samples.mean(dim=1)  # [B, L, 3]
        atoms = mean_centroid.unsqueeze(2) + offsets
        
        return atoms  # [B, L, 4, 3]
```

---

## Training Script

### [NEW] `scripts/train_resfold_e2e.py`

Key training loop:

```python
def train_step(model, batch, noiser, geom_loss_fn, args):
    result = model.forward_e2e(
        batch['centroids'], batch['aa_seq'], batch['chain_ids'],
        batch['res_idx'], batch['mask_res'], noiser
    )
    
    # Stage 1 loss: MSE on each of K samples
    loss_s1 = 0.0
    for k in range(model.n_samples):
        loss_s1 += compute_mse_loss(
            result['centroids_samples'][:, k], 
            batch['centroids'], 
            batch['mask_res']
        )
    loss_s1 /= model.n_samples
    
    # Stage 2 loss: atoms + geometry
    B, L = batch['centroids'].shape[:2]
    atoms_flat = result['atoms_pred'].view(B, -1, 3)
    gt_atoms_flat = batch['coords_res'].view(B, -1, 3)
    
    loss_mse = compute_mse_loss(atoms_flat, gt_atoms_flat, batch['mask_atom'])
    loss_geom = geom_loss_fn(result['atoms_pred'], batch['mask_res'])
    loss_s2 = loss_mse + args.geom_weight * loss_geom
    
    # Combined E2E loss
    total_loss = args.s1_weight * loss_s1 + args.s2_weight * loss_s2
    total_loss.backward()
    
    return {
        'stage1': loss_s1.item(),
        'stage2': loss_s2.item(),
        'total': total_loss.item(),
    }
```

---

## Memory Considerations

With K=5 samples per batch:
- Stage 1 forward: 5× memory for centroid predictions
- Gradient checkpointing can help if needed
- Consider gradient accumulation for larger effective batch

**Recommended settings for ~24GB GPU:**
```bash
python train_resfold_e2e.py \
    --batch_size 8 --grad_accum 4 \
    --n_samples 5 \
    --checkpoint outputs/resfold_stage1/best_model.pt
```

---

## Verification

1. **Gradient flow test:** Both stages receive gradients during E2E training
2. **Overfitting test:** Train on 5 samples, check both losses decrease
3. **Memory test:** Verify fit on target GPU with K=5 samples

```bash
# Quick overfit test
python train_resfold_e2e.py --n_train 5 --n_test 2 --n_steps 2000 \
    --checkpoint outputs/resfold_stage1/best_model.pt
```
