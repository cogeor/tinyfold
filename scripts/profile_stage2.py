"""Profile Stage 2 training to find bottlenecks."""
import torch
import time
import sys
sys.path.insert(0, 'scripts')

from models.resfold import ResidueDenoiser
from models.resfold_assembler import ResFoldAssembler
from models.diffusion import VENoiser, KarrasSchedule

device = torch.device('cuda')

# Simulate typical batch
B, L = 8, 200  # batch=8, ~200 residues avg
K = 5  # centroid samples

print(f"Profiling with B={B}, L={L}, K={K}")
print("=" * 60)

# Create models
trunk = ResidueDenoiser(c_token=256, trunk_layers=9, denoiser_blocks=7).to(device)
trunk.eval()
for p in trunk.parameters():
    p.requires_grad = False

# Test different assembler sizes
for n_layers in [6, 10, 14, 19]:
    assembler = ResFoldAssembler(c_token=256, n_samples=5, n_layers=n_layers).to(device)
    params = sum(p.numel() for p in assembler.parameters()) / 1e6

    # Inputs
    aa_seq = torch.randint(0, 20, (B, L), device=device)
    chain_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    gt_centroids = torch.randn(B, L, 3, device=device)
    gt_atoms = torch.randn(B, L, 4, 3, device=device)

    schedule = KarrasSchedule(n_steps=50)
    noiser = VENoiser(schedule)

    torch.cuda.synchronize()

    # Profile trunk tokens (once per batch)
    t0 = time.time()
    with torch.no_grad():
        trunk_tokens = trunk.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)
    torch.cuda.synchronize()
    t_trunk = time.time() - t0

    # Profile centroid generation (K forward passes)
    t0 = time.time()
    with torch.no_grad():
        samples = []
        for k in range(K):
            sigma = torch.full((B,), 0.5 + k * 0.5, device=device)
            noisy = gt_centroids + sigma.view(-1, 1, 1) * torch.randn_like(gt_centroids)
            pred = trunk.forward_sigma_with_trunk(noisy, trunk_tokens, sigma, mask)
            samples.append(pred)
        centroid_samples = torch.stack(samples, dim=1)
    torch.cuda.synchronize()
    t_centroids = time.time() - t0

    # Profile assembler forward
    t0 = time.time()
    pred_atoms = assembler(trunk_tokens, centroid_samples, mask)
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    # Profile backward
    loss = (pred_atoms - gt_atoms).pow(2).mean()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t0

    total = t_trunk + t_centroids + t_fwd + t_bwd
    print(f"\nn_layers={n_layers:2d} ({params:.1f}M params):")
    print(f"  Trunk tokens:     {t_trunk*1000:6.1f} ms")
    print(f"  Centroid gen (x5):{t_centroids*1000:6.1f} ms")
    print(f"  Assembler fwd:    {t_fwd*1000:6.1f} ms")
    print(f"  Assembler bwd:    {t_bwd*1000:6.1f} ms")
    print(f"  Total:            {total*1000:6.1f} ms ({1/total:.1f} it/s)")

    del assembler
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Conclusion: Centroid generation is likely the bottleneck")
print("Options:")
print("  1. Cache centroids during training epoch")
print("  2. Reduce K from 5 to 3")
print("  3. Use smaller Stage 2 model")
