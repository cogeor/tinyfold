"""Compare single-shot vs iterative inference methods."""
import torch
import json
import sys
sys.path.insert(0, 'scripts')

from models.resfold import ResidueDenoiser
from models.resfold_assembler import ResFoldAssembler
from models.diffusion import VENoiser, KarrasSchedule
from train_resfold_stage2 import generate_centroid_samples

def load_models(stage1_ckpt, stage2_ckpt, device, trunk_layers=9, denoiser_blocks=7, c_token=256):
    """Load Stage 1 and Stage 2 models."""
    # Stage 1
    stage1_state = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    state_dict = stage1_state.get('model_state_dict', stage1_state)

    # Handle pipeline checkpoint (stage1. prefix)
    if any(k.startswith('stage1.') for k in state_dict.keys()):
        state_dict = {k.replace('stage1.', ''): v for k, v in state_dict.items() if k.startswith('stage1.')}

    trunk_model = ResidueDenoiser(
        c_token=c_token,
        trunk_layers=trunk_layers,
        denoiser_blocks=denoiser_blocks,
    ).to(device)
    trunk_model.load_state_dict(state_dict, strict=True)
    trunk_model.eval()

    # Stage 2
    stage2_state = torch.load(stage2_ckpt, map_location=device, weights_only=False)
    args2 = stage2_state.get('args', None)

    # Extract config from args
    n_layers_s2 = getattr(args2, 'n_layers', 19) if args2 else 19
    n_samples_s2 = getattr(args2, 'n_samples', 5) if args2 else 5
    n_heads_s2 = getattr(args2, 'n_heads', 8) if args2 else 8

    assembler = ResFoldAssembler(
        c_token=c_token,
        n_samples=n_samples_s2,
        n_layers=n_layers_s2,
        n_heads=n_heads_s2,
    ).to(device)
    assembler.load_state_dict(stage2_state['assembler_state_dict'])
    assembler.eval()

    return trunk_model, assembler, {'c_token': c_token}

def kabsch_rmsd(pred, target, mask=None):
    """Compute RMSD after optimal rotation alignment."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    # Center
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)

    # Kabsch
    H = pred_c.T @ tgt_c
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    pred_aligned = pred_c @ R
    rmsd = ((pred_aligned - tgt_c) ** 2).sum(dim=-1).mean().sqrt()
    return rmsd.item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_ckpt', default='outputs/stage1_50k_small/best_model.pt')
    parser.add_argument('--stage2_ckpt', default='outputs/stage2_overfit_1sample/best_model.pt')
    parser.add_argument('--data_path', default='data/processed/samples.parquet')
    parser.add_argument('--split_file', default='outputs/stage2_overfit_1sample/split.json')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--use_train', action='store_true', help='Use train sample instead of test')
    parser.add_argument('--trunk_layers', type=int, default=9)
    parser.add_argument('--denoiser_blocks', type=int, default=7)
    parser.add_argument('--c_token', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load models
    print('\nLoading models...')
    trunk_model, assembler, cfg = load_models(
        args.stage1_ckpt, args.stage2_ckpt, device,
        trunk_layers=args.trunk_layers,
        denoiser_blocks=args.denoiser_blocks,
        c_token=args.c_token
    )

    params1 = sum(p.numel() for p in trunk_model.parameters())
    params2 = sum(p.numel() for p in assembler.parameters())
    print(f'  Stage 1: {params1/1e6:.2f}M params')
    print(f'  Stage 2: {params2/1e6:.2f}M params')

    # Load data
    print('\nLoading data...')
    import pyarrow.parquet as pq
    table = pq.read_table(args.data_path)

    with open(args.split_file) as f:
        split = json.load(f)

    if args.use_train:
        idx = split['train_indices'][args.sample_idx]
        sample_type = 'train'
    else:
        idx = split['test_indices'][args.sample_idx]
        sample_type = 'test'

    row = table.slice(idx, 1).to_pydict()
    aa_seq = torch.tensor(row['seq'][0], dtype=torch.long, device=device).unsqueeze(0)
    chain_ids = torch.tensor(row['chain_id_res'][0], dtype=torch.long, device=device).unsqueeze(0)
    res_idx_data = row['res_idx'][0]
    res_idx = torch.tensor(res_idx_data, dtype=torch.long, device=device).unsqueeze(0)
    coords = torch.tensor(row['atom_coords'][0], dtype=torch.float32, device=device)

    L = aa_seq.shape[1]
    B = 1
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Reshape and center coords
    coords = coords.view(-1, 3)
    coords = coords - coords.mean(dim=0, keepdim=True)
    coords = coords / coords.std()

    # Reshape coords to [L, 4, 3]
    gt_atoms = coords.view(L, 4, 3)
    gt_centroids = gt_atoms.mean(dim=1)  # [L, 3]

    print(f'  Sample: {sample_type}, L={L} residues, {L*4} atoms')

    # Create noiser for centroid sampling
    schedule = KarrasSchedule(n_steps=50, sigma_min=0.002, sigma_max=10.0)
    noiser = VENoiser(schedule)

    # Generate centroid samples from Stage 1
    print('\nGenerating centroid samples from Stage 1...')
    with torch.no_grad():
        centroid_samples = generate_centroid_samples(
            trunk_model, noiser, aa_seq, chain_ids, res_idx, mask,
            n_samples=5, gt_centroids=gt_centroids.unsqueeze(0)
        )
    print(f'  Centroid samples shape: {centroid_samples.shape}')

    # Get trunk tokens
    with torch.no_grad():
        trunk_tokens = trunk_model.get_trunk_tokens(aa_seq, chain_ids, res_idx, mask)

    # === Compare methods ===
    print('\n' + '='*60)
    print('COMPARING INFERENCE METHODS')
    print('='*60)

    gt_atoms_batch = gt_atoms.unsqueeze(0)  # [1, L, 4, 3]
    gt_flat = gt_atoms_batch.view(-1, 3)

    # 1. Single-shot
    print('\n1. Single-shot inference:')
    with torch.no_grad():
        pred_single = assembler(trunk_tokens, centroid_samples, mask)
    pred_flat = pred_single.view(-1, 3)
    rmsd_single = kabsch_rmsd(pred_flat, gt_flat)
    print(f'   RMSD: {rmsd_single:.2f} Å')

    # 2. Iterative (residue-level) with clustering
    print('\n2. Iterative (residue-level) with clustering:')
    for k in [1, 5, 10, 20]:
        with torch.no_grad():
            pred_iter = assembler.sample_iterative_residue(
                trunk_tokens, centroid_samples, mask,
                k_residues_per_step=k,
                update_centroids=True
            )
        pred_flat = pred_iter.view(-1, 3)
        rmsd = kabsch_rmsd(pred_flat, gt_flat)
        n_iters = (L + k - 1) // k
        print(f'   k={k:2d} ({n_iters:3d} iters): RMSD = {rmsd:.2f} Å')

    # 3. Iterative (atom-level) with clustering
    print('\n3. Iterative (atom-level) with clustering:')
    for k in [4, 20, 40, 80]:
        with torch.no_grad():
            pred_iter = assembler.sample_iterative(
                trunk_tokens, centroid_samples, mask,
                k_per_step=k,
                update_centroids=True
            )
        pred_flat = pred_iter.view(-1, 3)
        rmsd = kabsch_rmsd(pred_flat, gt_flat)
        n_iters = (L * 4 + k - 1) // k
        print(f'   k={k:2d} ({n_iters:3d} iters): RMSD = {rmsd:.2f} Å')

    print('\n' + '='*60)

if __name__ == '__main__':
    main()
