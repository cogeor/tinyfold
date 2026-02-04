#!/usr/bin/env python
"""TinyFold Web Visualizer - FastAPI backend.

Serves protein structures from the dataset and runs model predictions.

Usage:
    cd web
    python server.py
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add paths for imports
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent / "scripts"))
sys.path.insert(0, str(script_dir.parent / "src"))

from models import create_model, create_schedule, create_noiser
from tinyfold.training.data_split import DataSplitConfig, get_train_test_indices


# =============================================================================
# Response Models
# =============================================================================

class SampleMeta(BaseModel):
    sample_id: str
    split: str
    n_atoms: int
    n_residues: int
    has_prediction: bool = False  # True if pre-computed prediction available


class SampleListResponse(BaseModel):
    samples: list[SampleMeta]
    total: int


class SampleDetailResponse(BaseModel):
    sample_id: str
    split: str
    n_atoms: int
    n_residues: int
    pdb_string: str


class PredictionResponse(BaseModel):
    sample_id: str
    pdb_string: str
    rmsd: float
    inference_time_ms: float
    cached: bool = False  # True if loaded from pre-computed cache


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state initialized at startup."""
    config: dict = None
    device: torch.device = None
    model = None
    noiser = None
    schedule = None
    table = None
    train_indices: list[int] = []
    test_indices: list[int] = []
    sample_to_split: dict[str, str] = {}
    predictions_cache: dict = None  # Pre-computed predictions


state = AppState()


# =============================================================================
# Data Loading
# =============================================================================

def load_sample_raw(table, i: int) -> dict:
    """Load a single sample from parquet table."""
    coords = np.array(table['atom_coords'][i].as_py(), dtype=np.float32)
    atom_types = np.array(table['atom_type'][i].as_py(), dtype=np.int64)
    atom_to_res = np.array(table['atom_to_res'][i].as_py(), dtype=np.int64)
    seq_res = np.array(table['seq'][i].as_py(), dtype=np.int64)
    chain_res = np.array(table['chain_id_res'][i].as_py(), dtype=np.int64)
    res_idx = np.array(table['res_idx'][i].as_py(), dtype=np.int64)

    n_atoms = len(atom_types)
    coords = coords.reshape(n_atoms, 3)

    # Per-atom sequence and chain
    aa_seq = seq_res[atom_to_res]
    chain_ids = chain_res[atom_to_res]

    # Center and normalize
    centroid = coords.mean(axis=0, keepdims=True)
    coords_centered = coords - centroid
    std = coords_centered.std()
    coords_norm = coords_centered / std

    return {
        'sample_id': table['sample_id'][i].as_py(),
        'coords': coords,  # Original coords for PDB output
        'coords_norm': coords_norm,  # Normalized for model
        'atom_types': atom_types,
        'atom_to_res': atom_to_res,
        'aa_seq': aa_seq,
        'chain_ids': chain_ids,
        'seq_res': seq_res,
        'chain_res': chain_res,
        'res_idx': res_idx,
        'std': std,
        'centroid': centroid,
        'n_atoms': n_atoms,
        'n_residues': len(seq_res),
    }


def coords_to_pdb_string(
    xyz: np.ndarray,
    atom_to_res: np.ndarray,
    atom_types: np.ndarray,
    chain_res: np.ndarray,
    res_idx: np.ndarray,
    seq_res: np.ndarray,
) -> str:
    """Convert coordinates to PDB format string."""
    ATOM_NAMES = ['N', 'CA', 'C', 'O']
    AA_3LETTER = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
        'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'UNK'
    ]

    lines = []
    atom_serial = 1
    prev_chain = None

    for i in range(len(xyz)):
        x, y, z = xyz[i]
        res = atom_to_res[i]
        atype = atom_types[i]
        chain = chain_res[res]
        resnum = res_idx[res] + 1  # PDB uses 1-based

        atom_name = ATOM_NAMES[atype] if atype < 4 else 'X'
        element = atom_name[0]
        chain_label = 'A' if chain == 0 else 'B'
        resname = AA_3LETTER[seq_res[res]] if seq_res[res] < 21 else 'UNK'

        # Add TER at chain break
        if prev_chain is not None and chain != prev_chain:
            lines.append("TER")
        prev_chain = chain

        # Format atom name (left-justified for 1-2 char)
        if len(atom_name) < 4:
            atom_name_fmt = f" {atom_name:<3}"
        else:
            atom_name_fmt = f"{atom_name:<4}"

        line = (
            f"ATOM  {atom_serial:5d} {atom_name_fmt} {resname:>3} "
            f"{chain_label}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2}"
        )
        lines.append(line)
        atom_serial += 1

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)


# =============================================================================
# Inference
# =============================================================================

def kabsch_align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch alignment and RMSD computation.

    Aligns pred to target using optimal rotation (Kabsch algorithm).

    Args:
        pred: [N, 3] predicted coordinates
        target: [N, 3] target coordinates

    Returns:
        aligned_pred: [N, 3] aligned prediction
        rmsd: float
    """
    # Center both point clouds
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    P = pred - pred_mean    # centered prediction
    Q = target - target_mean  # centered target

    # Compute cross-covariance matrix
    H = P.T @ Q  # [3, 3]

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Correct for reflection
    d = np.linalg.det(V @ U.T)
    if d < 0:
        V[:, 2] *= -1

    # Optimal rotation matrix: R = V @ U^T
    # To align P to Q, apply as: P_aligned = P @ R.T (equiv. to R @ P^T transposed)
    R = V @ U.T
    P_aligned = P @ R.T

    # Translate to target center
    pred_aligned = P_aligned + target_mean

    # Compute RMSD
    diff = pred_aligned - target
    rmsd = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    return pred_aligned, rmsd


@torch.no_grad()
def run_inference(sample: dict) -> np.ndarray:
    """Run diffusion inference on a sample."""
    device = state.device

    # Prepare tensors
    coords = torch.tensor(sample['coords_norm'], dtype=torch.float32).unsqueeze(0).to(device)
    atom_types = torch.tensor(sample['atom_types'], dtype=torch.long).unsqueeze(0).to(device)
    atom_to_res = torch.tensor(sample['atom_to_res'], dtype=torch.long).unsqueeze(0).to(device)
    aa_seq = torch.tensor(sample['aa_seq'], dtype=torch.long).unsqueeze(0).to(device)
    chain_ids = torch.tensor(sample['chain_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.ones(1, sample['n_atoms'], dtype=torch.bool, device=device)

    noiser = state.noiser
    model = state.model
    config = state.config
    noise_type = config['inference']['noise_type']
    clamp_val = config['inference']['clamp_val']

    B, N = 1, sample['n_atoms']

    # Initialize based on noise type
    if noise_type in ("linear_chain", "linear_flow"):
        from models.diffusion import generate_extended_chain
        x_linear = generate_extended_chain(
            n_atoms=N,
            atom_to_res=atom_to_res[0],
            atom_type=atom_types[0],
            chain_ids=chain_ids[0],
            device=device,
            apply_rotation=False,
        ).unsqueeze(0)
        x = x_linear.clone()
        t_range = reversed(range(noiser.T + 1))
    else:
        x_linear = None
        x = torch.randn(B, N, 3, device=device)
        t_range = reversed(range(noiser.T))

    # Diffusion loop
    for t in t_range:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        if noise_type == "linear_chain" and hasattr(model, 'forward_direct'):
            x0_pred = model.forward_direct(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)
        else:
            x0_pred = model(x, atom_types, atom_to_res, aa_seq, chain_ids, t_batch, mask)

        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        if noise_type in ("linear_chain", "linear_flow"):
            x = noiser.reverse_step(x, x0_pred, t, x_linear)
        else:
            # DDPM reverse
            if t > 0:
                ab_t = noiser.alpha_bar[t]
                ab_prev = noiser.alpha_bar[t - 1]
                beta = noiser.betas[t]
                alpha = noiser.alphas[t]

                coef1 = torch.sqrt(ab_prev) * beta / (1 - ab_t)
                coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
                mean = coef1 * x0_pred + coef2 * x

                var = beta * (1 - ab_prev) / (1 - ab_t)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = x0_pred

    # Denormalize
    pred_norm = x[0].cpu().numpy()
    pred = pred_norm * sample['std'] + sample['centroid']

    return pred


# =============================================================================
# FastAPI App
# =============================================================================

def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def init_app():
    """Initialize model and data at startup."""
    state.config = load_config()
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {state.device}")

    # Load model
    print("Loading model...")
    checkpoint_path = Path(__file__).parent / state.config["model"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)

    # Infer n_timesteps from checkpoint's time_embed shape
    time_embed_key = None
    for k in checkpoint["model_state_dict"]:
        if "time_embed.weight" in k:
            time_embed_key = k
            break

    if time_embed_key:
        embed_size = checkpoint["model_state_dict"][time_embed_key].shape[0]
        n_timesteps = embed_size
        print(f"Inferred n_timesteps={n_timesteps} from checkpoint (embed_size={embed_size})")
    else:
        n_timesteps = state.config["inference"]["n_timesteps"]

    model = create_model(
        state.config["model"]["architecture"],
        n_timesteps=n_timesteps,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(state.device)
    model.eval()
    state.model = model
    state.config["inference"]["n_timesteps"] = n_timesteps  # Update config to match
    print(f"Loaded model: {state.config['model']['architecture']}")

    # Create noiser
    state.schedule = create_schedule("cosine", T=n_timesteps)
    state.noiser = create_noiser(
        state.config["inference"]["noise_type"],
        state.schedule,
    )
    print(f"Noise type: {state.config['inference']['noise_type']}")

    # Load data
    print("Loading data...")
    data_path = Path(__file__).parent / state.config["data"]["samples_path"]
    state.table = pq.read_table(data_path)
    print(f"Loaded {len(state.table)} samples")

    # Get train/test split
    split_config = DataSplitConfig(
        n_train=state.config["data"]["n_train"],
        n_test=state.config["data"]["n_test"],
        min_atoms=state.config["data"]["min_atoms"],
        max_atoms=state.config["data"]["max_atoms"],
    )
    state.train_indices, state.test_indices = get_train_test_indices(state.table, split_config)
    print(f"Train: {len(state.train_indices)}, Test: {len(state.test_indices)}")

    # Build sample_id -> split mapping
    for idx in state.train_indices:
        sample_id = state.table['sample_id'][idx].as_py()
        state.sample_to_split[sample_id] = "train"
    for idx in state.test_indices:
        sample_id = state.table['sample_id'][idx].as_py()
        state.sample_to_split[sample_id] = "test"

    # Load predictions cache if configured
    cache_path = state.config.get("predictions", {}).get("cache_path")
    if cache_path:
        cache_file = Path(__file__).parent / cache_path
        if cache_file.exists():
            import json
            with open(cache_file) as f:
                state.predictions_cache = json.load(f)
            print(f"Loaded {len(state.predictions_cache)} cached predictions from {cache_path}")
        else:
            print(f"Warning: Predictions cache not found: {cache_path}")
            state.predictions_cache = None
    else:
        state.predictions_cache = None

    print("Server ready!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app()
    yield


app = FastAPI(title="TinyFold Visualizer", lifespan=lifespan)


# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve main page."""
    return FileResponse(static_dir / "index.html")


@app.get("/api/samples", response_model=SampleListResponse)
def list_samples(split: str = "all", search: str = "", page: int = 1, per_page: int = 50):
    """List samples with optional filtering.

    Only shows samples that have pre-computed predictions.
    """
    samples = []

    # Determine which indices to use
    if split == "train":
        indices = state.train_indices
    elif split == "test":
        indices = state.test_indices
    else:
        indices = state.train_indices + state.test_indices

    for idx in indices:
        sample_id = state.table['sample_id'][idx].as_py()

        # Only show samples with predictions
        has_pred = state.predictions_cache is not None and sample_id in state.predictions_cache
        if not has_pred:
            continue

        if search and search.lower() not in sample_id.lower():
            continue

        n_atoms = len(state.table['atom_type'][idx].as_py())
        n_residues = len(state.table['seq'][idx].as_py())
        split_name = state.sample_to_split.get(sample_id, "unknown")

        samples.append(SampleMeta(
            sample_id=sample_id,
            split=split_name,
            n_atoms=n_atoms,
            n_residues=n_residues,
            has_prediction=True,
        ))

    total = len(samples)
    start = (page - 1) * per_page
    samples = samples[start:start + per_page]

    return SampleListResponse(samples=samples, total=total)


@app.get("/api/sample/{sample_id}", response_model=SampleDetailResponse)
def get_sample(sample_id: str):
    """Get sample details including PDB string."""
    # Find sample index
    idx = None
    for i in state.train_indices + state.test_indices:
        if state.table['sample_id'][i].as_py() == sample_id:
            idx = i
            break

    if idx is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = load_sample_raw(state.table, idx)
    split = state.sample_to_split.get(sample_id, "unknown")

    pdb = coords_to_pdb_string(
        xyz=sample['coords'],
        atom_to_res=sample['atom_to_res'],
        atom_types=sample['atom_types'],
        chain_res=sample['chain_res'],
        res_idx=sample['res_idx'],
        seq_res=sample['seq_res'],
    )

    return SampleDetailResponse(
        sample_id=sample_id,
        split=split,
        n_atoms=sample['n_atoms'],
        n_residues=sample['n_residues'],
        pdb_string=pdb,
    )


@app.get("/api/sample/{sample_id}/pdb", response_class=PlainTextResponse)
def get_sample_pdb(sample_id: str):
    """Get raw PDB string for a sample."""
    detail = get_sample(sample_id)
    return detail.pdb_string


@app.post("/api/predict/{sample_id}", response_model=PredictionResponse)
def predict(sample_id: str):
    """Run prediction on a sample."""
    # Find sample
    idx = None
    for i in state.train_indices + state.test_indices:
        if state.table['sample_id'][i].as_py() == sample_id:
            idx = i
            break

    if idx is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = load_sample_raw(state.table, idx)

    # Check for cached prediction
    if state.predictions_cache and sample_id in state.predictions_cache:
        cached = state.predictions_cache[sample_id]
        pred_coords = np.array(cached['coords'], dtype=np.float32)
        rmsd = cached['rmsd']
        inference_time = cached.get('inference_time', 0) * 1000  # Convert to ms

        # Generate PDB from cached coords
        pdb = coords_to_pdb_string(
            xyz=pred_coords,
            atom_to_res=sample['atom_to_res'],
            atom_types=sample['atom_types'],
            chain_res=sample['chain_res'],
            res_idx=sample['res_idx'],
            seq_res=sample['seq_res'],
        )

        return PredictionResponse(
            sample_id=sample_id,
            pdb_string=pdb,
            rmsd=rmsd,
            inference_time_ms=inference_time,
            cached=True,
        )

    # Run live inference
    start = time.time()
    pred_coords = run_inference(sample)
    inference_time = (time.time() - start) * 1000

    # Align prediction to ground truth
    pred_aligned, rmsd = kabsch_align(pred_coords, sample['coords'])

    # Generate PDB
    pdb = coords_to_pdb_string(
        xyz=pred_aligned,
        atom_to_res=sample['atom_to_res'],
        atom_types=sample['atom_types'],
        chain_res=sample['chain_res'],
        res_idx=sample['res_idx'],
        seq_res=sample['seq_res'],
    )

    return PredictionResponse(
        sample_id=sample_id,
        pdb_string=pdb,
        rmsd=rmsd,
        inference_time_ms=inference_time,
        cached=False,
    )


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
    )
