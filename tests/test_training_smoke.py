"""End-to-end smoke test of the live training data path.

Exercises parquet-row -> load_sample -> collate_batch -> one forward/backward
train step on ResFoldOneStep -> checkpoint save/load round-trip, all on tiny
synthetic data. This is the safety net for the training path (otherwise
untested) before any structural refactor touches it.
"""

import numpy as np
import pyarrow as pa
import torch

from tinyfold.model.resfold.onestep import ResFoldOneStep
from tinyfold.training.checkpointing import load_checkpoint, save_checkpoint
from tinyfold.training.data import collate_batch, load_sample


def _sample_columns(n_res_a: int, n_res_b: int, seed: int):
    rng = np.random.default_rng(seed)
    n_res = n_res_a + n_res_b
    n_atoms = n_res * 4
    return {
        "atom_coords": (rng.standard_normal(n_atoms * 3) * 10).tolist(),
        "atom_type": ([0, 1, 2, 3] * n_res),
        "atom_to_res": np.repeat(np.arange(n_res), 4).tolist(),
        "seq": rng.integers(0, 20, n_res).tolist(),
        "chain_id_res": ([0] * n_res_a + [1] * n_res_b),
        "res_idx": (list(range(n_res_a)) + list(range(n_res_b))),
        "sample_id": f"synthetic_{seed}",
    }


def _synthetic_table():
    rows = [_sample_columns(6, 4, seed=0), _sample_columns(5, 3, seed=1)]
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    return pa.table(cols)


def _tiny_model():
    return ResFoldOneStep(
        c_token=32, trunk_layers=1, trunk_heads=2, denoiser_blocks=1,
        denoiser_heads=2, atom_head_layers=1, atom_head_heads=2, n_timesteps=10,
    )


def test_load_sample_shapes():
    table = _synthetic_table()
    s = load_sample(table, 0)
    assert s["n_res"] == 10 and s["n_atoms"] == 40
    assert s["coords_res"].shape == (10, 4, 3)
    assert s["centroids"].shape == (10, 3)
    assert s["aa_seq"].shape == (10,)
    assert s["chain_ids"].tolist() == [0] * 6 + [1] * 4
    assert not torch.isnan(s["centroids"]).any()


def test_collate_pads_and_masks():
    table = _synthetic_table()
    batch = collate_batch([load_sample(table, 0), load_sample(table, 1)],
                          torch.device("cpu"))
    assert batch["centroids"].shape == (2, 10, 3)  # padded to max L=10
    assert batch["mask_res"][0].sum().item() == 10
    assert batch["mask_res"][1].sum().item() == 8
    assert not torch.isnan(batch["centroids"]).any()


def test_one_train_step_updates_params():
    torch.manual_seed(0)
    table = _synthetic_table()
    batch = collate_batch([load_sample(table, 0), load_sample(table, 1)],
                          torch.device("cpu"))
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sigma = torch.full((2,), 0.5)
    x_t = batch["centroids"] + torch.randn_like(batch["centroids"]) * 0.5
    centroid_pred, _, _ = model.forward_sigma(
        x_t, batch["aa_seq"], batch["chain_ids"], batch["res_idx"], sigma,
        batch["mask_res"],
    )
    m = batch["mask_res"].unsqueeze(-1).float()
    loss = ((centroid_pred - batch["centroids"]) ** 2 * m).sum() / m.sum()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    opt.step()

    changed = [k for k, v in model.state_dict().items()
               if not torch.equal(v, before[k])]
    assert changed, "no parameters changed after an optimizer step"


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=7, metrics={"train_rmse": 0.5})

    fresh = _tiny_model()
    meta = load_checkpoint(path, fresh)
    assert meta["step"] == 7
    for k, v in model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k])
