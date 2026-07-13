"""Legacy diffusion methods must actually return their computed values.

Guards two methods that previously fell off the end returning None:
DiffusionSchedule.predict_x0 and LinearChainFlow.add_noise.
"""

import torch

from tinyfold.model.diffusion.noise import LinearChainFlow
from tinyfold.model.diffusion.schedule import DiffusionSchedule


def test_predict_x0_inverts_forward_noising():
    sched = DiffusionSchedule(T=16)
    x0 = torch.randn(32, 3)
    eps = torch.randn(32, 3)
    t = 7
    x_t = sched.sqrt_alpha_bar[t] * x0 + sched.sqrt_one_minus_alpha_bar[t] * eps
    x0_hat = sched.predict_x0(x_t, t, eps)
    assert x0_hat is not None
    assert torch.allclose(x0_hat, x0, atol=1e-4)


def test_linear_chain_flow_add_noise_returns_tuple():
    sched = DiffusionSchedule(T=16)
    flow = LinearChainFlow(sched, noise_scale=0.0)
    B, n_res = 1, 3
    N = n_res * 4
    x0 = torch.randn(B, N, 3)
    atom_to_res = torch.arange(n_res).repeat_interleave(4).unsqueeze(0)
    atom_type = torch.tensor([0, 1, 2, 3] * n_res).unsqueeze(0)
    chain_ids = torch.zeros(B, N, dtype=torch.long)
    x_t, target = flow.add_noise(x0, torch.tensor([7]), atom_to_res, atom_type, chain_ids)
    assert x_t is not None and target is not None
    assert x_t.shape == x0.shape
    assert torch.equal(target, x0)
