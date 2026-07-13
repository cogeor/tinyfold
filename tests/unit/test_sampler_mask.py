"""DDPM sampler must accept a multi-element mask tensor without crashing.

Regression guard for `model_kwargs.get('mask') or model_kwargs.get('mask_res')`,
which raises "Boolean value of Tensor is ambiguous" whenever 'mask' holds a tensor.
"""

import torch
import torch.nn as nn

from tinyfold.model.diffusion.sampler import DDPMSampler


class _OneStepNoiser:
    T = 1  # single reverse step hits the t==0 branch (no schedule attrs needed)


def test_ddpm_sampler_accepts_mask_tensor():
    sampler = DDPMSampler()
    mask = torch.ones(2, 5)  # multi-element -> bool(mask) would raise

    def forward_fn(model, x, t, **kwargs):
        return torch.zeros_like(x)

    out = sampler.sample(
        nn.Identity(),
        (2, 5, 3),
        {"mask": mask},
        _OneStepNoiser(),
        torch.device("cpu"),
        forward_fn=forward_fn,
    )
    assert out.shape == (2, 5, 3)
