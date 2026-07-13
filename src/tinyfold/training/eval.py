"""Evaluation helpers shared by training/eval/plotting.

Extracted from train_resfold.py, where the continuous-sigma sampler dispatch was
copy-pasted three times (test-eval, plotting-stage1, plotting).
"""

from tinyfold.inference.samplers import sample_centroids_one_shot, sample_centroids_ve


def sample_centroids_continuous(
    model,
    batch,
    noiser,
    device,
    *,
    one_shot: bool,
    align_per_step: bool,
    recenter: bool,
    kabsch_interp: bool,
    is_onestep: bool,
):
    """Dispatch the continuous-sigma centroid sampler.

    Picks the one-shot sampler when ``one_shot`` else the VE Euler sampler, and
    returns the sampler's raw output ((centroids, atoms) when ``is_onestep``,
    else centroids).
    """
    if one_shot:
        return sample_centroids_one_shot(model, batch, noiser, device, is_onestep=is_onestep)
    return sample_centroids_ve(
        model,
        batch,
        noiser,
        device,
        align_per_step=align_per_step,
        recenter=recenter,
        kabsch_interp=kabsch_interp,
        is_onestep=is_onestep,
    )
