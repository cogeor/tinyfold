"""Diffusion sampling for ResFold inference.

Sampler functions moved verbatim out of scripts/train_resfold.py (loop 05 of the
refactor) so eval/showcase tooling can import them from the package instead of
importing the 2k-line training CLI. Behavior is unchanged.
"""
import torch

from tinyfold.model.diffusion import kabsch_align_to_target
from tinyfold.model.geometry import kabsch_rigid


@torch.no_grad()
def sample_atoms_diffusion(model, tokens, centroids, mask, n_steps=8,
                           sigma_min=0.002, sigma_max=1.0, generator=None):
    """Iterative atom-diffusion sampling of backbone offsets, given centroids.

    Deterministic Euler over a Karras sigma schedule; returns atoms
    ``[B, L, 4, 3] = centroids + delta``. ``tokens`` are the centroid-stage
    denoiser tokens (from ``model.centroid_tokens``).
    """
    from tinyfold.model.diffusion import KarrasSchedule
    B, L = mask.shape
    device = centroids.device
    sched = KarrasSchedule(n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0)
    sigmas = sched.sigmas.to(device)
    delta = sigmas[0] * torch.randn(B, L, 4, 3, device=device, generator=generator)
    for i in range(len(sigmas) - 1):
        s = sigmas[i].expand(B)
        d0 = model.denoise_atoms(delta, tokens, s, mask)
        d_dir = (delta - d0) / sigmas[i]
        delta = delta + d_dir * (sigmas[i + 1] - sigmas[i])
    # final clean estimate at the smallest sigma
    delta = model.denoise_atoms(delta, tokens, sigmas[-1].expand(B), mask)
    return centroids.unsqueeze(2) + delta


def _template_kwargs(batch):
    """Template forward-kwargs, only when a template is present in the batch.

    Returns an empty dict when there is no template, so denoisers that do not
    accept template kwargs (pipeline stage1, test doubles) are unaffected.
    """
    if batch.get('template_coords_res') is None:
        return {}
    return dict(
        template_coords_res=batch.get('template_coords_res'),
        template_mask=batch.get('template_mask'),
        template_frame_id=batch.get('template_frame_id'),
    )


@torch.no_grad()
def sample_centroids(model, batch, noiser, device, clamp_val=3.0,
                     align_per_step=False, recenter=False):
    """DDPM sampling for Stage 1 centroids only.

    Args:
        model: ResFoldPipeline model
        batch: Batch dict with aa_seq, chain_ids, res_idx, mask_res
        noiser: Diffusion noiser with schedule
        device: torch device
        clamp_val: Value to clamp predictions
        align_per_step: If True, Kabsch-align x0_pred to current x before update.
                        This fixes frame drift (Boltz-1 style). Default False for
                        backward compatibility.
        recenter: If True, re-center coordinates each step (avoids translation drift).
                  Default False for backward compatibility.

    Returns:
        centroids: [B, L, 3] predicted centroids
    """
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Start from noise
    x = torch.randn(B, L, 3, device=device)

    for t in reversed(range(noiser.T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict x0 (clean centroids)
        x0_pred = model.forward_stage1(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            t_batch, mask, esm_embed=batch.get('esm_embed'),
        )
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # NEW: Kabsch-align x0_pred to current x's frame (fixes drift)
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        # DDPM reverse step
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

        # NEW: Re-center to avoid translation drift
        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    return x


@torch.no_grad()
def sample_centroids_one_shot(model, batch, noiser, device, is_onestep=False,
                              sigma_init=None, generator=None):
    """Single-forward inference for EDM-preconditioned models.

    At high sigma the EDM model's output is dominated by F (the learned prior).
    With sigma_data=1 and a well-trained model, one forward at sigma~=sigma_max
    on pure-noise input recovers the structure directly. This avoids the
    multi-step trajectory drift that plagues Euler on overfit models, and is
    appropriate when the science target is "sequence -> structure" (the model
    learns a regression conditional on noise level).

    For ResFoldOneStep, also returns the atom-head output from the same forward.

    When ``generator`` is provided, the noise init draws from it instead of the
    global RNG so the caller can request K reproducible, distinct samples.
    """
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']
    sigmas = noiser.sigmas.to(device)
    if sigma_init is None:
        sigma_init = sigmas[0]
    sigma_init = torch.as_tensor(sigma_init, device=device).view(1).expand(B)
    x = sigma_init.view(B, 1, 1) * torch.randn(B, L, 3, device=device, generator=generator)
    denoiser = model if is_onestep else model.stage1
    # Atom-diffusion models: one-shot centroids, then iterative atom diffusion.
    if is_onestep and getattr(model, "atom_diffusion", False):
        centroid_pred, tokens, _ = model.centroid_tokens(
            x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
            sigma_init, mask, x0_prev=None, esm_embed=batch.get('esm_embed'),
            **_template_kwargs(batch),
        )
        atoms_pred = sample_atoms_diffusion(
            model, tokens, centroid_pred, mask,
            n_steps=getattr(model, "_atom_eval_steps", 8),
            sigma_min=model.atom_sigma_min, sigma_max=model.atom_sigma_max,
            generator=generator,
        )
        return centroid_pred, atoms_pred
    out = denoiser.forward_sigma(
        x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
        sigma_init, mask, x0_prev=None,
        esm_embed=batch.get('esm_embed'),
        **_template_kwargs(batch),
    )
    if is_onestep:
        # Loop 06: ResFoldOneStep.forward_sigma returns a 3-tuple
        # (centroid, atoms, pred_lddt_or_None). Drop pred_lddt here — single-
        # sample inference doesn't rank, and callers downstream expect the
        # legacy 2-tuple shape.
        centroid_pred, atoms_pred, _ = out
        return centroid_pred, atoms_pred
    return out


@torch.no_grad()
def sample_centroids_ve(model, batch, noiser, device, clamp_val=3.0,
                        align_per_step=True, recenter=True, kabsch_interp=False,
                        self_cond=True, is_onestep=False, generator=None):
    """VE (variance-exploding) sampling for continuous sigma models.

    Uses AF3-style Euler sampling with the Karras sigma schedule.
    Model must have been trained with continuous sigma (forward_sigma).

    When ``is_onestep`` is True the model's forward_sigma is expected to return
    a (centroid, atoms) tuple; sampling uses only the centroid for the
    diffusion update, and at the end of the loop a final forward at sigma_min
    is run to read the atom-head output for downstream visualization.

    Args:
        model: ResFoldPipeline (or ResFoldOneStep when is_onestep=True)
        batch: Batch dict with aa_seq, chain_ids, res_idx, mask_res
        noiser: VENoiser with KarrasSchedule (has .sigmas attribute)
        device: torch device
        clamp_val: Value to clamp predictions
        align_per_step: Kabsch-align x0_pred to x each step
        recenter: Re-center coordinates each step
        kabsch_interp: Boltz Kabsch-interpolation sampler. When True, after
            each Euler step the freshly-updated ``x`` is rigid-aligned onto
            the previous step's ``x``, so consecutive denoiser inputs share a
            rigid frame. Orthogonal to ``align_per_step`` (different line,
            different anchor); the two can coexist. See
            ``.delegate/work/20260524-051951-prio01-retrain/03/PLAN.md`` D2.
        self_cond: Use self-conditioning (pass previous x0_pred to model)
        is_onestep: Set True for ResFoldOneStep (tuple return, atom output)

    Returns:
        centroids if is_onestep is False, else (centroids, atoms)
    """
    B, L = batch['aa_seq'].shape
    mask = batch['mask_res']

    # Get sigma schedule from noiser
    sigmas = noiser.sigmas.to(device)  # Decreasing: [sigma_max, ..., sigma_min]

    # Initialize at highest noise level (VE: x = sigma * noise)
    x = sigmas[0] * torch.randn(B, L, 3, device=device, generator=generator)

    # Track previous x0_pred for self-conditioning
    x0_prev = None

    # OneStep models expose forward_sigma directly; pipeline exposes it on .stage1
    denoiser = model if is_onestep else model.stage1

    # Trunk caching: the trunk (sequence + template + relpos/pair track) does NOT
    # depend on x_t or sigma, so for onestep models we compute it ONCE and reuse
    # it across all Euler steps via forward_sigma_with_trunk. This is exact and
    # avoids re-running the O(L^2) pair track every step (~T x faster for
    # template-conditioned models). The pipeline path keeps forward_sigma.
    trunk_tokens = None
    if is_onestep:
        trunk_tokens = model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], mask,
            esm_embed=batch.get('esm_embed'), **_template_kwargs(batch),
        )

    # Euler sampling loop
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Create sigma tensor for batch
        sigma_batch = sigma.expand(B)

        # Snapshot x BEFORE the denoiser forward — clone is required because
        # x is rebound by the Euler step below. Used by the kabsch_interp
        # branch to align the freshly-stepped x onto the pre-step frame
        # (Boltz trajectory-frame trick; see PLAN D2).
        x_prev = x.clone() if kabsch_interp else None

        # Predict x0 using continuous sigma conditioning (with self-conditioning)
        if is_onestep:
            out = model.forward_sigma_with_trunk(
                x, trunk_tokens, sigma_batch, mask,
                x0_prev=x0_prev if self_cond else None,
                res_idx=batch['res_idx'], chain_ids=batch['chain_ids'],
            )
        else:
            out = denoiser.forward_sigma(
                x, batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                sigma_batch, mask, x0_prev=x0_prev if self_cond else None,
                esm_embed=batch.get('esm_embed'),
                **_template_kwargs(batch),
            )
        x0_pred = out[0] if is_onestep else out
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)

        # Kabsch-align x0_pred to current x
        if align_per_step:
            x0_pred = kabsch_align_to_target(x0_pred, x, mask)

        # Store for self-conditioning in next iteration. Note: we intentionally
        # store x0_pred in the PRE-kabsch_interp frame — self-cond is about
        # giving the denoiser a structural hint, not a frame-consistency one.
        x0_prev = x0_pred.detach()

        # Euler step: x_next = x + (sigma_next - sigma) * (x - x0_pred) / sigma
        # This is the score-based update: dx/dt = -sigma * score, where score = (x - x0_pred) / sigma^2
        d = (x - x0_pred) / sigma  # Direction toward x0
        dt = sigma_next - sigma    # Negative (decreasing sigma)
        x = x + d * dt

        # Boltz Kabsch-interpolation: rigid-align the freshly-stepped x onto
        # the pre-step x_prev so consecutive denoiser inputs share a frame.
        # Applied BEFORE the optional recenter (which becomes a no-op anyway
        # since R, t already match x_prev's centroid). See PLAN D2/D3.
        if kabsch_interp:
            _, _, x = kabsch_rigid(x, x_prev, mask)

        # Re-center
        if recenter:
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                n_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                centroid = (x * mask_exp).sum(dim=1, keepdim=True) / n_valid
            else:
                centroid = x.mean(dim=1, keepdim=True)
            x = x - centroid

    if is_onestep:
        # One extra forward at sigma_min to read the atom-head output.
        # NOTE: `x` here is the terminal Euler-step value, already kabsch-
        # aligned if kabsch_interp=True. The atom head emits per-residue
        # [L, 4, 3] offsets in this centroid frame, so atoms automatically
        # follow the aligned centroids — no extra Kabsch on atoms needed.
        # See PLAN D5.
        sigma_min_batch = sigmas[-1].expand(B)
        # Loop 06: ResFoldOneStep.forward_sigma is now a 3-tuple
        # (centroid, atoms, pred_lddt_or_None). Reuse the cached trunk.
        sigma_out = model.forward_sigma_with_trunk(
            x, trunk_tokens, sigma_min_batch, mask,
            x0_prev=x0_prev if self_cond else None,
            res_idx=batch['res_idx'], chain_ids=batch['chain_ids'],
        )
        atoms_pred = sigma_out[1]
        return x, atoms_pred

    return x


@torch.no_grad()
def sample_k_centroids(
    model,
    batch,
    noiser,
    device,
    K: int,
    base_seed: int,
    target_idx: int,
    is_onestep: bool,
    one_shot: bool,
    align_per_step: bool = False,
    recenter: bool = False,
    kabsch_interp: bool = False,
    self_cond: bool = True,
):
    """Draw K reproducible centroid samples for one target.

    For the (is_onestep, one_shot) path the trunk is run ONCE and only the
    denoiser runs K times via ``forward_sigma_with_trunk``. For the VE path
    the trunk runs K times — for L<=300, c_token=128 and a 4-block trunk
    this cost is negligible vs the K * (T-1) denoiser passes.

    The per-sample noise generator is seeded
    ``base_seed * 100003 + target_idx * 1009 + sample_idx`` so the same
    ``--seed`` reproduces the run bit-for-bit, and the K seeds within a
    target are independent across targets.

    Args:
        model: ResFoldOneStep or ResFoldPipeline.
        batch: collated batch dict (B=1 expected).
        noiser: VENoiser (continuous-sigma path).
        device: torch device.
        K: number of samples to draw.
        base_seed: typically ``args.seed``.
        target_idx: integer index for per-target seeding.
        is_onestep: True for ResFoldOneStep.
        one_shot: True for single-forward EDM inference (fast path).
        align_per_step: forwarded to the VE sampler.
        recenter: forwarded to the VE sampler.
        kabsch_interp: forwarded to the VE sampler (Boltz trajectory-frame
            alignment). Ignored on the one_shot fast path.
        self_cond: forwarded to the VE sampler.

    Returns:
        tuple ``(centroids, atoms_or_None, pred_lddts_or_None)`` where
            centroids:        ``[K, B, L, 3]``
            atoms_or_None:    ``[K, B, L, 4, 3]`` when ``is_onestep`` else ``None``.
            pred_lddts_or_None: ``[K, B]`` predicted lDDT per sample when the
                OneStep model has a confidence head, else ``None``. The
                non-onestep pipeline always returns ``None`` for this slot.
    """
    B, L = batch['aa_seq'].shape
    assert B == 1, "sample_k_centroids assumes one target per call"
    mask = batch['mask_res']

    centroid_list = []
    atom_list = []
    # Loop 06: per-sample predicted lDDT (only populated when the OneStep
    # model has a confidence head). For the slow VE path on the onestep model
    # we rerun forward_sigma at sigma_min (done inside sample_centroids_ve)
    # which loses the lddt scalar, so we only collect pred_lddt on the
    # one_shot fast path. The VE path falls back to None (the cluster ranker
    # remains active in that case via _run_test_eval).
    has_conf_head = (
        is_onestep
        and getattr(model, "confidence_head", None) is not None
    )
    pred_lddt_list: list = []

    # Fast path: reuse the trunk across K denoiser calls.
    if is_onestep and one_shot:
        trunk_tokens = model.get_trunk_tokens(
            batch['aa_seq'], batch['chain_ids'], batch['res_idx'], mask,
            esm_embed=batch.get('esm_embed'),
            **_template_kwargs(batch),
        )
        sigmas = noiser.sigmas.to(device)
        sigma_init = sigmas[0].view(1).expand(B)
        is_ad = getattr(model, "atom_diffusion", False)
        for i in range(K):
            seed_i = base_seed * 100003 + target_idx * 1009 + i
            gen = torch.Generator(device=device).manual_seed(seed_i)
            x = sigma_init.view(B, 1, 1) * torch.randn(
                B, L, 3, device=device, generator=gen
            )
            if is_ad:
                # Atom-diffusion models have an untrained regression atom_head;
                # their atoms MUST come from the iterative atom-diffusion sampler
                # conditioned on the denoiser tokens (reuse the trunk pass).
                centroid_pred, tokens, pred_lddt = model.centroid_tokens_with_trunk(
                    x, trunk_tokens, sigma_init, mask, x0_prev=None,
                    res_idx=batch['res_idx'], chain_ids=batch['chain_ids'],
                )
                atoms_pred = sample_atoms_diffusion(
                    model, tokens, centroid_pred, mask,
                    n_steps=getattr(model, "_atom_eval_steps", 8),
                    sigma_min=model.atom_sigma_min, sigma_max=model.atom_sigma_max,
                    generator=gen,
                )
            else:
                centroid_pred, atoms_pred, pred_lddt = model.forward_sigma_with_trunk(
                    x, trunk_tokens, sigma_init, mask, x0_prev=None,
                    res_idx=batch['res_idx'], chain_ids=batch['chain_ids'],
                )
            centroid_list.append(centroid_pred)
            atom_list.append(atoms_pred)
            if has_conf_head and pred_lddt is not None:
                pred_lddt_list.append(pred_lddt)
    else:
        for i in range(K):
            seed_i = base_seed * 100003 + target_idx * 1009 + i
            gen = torch.Generator(device=device).manual_seed(seed_i)
            if one_shot:
                # Non-onestep one_shot: trunk lives inside model.stage1; just rerun it.
                out = sample_centroids_one_shot(
                    model, batch, noiser, device, is_onestep=is_onestep, generator=gen,
                )
            else:
                out = sample_centroids_ve(
                    model, batch, noiser, device,
                    align_per_step=align_per_step, recenter=recenter,
                    kabsch_interp=kabsch_interp,
                    self_cond=self_cond, is_onestep=is_onestep, generator=gen,
                )
            if is_onestep:
                centroid_pred, atoms_pred = out
                centroid_list.append(centroid_pred)
                atom_list.append(atoms_pred)
                if has_conf_head:
                    # Recover pred_lddt with a single extra forward at the
                    # final centroid. The VE sampler already ran the denoiser
                    # T times, so this is cheap relative to the trajectory.
                    sigma_min_batch = noiser.sigmas.to(device)[-1].expand(B)
                    _, _, pred_lddt = model.forward_sigma(
                        centroid_pred,
                        batch['aa_seq'], batch['chain_ids'], batch['res_idx'],
                        sigma_min_batch, mask, x0_prev=None,
                        esm_embed=batch.get('esm_embed'),
                        **_template_kwargs(batch),
                    )
                    if pred_lddt is not None:
                        pred_lddt_list.append(pred_lddt)
            else:
                centroid_list.append(out)

    centroids = torch.stack(centroid_list, dim=0)  # [K, B, L, 3]
    atoms = torch.stack(atom_list, dim=0) if is_onestep else None
    pred_lddts = torch.stack(pred_lddt_list, dim=0) if pred_lddt_list else None
    return centroids, atoms, pred_lddts


@torch.no_grad()
def sample_centroids_with_sampler(model, batch, noiser, device, sampler):
    """Sample centroids using a sampler from the registry.

    Args:
        model: ResFoldPipeline model
        batch: Batch dict with aa_seq, chain_ids, res_idx, mask_res
        noiser: Diffusion noiser with schedule
        device: torch device
        sampler: Sampler instance from create_sampler()

    Returns:
        centroids: [B, L, 3] predicted centroids
    """
    B, L = batch['aa_seq'].shape

    # Create model kwargs for sampler
    model_kwargs = {
        'aa_seq': batch['aa_seq'],
        'chain_ids': batch['chain_ids'],
        'res_idx': batch['res_idx'],
        'mask_res': batch['mask_res'],
    }
    if 'esm_embed' in batch:
        model_kwargs['esm_embed'] = batch['esm_embed']

    # Custom forward function that adapts ResFold interface to sampler interface
    def forward_fn(mdl, x, t, aa_seq, chain_ids, res_idx, mask_res, esm_embed=None, **kwargs):
        return mdl.forward_stage1(x, aa_seq, chain_ids, res_idx, t, mask_res, esm_embed=esm_embed)

    # Run sampler
    return sampler.sample(
        model=model,
        shape=(B, L, 3),
        model_kwargs=model_kwargs,
        noiser=noiser,
        device=device,
        forward_fn=forward_fn,
    )
