"""Evaluation helpers shared by training/eval/plotting.

Extracted from train_resfold.py, where the continuous-sigma sampler dispatch was
copy-pasted three times (test-eval, plotting-stage1, plotting).
"""

import torch

from tinyfold.inference.samplers import sample_centroids_one_shot, sample_centroids_ve


def summarize_eval_metrics(
    *,
    mode: str,
    k_list: list,
    test_rmses: list,
    test_dockq_scores: list,
    test_lddt_scores: list,
    test_ilddt_scores: list,
    test_atom_rmses: list,
    test_c_rmsds: list,
    per_k_oracle: dict,
    per_k_mean: dict,
    per_k_ranked: dict,
    per_k_ranked_conf: dict,
    per_k_ranked_consistency: dict,
    per_k_ranked_energy: dict,
    all_pred_lddts: list,
    all_neg_rmses: list,
    pred_lddt_std_per_target: list,
    n_test: int,
    train_avg=None,
    n_eval=None,
):
    """Aggregate the per-target eval lists into summary metrics + a log line.

    Pure w.r.t. I/O: it reads only the collected Python lists/dicts and a few
    scalars, and returns the numbers plus the pre-formatted log string. The
    caller is responsible for emitting ``log_msg`` (kept out of here so this is
    unit-testable without a logger).

    Returns:
        tuple ``(test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg,
        extra_tokens, log_msg)``. ``dockq_avg`` / ``dockq_success_pct`` /
        ``c_rmsd_avg`` may be ``None`` when the metric was not collected.
    """
    test_avg = sum(test_rmses) / len(test_rmses)

    metric_name = "Centroid RMSE" if mode == "stage1_only" else "Atom RMSE"
    if train_avg is not None and n_eval is not None:
        log_msg = (
            f"         >>> Train {metric_name} ({n_eval}): {train_avg:.4f} A "
            f"| Test {metric_name} ({n_test}): {test_avg:.4f} A"
        )
    else:
        # --eval_only: no train-side numbers to print.
        log_msg = f"         >>> Test {metric_name} ({n_test}): {test_avg:.4f} A"
    dockq_avg = None
    dockq_success_pct = None
    c_rmsd_avg = None
    if test_dockq_scores:
        dockq_avg = sum(test_dockq_scores) / len(test_dockq_scores)
        dockq_success_pct = 100.0 * sum(
            1 for d in test_dockq_scores if d >= 0.23
        ) / len(test_dockq_scores)
        log_msg += f" | DockQ: {dockq_avg:.4f} (succ {dockq_success_pct:.1f}%)"
    if test_lddt_scores:
        lddt_avg = sum(test_lddt_scores) / len(test_lddt_scores)
        log_msg += f" | lDDT: {lddt_avg:.4f}"
    if test_ilddt_scores:
        ilddt_avg = sum(test_ilddt_scores) / len(test_ilddt_scores)
        log_msg += f" | ilDDT: {ilddt_avg:.4f}"
    if test_atom_rmses:
        atom_avg = sum(test_atom_rmses) / len(test_atom_rmses)
        log_msg += f" | Atom RMSE: {atom_avg:.4f}"
    if test_c_rmsds:
        c_rmsd_avg = sum(test_c_rmsds) / len(test_c_rmsds)
        log_msg += f" | C-RMSD: {c_rmsd_avg:.4f} A"
    # Multi-sample (Loop 02) tokens: K=1 oracle/mean/ranked all equal the
    # printed test RMSE, so skip K=1 to keep the cell tidy.
    extra_tokens: list = []
    for k in k_list:
        if k == 1:
            continue
        o = sum(per_k_oracle[k]) / len(per_k_oracle[k])
        m = sum(per_k_mean[k]) / len(per_k_mean[k])
        r = sum(per_k_ranked[k]) / len(per_k_ranked[k])
        log_msg += f" | oracle@{k}: {o:.4f} A | mean@{k}: {m:.4f} A | ranked@{k}: {r:.4f} A"
        extra_tokens.append(f"oracle@{k} {o:.3f} A")
        extra_tokens.append(f"mean@{k} {m:.3f} A")
        extra_tokens.append(f"ranked@{k} {r:.3f} A")
        # Loop 06: confidence-ranked sample (only when the head fired).
        if per_k_ranked_conf[k]:
            rc = sum(per_k_ranked_conf[k]) / len(per_k_ranked_conf[k])
            log_msg += f" | ranked_conf@{k}: {rc:.4f} A"
            extra_tokens.append(f"ranked_conf@{k} {rc:.3f} A")
        # Energy-style rankers: self-consistency and geometric clash/contact.
        if per_k_ranked_consistency[k]:
            rcs = sum(per_k_ranked_consistency[k]) / len(per_k_ranked_consistency[k])
            log_msg += f" | ranked_consistency@{k}: {rcs:.4f} A"
            extra_tokens.append(f"ranked_consistency@{k} {rcs:.3f} A")
        if per_k_ranked_energy[k]:
            re = sum(per_k_ranked_energy[k]) / len(per_k_ranked_energy[k])
            log_msg += f" | ranked_energy@{k}: {re:.4f} A"
            extra_tokens.append(f"ranked_energy@{k} {re:.3f} A")
    # Loop 06: Spearman(pred_lddt, -RMSE) across all (target, sample) pairs.
    # Falls back to torch.corrcoef Pearson if scipy isn't installed.
    if all_pred_lddts:
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(all_pred_lddts, all_neg_rmses)
            log_msg += f" | Spearman(pred_lddt,-RMSE): {rho:.3f}"
            extra_tokens.append(f"spearman {rho:.3f}")
        except Exception:
            a = torch.tensor(all_pred_lddts, dtype=torch.float32)
            b = torch.tensor(all_neg_rmses, dtype=torch.float32)
            stacked = torch.stack([a, b], dim=0)
            if a.numel() > 1 and a.std() > 0 and b.std() > 0:
                rho = torch.corrcoef(stacked)[0, 1].item()
                log_msg += f" | Pearson(pred_lddt,-RMSE): {rho:.3f}"
                extra_tokens.append(f"pearson {rho:.3f}")
        # Mean of per-target stddev: a value near 0 means the head's score
        # is invariant to which sample it sees (collapse-to-mean indicator).
        if pred_lddt_std_per_target:
            std_mean = sum(pred_lddt_std_per_target) / len(pred_lddt_std_per_target)
            log_msg += f" | pred_lddt_std(per-target): {std_mean:.4f}"
            extra_tokens.append(f"pred_lddt_std {std_mean:.4f}")

    return test_avg, dockq_avg, dockq_success_pct, c_rmsd_avg, extra_tokens, log_msg


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
