"""Unit tests for summarize_eval_metrics (extracted from train_resfold eval).

Locks the aggregation + log-line formatting behavior so the L28 extraction
stays byte-identical to the inlined version.
"""

from tinyfold.training.eval import summarize_eval_metrics


def _base_kwargs(**over):
    kw = dict(
        mode="stage1_only",
        k_list=[1],
        test_rmses=[1.0, 3.0],
        test_dockq_scores=[],
        test_lddt_scores=[],
        test_ilddt_scores=[],
        test_atom_rmses=[],
        test_c_rmsds=[],
        per_k_oracle={1: []},
        per_k_mean={1: []},
        per_k_ranked={1: []},
        per_k_ranked_conf={1: []},
        per_k_ranked_consistency={1: []},
        per_k_ranked_energy={1: []},
        all_pred_lddts=[],
        all_neg_rmses=[],
        pred_lddt_std_per_target=[],
        n_test=2,
    )
    kw.update(over)
    return kw


def test_basic_test_avg_and_eval_only_prefix():
    test_avg, dockq_avg, succ, c_rmsd, tokens, msg = summarize_eval_metrics(
        **_base_kwargs()
    )
    assert test_avg == 2.0
    assert dockq_avg is None and succ is None and c_rmsd is None
    assert tokens == []
    # --eval_only prefix (no train numbers).
    assert "Test Centroid RMSE (2): 2.0000 A" in msg
    assert "Train" not in msg


def test_train_prefix_when_train_avg_present():
    _, _, _, _, _, msg = summarize_eval_metrics(
        **_base_kwargs(train_avg=1.5, n_eval=10)
    )
    assert "Train Centroid RMSE (10): 1.5000 A" in msg
    assert "Test Centroid RMSE (2): 2.0000 A" in msg


def test_atom_rmse_metric_name_for_non_stage1():
    _, _, _, _, _, msg = summarize_eval_metrics(**_base_kwargs(mode="end_to_end"))
    assert "Atom RMSE" in msg
    assert "Centroid RMSE" not in msg


def test_dockq_avg_and_success_threshold():
    # 0.5 >= 0.23 (success), 0.1 < 0.23 (fail) -> avg 0.3, 50% success.
    test_avg, dockq_avg, succ, *_ , msg = summarize_eval_metrics(
        **_base_kwargs(test_dockq_scores=[0.5, 0.1])
    )
    assert abs(dockq_avg - 0.3) < 1e-9
    assert abs(succ - 50.0) < 1e-9
    assert "DockQ: 0.3000 (succ 50.0%)" in msg


def test_c_rmsd_returned_and_logged():
    _, _, _, c_rmsd, _, msg = summarize_eval_metrics(
        **_base_kwargs(test_c_rmsds=[2.0, 4.0])
    )
    assert c_rmsd == 3.0
    assert "C-RMSD: 3.0000 A" in msg


def test_multisample_extra_tokens_skip_k1():
    kw = _base_kwargs(
        k_list=[1, 2],
        per_k_oracle={1: [], 2: [1.0, 2.0]},
        per_k_mean={1: [], 2: [2.0, 4.0]},
        per_k_ranked={1: [], 2: [1.5, 2.5]},
        per_k_ranked_conf={1: [], 2: []},
        per_k_ranked_consistency={1: [], 2: []},
        per_k_ranked_energy={1: [], 2: []},
    )
    _, _, _, _, tokens, msg = summarize_eval_metrics(**kw)
    # K=1 is skipped; K=2 emits oracle/mean/ranked.
    assert "oracle@2 1.500 A" in tokens
    assert "mean@2 3.000 A" in tokens
    assert "ranked@2 2.000 A" in tokens
    assert "oracle@2: 1.5000 A" in msg
