"""Unit tests for the loop-01 additions to ``append_registry_row``.

The function gained three optional kwargs (``dockq_avg``, ``dockq_success_pct``,
``c_rmsd``). When supplied they are appended to the existing Outcome cell as
``"... ; C-RMSD X.XXXX A; DockQ X.XXX succ Y.Y%"`` so old rows stay readable
and grep-friendly. When all three are ``None`` the row format is unchanged
(back-compat with the 36 historical rows). The ``;`` separator avoids the
bare ``|`` that would otherwise be parsed as a new markdown column.
"""

import pytest

from tinyfold.training.registry_append import append_registry_row


REGISTRY_HEADER = (
    "# Experiment Registry\n"
    "\n"
    "Test fixture.\n"
    "\n"
    "| Date | Run | Model | What was tried | Outcome | Why stopped | Files |\n"
    "|------|-----|-------|----------------|---------|-------------|-------|\n"
)


@pytest.fixture
def tmp_registry(tmp_path):
    """Pre-seed a temporary REGISTRY.md with the production header."""
    path = tmp_path / "REGISTRY.md"
    path.write_text(REGISTRY_HEADER, encoding="utf-8")
    return path


def test_appends_dockq_and_c_rmsd_tokens(tmp_registry):
    """All three kwargs present -> tokens appear in the Outcome cell."""
    append_registry_row(
        run_name="loop01_dockq_test",
        model="resfold",
        config_path="configs/train/resfold/phase_c_n8600.yaml",
        final_metric=9.8693,
        outcome="converged",
        registry_path=tmp_registry,
        dockq_avg=0.42,
        dockq_success_pct=35.0,
        c_rmsd=8.7,
    )

    last_line = tmp_registry.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    # Outcome cell wording fixed by registry_append: "test RMSE X A — outcome".
    assert "test RMSE 9.8693 A" in last_line
    # Loop-01 metric tokens (formatting is locked: 4 decimals A for C-RMSD,
    # 3 decimals for DockQ + 1 decimal for the success rate).
    assert "C-RMSD 8.7000 A" in last_line
    assert "DockQ 0.420 succ 35.0%" in last_line


def test_appends_dockq_without_success_rate(tmp_registry):
    """``dockq_avg`` without ``dockq_success_pct`` -> no ``succ`` token."""
    append_registry_row(
        run_name="loop01_dockq_only",
        model="resfold",
        config_path=None,
        final_metric=12.3,
        outcome="converged",
        registry_path=tmp_registry,
        dockq_avg=0.15,
    )

    last_line = tmp_registry.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert "DockQ 0.150" in last_line
    # The full success-rate token uses the format "succ Y.Y%"; absent here.
    assert " succ " not in last_line
    assert "%" not in last_line


def test_back_compat_when_none(tmp_registry):
    """All loop-01 kwargs left at ``None`` -> row is identical to today's format."""
    append_registry_row(
        run_name="legacy_call",
        model="resfold",
        config_path=None,
        final_metric=10.0,
        outcome="converged",
        registry_path=tmp_registry,
    )

    last_line = tmp_registry.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert "C-RMSD" not in last_line
    assert "DockQ" not in last_line
    # The pre-existing format must still be intact.
    assert "test RMSE 10.0000 A" in last_line
    assert "converged" in last_line
