"""Append one markdown row to ``experiments/REGISTRY.md`` per training run.

This helper is called from the ``finally:`` block of training scripts so that
every run — converged, crashed, or killed by the user — leaves a single row
behind. It is append-only and assumes a single writer per process (no file
locking).
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def _escape_pipes(value: str) -> str:
    """Escape literal pipe characters so they do not break the markdown row."""
    return value.replace("|", "\\|")


def append_registry_row(
    run_name: str,
    model: str,
    config_path: Optional[str],
    final_metric: Optional[float],
    outcome: str,
    registry_path: Optional[Path] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Append one row to ``experiments/REGISTRY.md`` and return the resolved path.

    Columns (must match the existing header):
        | Date | Run | Model | What was tried | Outcome | Why stopped | Files |

    Args:
        run_name: Timestamped run name (used for the Run column and the Files link).
        model: Short model identifier (e.g. ``"resfold"`` or ``"resfold_stage2"``).
        config_path: Path to the YAML config profile (if any).
        final_metric: Best test RMSE in Angstroms, or ``None`` if no metric was reached.
        outcome: Short outcome string (``"converged"``, ``"killed by user"``,
            ``"crashed: <ExcName>"``). Also written verbatim into the
            ``Why stopped`` column.
        registry_path: Override path to the registry. Defaults to
            ``<repo_root>/experiments/REGISTRY.md`` where ``repo_root`` is
            resolved as ``Path(__file__).resolve().parents[3]``.

    Returns:
        Resolved ``Path`` to the registry file that was written to.

    Raises:
        FileNotFoundError: if ``registry_path`` does not exist. We never create
            the registry from scratch here — Loop 03 owns its creation.
    """
    if registry_path is None:
        registry_path = Path(__file__).resolve().parents[3] / "experiments" / "REGISTRY.md"
    else:
        registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry not found at {registry_path}; Loop 03 should have created it."
        )

    date_str = datetime.now().strftime("%Y-%m-%d")
    run_str = _escape_pipes(run_name)
    model_str = _escape_pipes(model)
    what_tried = (
        f"config: {_escape_pipes(config_path)}" if config_path else "ad-hoc CLI args"
    )
    outcome_clean = _escape_pipes(outcome)
    if final_metric is not None:
        outcome_cell = f"test RMSE {final_metric:.4f} A — {outcome_clean}"
    else:
        outcome_cell = outcome_clean
    if output_dir is not None:
        rel = Path(output_dir).as_posix().lstrip("./")
        if rel.startswith("outputs/"):
            files_cell = f"[{rel}/](../{rel}/)"
        else:
            files_cell = f"[{rel}/]({rel}/)"
    else:
        files_cell = f"[outputs/{run_str}/](../outputs/{run_str}/)"

    row = (
        f"| {date_str} | {run_str} | {model_str} | {what_tried} "
        f"| {outcome_cell} | {outcome_clean} | {files_cell} |\n"
    )

    # Ensure the existing file ends with a newline before we append.
    needs_leading_newline = False
    if registry_path.stat().st_size > 0:
        with open(registry_path, "rb") as f:
            f.seek(-1, 2)
            last_byte = f.read(1)
        if last_byte not in (b"\n", b"\r"):
            needs_leading_newline = True

    with open(registry_path, "a", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        f.write(row)

    return registry_path
