"""Deterministic run naming for TinyFold training runs.

Generates descriptive, timestamped run names that capture key configuration
parameters for easy identification and organization.

Format: {model}_{mode}_{n_train}_{timestamp}

Examples:
    - resfold_s1_20K_20260122_220043
    - af3_std_5K_20260122_140512
    - attention_v2_gaussian_1K_20260122_093015
"""

from datetime import datetime
from typing import Dict, Any, Optional


def _format_count(n: int) -> str:
    """Format sample count for display.

    Args:
        n: Sample count

    Returns:
        Formatted string (e.g., "1K", "20K", "100")
    """
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


def _extract_key_params(model_name: str, config: Dict[str, Any]) -> str:
    """Extract key parameters from config for run name.

    Args:
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        Abbreviated parameter string
    """
    parts = []

    # Mode (for ResFold)
    if "mode" in config:
        mode = config["mode"]
        mode_abbrev = {
            "stage1_only": "s1",
            "stage2_only": "s2",
            "end_to_end": "e2e",
        }
        parts.append(mode_abbrev.get(mode, mode[:3]))

    # Noise type (for AF3/attention models)
    elif "noise_type" in config:
        noise = config["noise_type"]
        noise_abbrev = {
            "gaussian": "gau",
            "linear_chain": "lc",
            "linear_flow": "lf",
        }
        parts.append(noise_abbrev.get(noise, noise[:3]))

    # Sample count
    if "n_train" in config:
        parts.append(_format_count(config["n_train"]))

    return "_".join(parts) if parts else "default"


def generate_run_name(
    model_name: str,
    config: Dict[str, Any],
    timestamp: bool = True,
    custom_suffix: Optional[str] = None,
) -> str:
    """Generate deterministic, descriptive run name.

    Args:
        model_name: Name of the model (e.g., "resfold", "af3", "attention_v2")
        config: Configuration dictionary
        timestamp: Whether to include timestamp (default True)
        custom_suffix: Optional custom suffix to append

    Returns:
        Run name string

    Examples:
        >>> generate_run_name("resfold", {"mode": "stage1_only", "n_train": 20000})
        "resfold_s1_20K_20260122_220043"

        >>> generate_run_name("af3", {"noise_type": "gaussian", "n_train": 5000})
        "af3_gau_5K_20260122_140512"
    """
    params = _extract_key_params(model_name, config)

    parts = [model_name, params]

    if custom_suffix:
        parts.append(custom_suffix)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)

    return "_".join(parts)


def parse_run_name(run_name: str) -> Dict[str, str]:
    """Parse run name to extract components.

    Args:
        run_name: Run name string

    Returns:
        Dict with parsed components
    """
    parts = run_name.split("_")

    result = {"model": parts[0] if parts else "unknown"}

    # Try to find timestamp (format: YYYYMMDD_HHMMSS)
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            # Found date part, next should be time
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                result["timestamp"] = f"{part}_{parts[i + 1]}"
                result["params"] = "_".join(parts[1:i])
                break
    else:
        result["params"] = "_".join(parts[1:])

    return result
