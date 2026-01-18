"""Consistent styling for visualizations."""

# Chain colors (py3Dmol hex format)
CHAIN_COLORS = {
    "A": "#3B82F6",  # Blue
    "B": "#F97316",  # Orange
    "ref": "#9CA3AF",  # Gray for reference
}

# Style presets for py3Dmol
STYLE_PRESETS = {
    "cartoon": {
        "cartoon": {"color": "spectrum"}
    },
    "cartoon_chain_colored": lambda chain: {
        "cartoon": {"color": CHAIN_COLORS.get(chain, "#888888")}
    },
    "stick": {
        "stick": {"radius": 0.15}
    },
    "sphere_ca": {
        "sphere": {"radius": 0.8}
    },
    "transparent": {
        "cartoon": {"opacity": 0.5, "color": CHAIN_COLORS["ref"]}
    },
}


def get_chain_style(chain: str, style_type: str = "cartoon") -> dict:
    """Get style dict for a chain.

    Args:
        chain: Chain label (A, B, or ref)
        style_type: One of cartoon, stick, sphere_ca

    Returns:
        py3Dmol style dict
    """
    color = CHAIN_COLORS.get(chain, "#888888")

    if style_type == "cartoon":
        return {"cartoon": {"color": color}}
    elif style_type == "stick":
        return {"stick": {"radius": 0.15, "color": color}}
    elif style_type == "sphere_ca":
        return {"sphere": {"radius": 0.8, "color": color}}
    elif style_type == "transparent":
        return {"cartoon": {"opacity": 0.4, "color": color}}
    else:
        return {"cartoon": {"color": color}}


def get_highlight_style(highlight_type: str = "interface") -> dict:
    """Get style for highlighting residues.

    Args:
        highlight_type: One of interface, clash, error

    Returns:
        py3Dmol style dict
    """
    if highlight_type == "interface":
        return {"sphere": {"radius": 1.0, "color": "#22C55E"}}  # Green
    elif highlight_type == "clash":
        return {"sphere": {"radius": 1.2, "color": "#EF4444"}}  # Red
    elif highlight_type == "error":
        return {"sphere": {"radius": 1.0, "color": "#EAB308"}}  # Yellow
    else:
        return {"sphere": {"radius": 1.0, "color": "#888888"}}
