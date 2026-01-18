"""Contact map and matrix plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_contact_map(
    pred_contacts: np.ndarray,
    ref_contacts: np.ndarray,
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot predicted vs reference contact maps.

    Args:
        pred_contacts: [LA, LB] predicted contact matrix
        ref_contacts: [LA, LB] reference contact matrix
        out_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Predicted contacts
    axes[0].imshow(pred_contacts.T, cmap="Blues", aspect="auto", origin="lower")
    axes[0].set_xlabel("Chain A residue")
    axes[0].set_ylabel("Chain B residue")
    axes[0].set_title(f"Predicted ({pred_contacts.sum()} contacts)")

    # Reference contacts
    axes[1].imshow(ref_contacts.T, cmap="Blues", aspect="auto", origin="lower")
    axes[1].set_xlabel("Chain A residue")
    axes[1].set_ylabel("Chain B residue")
    axes[1].set_title(f"Reference ({ref_contacts.sum()} contacts)")

    # Difference map
    # Green = true positive, Red = false positive, Yellow = false negative
    diff_map = np.zeros((*pred_contacts.shape, 3))
    tp = pred_contacts & ref_contacts
    fp = pred_contacts & ~ref_contacts
    fn = ~pred_contacts & ref_contacts

    diff_map[tp] = [0.2, 0.8, 0.2]  # Green
    diff_map[fp] = [0.9, 0.2, 0.2]  # Red
    diff_map[fn] = [0.9, 0.9, 0.2]  # Yellow

    axes[2].imshow(np.transpose(diff_map, (1, 0, 2)), aspect="auto", origin="lower")
    axes[2].set_xlabel("Chain A residue")
    axes[2].set_ylabel("Chain B residue")
    axes[2].set_title("Difference (G=TP, R=FP, Y=FN)")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distance_map(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    atom_type: np.ndarray,
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot CA distance maps for predicted and reference.

    Args:
        pred_xyz: [N_atom, 3] predicted coordinates
        ref_xyz: [N_atom, 3] reference coordinates
        atom_type: [N_atom] atom types (1 = CA)
        out_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Extract CA atoms
    ca_mask = atom_type == 1
    pred_ca = pred_xyz[ca_mask]
    ref_ca = ref_xyz[ca_mask]

    # Compute distance matrices
    pred_dist = np.sqrt(((pred_ca[:, None] - pred_ca[None, :]) ** 2).sum(-1))
    ref_dist = np.sqrt(((ref_ca[:, None] - ref_ca[None, :]) ** 2).sum(-1))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im1 = axes[0].imshow(pred_dist, cmap="viridis", vmin=0, vmax=30)
    axes[0].set_title("Predicted CA distances")
    axes[0].set_xlabel("Residue")
    axes[0].set_ylabel("Residue")
    plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

    im2 = axes[1].imshow(ref_dist, cmap="viridis", vmin=0, vmax=30)
    axes[1].set_title("Reference CA distances")
    axes[1].set_xlabel("Residue")
    axes[1].set_ylabel("Residue")
    plt.colorbar(im2, ax=axes[1], label="Distance (Å)")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
