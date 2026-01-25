#!/usr/bin/env python
"""
Generate architecture diagrams for TinyFold/ResFold.

Creates publication-quality architecture visualizations using matplotlib.
Inspired by "The Illustrated AlphaFold" visual style.

Usage:
    python scripts/plot_architecture.py
    python scripts/plot_architecture.py --output docs/images/architecture.png
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

# =============================================================================
# Color Scheme
# =============================================================================

COLORS = {
    'input': '#FFF3E0',       # Light orange
    'input_border': '#FF9800',
    'stage1': '#E3F2FD',      # Light blue
    'stage1_border': '#2196F3',
    'stage1_dark': '#BBDEFB',
    'stage2': '#E8F5E9',      # Light green
    'stage2_border': '#4CAF50',
    'stage2_dark': '#C8E6C9',
    'output': '#F3E5F5',      # Light purple
    'output_border': '#9C27B0',
    'module': '#FAFAFA',      # Near white
    'module_border': '#9E9E9E',
    'arrow': '#424242',       # Dark gray
    'text': '#212121',        # Near black
    'text_light': '#757575',  # Gray
    'diffusion': '#E1F5FE',   # Cyan tint for diffusion loop
}

# =============================================================================
# Drawing Primitives
# =============================================================================

def draw_rounded_box(ax, x, y, width, height, label, sublabel=None,
                     facecolor='white', edgecolor='gray', linewidth=1.5,
                     fontsize=11, sublabel_fontsize=9, pad=0.02):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes
    )
    ax.add_patch(box)

    # Main label
    label_y = y + height/2 + (0.01 if sublabel else 0)
    ax.text(x + width/2, label_y, label,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=COLORS['text'],
            transform=ax.transAxes)

    # Sublabel (smaller, below main label)
    if sublabel:
        ax.text(x + width/2, y + height/2 - 0.015, sublabel,
                ha='center', va='center',
                fontsize=sublabel_fontsize,
                color=COLORS['text_light'],
                transform=ax.transAxes)

    return box


def draw_arrow(ax, start, end, color=None, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=12,
        color=color,
        linewidth=1.5,
        connectionstyle=connectionstyle,
        transform=ax.transAxes
    )
    ax.add_patch(arrow)
    return arrow


def draw_module(ax, x, y, width, height, title, details,
                facecolor=None, edgecolor=None):
    """Draw a module box with title and detail lines."""
    if facecolor is None:
        facecolor = COLORS['module']
    if edgecolor is None:
        edgecolor = COLORS['module_border']

    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(box)

    # Title
    ax.text(x + width/2, y + height - 0.02, title,
            ha='center', va='top',
            fontsize=10, fontweight='bold',
            color=COLORS['text'],
            transform=ax.transAxes)

    # Details (smaller text)
    detail_y = y + height - 0.045
    for i, detail in enumerate(details):
        ax.text(x + 0.015, detail_y - i * 0.022, f"• {detail}",
                ha='left', va='top',
                fontsize=8,
                color=COLORS['text_light'],
                transform=ax.transAxes,
                family='monospace')

    return box


# =============================================================================
# Main Architecture Diagram
# =============================================================================

def plot_resfold_architecture(save_path='docs/images/architecture.png', dpi=150):
    """Generate the full ResFold architecture diagram."""

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Layout constants
    box_width = 0.8
    box_x = 0.1
    margin = 0.015

    # ==========================================================================
    # Title
    # ==========================================================================
    ax.text(0.5, 0.97, 'ResFold Architecture',
            ha='center', va='top',
            fontsize=16, fontweight='bold',
            color=COLORS['text'],
            transform=ax.transAxes)
    ax.text(0.5, 0.945, 'Two-Stage Protein Structure Prediction',
            ha='center', va='top',
            fontsize=11,
            color=COLORS['text_light'],
            transform=ax.transAxes)

    # ==========================================================================
    # Input Box
    # ==========================================================================
    input_y = 0.87
    input_h = 0.055

    draw_rounded_box(ax, box_x, input_y, box_width, input_h,
                     "Input",
                     "Sequence [L] • Chain IDs [L] • Noisy Centroids x_t [L, 3]",
                     facecolor=COLORS['input'],
                     edgecolor=COLORS['input_border'])

    # Arrow: Input -> Stage 1
    draw_arrow(ax, (0.5, input_y), (0.5, input_y - 0.02))

    # ==========================================================================
    # Stage 1: Residue Diffusion
    # ==========================================================================
    stage1_y = 0.52
    stage1_h = 0.32

    # Stage 1 outer box
    stage1_box = FancyBboxPatch(
        (box_x, stage1_y), box_width, stage1_h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=COLORS['stage1'],
        edgecolor=COLORS['stage1_border'],
        linewidth=2,
        transform=ax.transAxes
    )
    ax.add_patch(stage1_box)

    # Stage 1 title
    ax.text(box_x + 0.015, stage1_y + stage1_h - 0.015,
            "Stage 1: Residue Diffusion",
            ha='left', va='top',
            fontsize=12, fontweight='bold',
            color=COLORS['stage1_border'],
            transform=ax.transAxes)

    # Trunk module
    trunk_x = box_x + 0.03
    trunk_y = stage1_y + 0.17
    trunk_w = box_width - 0.06
    trunk_h = 0.11

    draw_module(ax, trunk_x, trunk_y, trunk_w, trunk_h,
                "ResidueEncoder (Trunk) — runs once per sample",
                ["AA embedding + Chain embedding + Position encoding",
                 "Coordinate projection: x_t → [L, c]",
                 "Transformer encoder (9 layers) → trunk_tokens [L, 256]"],
                facecolor=COLORS['stage1_dark'],
                edgecolor=COLORS['stage1_border'])

    # Arrow: Trunk -> Denoiser
    draw_arrow(ax, (0.5, trunk_y), (0.5, trunk_y - 0.015))

    # Denoiser module
    denoiser_y = stage1_y + 0.025
    denoiser_h = 0.12

    draw_module(ax, trunk_x, denoiser_y, trunk_w, denoiser_h,
                "DiffusionTransformer (Denoiser) — runs T times",
                ["coord_embed(x_t) + trunk_tokens + time_embed(t)",
                 "Transformer (7 blocks) with Adaptive LayerNorm",
                 "Output: predicted clean centroids x₀ [L, 3]"],
                facecolor=COLORS['diffusion'],
                edgecolor=COLORS['stage1_border'])

    # Diffusion loop indicator
    loop_x = box_x + box_width - 0.025
    ax.annotate('', xy=(loop_x + 0.015, denoiser_y + 0.03),
                xytext=(loop_x + 0.015, denoiser_y + denoiser_h - 0.02),
                arrowprops=dict(arrowstyle='->', color=COLORS['stage1_border'],
                               connectionstyle='arc3,rad=0.3'),
                transform=ax.transAxes)
    ax.text(loop_x + 0.03, denoiser_y + denoiser_h/2, 'T=50\nsteps',
            ha='left', va='center', fontsize=7,
            color=COLORS['stage1_border'],
            transform=ax.transAxes)

    # Arrow: Stage 1 -> Centroids
    draw_arrow(ax, (0.5, stage1_y), (0.5, stage1_y - 0.015))

    # Intermediate output: Centroids
    cent_y = 0.46
    cent_h = 0.045
    draw_rounded_box(ax, box_x + 0.15, cent_y, box_width - 0.3, cent_h,
                     "Predicted Centroids [L, 3]",
                     facecolor='white',
                     edgecolor=COLORS['arrow'],
                     linewidth=1)

    # Arrow: Centroids -> Stage 2
    draw_arrow(ax, (0.5, cent_y), (0.5, cent_y - 0.015))

    # ==========================================================================
    # Stage 2: Atom Refinement
    # ==========================================================================
    stage2_y = 0.18
    stage2_h = 0.26

    # Stage 2 outer box
    stage2_box = FancyBboxPatch(
        (box_x, stage2_y), box_width, stage2_h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=COLORS['stage2'],
        edgecolor=COLORS['stage2_border'],
        linewidth=2,
        transform=ax.transAxes
    )
    ax.add_patch(stage2_box)

    # Stage 2 title
    ax.text(box_x + 0.015, stage2_y + stage2_h - 0.015,
            "Stage 2: Atom Refinement",
            ha='left', va='top',
            fontsize=12, fontweight='bold',
            color=COLORS['stage2_border'],
            transform=ax.transAxes)

    # Global transformer module
    global_y = stage2_y + 0.13
    global_h = 0.095

    draw_module(ax, trunk_x, global_y, trunk_w, global_h,
                "GlobalTransformer — inter-residue context",
                ["Centroid + sequence + position embeddings",
                 "Transformer encoder (6 layers) → tokens [L, 192]"],
                facecolor=COLORS['stage2_dark'],
                edgecolor=COLORS['stage2_border'])

    # Arrow
    draw_arrow(ax, (0.5, global_y), (0.5, global_y - 0.015))

    # Local atom attention module
    local_y = stage2_y + 0.025
    local_h = 0.085

    draw_module(ax, trunk_x, local_y, trunk_w, local_h,
                "LocalAtomAttention — within-residue refinement",
                ["Broadcast tokens → [L, 4, c_atom]",
                 "Local attention (4 atoms per residue) → offsets [L, 4, 3]"],
                facecolor=COLORS['stage2_dark'],
                edgecolor=COLORS['stage2_border'])

    # Arrow: Stage 2 -> Output
    draw_arrow(ax, (0.5, stage2_y), (0.5, stage2_y - 0.015))

    # ==========================================================================
    # Output Box
    # ==========================================================================
    output_y = 0.10
    output_h = 0.06

    draw_rounded_box(ax, box_x, output_y, box_width, output_h,
                     "Output: Backbone Atoms [L, 4, 3]",
                     "N, CA, C, O positions for each residue",
                     facecolor=COLORS['output'],
                     edgecolor=COLORS['output_border'])

    # ==========================================================================
    # Legend / Stats
    # ==========================================================================
    legend_y = 0.03
    ax.text(0.5, legend_y,
            "Stage 1: 13.8M params (49%) • Stage 2: 14.2M params (51%) • Total: 28M params",
            ha='center', va='center',
            fontsize=9,
            color=COLORS['text_light'],
            transform=ax.transAxes)

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Architecture diagram saved to: {save_path}")
    return save_path


# =============================================================================
# Example Prediction Plot
# =============================================================================

def plot_prediction_example(pred_centroids, gt_centroids, chain_ids,
                            save_path='docs/images/example_prediction.png',
                            title="Example Prediction"):
    """
    Create a 3D visualization comparing predicted vs ground truth structure.

    Args:
        pred_centroids: [L, 3] predicted centroid positions
        gt_centroids: [L, 3] ground truth centroid positions
        chain_ids: [L] chain assignments (0 or 1)
        save_path: output file path
        title: plot title
    """
    import numpy as np

    fig = plt.figure(figsize=(12, 5))

    # Convert to numpy if needed
    if hasattr(pred_centroids, 'numpy'):
        pred_centroids = pred_centroids.numpy()
    if hasattr(gt_centroids, 'numpy'):
        gt_centroids = gt_centroids.numpy()
    if hasattr(chain_ids, 'numpy'):
        chain_ids = chain_ids.numpy()

    # Colors for chains
    chain_colors = ['#2196F3', '#FF9800']  # Blue, Orange

    # --- Ground Truth ---
    ax1 = fig.add_subplot(131, projection='3d')
    for c in [0, 1]:
        mask = chain_ids == c
        if mask.sum() > 0:
            coords = gt_centroids[mask]
            ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=chain_colors[c], s=20, alpha=0.8,
                       label=f'Chain {"A" if c == 0 else "B"}')
            # Connect consecutive residues
            ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=chain_colors[c], alpha=0.4, linewidth=1)
    ax1.set_title('Ground Truth', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')

    # --- Prediction ---
    ax2 = fig.add_subplot(132, projection='3d')
    for c in [0, 1]:
        mask = chain_ids == c
        if mask.sum() > 0:
            coords = pred_centroids[mask]
            ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=chain_colors[c], s=20, alpha=0.8,
                       label=f'Chain {"A" if c == 0 else "B"}')
            ax2.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=chain_colors[c], alpha=0.4, linewidth=1)
    ax2.set_title('Prediction', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')

    # --- Overlay ---
    ax3 = fig.add_subplot(133, projection='3d')
    for c in [0, 1]:
        mask = chain_ids == c
        if mask.sum() > 0:
            gt_c = gt_centroids[mask]
            pred_c = pred_centroids[mask]
            # GT: solid
            ax3.scatter(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
                       c=chain_colors[c], s=20, alpha=0.8, marker='o')
            ax3.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
                    c=chain_colors[c], alpha=0.5, linewidth=1.5)
            # Pred: hollow
            ax3.scatter(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2],
                       facecolors='none', edgecolors=chain_colors[c],
                       s=30, alpha=0.8, marker='o', linewidth=1)
            ax3.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2],
                    c=chain_colors[c], alpha=0.3, linewidth=1, linestyle='--')

    # Custom legend for overlay
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Ground Truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=8, label='Prediction'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax3.set_title('Overlay', fontsize=11, fontweight='bold')
    ax3.set_xlabel('X (Å)')
    ax3.set_ylabel('Y (Å)')
    ax3.set_zlabel('Z (Å)')

    # Match axis limits across all plots
    all_coords = np.concatenate([gt_centroids, pred_centroids], axis=0)
    max_range = (all_coords.max(axis=0) - all_coords.min(axis=0)).max() / 2
    mid = all_coords.mean(axis=0)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Prediction example saved to: {save_path}")
    return save_path


def generate_dummy_prediction():
    """Generate a dummy prediction for testing the plot."""
    np.random.seed(42)

    # Create a simple two-chain structure
    L_a, L_b = 50, 40
    L = L_a + L_b

    # Chain A: helix-like
    t_a = np.linspace(0, 4*np.pi, L_a)
    gt_a = np.stack([
        5 * np.cos(t_a),
        5 * np.sin(t_a),
        t_a * 1.5
    ], axis=1)

    # Chain B: nearby helix
    t_b = np.linspace(0, 3*np.pi, L_b)
    gt_b = np.stack([
        5 * np.cos(t_b) + 8,
        5 * np.sin(t_b),
        t_b * 1.5 + 5
    ], axis=1)

    gt_centroids = np.concatenate([gt_a, gt_b], axis=0)
    chain_ids = np.array([0]*L_a + [1]*L_b)

    # Prediction: GT + noise
    pred_centroids = gt_centroids + np.random.randn(L, 3) * 1.5

    return pred_centroids, gt_centroids, chain_ids


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate TinyFold architecture diagrams')
    parser.add_argument('--output', '-o', type=str,
                        default='docs/images/architecture.png',
                        help='Output path for architecture diagram')
    parser.add_argument('--example', '-e', action='store_true',
                        help='Also generate example prediction plot')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    args = parser.parse_args()

    # Generate architecture diagram
    plot_resfold_architecture(args.output, dpi=args.dpi)

    # Generate example prediction
    if args.example:
        pred, gt, chains = generate_dummy_prediction()
        plot_prediction_example(pred, gt, chains,
                               save_path='docs/images/example_prediction.png',
                               title='Example: Two-Chain Complex')


if __name__ == '__main__':
    main()
