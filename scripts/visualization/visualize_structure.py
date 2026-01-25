#!/usr/bin/env python
"""Publication-ready protein structure visualization.

Generates high-quality figures with secondary structure (cartoon) rendering
for ground truth and predicted structures.

Usage:
    # Single structure visualization
    python visualize_structure.py --sample_id 1a2k_A_B

    # Multiple structures in a grid
    python visualize_structure.py --sample_ids 1a2k_A_B,3hfm_H_Y --grid

    # First N samples from dataset
    python visualize_structure.py --num_samples 5

    # Custom output directory
    python visualize_structure.py --sample_id 1a2k_A_B --output outputs/viz/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from tinyfold.data.cache import dict_to_sample
from tinyfold.viz.io.structure_writer import coords_to_pdb_string, write_pdb
from tinyfold.viz.render.styles import CHAIN_COLORS


# ============================================================================
# Data Loading
# ============================================================================

def load_sample_by_id(table, sample_id: str) -> dict | None:
    """Load a sample from parquet table by sample_id."""
    df = table.to_pandas()
    mask = df["sample_id"] == sample_id
    if not mask.any():
        return None
    row = df[mask].iloc[0].to_dict()
    return dict_to_sample(row)


def load_sample_by_index(table, idx: int) -> dict:
    """Load a sample from parquet table by index."""
    df = table.to_pandas()
    row = df.iloc[idx].to_dict()
    return dict_to_sample(row)


def get_sample_ids(table, num_samples: int | None = None) -> list[str]:
    """Get list of sample IDs from table."""
    ids = table["sample_id"].to_pylist()
    if num_samples is not None:
        ids = ids[:num_samples]
    return ids


# ============================================================================
# py3Dmol HTML Generation
# ============================================================================

def make_publication_viewer_html(
    pdb_string: str,
    title: str = "Protein Structure",
    width: int = 800,
    height: int = 600,
    background: str = "white",
    style: str = "cartoon",
    color_scheme: str = "chain",
    interface_residues: list[int] | None = None,
    show_controls: bool = True,
) -> str:
    """Create publication-ready HTML viewer with py3Dmol.

    Args:
        pdb_string: PDB format structure
        title: Viewer title
        width: Width in pixels
        height: Height in pixels
        background: Background color
        style: Visualization style (cartoon, stick, sphere)
        color_scheme: Coloring scheme (chain, ss, interface)
        interface_residues: List of interface residue indices to highlight
        show_controls: Show rotation/zoom controls

    Returns:
        HTML string
    """
    pdb_escaped = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    # Build style JavaScript based on color scheme
    if color_scheme == "chain":
        style_js = f"""
            viewer.setStyle({{chain: 'A'}}, {{{style}: {{color: '{CHAIN_COLORS["A"]}'}}}});
            viewer.setStyle({{chain: 'B'}}, {{{style}: {{color: '{CHAIN_COLORS["B"]}'}}}});
        """
    elif color_scheme == "ss":
        # Color by secondary structure
        style_js = f"""
            viewer.setStyle({{}}, {{{style}: {{colorscheme: 'ssJmol'}}}});
        """
    elif color_scheme == "spectrum":
        style_js = f"""
            viewer.setStyle({{}}, {{{style}: {{color: 'spectrum'}}}});
        """
    else:
        style_js = f"""
            viewer.setStyle({{chain: 'A'}}, {{{style}: {{color: '{CHAIN_COLORS["A"]}'}}}});
            viewer.setStyle({{chain: 'B'}}, {{{style}: {{color: '{CHAIN_COLORS["B"]}'}}}});
        """

    # Add interface highlighting if provided
    interface_js = ""
    if interface_residues:
        res_str = ",".join(str(r + 1) for r in interface_residues)  # 1-based
        interface_js = f"""
            viewer.setStyle({{resi: [{res_str}]}}, {{
                {style}: {{color: '#22C55E'}},
                stick: {{radius: 0.1, color: '#22C55E'}}
            }});
        """

    controls_html = ""
    controls_js = ""
    if show_controls:
        controls_html = """
        <div class="controls">
            <button class="btn" onclick="resetView()">Reset View</button>
            <button class="btn" onclick="toggleSpin()">Spin</button>
            <button class="btn" onclick="setView('front')">Front</button>
            <button class="btn" onclick="setView('side')">Side</button>
            <button class="btn" onclick="setView('top')">Top</button>
        </div>
        """
        controls_js = """
        let spinning = false;

        function resetView() {
            viewer.zoomTo();
            viewer.render();
        }

        function toggleSpin() {
            spinning = !spinning;
            viewer.spin(spinning ? 'y' : false);
        }

        function setView(name) {
            viewer.zoomTo();
            if (name === 'front') {
                // Default front view
            } else if (name === 'side') {
                viewer.rotate(90, 'y');
            } else if (name === 'top') {
                viewer.rotate(90, 'x');
            }
            viewer.render();
        }
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f8f9fa;
            padding: 24px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .viewer-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            overflow: hidden;
        }}
        .viewer-header {{
            padding: 16px 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        .viewer-title {{
            font-size: 18px;
            font-weight: 600;
            color: #212529;
        }}
        .viewer-box {{
            width: {width}px;
            height: {height}px;
            margin: 0 auto;
        }}
        .viewer-footer {{
            padding: 12px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #495057;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .controls {{
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }}
        .btn {{
            padding: 6px 14px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            color: #495057;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #e9ecef;
            border-color: #ced4da;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="viewer-card">
            <div class="viewer-header">
                <div class="viewer-title">{title}</div>
            </div>
            <div id="viewer" class="viewer-box"></div>
            <div class="viewer-footer">
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background: {CHAIN_COLORS['A']}"></div>
                        <span>Chain A</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: {CHAIN_COLORS['B']}"></div>
                        <span>Chain B</span>
                    </div>
                </div>
                {controls_html}
            </div>
        </div>
    </div>

    <script>
        const pdbData = `{pdb_escaped}`;

        let viewer = $3Dmol.createViewer('viewer', {{
            backgroundColor: '{background}'
        }});

        viewer.addModel(pdbData, 'pdb');
        {style_js}
        {interface_js}
        viewer.zoomTo();
        viewer.render();

        {controls_js}
    </script>
</body>
</html>
"""
    return html


def make_grid_viewer_html(
    structures: list[dict],
    title: str = "Structure Gallery",
    cols: int = 2,
    width: int = 400,
    height: int = 350,
) -> str:
    """Create HTML with multiple structures in a grid.

    Args:
        structures: List of dicts with 'pdb', 'label' keys
        title: Page title
        cols: Number of columns
        width: Per-viewer width
        height: Per-viewer height

    Returns:
        HTML string
    """
    viewer_divs = []
    viewer_scripts = []

    for i, struct in enumerate(structures):
        pdb_escaped = struct['pdb'].replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        label = struct.get('label', f'Structure {i+1}')

        viewer_divs.append(f"""
            <div class="viewer-cell">
                <div class="viewer-label">{label}</div>
                <div id="viewer{i}" class="viewer-box"></div>
            </div>
        """)

        viewer_scripts.append(f"""
            let v{i} = $3Dmol.createViewer('viewer{i}', {{backgroundColor: 'white'}});
            v{i}.addModel(`{pdb_escaped}`, 'pdb');
            v{i}.setStyle({{chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
            v{i}.setStyle({{chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});
            v{i}.zoomTo();
            v{i}.render();
        """)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            padding: 24px;
            margin: 0;
        }}
        .page-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #212529;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            gap: 16px;
            max-width: {cols * (width + 40)}px;
        }}
        .viewer-cell {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 12px;
        }}
        .viewer-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            color: #495057;
        }}
        .viewer-box {{
            width: {width}px;
            height: {height}px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            margin-top: 16px;
            font-size: 13px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="page-title">{title}</div>
    <div class="grid">
        {''.join(viewer_divs)}
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: {CHAIN_COLORS['A']}"></div>
            <span>Chain A</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {CHAIN_COLORS['B']}"></div>
            <span>Chain B</span>
        </div>
    </div>
    <script>
        {''.join(viewer_scripts)}
    </script>
</body>
</html>
"""
    return html


# ============================================================================
# Matplotlib Static Rendering
# ============================================================================

def plot_backbone_structure(
    coords: np.ndarray,
    chain_ids: np.ndarray,
    atom_types: np.ndarray,
    ax=None,
    title: str = "",
    view: str = "default",
    show_ca_trace: bool = True,
    alpha: float = 0.8,
):
    """Plot protein backbone structure with matplotlib.

    Args:
        coords: [N_atom, 3] coordinates
        chain_ids: [N_atom] chain IDs (0 or 1)
        atom_types: [N_atom] atom types (0=N, 1=CA, 2=C, 3=O)
        ax: Matplotlib 3D axis (created if None)
        title: Plot title
        view: Camera view (default, front, side, top)
        show_ca_trace: Draw lines connecting CA atoms
        alpha: Point transparency

    Returns:
        axis object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Color scheme matching py3Dmol
    colors = {0: '#3B82F6', 1: '#F97316'}  # Chain A: blue, Chain B: orange

    # Plot atoms colored by chain
    for chain_id in [0, 1]:
        mask = chain_ids == chain_id
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=colors[chain_id],
                s=20,
                alpha=alpha,
                label=f'Chain {"A" if chain_id == 0 else "B"}',
            )

    # Draw CA trace (backbone ribbon approximation)
    if show_ca_trace:
        ca_mask = atom_types == 1  # CA atoms
        for chain_id in [0, 1]:
            chain_mask = chain_ids == chain_id
            both_mask = ca_mask & chain_mask
            if both_mask.sum() > 1:
                ca_coords = coords[both_mask]
                ax.plot(
                    ca_coords[:, 0],
                    ca_coords[:, 1],
                    ca_coords[:, 2],
                    c=colors[chain_id],
                    linewidth=1.5,
                    alpha=0.6,
                )

    # Set equal aspect ratio
    max_range = np.max(coords.max(axis=0) - coords.min(axis=0)) / 2
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Set view angle
    if view == "front":
        ax.view_init(elev=0, azim=0)
    elif view == "side":
        ax.view_init(elev=0, azim=90)
    elif view == "top":
        ax.view_init(elev=90, azim=0)
    else:
        ax.view_init(elev=20, azim=45)

    # Clean up axes for publication
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove axis panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    if title:
        ax.set_title(title, fontsize=14, fontweight='medium', pad=10)

    return ax


def create_multi_view_figure(
    coords: np.ndarray,
    chain_ids: np.ndarray,
    atom_types: np.ndarray,
    title: str = "",
    dpi: int = 150,
) -> 'plt.Figure':
    """Create figure with multiple view angles.

    Args:
        coords: [N_atom, 3] coordinates
        chain_ids: [N_atom] chain IDs
        atom_types: [N_atom] atom types
        title: Figure title
        dpi: Output DPI

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), dpi=dpi)

    views = [('Front', 'front'), ('Side', 'side'), ('Top', 'top')]

    for i, (label, view) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        plot_backbone_structure(
            coords, chain_ids, atom_types,
            ax=ax, title=label, view=view
        )

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# ============================================================================
# Main Script
# ============================================================================

def visualize_single_sample(
    sample: dict,
    output_dir: Path,
    style: str = "cartoon",
    color_scheme: str = "chain",
    dpi: int = 300,
    create_png: bool = True,
    create_html: bool = True,
    create_pdb: bool = True,
) -> dict:
    """Generate all visualizations for a single sample.

    Args:
        sample: Sample dict from parquet
        output_dir: Output directory
        style: py3Dmol style
        color_scheme: Coloring scheme
        dpi: PNG output DPI
        create_png: Generate PNG files
        create_html: Generate HTML viewer
        create_pdb: Save PDB file

    Returns:
        Dict with output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    sample_id = sample["sample_id"]
    coords = sample["atom_coords"]
    atom_to_res = sample["atom_to_res"]
    atom_type = sample["atom_type"]
    chain_id_res = sample["chain_id_res"]
    res_idx = sample["res_idx"]
    seq = sample["seq"]
    iface_mask = sample["iface_mask"]

    # Get chain IDs per atom
    chain_ids = chain_id_res[atom_to_res]

    # Get interface residues for highlighting
    interface_res = np.where(iface_mask)[0].tolist() if iface_mask.any() else None

    # Generate PDB string
    pdb_string = coords_to_pdb_string(
        xyz=coords,
        atom_to_res=atom_to_res,
        atom_type=atom_type,
        chain_id_res=chain_id_res,
        res_idx=res_idx,
        seq=seq,
    )

    # Save PDB file
    if create_pdb:
        pdb_path = output_dir / "structure.pdb"
        with open(pdb_path, "w") as f:
            f.write(pdb_string)
        outputs["pdb"] = str(pdb_path)
        print(f"  PDB: {pdb_path}")

    # Create HTML viewer
    if create_html:
        html = make_publication_viewer_html(
            pdb_string=pdb_string,
            title=f"Structure: {sample_id}",
            style=style,
            color_scheme=color_scheme,
            interface_residues=interface_res if color_scheme == "interface" else None,
        )
        html_path = output_dir / "structure.html"
        with open(html_path, "w") as f:
            f.write(html)
        outputs["html"] = str(html_path)
        print(f"  HTML: {html_path}")

    # Create PNG with matplotlib
    if create_png:
        import matplotlib.pyplot as plt

        # Single view
        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        plot_backbone_structure(
            coords, chain_ids, atom_type,
            ax=ax, title=sample_id, view="default"
        )
        ax.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()

        png_path = output_dir / "structure.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        outputs["png"] = str(png_path)
        print(f"  PNG: {png_path}")

        # Multi-view figure
        fig = create_multi_view_figure(
            coords, chain_ids, atom_type,
            title=sample_id, dpi=dpi
        )
        multiview_path = output_dir / "structure_multiview.png"
        fig.savefig(multiview_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        outputs["png_multiview"] = str(multiview_path)
        print(f"  Multi-view PNG: {multiview_path}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready protein structure visualization"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to samples.parquet (default: data/processed/samples.parquet)",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Single sample ID to visualize",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Comma-separated sample IDs for grid view",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Visualize first N samples from dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/viz",
        help="Output directory",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="cartoon",
        choices=["cartoon", "stick", "sphere", "line"],
        help="Visualization style",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="chain",
        choices=["chain", "ss", "interface", "spectrum"],
        help="Color scheme",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output DPI",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create grid view for multiple samples",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG generation",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML generation",
    )
    args = parser.parse_args()

    # Find data file
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = project_root / "data" / "processed" / "samples.parquet"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    table = pq.read_table(data_path)
    print(f"Loaded {len(table)} samples")

    output_dir = Path(args.output)

    # Determine which samples to visualize
    if args.sample_id:
        sample_ids = [args.sample_id]
    elif args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]
    elif args.num_samples:
        sample_ids = get_sample_ids(table, args.num_samples)
    else:
        # Default: first sample
        sample_ids = get_sample_ids(table, 1)

    print(f"\nVisualizing {len(sample_ids)} sample(s)")

    # Process each sample
    all_structures = []
    for sample_id in sample_ids:
        print(f"\nProcessing: {sample_id}")
        sample = load_sample_by_id(table, sample_id)
        if sample is None:
            print(f"  Warning: Sample not found: {sample_id}")
            continue

        sample_output_dir = output_dir / sample_id
        outputs = visualize_single_sample(
            sample=sample,
            output_dir=sample_output_dir,
            style=args.style,
            color_scheme=args.color,
            dpi=args.dpi,
            create_png=not args.no_png,
            create_html=not args.no_html,
        )

        # Collect for grid view
        if args.grid and "pdb" in outputs:
            with open(outputs["pdb"]) as f:
                pdb_str = f.read()
            all_structures.append({
                "pdb": pdb_str,
                "label": sample_id,
            })

    # Create grid view if requested
    if args.grid and len(all_structures) > 1:
        print(f"\nCreating grid view with {len(all_structures)} structures")
        cols = min(3, len(all_structures))
        html = make_grid_viewer_html(
            structures=all_structures,
            title="Structure Gallery",
            cols=cols,
        )
        grid_path = output_dir / "grid_view.html"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grid_path, "w") as f:
            f.write(html)
        print(f"  Grid HTML: {grid_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
