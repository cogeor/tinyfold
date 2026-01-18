"""py3Dmol-based 3D viewer for protein structures."""

from tinyfold.viz.render.styles import CHAIN_COLORS, get_chain_style


def make_viewer_html(
    pred_pdb: str,
    ref_pdb: str | None = None,
    title: str = "Structure Viewer",
    highlights: dict | None = None,
    width: int = 800,
    height: int = 500,
    show_both_alignments: bool = True,
) -> str:
    """Create HTML with embedded py3Dmol viewer.

    Args:
        pred_pdb: PDB string for predicted structure
        ref_pdb: PDB string for reference structure (optional)
        title: Viewer title
        highlights: dict with keys like 'interface_residues', 'clash_atoms'
        width: Viewer width in pixels
        height: Viewer height in pixels
        show_both_alignments: If True, show tabs for complex/receptor alignment

    Returns:
        HTML string with embedded viewer
    """
    # Escape PDB strings for JavaScript
    pred_pdb_escaped = pred_pdb.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    ref_pdb_escaped = ref_pdb.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$") if ref_pdb else ""

    # Build highlight JavaScript
    highlight_js = ""
    if highlights:
        if "interface_residues" in highlights:
            res_list = highlights["interface_residues"]
            if res_list:
                res_str = ",".join(str(r + 1) for r in res_list)  # 1-based for PDB
                highlight_js += f"""
                viewer.setStyle({{resi: [{res_str}], chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}, stick: {{radius: 0.1, color: '#22C55E'}}}});
                viewer.setStyle({{resi: [{res_str}], chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}, stick: {{radius: 0.1, color: '#22C55E'}}}});
                """

    has_ref = ref_pdb is not None

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .viewer-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 16px;
            margin-bottom: 16px;
        }}
        .viewer-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #333;
        }}
        .viewer-box {{
            width: {width}px;
            height: {height}px;
            position: relative;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            font-size: 14px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .tab {{
            padding: 8px 16px;
            border: none;
            background: #e5e7eb;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}
        .tab.active {{
            background: #3B82F6;
            color: white;
        }}
        .controls {{
            margin-top: 12px;
            display: flex;
            gap: 8px;
        }}
        .btn {{
            padding: 6px 12px;
            border: 1px solid #d1d5db;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        .btn:hover {{
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <div class="viewer-container">
        <div class="viewer-title">{title}</div>

        {"<div class='tabs'><button class='tab active' onclick='showPred()'>Predicted</button><button class='tab' onclick='showRef()'>Reference</button><button class='tab' onclick='showOverlay()'>Overlay</button></div>" if has_ref else ""}

        <div id="viewer" class="viewer-box"></div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: {CHAIN_COLORS['A']}"></div>
                <span>Chain A</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {CHAIN_COLORS['B']}"></div>
                <span>Chain B</span>
            </div>
            {"<div class='legend-item'><div class='legend-color' style='background: " + CHAIN_COLORS['ref'] + "'></div><span>Reference</span></div>" if has_ref else ""}
        </div>

        <div class="controls">
            <button class="btn" onclick="resetView()">Reset View</button>
            <button class="btn" onclick="toggleSpin()">Toggle Spin</button>
        </div>
    </div>

    <script>
        const predPdb = `{pred_pdb_escaped}`;
        const refPdb = `{ref_pdb_escaped}`;

        let viewer = $3Dmol.createViewer('viewer', {{
            backgroundColor: 'white'
        }});

        let spinning = false;

        function showPred() {{
            viewer.removeAllModels();
            viewer.addModel(predPdb, 'pdb');
            viewer.setStyle({{chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
            viewer.setStyle({{chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});
            {highlight_js}
            viewer.zoomTo();
            viewer.render();
            updateTabs('pred');
        }}

        function showRef() {{
            if (!refPdb) return;
            viewer.removeAllModels();
            viewer.addModel(refPdb, 'pdb');
            viewer.setStyle({{chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
            viewer.setStyle({{chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});
            viewer.zoomTo();
            viewer.render();
            updateTabs('ref');
        }}

        function showOverlay() {{
            if (!refPdb) return;
            viewer.removeAllModels();

            // Add reference (transparent)
            viewer.addModel(refPdb, 'pdb');
            viewer.setStyle({{model: 0}}, {{cartoon: {{opacity: 0.4, color: '{CHAIN_COLORS["ref"]}'}}}});

            // Add prediction
            viewer.addModel(predPdb, 'pdb');
            viewer.setStyle({{model: 1, chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
            viewer.setStyle({{model: 1, chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});

            viewer.zoomTo();
            viewer.render();
            updateTabs('overlay');
        }}

        function updateTabs(active) {{
            document.querySelectorAll('.tab').forEach((tab, i) => {{
                const names = ['pred', 'ref', 'overlay'];
                tab.classList.toggle('active', names[i] === active);
            }});
        }}

        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}

        function toggleSpin() {{
            spinning = !spinning;
            if (spinning) {{
                viewer.spin('y', 1);
            }} else {{
                viewer.spin(false);
            }}
        }}

        // Initial view
        showPred();
    </script>
</body>
</html>
"""
    return html


def make_dual_viewer_html(
    pred_pdb_complex_aligned: str,
    pred_pdb_receptor_aligned: str,
    ref_pdb: str,
    title: str = "Structure Comparison",
    width: int = 400,
    height: int = 400,
) -> str:
    """Create HTML with two viewers: complex-aligned and receptor-aligned.

    Args:
        pred_pdb_complex_aligned: PDB string aligned on full complex
        pred_pdb_receptor_aligned: PDB string aligned on receptor (chain A)
        ref_pdb: PDB string for reference
        title: Title
        width: Per-viewer width
        height: Per-viewer height

    Returns:
        HTML string with dual viewers
    """
    pred_complex_escaped = pred_pdb_complex_aligned.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    pred_receptor_escaped = pred_pdb_receptor_aligned.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    ref_escaped = ref_pdb.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        .viewers-row {{
            display: flex;
            gap: 20px;
        }}
        .viewer-panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 16px;
        }}
        .viewer-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            color: #555;
        }}
        .viewer-box {{
            width: {width}px;
            height: {height}px;
            border: 1px solid #e0e0e0;
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
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="title">{title}</div>

    <div class="viewers-row">
        <div class="viewer-panel">
            <div class="viewer-label">Complex-Aligned (Global RMSD)</div>
            <div id="viewer1" class="viewer-box"></div>
        </div>
        <div class="viewer-panel">
            <div class="viewer-label">Receptor-Aligned (LRMSD)</div>
            <div id="viewer2" class="viewer-box"></div>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: {CHAIN_COLORS['A']}"></div>
            <span>Chain A (pred)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {CHAIN_COLORS['B']}"></div>
            <span>Chain B (pred)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {CHAIN_COLORS['ref']}"></div>
            <span>Reference</span>
        </div>
    </div>

    <script>
        const predComplex = `{pred_complex_escaped}`;
        const predReceptor = `{pred_receptor_escaped}`;
        const refPdb = `{ref_escaped}`;

        // Viewer 1: Complex-aligned
        let v1 = $3Dmol.createViewer('viewer1', {{backgroundColor: 'white'}});
        v1.addModel(refPdb, 'pdb');
        v1.setStyle({{model: 0}}, {{cartoon: {{opacity: 0.4, color: '{CHAIN_COLORS["ref"]}'}}}});
        v1.addModel(predComplex, 'pdb');
        v1.setStyle({{model: 1, chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
        v1.setStyle({{model: 1, chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});
        v1.zoomTo();
        v1.render();

        // Viewer 2: Receptor-aligned
        let v2 = $3Dmol.createViewer('viewer2', {{backgroundColor: 'white'}});
        v2.addModel(refPdb, 'pdb');
        v2.setStyle({{model: 0}}, {{cartoon: {{opacity: 0.4, color: '{CHAIN_COLORS["ref"]}'}}}});
        v2.addModel(predReceptor, 'pdb');
        v2.setStyle({{model: 1, chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS["A"]}'}}}});
        v2.setStyle({{model: 1, chain: 'B'}}, {{cartoon: {{color: '{CHAIN_COLORS["B"]}'}}}});
        v2.zoomTo();
        v2.render();
    </script>
</body>
</html>
"""
    return html


def display_in_notebook(html: str) -> None:
    """Display HTML viewer in Jupyter/Colab notebook.

    Args:
        html: HTML string from make_viewer_html
    """
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except ImportError:
        print("IPython not available. Save HTML to file instead.")
