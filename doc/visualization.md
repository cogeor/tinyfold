# Visualization and Metrics

## Overview

The visualization module provides tools for:
1. **Structural alignment** using Kabsch algorithm
2. **Quality metrics**: RMSD, LRMSD, iRMSD, contact prediction
3. **PDB output** for molecular viewers
4. **Interactive 3D visualization** with py3Dmol
5. **HTML reports** with plots and metrics

```
viz/
├── mapping/
│   └── atom_schema.py     # Atom naming conventions
├── metrics/
│   ├── align.py           # Kabsch alignment
│   ├── rmsd.py            # RMSD computations
│   └── contacts.py        # Contact map analysis
├── io/
│   └── structure_writer.py # PDB file writing
├── plots/
│   ├── matrices.py        # Contact/distance maps
│   └── distributions.py   # Metric distributions
├── render/
│   ├── styles.py          # Color schemes
│   └── py3dmol_viewer.py  # Interactive viewer
└── report/
    └── html_report.py     # Report generation
```

## Kabsch Alignment (metrics/align.py)

The Kabsch algorithm finds the optimal rotation to superimpose two point sets.

### Algorithm

Given two sets of corresponding points P (predicted) and Q (reference):

1. **Center both point sets**:
   ```
   P_centered = P - mean(P)
   Q_centered = Q - mean(Q)
   ```

2. **Compute cross-covariance matrix**:
   ```
   H = P_centered.T @ Q_centered
   ```

3. **SVD decomposition**:
   ```
   U, S, Vt = svd(H)
   ```

4. **Optimal rotation**:
   ```
   R = Vt.T @ U.T
   ```

5. **Handle reflection** (det(R) = -1):
   ```
   if det(R) < 0:
       Vt[-1, :] *= -1
       R = Vt.T @ U.T
   ```

6. **Apply transformation**:
   ```
   P_aligned = (R @ P_centered.T).T + mean(Q)
   ```

### Usage

```python
from tinyfold.viz.metrics.align import kabsch_align

aligned_coords, R, t = kabsch_align(pred_coords, ref_coords, mask=valid_mask)
```

### Masked Alignment

When some atoms are missing or should be excluded:

```python
# Only use CA atoms for alignment
ca_mask = atom_type == 1
aligned, R, t = kabsch_align(pred, ref, mask=ca_mask)
```

## RMSD Metrics (metrics/rmsd.py)

Root Mean Square Deviation measures structural similarity:

```
RMSD = sqrt(mean((pred - ref)²))
```

### Metric Types

| Metric | Description |
|--------|-------------|
| **RMSD** | Global RMSD after optimal alignment |
| **LRMSD** | Ligand RMSD (chain B after aligning chain A) |
| **iRMSD** | Interface RMSD (only interface residues) |

### LRMSD (Docking Quality)

```python
def compute_lrmsd(pred, ref, chain_a_mask, chain_b_mask):
    # 1. Align on receptor (chain A)
    pred_aligned = align_on_subset(pred, ref, chain_a_mask)

    # 2. Compute RMSD of ligand (chain B)
    lrmsd = sqrt(mean((pred_aligned[chain_b_mask] - ref[chain_b_mask])²))
    return lrmsd
```

**Why LRMSD?** Global RMSD can be low even with bad docking if both chains are internally correct. LRMSD specifically measures interface quality.

### Interface RMSD

```python
def compute_irmsd(pred, ref, interface_mask):
    # Only consider residues at the interface
    pred_iface = pred[interface_mask]
    ref_iface = ref[interface_mask]

    # Align and compute RMSD on interface only
    aligned, _, _ = kabsch_align(pred_iface, ref_iface)
    return sqrt(mean((aligned - ref_iface)²))
```

### Backbone RMSD Function

```python
def backbone_rmsd(pred, ref, atom_to_res, chain_id_res, atom_mask=None, interface_mask=None):
    """Compute multiple RMSD metrics.

    Returns:
        {
            "rmsd_complex": float,   # Overall aligned RMSD
            "rmsd_chain_a": float,   # Chain A only
            "rmsd_chain_b": float,   # Chain B only
            "lrmsd": float,          # Ligand RMSD
            "irmsd": float,          # Interface RMSD (or NaN if no interface)
        }
    """
```

## Contact Maps (metrics/contacts.py)

Contact maps capture which residue pairs are spatially close.

### Definition

```python
def contact_map_CA(coords, atom_type, atom_to_res, chain_id_res, threshold=8.0):
    """Compute inter-chain CA-CA contact map.

    Returns:
        A_idx: indices of chain A residues
        B_idx: indices of chain B residues
        contacts: [LA, LB] boolean matrix
    """
    # Extract CA atoms (atom_type == 1)
    ca_coords_a = coords[CA atoms of chain A]
    ca_coords_b = coords[CA atoms of chain B]

    # Compute distances
    dist = cdist(ca_coords_a, ca_coords_b)

    # Threshold
    contacts = dist < threshold
    return contacts
```

### Contact Metrics

```python
def contact_metrics(pred_contacts, ref_contacts):
    """Precision, recall, F1 for contact prediction."""
    tp = (pred_contacts & ref_contacts).sum()
    fp = (pred_contacts & ~ref_contacts).sum()
    fn = (~pred_contacts & ref_contacts).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": pred_contacts.sum(),
        "n_ref": ref_contacts.sum(),
    }
```

### Interface Residues

```python
def interface_residues(contact_map, chain_id_res):
    """Identify residues at the interface."""
    # Chain A residues that contact chain B
    A_interface = contact_map.any(axis=1)
    # Chain B residues that contact chain A
    B_interface = contact_map.any(axis=0)

    # Combine into full mask
    interface_mask = zeros(len(chain_id_res), dtype=bool)
    interface_mask[chain_a_indices] = A_interface
    interface_mask[chain_b_indices] = B_interface

    return interface_mask
```

## PDB Writing (io/structure_writer.py)

### PDB Format

```
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
```

Columns:
- 1-6: Record name (ATOM)
- 7-11: Atom serial number
- 13-16: Atom name
- 17: Alternate location (usually blank)
- 18-20: Residue name
- 22: Chain ID
- 23-26: Residue number
- 31-54: X, Y, Z coordinates
- 55-60: Occupancy
- 61-66: Temperature factor
- 77-78: Element symbol

### Usage

```python
from tinyfold.viz.io.structure_writer import write_pdb, coords_to_pdb_string

# Write to file
write_pdb("output.pdb", coords, atom_to_res, atom_type, chain_id_res, res_idx)

# Get as string
pdb_str = coords_to_pdb_string(coords, atom_to_res, atom_type, chain_id_res, res_idx)
```

### Atom Schema

```python
@dataclass
class AtomSchema:
    atom_type_to_name = {0: "N", 1: "CA", 2: "C", 3: "O"}
    atom_type_to_element = {0: "N", 1: "C", 2: "C", 3: "O"}
    chain_labels = ("A", "B")
```

## Interactive Viewer (render/py3dmol_viewer.py)

Uses py3Dmol for in-browser 3D visualization.

### Single Structure

```python
from tinyfold.viz.render.py3dmol_viewer import make_viewer_html

html = make_viewer_html(pdb_string, title="Structure")
```

### Dual View (Pred vs Ref)

```python
html = make_dual_viewer_html(
    pred_pdb,
    pred_receptor_aligned_pdb,
    ref_pdb,
    title="Comparison"
)
```

Features:
- Toggle between complex-aligned and receptor-aligned views
- Chain coloring (A=blue, B=orange)
- Cartoon + lines representation

## Plotting (plots/)

### Contact Map

```python
from tinyfold.viz.plots.matrices import plot_contact_map

fig = plot_contact_map(pred_contacts, ref_contacts, save_path="contacts.png")
```

Features:
- Side-by-side predicted vs reference
- True positive highlighting
- Precision/recall annotation

### RMSD Comparison

```python
from tinyfold.viz.plots.distributions import plot_rmsd_comparison

metrics = {"rmsd_complex": 2.5, "lrmsd": 4.1, "irmsd": 3.2}
fig = plot_rmsd_comparison(metrics, save_path="rmsd.png")
```

### Multi-Sample Distribution

```python
from tinyfold.viz.plots.distributions import plot_metric_distribution

# Show distribution across multiple samples
fig = plot_metric_distribution(
    values=[2.1, 2.5, 3.0, 2.8],
    metric_name="RMSD",
    unit="Å",
    highlight_idx=0,  # Best sample
)
```

## HTML Reports (report/html_report.py)

Generates comprehensive HTML reports with:
- 3D viewer (py3Dmol)
- RMSD metrics cards
- Contact prediction metrics
- Plots (contact map, RMSD comparison)
- Sample ranking table (if multiple predictions)

### Usage

```python
from tinyfold.viz.report.html_report import make_report

report_path = make_report(
    sample_id="1a2k_A_B",
    pred_xyz=pred_coords,
    ref_xyz=ref_coords,
    atom_to_res=atom_to_res,
    atom_type=atom_type,
    chain_id_res=chain_id_res,
    res_idx=res_idx,
    out_dir="reports/",
)
```

### Output Structure

```
reports/1a2k_A_B/
├── report.html      # Main report
├── viewer.html      # 3D viewer (iframe)
├── pred.pdb         # Predicted structure
├── ref.pdb          # Reference structure
└── plots/
    ├── contact_map.png
    ├── rmsd_comparison.png
    └── contact_metrics.png
```

### Template System

Uses Jinja2 for HTML templating with fallback to simple string formatting:

```python
if HAS_JINJA2:
    template = Template(REPORT_TEMPLATE)
    html = template.render(sample_id=..., rmsd=..., contacts=...)
else:
    html = _render_simple_report(...)
```

## Quality Assessment Guidelines

### RMSD Thresholds

| RMSD | Quality |
|------|---------|
| < 2Å | High quality |
| 2-5Å | Medium quality |
| > 5Å | Low quality |

### LRMSD for Docking

| LRMSD | Classification |
|-------|----------------|
| < 4Å | Acceptable |
| < 2Å | High quality |
| < 1Å | Near-native |

### Contact F1

| F1 Score | Interpretation |
|----------|----------------|
| > 0.8 | Excellent |
| 0.5-0.8 | Good |
| < 0.5 | Poor |
