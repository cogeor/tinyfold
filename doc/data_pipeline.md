# Data Pipeline

## Overview

The data pipeline transforms raw PDB structures into tensors suitable for training. It handles the complexities of protein structure data: missing atoms, modified residues, chain extraction, and efficient storage.

```
PDB/mmCIF files
      │
      ▼
┌─────────────────┐
│  Structure I/O  │  gemmi library
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Cleaning     │  Handle modified residues, missing atoms
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Atomization    │  Residue → Atom level, build bonds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │  Check bond lengths, completeness
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parquet Cache  │  Efficient columnar storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PPIDataset    │  PyTorch Dataset + collation
└─────────────────┘
```

## DIPS-Plus Dataset

We use DIPS-Plus, a curated dataset of protein-protein interfaces:
- **Source**: https://zenodo.org/records/8140981
- **Size**: ~42,000 PPI complexes
- **Format**: PDB files with chain pairs

### Manifest Structure

```json
{
  "sample_id": "1a2k_A_B",
  "pdb_id": "1a2k",
  "chain_a": "A",
  "chain_b": "B",
  "path": "data/raw/1a2k.pdb"
}
```

## Structure Loading (parsing/structure_io.py)

Uses the `gemmi` library for robust PDB/mmCIF parsing:

```python
def load_structure(path: str) -> gemmi.Structure:
    """Load structure with automatic format detection."""
    if path.endswith('.cif'):
        return gemmi.read_structure(path)
    return gemmi.read_pdb(path)

def get_backbone_atoms(chain: gemmi.Chain) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract backbone atoms (N, CA, C, O) from chain.

    Returns:
        coords: [L, 4, 3] backbone coordinates
        mask: [L, 4] boolean mask for present atoms
        sequence: list of 1-letter AA codes
    """
```

### Coordinate Layout

```
coords[residue_idx, atom_type, xyz]

atom_type:
  0 = N   (amide nitrogen)
  1 = CA  (alpha carbon)
  2 = C   (carbonyl carbon)
  3 = O   (carbonyl oxygen)
```

## Cleaning (processing/cleaning.py)

### Modified Residue Handling

Non-standard residues are mapped to their parent amino acid:

```python
MODIFIED_AA_MAP = {
    "MSE": "M",  # Selenomethionine → Methionine
    "HYP": "P",  # Hydroxyproline → Proline
    "SEP": "S",  # Phosphoserine → Serine
    ...
}
```

### Missing Atom Handling

The mask tensor tracks which atoms are present:

```python
mask[i, j] = True   # Atom j of residue i is present
mask[i, j] = False  # Atom j of residue i is missing
```

Missing atoms are filled with NaN coordinates and masked out during processing.

## Atomization (processing/atomization.py)

Converts residue-level data to atom-level for the EGNN:

### Coordinate Flattening

```python
# Input: coords[L, 4, 3] per-residue
# Output: atom_coords[N_atom, 3] flattened

atom_coords = coords.reshape(L * 4, 3)
```

### Atom Indexing

```python
# Global atom index from residue and atom type
def atom_idx(res_idx: int, atom_type: int) -> int:
    return res_idx * 4 + atom_type

# Inverse mapping
atom_to_res = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
atom_type = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...]
```

### Bond Graph Construction

Covalent bonds are built as a sparse graph:

```python
# Within-residue bonds (per residue)
N-CA   (atoms 0-1, type 0)
CA-C   (atoms 1-2, type 1)
C-O    (atoms 2-3, type 2)

# Peptide bonds (between consecutive residues)
C(i)-N(i+1)  (type 3)
```

**Important**: Peptide bonds only connect residues within the same chain. Chain A and Chain B are not covalently bonded.

```python
# Bond type encoding
BOND_TYPE_N_CA = 0
BOND_TYPE_CA_C = 1
BOND_TYPE_C_O = 2
BOND_TYPE_PEPTIDE = 3
```

## Interface Detection (processing/interface.py)

Identifies residues at the protein-protein interface:

```python
def compute_interface_mask(
    coords_a: np.ndarray,  # [LA, 4, 3]
    coords_b: np.ndarray,  # [LB, 4, 3]
    threshold: float = 10.0,  # Angstroms
) -> np.ndarray:
    """Find residues with CA atoms within threshold of other chain."""
```

The interface mask is used to compute interface RMSD (iRMSD), a key metric for docking quality.

## Validation (processing/filters.py)

### Filter Criteria

```python
@dataclass
class FilterResult:
    passed: bool
    reason: str | None

def validate_sample(sample: dict) -> FilterResult:
    # Chain length: 40-300 residues per chain
    # Backbone completeness: >95% atoms present
    # Bond lengths: within tolerance of ideal
    # No NaN/Inf coordinates
    # Chains must interact (min distance < 15Å)
```

### Bond Length Validation

```python
BOND_LENGTHS = {
    "N-CA": 1.458,  # ± 0.15 Å
    "CA-C": 1.525,
    "C-O": 1.229,
    "C-N": 1.329,  # Peptide bond
}
```

Samples with bond lengths outside tolerance are rejected—they likely have structural errors.

## Parquet Serialization (cache.py)

Samples are stored in Apache Parquet format for efficient I/O:

### Schema

```python
{
    "sample_id": str,
    "pdb_id": str,
    "seq": bytes,           # np.int64 array
    "chain_id_res": bytes,  # np.int64 array
    "res_idx": bytes,       # np.int64 array
    "atom_coords": bytes,   # np.float32 array
    "atom_mask": bytes,     # bool array
    "atom_to_res": bytes,   # np.int64 array
    "atom_type": bytes,     # np.int64 array
    "bonds_src": bytes,     # np.int64 array
    "bonds_dst": bytes,     # np.int64 array
    "bond_type": bytes,     # np.int64 array
    "iface_mask": bytes,    # bool array
    "LA": int,
    "LB": int,
}
```

Arrays are serialized with `array.tobytes()` and deserialized with `np.frombuffer()`.

## Dataset and Collation (datasets/, collate.py)

### PPIDataset

```python
class PPIDataset(Dataset):
    def __init__(self, parquet_path, split_file=None):
        # Load all samples into memory
        # Optionally filter to specific split

    def __getitem__(self, idx) -> dict[str, Tensor]:
        # Convert numpy arrays to torch tensors
        return {
            "seq": torch.from_numpy(seq),
            "atom_coords": torch.from_numpy(coords),
            ...
        }
```

### Batch Collation

Variable-length samples require padding:

```python
def collate_ppi(batch: list[dict]) -> dict:
    """Collate variable-length PPI samples.

    Handles:
    - Padding residue-level tensors to max length
    - Padding atom-level tensors to max atoms
    - Merging bond graphs with offset indices
    """
```

#### Edge Index Offsetting

When batching, atom indices must be offset:

```
Sample 1: atoms 0-47,  bonds reference [0, 47]
Sample 2: atoms 48-95, bonds reference [48, 95]

Merged edge_index = cat([edges1, edges2 + 48])
```

#### Mask Tensors

```python
res_mask:  [B, Lmax]      # True for real residues
atom_mask: [B, Natom_max] # True for real/valid atoms
```

## Data Preparation Script

```bash
python scripts/prepare_data.py --output-dir data/processed
```

This downloads DIPS-Plus from Zenodo and processes it into Parquet format.

## Training Data Loading

The training script (`scripts/train.py`) loads data directly from Parquet:

```python
table = pq.read_table("data/processed/samples.parquet")

# Filter to medium-sized samples (200-400 atoms)
medium_indices = find_medium_samples(table, 200, 400)

# Load a sample
def load_sample_raw(table, i):
    coords = torch.tensor(table['atom_coords'][i].as_py())
    atom_types = torch.tensor(table['atom_type'][i].as_py())
    atom_to_res = torch.tensor(table['atom_to_res'][i].as_py())
    seq_res = torch.tensor(table['seq'][i].as_py())
    chain_res = torch.tensor(table['chain_id_res'][i].as_py())
    ...
```

Note: The training script defines its data loading inline rather than using the `src/tinyfold/data/` modules.

## Common Issues

### Missing Backbone Atoms

Some PDB files have missing atoms due to poor electron density. We mask these atoms but keep the residue if most atoms are present.

### Alternate Conformations

We take only the first alternate conformation (altloc 'A' or ' ').

### Chain Breaks

Physical chain breaks (missing residues in sequence) are handled by not creating peptide bonds across gaps.

### Very Long Chains

Chains > 300 residues are filtered out for memory efficiency. The Pairformer's O(L²) memory makes long sequences expensive.
