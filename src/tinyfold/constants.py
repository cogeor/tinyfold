"""Constants for amino acids, atoms, and bond geometry."""

# Standard amino acids (1-letter codes)
AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CODES)}
AA_TO_IDX["X"] = 20  # Unknown/non-standard
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
NUM_AA = 21

# 3-letter to 1-letter mapping
AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Common modified residues -> standard mapping
MODIFIED_AA_MAP = {
    "MSE": "M",  # Selenomethionine
    "HYP": "P",  # Hydroxyproline
    "MLY": "K",  # N-dimethyl-lysine
    "SEP": "S",  # Phosphoserine
    "TPO": "T",  # Phosphothreonine
    "PTR": "Y",  # Phosphotyrosine
    "CSO": "C",  # S-hydroxycysteine
    "SEC": "C",  # Selenocysteine
    "PCA": "E",  # Pyroglutamic acid
}

# Backbone atoms (in fixed order)
BACKBONE_ATOMS = ["N", "CA", "C", "O"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(BACKBONE_ATOMS)}
NUM_ATOM_TYPES = 4

# Bond types (4 distinct types for edge attribute encoding)
BOND_TYPE_N_CA = 0      # N-CA bond within residue
BOND_TYPE_CA_C = 1      # CA-C bond within residue
BOND_TYPE_C_O = 2       # C-O bond within residue
BOND_TYPE_PEPTIDE = 3   # Peptide bond C(i)-N(i+1)
NUM_BOND_TYPES = 4

# Legacy alias for backwards compatibility
BOND_TYPE_BACKBONE = 0  # Deprecated: use specific bond types above

# Within-residue backbone bonds (atom index pairs, bond type)
# N=0, CA=1, C=2, O=3
BACKBONE_BONDS = [
    (0, 1, BOND_TYPE_N_CA),   # N-CA
    (1, 2, BOND_TYPE_CA_C),   # CA-C
    (2, 3, BOND_TYPE_C_O),    # C-O
]

# Expected bond lengths in Angstroms (for validation)
BOND_LENGTHS = {
    "N-CA": 1.458,
    "CA-C": 1.525,
    "C-O": 1.229,
    "C-N": 1.329,  # Peptide bond
}

# Tolerance for bond length validation (Angstroms)
BOND_LENGTH_TOLERANCE = 0.15

# Interface distance threshold (CA-CA in Angstroms)
INTERFACE_DISTANCE_THRESHOLD = 10.0

# Filter thresholds
MIN_CHAIN_LENGTH = 40
MAX_CHAIN_LENGTH = 300
MIN_BACKBONE_COMPLETENESS = 0.95
MAX_INTER_CHAIN_DISTANCE = 15.0  # Reject if chains don't interact
