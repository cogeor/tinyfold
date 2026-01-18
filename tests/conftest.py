"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_pdb_content() -> str:
    """Minimal valid PDB file content for a small complex."""
    # Two small chains (3 residues each) with backbone atoms
    return """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.320   1.540   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.970   2.840   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.480   2.740   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.080   1.670   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.080   3.870   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.520   3.990   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.070   5.410   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.310   6.380   0.000  1.00  0.00           O
ATOM     13  N   VAL B   1       5.000  10.000   0.000  1.00  0.00           N
ATOM     14  CA  VAL B   1       6.458  10.000   0.000  1.00  0.00           C
ATOM     15  C   VAL B   1       7.009  11.420   0.000  1.00  0.00           C
ATOM     16  O   VAL B   1       6.251  12.390   0.000  1.00  0.00           O
ATOM     17  N   LEU B   2       8.320  11.540   0.000  1.00  0.00           N
ATOM     18  CA  LEU B   2       8.970  12.840   0.000  1.00  0.00           C
ATOM     19  C   LEU B   2      10.480  12.740   0.000  1.00  0.00           C
ATOM     20  O   LEU B   2      11.080  11.670   0.000  1.00  0.00           O
ATOM     21  N   ILE B   3      11.080  13.870   0.000  1.00  0.00           N
ATOM     22  CA  ILE B   3      12.520  13.990   0.000  1.00  0.00           C
ATOM     23  C   ILE B   3      13.070  15.410   0.000  1.00  0.00           C
ATOM     24  O   ILE B   3      12.310  16.380   0.000  1.00  0.00           O
END
"""


@pytest.fixture
def sample_pdb_file(tmp_path, sample_pdb_content) -> Path:
    """Create a temporary PDB file."""
    pdb_path = tmp_path / "test_complex.pdb"
    pdb_path.write_text(sample_pdb_content)
    return pdb_path
