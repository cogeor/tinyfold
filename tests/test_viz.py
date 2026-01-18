"""Tests for visualization module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tinyfold.viz.io.structure_writer import coords_to_pdb_string
from tinyfold.viz.mapping.atom_schema import AtomSchema
from tinyfold.viz.metrics.align import kabsch_align
from tinyfold.viz.metrics.contacts import contact_map_CA, contact_metrics
from tinyfold.viz.metrics.rmsd import backbone_rmsd, compute_rmsd


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_structure():
    """Create a minimal protein structure for testing."""
    L = 10  # 5 residues per chain
    N_atom = L * 4

    # Simple coordinates
    coords = np.zeros((N_atom, 3))
    for i in range(L):
        base_x = (i % 5) * 3.8  # ~3.8 Å between residues
        base_y = 0 if i < 5 else 10  # Chain B offset by 10 Å
        for j in range(4):
            atom_idx = i * 4 + j
            coords[atom_idx] = [base_x + j * 0.5, base_y + j * 0.3, j * 0.2]

    atom_to_res = np.arange(L).repeat(4)
    atom_type = np.tile([0, 1, 2, 3], L)  # N, CA, C, O
    chain_id_res = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    res_idx = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    seq = np.zeros(L, dtype=np.int64)  # All ALA
    atom_mask = np.ones(N_atom, dtype=bool)

    return {
        "coords": coords,
        "atom_to_res": atom_to_res,
        "atom_type": atom_type,
        "chain_id_res": chain_id_res,
        "res_idx": res_idx,
        "seq": seq,
        "atom_mask": atom_mask,
        "L": L,
        "N_atom": N_atom,
    }


# ============================================================================
# Kabsch Alignment Tests
# ============================================================================


class TestKabschAlign:
    """Tests for Kabsch alignment."""

    def test_identity_alignment(self):
        """Aligning identical structures should give zero RMSD."""
        coords = np.random.randn(100, 3)
        aligned, R, t = kabsch_align(coords, coords)

        rmsd = compute_rmsd(aligned, coords)
        assert rmsd < 1e-10, f"Identity alignment RMSD should be ~0, got {rmsd}"

    def test_translation_recovery(self):
        """Should correctly align translated structures."""
        coords = np.random.randn(50, 3)
        translation = np.array([10, -5, 3])
        translated = coords + translation

        aligned, R, t = kabsch_align(translated, coords)
        rmsd = compute_rmsd(aligned, coords)

        assert rmsd < 1e-10, f"Translation alignment RMSD should be ~0, got {rmsd}"

    def test_rotation_recovery(self):
        """Should correctly align rotated structures."""
        coords = np.random.randn(50, 3)

        # Simple 90-degree rotation around Z axis
        R_true = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        rotated = (R_true @ coords.T).T

        aligned, R, t = kabsch_align(rotated, coords)
        rmsd = compute_rmsd(aligned, coords)

        assert rmsd < 1e-10, f"Rotation alignment RMSD should be ~0, got {rmsd}"

    def test_with_mask(self):
        """Alignment with mask should only use masked atoms."""
        coords = np.random.randn(100, 3)

        # Create mask for first 50 atoms
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True

        # Translate only first 50 atoms
        modified = coords.copy()
        modified[:50] += 5

        aligned, _, _ = kabsch_align(modified, coords, mask=mask)

        # First 50 should be well-aligned
        rmsd_masked = compute_rmsd(aligned, coords, mask)
        assert rmsd_masked < 1e-10


# ============================================================================
# RMSD Tests
# ============================================================================


class TestRMSD:
    """Tests for RMSD computation."""

    def test_backbone_rmsd_shapes(self, sample_structure):
        """Test that backbone_rmsd returns expected keys."""
        s = sample_structure
        ref = s["coords"]
        pred = ref + np.random.randn(*ref.shape) * 0.5  # Add noise

        metrics = backbone_rmsd(
            pred, ref,
            s["atom_to_res"], s["chain_id_res"],
            s["atom_mask"]
        )

        expected_keys = ["rmsd_complex", "rmsd_chain_a", "rmsd_chain_b", "lrmsd", "irmsd"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_rmsd_identical(self, sample_structure):
        """Identical structures should have zero RMSD."""
        s = sample_structure
        metrics = backbone_rmsd(
            s["coords"], s["coords"],
            s["atom_to_res"], s["chain_id_res"],
            s["atom_mask"]
        )

        assert metrics["rmsd_complex"] < 1e-10
        assert metrics["rmsd_chain_a"] < 1e-10
        assert metrics["rmsd_chain_b"] < 1e-10


# ============================================================================
# Contact Map Tests
# ============================================================================


class TestContactMap:
    """Tests for contact map computation."""

    def test_contact_map_shape(self, sample_structure):
        """Contact map should have correct shape."""
        s = sample_structure
        A_idx, B_idx, contact_matrix = contact_map_CA(
            s["coords"], s["atom_type"], s["atom_to_res"], s["chain_id_res"]
        )

        LA = (s["chain_id_res"] == 0).sum()
        LB = (s["chain_id_res"] == 1).sum()

        assert contact_matrix.shape == (LA, LB)

    def test_contact_metrics_perfect(self):
        """Perfect prediction should have P=R=F1=1."""
        contacts = np.array([[True, False], [False, True]])

        metrics = contact_metrics(contacts, contacts)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_contact_metrics_empty(self):
        """No contacts should handle gracefully."""
        pred = np.zeros((5, 5), dtype=bool)
        ref = np.zeros((5, 5), dtype=bool)

        metrics = contact_metrics(pred, ref)

        # With no contacts, precision and recall are 0/0 -> 0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


# ============================================================================
# PDB Writer Tests
# ============================================================================


class TestPDBWriter:
    """Tests for PDB writing."""

    def test_pdb_string_format(self, sample_structure):
        """PDB string should have valid format."""
        s = sample_structure
        pdb_str = coords_to_pdb_string(
            s["coords"], s["atom_to_res"], s["atom_type"],
            s["chain_id_res"], s["res_idx"], s["seq"], s["atom_mask"]
        )

        lines = pdb_str.strip().split("\n")

        # Should have ATOM lines
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        assert len(atom_lines) == s["N_atom"]

        # Should have TER and END
        assert any(l.startswith("TER") for l in lines)
        assert lines[-1] == "END"

    def test_pdb_chain_labels(self, sample_structure):
        """PDB should have correct chain labels."""
        s = sample_structure
        pdb_str = coords_to_pdb_string(
            s["coords"], s["atom_to_res"], s["atom_type"],
            s["chain_id_res"], s["res_idx"]
        )

        lines = pdb_str.split("\n")
        atom_lines = [l for l in lines if l.startswith("ATOM")]

        # Check chain IDs (column 22 in PDB format)
        chains = set(l[21] for l in atom_lines)
        assert "A" in chains
        assert "B" in chains

    def test_pdb_atom_names(self, sample_structure):
        """PDB should have correct atom names."""
        s = sample_structure
        schema = AtomSchema()
        pdb_str = coords_to_pdb_string(
            s["coords"], s["atom_to_res"], s["atom_type"],
            s["chain_id_res"], s["res_idx"], schema=schema
        )

        # Check for backbone atoms
        assert " N  " in pdb_str or " N " in pdb_str
        assert " CA " in pdb_str
        assert " C  " in pdb_str or " C " in pdb_str
        assert " O  " in pdb_str or " O " in pdb_str

    def test_pdb_atom_count_with_mask(self, sample_structure):
        """Masked atoms should be excluded from PDB."""
        s = sample_structure
        mask = s["atom_mask"].copy()
        mask[::2] = False  # Mask every other atom

        pdb_str = coords_to_pdb_string(
            s["coords"], s["atom_to_res"], s["atom_type"],
            s["chain_id_res"], s["res_idx"], atom_mask=mask
        )

        atom_lines = [l for l in pdb_str.split("\n") if l.startswith("ATOM")]
        assert len(atom_lines) == mask.sum()


# ============================================================================
# Integration Test
# ============================================================================


class TestReportGeneration:
    """Integration test for report generation."""

    def test_make_report_creates_files(self, sample_structure):
        """make_report should create expected output files."""
        from tinyfold.viz.report.html_report import make_report

        s = sample_structure
        ref = s["coords"]
        pred = ref + np.random.randn(*ref.shape) * 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = make_report(
                sample_id="test_sample",
                pred_xyz=pred,
                ref_xyz=ref,
                atom_to_res=s["atom_to_res"],
                atom_type=s["atom_type"],
                chain_id_res=s["chain_id_res"],
                res_idx=s["res_idx"],
                out_dir=tmpdir,
                seq=s["seq"],
                atom_mask=s["atom_mask"],
            )

            out_dir = Path(tmpdir) / "test_sample"

            # Check files exist
            assert (out_dir / "report.html").exists()
            assert (out_dir / "viewer.html").exists()
            assert (out_dir / "pred.pdb").exists()
            assert (out_dir / "ref.pdb").exists()
            assert (out_dir / "plots" / "contact_map.png").exists()
            assert (out_dir / "plots" / "rmsd_comparison.png").exists()


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
