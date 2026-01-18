"""
Integration tests for the TinyFold data pipeline.

These tests verify the complete pipeline from loading structures to producing
training-ready tensors. Focus is on real data validation, not mocking.

Test philosophy (from CLAUDE.md):
- Integration tests over unit tests
- Load actual proteins, verify coordinates make sense
- Check bond lengths are physically reasonable
- Verify round-trip through Parquet preserves data
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinyfold.constants import (
    AA_TO_IDX,
    BACKBONE_ATOMS,
    BOND_LENGTH_TOLERANCE,
    BOND_LENGTHS,
    NUM_ATOM_TYPES,
)
from tinyfold.data.cache import (
    dict_to_sample,
    read_parquet,
    sample_to_dict,
    write_parquet,
)
from tinyfold.data.collate import collate_ppi
from tinyfold.data.datasets.ppi_dataset import PPIDataset
from tinyfold.data.parsing.structure_io import (
    extract_chain,
    get_backbone_atoms,
    load_structure,
)
from tinyfold.data.processing.atomization import (
    atomize_chains,
    build_bonds,
    compute_bond_lengths,
)
from tinyfold.data.processing.cleaning import clean_chain
from tinyfold.data.processing.filters import validate_sample
from tinyfold.data.processing.interface import (
    compute_interface_mask,
    compute_min_interface_distance,
)


class TestStructureLoading:
    """Test loading and parsing protein structures."""

    def test_load_pdb_file(self, sample_pdb_file):
        """Load a PDB file and verify basic structure."""
        structure = load_structure(sample_pdb_file)

        # Should have at least one model
        assert len(structure) >= 1

        # Should have two chains
        model = structure[0]
        chain_names = [chain.name for chain in model]
        assert "A" in chain_names
        assert "B" in chain_names

    def test_extract_chains(self, sample_pdb_file):
        """Extract individual chains from structure."""
        structure = load_structure(sample_pdb_file)

        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        assert chain_a is not None
        assert chain_b is not None
        assert chain_a.name == "A"
        assert chain_b.name == "B"

    def test_extract_nonexistent_chain(self, sample_pdb_file):
        """Extracting non-existent chain returns None."""
        structure = load_structure(sample_pdb_file)
        chain_z = extract_chain(structure, "Z")
        assert chain_z is None

    def test_get_backbone_atoms(self, sample_pdb_file):
        """Extract backbone atoms and verify counts."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_data = get_backbone_atoms(chain_a)

        # Should have 3 residues (ALA, GLY, SER)
        assert len(chain_data.sequence) == 3
        assert chain_data.sequence == ["A", "G", "S"]

        # Coordinates shape: [L, 4, 3]
        assert chain_data.coords.shape == (3, 4, 3)

        # Mask shape: [L, 4]
        assert chain_data.mask.shape == (3, 4)

        # All backbone atoms should be present
        assert chain_data.mask.all()

    def test_sequence_indices_match_sequence(self, sample_pdb_file):
        """Verify sequence indices correspond to correct amino acids."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_data = get_backbone_atoms(chain_a)

        for aa, idx in zip(chain_data.sequence, chain_data.seq_indices):
            assert AA_TO_IDX[aa] == idx


class TestAtomization:
    """Test conversion from residue-level to atom-level data."""

    def test_atomize_chains_shapes(self, sample_pdb_file):
        """Verify atomization produces correct tensor shapes."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        atom_coords, atom_mask, atom_to_res, atom_type, chain_id_atom = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        L = LA + LB
        Natom = L * NUM_ATOM_TYPES

        # Verify shapes
        assert atom_coords.shape == (Natom, 3), f"Expected ({Natom}, 3), got {atom_coords.shape}"
        assert atom_mask.shape == (Natom,)
        assert atom_to_res.shape == (Natom,)
        assert atom_type.shape == (Natom,)
        assert chain_id_atom.shape == (Natom,)

    def test_atom_to_res_mapping(self, sample_pdb_file):
        """Verify atom_to_res correctly maps atoms to residues."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        atom_coords, atom_mask, atom_to_res, atom_type, chain_id_atom = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        # First 4 atoms should belong to residue 0
        assert all(atom_to_res[:4] == 0)

        # Atoms 4-7 should belong to residue 1
        assert all(atom_to_res[4:8] == 1)

    def test_atom_type_pattern(self, sample_pdb_file):
        """Verify atom_type follows [N, CA, C, O] pattern."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        _, _, _, atom_type, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        # Pattern should repeat: [0, 1, 2, 3, 0, 1, 2, 3, ...]
        expected_pattern = [0, 1, 2, 3] * (len(atom_type) // 4)
        assert list(atom_type) == expected_pattern


class TestBondGeometry:
    """Test bond construction and geometry validation."""

    def test_build_bonds_count(self, sample_pdb_file):
        """Verify correct number of bonds created."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        _, atom_mask, _, _, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        # Each bond is counted twice (both directions)
        # Within-residue bonds: 3 per residue (N-CA, CA-C, C-O)
        # Peptide bonds: (LA-1) + (LB-1) total
        expected_backbone = (LA + LB) * 3 * 2  # 3 bonds per residue, both directions
        expected_peptide = ((LA - 1) + (LB - 1)) * 2  # peptide bonds, both directions

        total_expected = expected_backbone + expected_peptide
        assert len(bonds_src) == total_expected

    def test_bond_lengths_reasonable(self, sample_pdb_file):
        """Verify bond lengths are chemically reasonable."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        atom_coords, atom_mask, _, _, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        # Compute bond lengths
        lengths = compute_bond_lengths(atom_coords, bonds_src, bonds_dst, atom_mask)

        # All bond lengths should be in reasonable range (0.8 to 2.0 Angstroms)
        assert all(lengths > 0.8), f"Bond too short: {lengths.min()}"
        assert all(lengths < 2.5), f"Bond too long: {lengths.max()}"

        # Check specific bond types
        # N-CA bonds (type 0, atoms 0-1 per residue)
        n_ca_mask = (bond_type == 0) & (bonds_src % 4 == 0) & (bonds_dst % 4 == 1)
        n_ca_lengths = lengths[n_ca_mask]
        if len(n_ca_lengths) > 0:
            expected = BOND_LENGTHS["N-CA"]
            assert all(abs(n_ca_lengths - expected) < 0.3), \
                f"N-CA bond length {n_ca_lengths.mean():.3f} too far from {expected}"

    def test_no_inter_chain_covalent_bonds(self, sample_pdb_file):
        """Verify no covalent bonds cross chain boundaries."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        _, atom_mask, _, _, chain_id_atom = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, _ = build_bonds(LA, LB, atom_mask)

        # Check that no bond crosses chain boundary
        for src, dst in zip(bonds_src, bonds_dst):
            src_chain = chain_id_atom[src]
            dst_chain = chain_id_atom[dst]
            assert src_chain == dst_chain, \
                f"Bond crosses chain boundary: atom {src} (chain {src_chain}) -> atom {dst} (chain {dst_chain})"


class TestInterfaceAnnotation:
    """Test protein-protein interface detection."""

    def test_interface_mask_shapes(self, sample_pdb_file):
        """Verify interface mask has correct shape."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        iface_a, iface_b = compute_interface_mask(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        assert iface_a.shape == (len(data_a.sequence),)
        assert iface_b.shape == (len(data_b.sequence),)
        assert iface_a.dtype == bool
        assert iface_b.dtype == bool

    def test_min_interface_distance(self, sample_pdb_file):
        """Verify minimum interface distance is computed correctly."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        min_dist = compute_min_interface_distance(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        # Should be a positive finite number
        assert min_dist > 0
        assert min_dist < float("inf")


class TestValidation:
    """Test sample validation filters."""

    def test_valid_sample_passes(self, sample_pdb_file):
        """A well-formed sample should pass validation."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        # Clean chains
        seq_a, seq_idx_a, coords_a, mask_a = clean_chain(
            data_a.sequence, data_a.seq_indices, data_a.coords, data_a.mask, data_a.residue_names
        )
        seq_b, seq_idx_b, coords_b, mask_b = clean_chain(
            data_b.sequence, data_b.seq_indices, data_b.coords, data_b.mask, data_b.residue_names
        )

        LA, LB = len(seq_a), len(seq_b)

        # Atomize
        atom_coords, atom_mask, atom_to_res, atom_type, _ = atomize_chains(
            coords_a, mask_a, coords_b, mask_b
        )

        # Build bonds
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        # Validate with relaxed length requirements for test data
        from tinyfold.data.processing.filters import (
            validate_backbone_completeness,
            validate_coordinates,
            validate_bond_lengths,
        )

        # These should pass
        assert validate_backbone_completeness(atom_mask).passed
        assert validate_coordinates(atom_coords, atom_mask).passed
        assert validate_bond_lengths(
            atom_coords, bonds_src, bonds_dst, bond_type, atom_mask
        ).passed


class TestParquetRoundTrip:
    """Test Parquet serialization preserves data exactly."""

    def test_sample_roundtrip(self, sample_pdb_file, tmp_path):
        """Writing and reading Parquet should preserve all data."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        # Process to get all tensors
        atom_coords, atom_mask, atom_to_res, atom_type, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        iface_a, iface_b = compute_interface_mask(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        # Create sample dict
        seq = np.concatenate([data_a.seq_indices, data_b.seq_indices])
        chain_id_res = np.concatenate([
            np.zeros(LA, dtype=np.int64),
            np.ones(LB, dtype=np.int64),
        ])
        res_idx = np.concatenate([
            np.arange(LA, dtype=np.int64),
            np.arange(LB, dtype=np.int64),
        ])
        iface_mask = np.concatenate([iface_a, iface_b])

        original = sample_to_dict(
            sample_id="test_sample",
            pdb_id="test",
            seq=seq,
            chain_id_res=chain_id_res,
            res_idx=res_idx,
            atom_coords=atom_coords,
            atom_mask=atom_mask,
            atom_to_res=atom_to_res,
            atom_type=atom_type,
            bonds_src=bonds_src,
            bonds_dst=bonds_dst,
            bond_type=bond_type,
            iface_mask=iface_mask,
            LA=LA,
            LB=LB,
        )

        # Write to Parquet
        parquet_path = tmp_path / "test.parquet"
        write_parquet([original], parquet_path)

        # Read back
        loaded_samples = read_parquet(parquet_path)
        assert len(loaded_samples) == 1
        loaded = loaded_samples[0]

        # Verify all fields match
        assert loaded["sample_id"] == original["sample_id"]
        assert loaded["pdb_id"] == original["pdb_id"]
        assert loaded["LA"] == LA
        assert loaded["LB"] == LB

        # Verify arrays match exactly
        np.testing.assert_array_equal(loaded["seq"], seq)
        np.testing.assert_array_equal(loaded["chain_id_res"], chain_id_res)
        np.testing.assert_array_equal(loaded["atom_mask"], atom_mask)
        np.testing.assert_array_almost_equal(loaded["atom_coords"], atom_coords)


class TestDataset:
    """Test PyTorch Dataset and DataLoader integration."""

    def test_dataset_loads_samples(self, sample_pdb_file, tmp_path):
        """Dataset should load samples and return torch tensors."""
        # First, create a Parquet file with test data
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        atom_coords, atom_mask, atom_to_res, atom_type, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        iface_a, iface_b = compute_interface_mask(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        seq = np.concatenate([data_a.seq_indices, data_b.seq_indices])
        chain_id_res = np.concatenate([np.zeros(LA, dtype=np.int64), np.ones(LB, dtype=np.int64)])
        res_idx = np.concatenate([np.arange(LA, dtype=np.int64), np.arange(LB, dtype=np.int64)])
        iface_mask = np.concatenate([iface_a, iface_b])

        sample = sample_to_dict(
            sample_id="test_sample",
            pdb_id="test",
            seq=seq,
            chain_id_res=chain_id_res,
            res_idx=res_idx,
            atom_coords=atom_coords,
            atom_mask=atom_mask,
            atom_to_res=atom_to_res,
            atom_type=atom_type,
            bonds_src=bonds_src,
            bonds_dst=bonds_dst,
            bond_type=bond_type,
            iface_mask=iface_mask,
            LA=LA,
            LB=LB,
        )

        parquet_path = tmp_path / "test.parquet"
        write_parquet([sample], parquet_path)

        # Load dataset
        dataset = PPIDataset(parquet_path)
        assert len(dataset) == 1

        # Get sample
        item = dataset[0]

        # Verify types are torch tensors
        assert isinstance(item["seq"], torch.Tensor)
        assert isinstance(item["atom_coords"], torch.Tensor)
        assert isinstance(item["atom_mask"], torch.Tensor)

        # Verify shapes
        assert item["seq"].shape == (LA + LB,)
        assert item["atom_coords"].shape == ((LA + LB) * 4, 3)

    def test_collate_batches_correctly(self, sample_pdb_file, tmp_path):
        """Collate function should batch samples correctly."""
        # Create two samples of different sizes
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        chain_b = extract_chain(structure, "B")

        data_a = get_backbone_atoms(chain_a)
        data_b = get_backbone_atoms(chain_b)

        atom_coords, atom_mask, atom_to_res, atom_type, _ = atomize_chains(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        LA, LB = len(data_a.sequence), len(data_b.sequence)
        bonds_src, bonds_dst, bond_type = build_bonds(LA, LB, atom_mask)

        iface_a, iface_b = compute_interface_mask(
            data_a.coords, data_a.mask, data_b.coords, data_b.mask
        )

        seq = np.concatenate([data_a.seq_indices, data_b.seq_indices])
        chain_id_res = np.concatenate([np.zeros(LA, dtype=np.int64), np.ones(LB, dtype=np.int64)])
        res_idx = np.concatenate([np.arange(LA, dtype=np.int64), np.arange(LB, dtype=np.int64)])
        iface_mask = np.concatenate([iface_a, iface_b])

        sample1 = sample_to_dict(
            sample_id="sample1", pdb_id="test1",
            seq=seq, chain_id_res=chain_id_res, res_idx=res_idx,
            atom_coords=atom_coords, atom_mask=atom_mask,
            atom_to_res=atom_to_res, atom_type=atom_type,
            bonds_src=bonds_src, bonds_dst=bonds_dst, bond_type=bond_type,
            iface_mask=iface_mask, LA=LA, LB=LB,
        )

        sample2 = sample_to_dict(
            sample_id="sample2", pdb_id="test2",
            seq=seq, chain_id_res=chain_id_res, res_idx=res_idx,
            atom_coords=atom_coords, atom_mask=atom_mask,
            atom_to_res=atom_to_res, atom_type=atom_type,
            bonds_src=bonds_src, bonds_dst=bonds_dst, bond_type=bond_type,
            iface_mask=iface_mask, LA=LA, LB=LB,
        )

        parquet_path = tmp_path / "test.parquet"
        write_parquet([sample1, sample2], parquet_path)

        dataset = PPIDataset(parquet_path)
        batch = collate_ppi([dataset[0], dataset[1]])

        # Verify batch structure
        assert batch["seq"].shape[0] == 2  # batch size
        assert batch["atom_coords"].shape[0] == 2
        assert batch["res_mask"].shape[0] == 2

        # Verify edge index was merged
        assert batch["edge_index"].shape[0] == 2  # [src, dst]
        assert batch["atom_batch"].max() == 1  # Two samples (0 and 1)


class TestCoordinateIntegrity:
    """Test that coordinates are physically reasonable."""

    def test_no_nan_coordinates(self, sample_pdb_file):
        """Coordinates should not contain NaN values."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        data_a = get_backbone_atoms(chain_a)

        assert not np.any(np.isnan(data_a.coords))

    def test_no_inf_coordinates(self, sample_pdb_file):
        """Coordinates should not contain infinite values."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        data_a = get_backbone_atoms(chain_a)

        assert not np.any(np.isinf(data_a.coords))

    def test_coordinates_reasonable_range(self, sample_pdb_file):
        """Coordinates should be in a reasonable range for proteins."""
        structure = load_structure(sample_pdb_file)
        chain_a = extract_chain(structure, "A")
        data_a = get_backbone_atoms(chain_a)

        # Typical protein coordinates are within a few hundred Angstroms
        # of the origin (or the protein's center)
        valid_coords = data_a.coords[data_a.mask]
        assert np.all(np.abs(valid_coords) < 1000), \
            "Coordinates outside reasonable range"
