"""One canonical residue->AA mapping, reused everywhere."""

from tinyfold.constants import map_residue_to_aa
from tinyfold.data.parsing.dips_loader import map_residue_to_aa as dips_map
from tinyfold.data.parsing.structure_io import map_residue_to_aa as io_map
from tinyfold.data.processing.cleaning import map_modified_residue


def test_standard_modified_and_unknown():
    assert map_residue_to_aa("ALA") == "A"
    assert map_residue_to_aa("gly") == "G"  # case-insensitive
    assert map_residue_to_aa("MSE") == "M"  # modified -> parent
    assert map_residue_to_aa("ZZZ") == "X"  # unknown


def test_all_call_sites_are_the_same_function():
    # The parsers and the cleaning alias must all resolve to the one canonical impl.
    assert dips_map is map_residue_to_aa
    assert io_map is map_residue_to_aa
    assert map_modified_residue is map_residue_to_aa
