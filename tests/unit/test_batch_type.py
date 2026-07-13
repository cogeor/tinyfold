"""The Batch TypedDict documents the residue-batch shape used across the model."""

from tinyfold.types import Batch


def test_batch_declares_core_keys():
    keys = set(Batch.__annotations__)
    # The tensors the live model always reads.
    for k in ("centroids", "coords_res", "aa_seq", "chain_ids", "res_idx", "mask_res"):
        assert k in keys
    # Optional feature keys are declared too (total=False).
    for k in ("esm_embed", "template_coords_res"):
        assert k in keys


def test_batch_is_total_false():
    # Feature keys are optional, so no key is "required".
    assert Batch.__total__ is False
